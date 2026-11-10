import sys
from pathlib import Path
import grpc
import time

# harness/ is not a package. The client lives under src/.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HARNESS = Path(__file__).resolve().parent
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from parameter_server.client import ParameterServerClient
from workload import Workload


# One thread or process per emulated worker. It talks to the server only
# through ParameterServerClient. Do not construct a ParameterServer here.


class Worker:
    """Emulated training worker.

    Own: worker_id, one client, an iteration counter, and the push handles
    that are still outstanding.

    Do not own: the server, the learning rate, or how keys and gradients are
    sampled. Those come from the server process and the workload generator.
    """

    def __init__(self, worker_id: int, target: str = "localhost:50051", iterations: int = 1, width: int = 1):
        self.worker_id = worker_id
        self.client = ParameterServerClient(target=target, worker_id=worker_id)
        self.iterations = iterations
        self.width = width
        self.workload = Workload(iterations=iterations, width=width, worker_id=worker_id)
        self.tau = 0
        self.pushes = {}
        self.log = []
        self.baseline = False

    def run(self):
        """Run iterations 1..T, then wait for any push still outstanding.

        Close the client when the worker stops, including when a call fails.
        """
        try:
            for t in range(1, self.iterations + 1):
                try:
                    self._iteration(t)
                except grpc.RpcError as e:
                    self.log.append(f"Iteration {t}: {e}")
                    break
            self._drain()
        finally:
            self.client.close()

    def _iteration(self, t: int):
        """One pull-compute-push cycle.

        1. Ask the workload for this iteration's keys, or a dense [begin, end).
        2. Apply the ordering rule: iteration t may start only after
           iteration t - tau - 1 has completed. tau = 0 waits for the
           previous push.
        3. client.pull(...) returns a Task immediately. Record that as submit.
           task.wait() returns (flat weights, width). Record that as complete.
        4. Sleep for the workload's simulated compute time.
        5. Build gradients for the same keys. Their length is n_keys * width.
        6. client.push(..., deps=[pull_task]) returns immediately. Remember
           this handle under iteration t, and record the submit.
        7. For the baseline, wait() on that push and record the complete
           before returning. Later, skip this wait and let step 2 do it.
        """
        # 1. Sparse Selection has keys; dense Selection has begin and end.
        step = self.workload.selection(t)
        # 2. tau = 0 waits for the previous push. A negative index is a no-op.
        self._wait_for_iteration(t - self.tau - 1)
        # 3. pull returns a handle. The weights come from wait(), not from the handle.
        pull = self.client.pull(keys=step.keys, begin=step.begin, end=step.end)
        self.log.append(f"Iteration {t}: pull submitted")
        weights, width = pull.wait()
        self.log.append(f"Iteration {t}: pull completed")
        # 4. Compute happens after the weights are local.
        time.sleep(self.workload.compute_delay_s(t))
        # 5. Same selection the pull used, so the gradient length matches.
        gradients = self.workload.gradients(t, step, weights)
        # 6. Pass the same keys or range. deps is keyword-only here.
        push = self.client.push(
            gradients,
            keys=step.keys,
            begin=step.begin,
            end=step.end,
            width=width,
            deps=[pull],
        )
        self.pushes[t] = push
        self.log.append(f"Iteration {t}: push submitted")
        # 7. Baseline waits here. Otherwise the next iteration, or _drain, waits.
        if self.baseline:
            push.wait()
            self.log.append(f"Iteration {t}: push completed")
            del self.pushes[t]
            return push
        return push

    def _wait_for_iteration(self, j: int):
        """If j >= 1, wait for the push recorded for iteration j.

        A second wait on a task that already finished should not record
        completion twice.
        """
        if j < 1:
            return
        push = self.pushes.get(j)
        if push is None:
            return
        push.wait()
        self.log.append(f"Iteration {j}: push completed")
        del self.pushes[j]

    def _drain(self):
        """Wait for every push that has been submitted but not completed."""
        for j in list(self.pushes):
            self._wait_for_iteration(j)
