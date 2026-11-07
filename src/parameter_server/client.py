import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

# The generated grpc file imports `parameter_server_pb2` as a top-level module.
_GENERATED = Path(__file__).resolve().parents[1] / "generated"
if str(_GENERATED) not in sys.path:
    sys.path.insert(0, str(_GENERATED))

import parameter_server_pb2 as ps_pb2
import parameter_server_pb2_grpc as ps_pb2_grpc


class Task:
    """Handle returned immediately by pull and push. wait() blocks on the RPC."""

    def __init__(self, future, task_id):
        self._future = future
        self.task_id = task_id

    def wait(self, timeout=None):
        # A timeout leaves the RPC running. Calling wait again is allowed.
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()


class ParameterServerClient:
    """Worker-facing client for one parameter server."""

    def __init__(self, target: str, worker_id: int = 0):
        self.channel = grpc.insecure_channel(target)
        self.stub = ps_pb2_grpc.ParameterServerStub(self.channel)
        self.worker_id = worker_id
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4)

    def close(self):
        self._pool.shutdown(wait=False)
        self.channel.close()

    def _next_task_id(self) -> ps_pb2.TaskId:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        return ps_pb2.TaskId(worker_id=self.worker_id, seq=seq)

    def _selector(self, keys, begin, end):
        selector = ps_pb2.KeySelector()
        if keys is not None:
            selector.sparse.keys.extend(keys)
            return selector, len(list(keys))
        if begin is not None and end is not None:
            if end < begin:
                raise ValueError("dense range end is before begin")
            selector.dense.begin = begin
            selector.dense.end = end
            return selector, end - begin
        raise ValueError("pass keys, or begin and end")

    def pull(self, keys=None, begin=None, end=None, deps=None) -> Task:
        """Return a task whose result is (flat weights, width), in request order."""
        selector, _ = self._selector(keys, begin, end)
        task_id = self._next_task_id()
        request = ps_pb2.PullRequest(target=selector, task_id=task_id)
        for dep in deps or []:
            request.deps.add().CopyFrom(dep.task_id)

        def _call():
            response = self.stub.Pull(request)
            return list(response.weights.data), response.weights.width

        return Task(self._pool.submit(_call), task_id)

    def push(self, gradients, keys=None, begin=None, end=None, width: int = 1, deps=None) -> Task:
        """Return a task whose result is the server's TaskId once SGD has been applied."""
        selector, n_keys = self._selector(keys, begin, end)
        values = list(gradients)
        if width < 1:
            raise ValueError("width must be >= 1")
        if len(values) != n_keys * width:
            raise ValueError(f"expected {n_keys * width} gradient values, got {len(values)}")

        task_id = self._next_task_id()
        request = ps_pb2.PushRequest(
            target=selector,
            gradients=ps_pb2.Tensor(width=width, data=values),
            task_id=task_id,
        )
        for dep in deps or []:
            request.deps.add().CopyFrom(dep.task_id)

        def _call():
            response = self.stub.Push(request)
            return response.task_id

        return Task(self._pool.submit(_call), task_id)
