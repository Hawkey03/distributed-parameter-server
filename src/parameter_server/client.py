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
        """Initialize a Task with a future and a task ID.
        
        Args:
            future: A future object that will be used to store the result of the task.
            task_id: A task ID that will be used to identify the task.
        
        - _future tracks the background operation inside this client process.
        - task_id identifies the operation in protobuf requests, including dependency lists.
        """
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
        """This stops the executor from accepting new work and closes the gRPC channel."""
        self._pool.shutdown(wait=False)
        self.channel.close()

    def _next_task_id(self) -> ps_pb2.TaskId:
        """Generate a new task ID.

        The lock makes counter updates safe if multiple threads share this client.
        
        Returns:
            A TaskId object with the worker ID and sequence number.
        """
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        return ps_pb2.TaskId(worker_id=self.worker_id, seq=seq)

    def _selector(self, keys, begin, end):
        """This helper converts the public Python arguments into a protobuf KeySelector.

        Args:
            keys: A list of keys to select.
            begin: The beginning of the dense range.
            end: The end of the dense range.

        Returns:
            A tuple of (selector, number_of_selected_keys).
        """
        selector = ps_pb2.KeySelector()
        # Sparse access
        if keys is not None:
            selector.sparse.keys.extend(keys)
            return selector, len(list(keys))
        # Dense access
        if begin is not None and end is not None:
            if end < begin:
                raise ValueError("dense range end is before begin")
            selector.dense.begin = begin
            selector.dense.end = end
            return selector, end - begin
        raise ValueError("pass keys, or begin and end")

    def pull(self, keys=None, begin=None, end=None, deps=None) -> Task:
        """Return a task whose result is (flat weights, width), in request order.

        Args:
            keys: A list of keys to select.
            begin: The beginning of the dense range.
            end: The end of the dense range.
            deps: A list of dependencies.
        """

        # construct request
        selector, _ = self._selector(keys, begin, end)
        task_id = self._next_task_id()
        request = ps_pb2.PullRequest(target=selector, task_id=task_id)
        # add dependencies
        for dep in deps or []:
            request.deps.add().CopyFrom(dep.task_id)
        # defining background operation
        def _call():
            response = self.stub.Pull(request)
            return list(response.weights.data), response.weights.width
        # schedule function with a thread pool
        return Task(self._pool.submit(_call), task_id)

    def push(self, gradients, keys=None, begin=None, end=None, width: int = 1, deps=None) -> Task:
        """Return a task whose result is the server's TaskId once SGD has been applied."""

        # selects keys and materializes the gradients
        selector, n_keys = self._selector(keys, begin, end)
        values = list(gradients)
        # validate shapes
        if width < 1:
            raise ValueError("width must be >= 1")
        if len(values) != n_keys * width:
            raise ValueError(f"expected {n_keys * width} gradient values, got {len(values)}")

        # build request
        task_id = self._next_task_id()
        request = ps_pb2.PushRequest(
            target=selector,
            gradients=ps_pb2.Tensor(width=width, data=values),
            task_id=task_id,
        )
        # copy over dependencies
        for dep in deps or []:
            request.deps.add().CopyFrom(dep.task_id)

        # defining background operation
        def _call():
            response = self.stub.Push(request)
            return response.task_id
        # schedule function with a thread pool
        return Task(self._pool.submit(_call), task_id)
