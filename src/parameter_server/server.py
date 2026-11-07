import logging
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


def _reject(context, details: str):
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(details)


def _keys_for(request):
    """Return (keys, error). Dense ranges are half-open [begin, end)."""
    kind = request.target.WhichOneof("target")
    if kind == "sparse":
        return list(request.target.sparse.keys), None
    if kind == "dense":
        begin = request.target.dense.begin
        end = request.target.dense.end
        if end < begin:
            return None, "dense range end is before begin"
        return range(begin, end), None
    return None, "request is missing a sparse key list or a dense range"


class ParameterServer(ps_pb2_grpc.ParameterServerServicer):
    """Single-server weight store. Push applies SGD under a lock."""

    def __init__(self, learning_rate: float = 0.01):
        # key -> vector of floats. Missing keys read as zeros of self.width.
        self.weights = {}
        self.width = 1
        self.learning_rate = learning_rate
        self.lock = threading.Lock()

    def _vector(self, key: int):
        value = self.weights.get(key)
        if value is None or len(value) != self.width:
            return [0.0] * self.width
        return value

    def Pull(self, request, context):
        keys, error = _keys_for(request)
        if error:
            _reject(context, error)
            return ps_pb2.PullResponse()

        # deps are carried on the request for later, and are not enforced yet.
        response = ps_pb2.PullResponse()
        with self.lock:
            response.weights.width = self.width
            for key in keys:
                response.weights.data.extend(self._vector(key))
                if request.target.WhichOneof("target") == "sparse":
                    response.keys.append(key)
        return response

    def Push(self, request, context):
        keys, error = _keys_for(request)
        if error:
            _reject(context, error)
            return ps_pb2.PushResponse()

        width = request.gradients.width
        data = list(request.gradients.data)
        n_keys = len(keys)
        if width < 1:
            _reject(context, "gradient width must be >= 1")
            return ps_pb2.PushResponse()
        if len(data) != n_keys * width:
            _reject(
                context,
                f"expected {n_keys * width} gradient values, got {len(data)}",
            )
            return ps_pb2.PushResponse()

        with self.lock:
            if self.weights and width != self.width:
                _reject(context, f"gradient width {width} does not match stored width {self.width}")
                return ps_pb2.PushResponse()
            self.width = width
            for i, key in enumerate(keys):
                start = i * width
                current = self._vector(key)
                self.weights[key] = [
                    current[j] - self.learning_rate * data[start + j] for j in range(width)
                ]

        response = ps_pb2.PushResponse()
        response.task_id.CopyFrom(request.task_id)
        return response


def serve(port: int):
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    ps_pb2_grpc.add_ParameterServerServicer_to_server(ParameterServer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


def main(port: int = 50051):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting parameter server on port %s", port)
    serve(port)


if __name__ == "__main__":
    main()
