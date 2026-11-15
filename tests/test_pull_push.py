"""One in-process server. Missing keys read as zeros, and a push applies SGD."""

from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

import parameter_server_pb2_grpc as pb2_grpc
from parameter_server.client import ParameterServerClient
from parameter_server.server import ParameterServer
from worker import Worker


@pytest.fixture
def running_server():
    servicer = ParameterServer(learning_rate=0.1, width=1)
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_ParameterServerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield servicer, f"127.0.0.1:{port}"
    server.stop(0)


def test_pull_of_missing_keys_then_push_applies_sgd(running_server):
    servicer, target = running_server
    client = ParameterServerClient(target=target, worker_id=1)
    try:
        weights, width = client.pull(keys=[0, 2]).wait()
        assert width == 1
        assert weights == [0.0, 0.0]

        ack = client.push([1.0, 3.0], keys=[0, 2], width=1).wait()
        assert ack.worker_id == 1
        assert ack.seq == 2

        weights, width = client.pull(keys=[0, 2]).wait()
        assert width == 1
        assert weights == pytest.approx([-0.1, -0.3])
        assert servicer.weights[0] == pytest.approx([-0.1])
        assert servicer.weights[2] == pytest.approx([-0.3])
    finally:
        client.close()


def test_one_worker_iteration_pushes_fixed_gradients(running_server):
    servicer, target = running_server
    worker = Worker(worker_id=3, target=target, iterations=1, width=1)
    worker.workload.num_keys = 4
    worker.workload.access_mode = "dense"
    worker.workload.range_size = 4
    worker.workload.compute_median_ms = 0.1
    worker.workload.gradient_type = "fixed"

    worker.run()

    assert worker.log == [
        "Iteration 1: pull submitted",
        "Iteration 1: pull completed",
        "Iteration 1: push submitted",
        "Iteration 1: push completed",
    ]
    # Same formula as Workload._fixed_gradients, then w = 0 - learning_rate * g.
    for key in range(4):
        gradient = (3 + 1) * 1e-2 + 1 * 1e-4 + key * 1e-6
        assert servicer.weights[key] == pytest.approx([-0.1 * gradient])
