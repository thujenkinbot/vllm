import pytest

from vllm.config import ParallelConfig
from vllm.v1.core.sched.output import SchedulerOutput


def test_multi_edge_parallel_config_for_edge() -> None:
    config = ParallelConfig(
        enable_edge_cloud=True,
        cloud_npu_count=8,
        num_edges=2,
        is_edge_node=True,
        nnodes=3,
        node_rank=0,
        distributed_executor_backend="mp",
    )

    assert config.edge_npu_count == 2
    assert config.cloud_npu_count == 8
    assert config.world_size == 10
    assert config.tensor_parallel_size == 1
    assert config.pipeline_parallel_size == 2
    assert config.local_world_size == 1
    assert config.edge_cloud_global_start_rank == 0


def test_second_edge_uses_node_rank_as_global_start_rank() -> None:
    config = ParallelConfig(
        enable_edge_cloud=True,
        cloud_npu_count=4,
        num_edges=2,
        is_edge_node=True,
        nnodes=3,
        node_rank=1,
        distributed_executor_backend="mp",
    )

    assert config.edge_cloud_global_start_rank == 1


def test_multi_edge_parallel_config_for_cloud() -> None:
    config = ParallelConfig(
        enable_edge_cloud=True,
        cloud_npu_count=8,
        num_edges=2,
        is_edge_node=False,
        nnodes=3,
        node_rank=2,
        distributed_executor_backend="mp",
    )

    assert config.world_size == 10
    assert config.tensor_parallel_size == 8
    assert config.pipeline_parallel_size == 2
    assert config.local_world_size == 8
    assert config.edge_cloud_global_start_rank == 2


@pytest.mark.parametrize("data_parallel_size", [2, 4])
def test_multi_edge_rejects_data_parallelism(data_parallel_size: int) -> None:
    with pytest.raises(ValueError, match="data_parallel_size must be 1"):
        ParallelConfig(
            enable_edge_cloud=True,
            cloud_npu_count=8,
            num_edges=2,
            data_parallel_size=data_parallel_size,
            nnodes=3,
            node_rank=2,
            distributed_executor_backend="mp",
        )


def test_scheduler_output_defaults_to_first_edge() -> None:
    assert SchedulerOutput.make_empty().edge_id == 0
