# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.entrypoints.cli.serve import _get_headless_executor_class
from vllm.v1.executor import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor


def test_multi_edge_headless_uses_configured_executor() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_edge_cloud=True, num_edges=2)
    )
    configured_executor = type("ConfiguredExecutor", (), {})

    with patch.object(
        Executor,
        "get_class",
        return_value=configured_executor,
    ) as get_class:
        executor_class = _get_headless_executor_class(vllm_config)

    assert executor_class is configured_executor
    get_class.assert_called_once_with(vllm_config)


def test_regular_headless_keeps_multiproc_executor() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_edge_cloud=False, num_edges=1)
    )

    assert _get_headless_executor_class(vllm_config) is MultiprocExecutor
