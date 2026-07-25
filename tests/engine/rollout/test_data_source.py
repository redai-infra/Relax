# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Integration tests for data_source.py and eager Dataset global-slice
semantics.

Extracted from test_streaming_dataset.py during tests/ directory
restructuring.

Run with: pytest tests/engine/rollout/test_data_source.py -v
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.engine.rollout.data_source import RolloutDataSource
from relax.utils.types import Sample


def _make_preflight_data_source(
    samples,
    *,
    rm_type=None,
    reward_route=None,
    use_streaming=False,
    prompt_data=None,
):
    data_source = RolloutDataSource.__new__(RolloutDataSource)
    data_source.args = SimpleNamespace(
        custom_rm_path=None,
        label_key="label",
        metadata_key="metadata",
        prompt_data=prompt_data,
        rm_type=rm_type,
        reward_route=reward_route,
    )
    data_source._use_streaming = use_streaming
    data_source.dataset = SimpleNamespace(samples=samples)
    return data_source


class TestRewardRoutePreflight:
    def test_eager_dataset_reports_mixed_reward_assignments(self):
        data_source = _make_preflight_data_source(
            [
                Sample(label="42", metadata={"rm_type": "math"}),
                Sample(label="<answer>B</answer>", metadata={}),
            ]
        )

        report = data_source.validate_reward_routes()

        assert report["total"] == 2
        assert report["assignments"] == {"label/multiple_choice": 1, "metadata/math": 1}
        assert report["unresolved_count"] == 0

    def test_eager_dataset_blocks_unresolved_records(self):
        data_source = _make_preflight_data_source([Sample(label="unsupported", metadata={})])

        with pytest.raises(ValueError, match="1 sample.*no unambiguous reward assignment"):
            data_source.validate_reward_routes()

    def test_unknown_metadata_uses_global_fallback(self):
        data_source = _make_preflight_data_source(
            [Sample(label="unsupported", metadata={"rm_type": "unknown"})],
            rm_type="math",
        )

        report = data_source.validate_reward_routes()

        assert report["assignments"] == {"global/math": 1}
        assert report["fallback_count"] == 1

    def test_yaml_config_can_place_global_rm_type_before_label(self):
        data_source = _make_preflight_data_source(
            [Sample(label="42", metadata={})],
            rm_type="openr1mm",
            reward_route={"priority": ["metadata", "rm_type", "label"]},
        )

        report = data_source.validate_reward_routes()

        assert report["assignments"] == {"global/openr1mm": 1}
        assert report["fallback_count"] == 0

    def test_streaming_preflight_scans_raw_route_fields(self):
        rows = [
            {"text": "math", "label": "42", "metadata": {"rm_type": "math"}},
            {"text": "choice", "label": "<answer>C</answer>", "metadata": {}},
        ]
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as stream:
                path = stream.name
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            data_source = _make_preflight_data_source([], use_streaming=True, prompt_data=path)

            report = data_source.validate_reward_routes()

            assert report["total"] == 2
            assert report["assignments"] == {"label/multiple_choice": 1, "metadata/math": 1}
        finally:
            if path is not None and os.path.exists(path):
                os.unlink(path)


class TestEagerDataset:
    """Tests for eager Dataset global-slice semantics."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        return tokenizer

    def test_dataset_multi_file_global_slice(self, mock_tokenizer):
        from relax.utils.data.data import Dataset

        data1 = [{"text": f"A{i}", "label": f"a{i}"} for i in range(3)]
        data2 = [{"text": f"B{i}", "label": f"b{i}"} for i in range(3)]
        files = []
        try:
            for data in (data1, data2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                    files.append(f.name)

            path = f"[{files[0]},{files[1]}]@[1:5]"
            dataset = Dataset(
                path=path,
                tokenizer=mock_tokenizer,
                processor=None,
                max_length=None,
                prompt_key="text",
                label_key="label",
            )

            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A1", "A2", "B0", "B1"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


class TestDataSourceIntegration:
    """Integration tests for data_source.py with StreamingDataset."""

    @pytest.fixture
    def jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        data = [{"text": f"Sample {i}", "label": f"label_{i}"} for i in range(10)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            filepath = f.name

        yield filepath, data
        os.unlink(filepath)

    def test_factory_function_streaming(self, jsonl_file):
        """Test _create_dataset factory with streaming enabled."""
        from relax.engine.rollout.data_source import _create_dataset
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file

        args = MagicMock()
        args.use_streaming_dataset = True
        args.streaming_buffer_size = 100
        args.prompt_data = filepath
        args.rollout_max_prompt_len = None
        args.input_key = "text"
        args.multimodal_keys = None
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.system_prompt = None
        args.tool_key = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = None
        args.rollout_seed = 42
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = _create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, StreamingDataset)
        assert len(dataset) == len(data)

    def test_factory_function_traditional(self, jsonl_file):
        """Test _create_dataset factory with streaming disabled."""
        from relax.engine.rollout.data_source import _create_dataset
        from relax.utils.data.data import Dataset

        filepath, data = jsonl_file

        args = MagicMock()
        args.use_streaming_dataset = False
        args.prompt_data = filepath
        args.rollout_max_prompt_len = None
        args.input_key = "text"
        args.multimodal_keys = None
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.system_prompt = None
        args.tool_key = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = None
        args.rollout_seed = 42
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = _create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, Dataset)

    def test_factory_function_streaming_multi_file_slice(self):
        """Test _create_dataset factory with streaming dataset over multiple
        files and outer slice."""
        from relax.engine.rollout.data_source import _create_dataset
        from relax.utils.data.streaming_dataset import StreamingDataset

        data1 = [{"text": f"A{i}", "label": f"a{i}"} for i in range(3)]
        data2 = [{"text": f"B{i}", "label": f"b{i}"} for i in range(3)]
        files = []
        try:
            for data in (data1, data2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                    files.append(f.name)

            args = MagicMock()
            args.use_streaming_dataset = True
            args.streaming_buffer_size = 100
            args.prompt_data = f"[{files[0]},{files[1]}]@[2:6]"
            args.rollout_max_prompt_len = None
            args.input_key = "text"
            args.multimodal_keys = None
            args.label_key = "label"
            args.metadata_key = "metadata"
            args.system_prompt = None
            args.tool_key = None
            args.apply_chat_template = False
            args.apply_chat_template_kwargs = None
            args.rollout_seed = 42
            args.custom_prompt_path = None

            tokenizer = MagicMock()
            dataset = _create_dataset(args, tokenizer, processor=None)

            assert isinstance(dataset, StreamingDataset)
            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A2", "B0", "B1", "B2"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
