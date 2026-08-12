# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Integration tests for data_source.py and eager Dataset global-slice
semantics.

Extracted from test_streaming_dataset.py during tests/ directory
restructuring.

Run with: pytest tests/engine/rollout/test_data_source.py -v
"""

import json
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.utils.types import Sample


class _EagerSamples:
    def __init__(self, prompts):
        self.samples = [Sample(prompt=prompt) for prompt in prompts]
        self.shuffle_calls = []

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch_id):
        self.shuffle_calls.append(epoch_id)


@pytest.fixture
def data_source_module(monkeypatch):
    processing_utils = ModuleType("relax.utils.data.processing_utils")
    processing_utils.load_processor = MagicMock(return_value=None)
    processing_utils.load_tokenizer = MagicMock()
    monkeypatch.setitem(sys.modules, "relax.utils.data.processing_utils", processing_utils)
    monkeypatch.delitem(sys.modules, "relax.engine.rollout.data_source", raising=False)

    from relax.engine.rollout import data_source

    yield data_source
    monkeypatch.delitem(sys.modules, "relax.engine.rollout.data_source", raising=False)


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

    def test_factory_function_streaming(self, jsonl_file, data_source_module):
        """Test _create_dataset factory with streaming enabled."""
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
        args.rollout_shuffle = False
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = data_source_module._create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, StreamingDataset)
        assert len(dataset) == len(data)
        assert dataset.index_manager.shuffle_enabled is False

    def test_factory_function_traditional(self, jsonl_file, data_source_module):
        """Test _create_dataset factory with streaming disabled."""
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

        dataset = data_source_module._create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, Dataset)

    def test_eager_data_source_repeatedly_wraps_to_fill_batch(self, data_source_module):
        source = data_source_module.RolloutDataSource.__new__(data_source_module.RolloutDataSource)
        source.args = SimpleNamespace(n_samples_per_prompt=1, rollout_shuffle=False)
        source.epoch_id = 0
        source.sample_group_index = 0
        source.sample_index = 0
        source.sample_offset = 0
        source._use_streaming = False
        source.dataset = _EagerSamples(["a", "b", "c"])

        first = source.get_samples(8)
        second = source.get_samples(8)

        assert [group[0].prompt for group in first] == ["a", "b", "c", "a", "b", "c", "a", "b"]
        assert [group[0].prompt for group in second] == ["c", "a", "b", "c", "a", "b", "c", "a"]
        assert source.sample_offset == 1
        assert source.epoch_id == 5

    def test_eager_data_source_rejects_empty_dataset(self, data_source_module):
        source = data_source_module.RolloutDataSource.__new__(data_source_module.RolloutDataSource)
        source.args = SimpleNamespace(n_samples_per_prompt=1, rollout_shuffle=False)
        source._use_streaming = False
        source.dataset = _EagerSamples([])

        with pytest.raises(ValueError, match="empty dataset"):
            source.get_samples(1)

    def test_streaming_data_source_uses_exact_internal_epoch(self, data_source_module):
        class _StreamingSamples:
            def get_batch(self, num_samples):
                return [Sample(prompt=str(index)) for index in range(num_samples)], True

            def get_state(self):
                return {"epoch_id": 3}

        source = data_source_module.RolloutDataSource.__new__(data_source_module.RolloutDataSource)
        source.args = SimpleNamespace(n_samples_per_prompt=1)
        source.epoch_id = 0
        source.sample_group_index = 0
        source.sample_index = 0
        source.sample_offset = 0
        source._use_streaming = True
        source.dataset = _StreamingSamples()

        samples = source.get_samples(8)

        assert len(samples) == 8
        assert source.epoch_id == 3

    def test_factory_function_streaming_multi_file_slice(self, data_source_module):
        """Test _create_dataset factory with streaming dataset over multiple
        files and outer slice."""
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
            dataset = data_source_module._create_dataset(args, tokenizer, processor=None)

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
