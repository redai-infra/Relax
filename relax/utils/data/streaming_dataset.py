# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Streaming Dataset Implementation.

This module provides a memory-efficient streaming dataset implementation
that loads data on-demand instead of loading everything into memory at once.

Key features:
- Lazy loading: Data is loaded only when needed
- Buffer caching: Recently accessed samples are cached
- Shuffle support: Global shuffle with epoch-based seeding
- Filter support: Length filtering is done at access time

Usage:
    from relax.utils.data.streaming_dataset import StreamingDataset

    dataset = StreamingDataset(
        path="data.jsonl",
        tokenizer=tokenizer,
        processor=processor,
        max_length=2048,
        buffer_size=10000,
    )

    # Get a batch of samples
    samples, crossed_epoch = dataset.get_batch(32)
"""

import json
import os
import random
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Iterator, Optional


try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from relax.utils.data.data_utils import (
    BaseDataset,
    check_sample_length,
    parse_generalized_path,
    resolve_path_plan,
)
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.config import MultimodalConfig
from relax.utils.types import Sample


logger = get_logger(__name__)

__all__ = [
    "StreamingDataset",
    "StreamingReader",
    "CompositeStreamingReader",
    "SampleBuffer",
    "IndexManager",
]


class StreamingReader:
    """Streaming file reader with random access support.

    Builds an index of line offsets on first access to enable efficient random
    access to any line in the file.
    """

    def __init__(self, path: str):
        """Initialize the streaming reader.

        Args:
            path: Path to the data file (JSONL or Parquet)
        """
        self.path, self.row_slice = parse_generalized_path(path)

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset path '{self.path}' does not exist.")

        self._line_offsets: Optional[list[int]] = None
        self._total_lines: Optional[int] = None
        self._is_parquet = self.path.endswith(".parquet")
        self._parquet_data: Optional[list] = None  # For parquet, we cache in memory

        # Validate file format
        if not self.path.endswith((".jsonl", ".parquet")):
            raise ValueError(f"Unsupported file format: {self.path}. Supported formats are .jsonl and .parquet.")

    def _build_index(self) -> None:
        """Build line offset index for JSONL files.

        This scans the file once to record the byte offset of each line,
        enabling efficient random access later.
        """
        if self._line_offsets is not None:
            return

        if self._is_parquet:
            self._load_parquet()
            return

        logger.info(f"Building index for {self.path}...")
        self._line_offsets = []

        with open(self.path, "rb") as f:
            offset = 0
            for line in f:
                line_stripped = line.strip()
                if line_stripped:  # Skip empty lines
                    self._line_offsets.append(offset)
                offset += len(line)

        # Apply row slice if specified
        if self.row_slice is not None:
            start = self.row_slice.start or 0
            stop = self.row_slice.stop or len(self._line_offsets)
            self._line_offsets = self._line_offsets[start:stop]

        self._total_lines = len(self._line_offsets)
        logger.info(f"Index built: {self._total_lines} lines")

    def _load_parquet(self) -> None:
        """Load parquet file into memory."""
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        logger.info(f"Loading parquet file {self.path}...")
        pf = pq.ParquetFile(self.path)
        self._parquet_data = []

        # Read row groups individually instead of using iter_batches().
        # iter_batches() creates chunked arrays for multi-row-group files,
        # which fails with ArrowNotImplementedError on nested types
        # (e.g. list<struct<...>>, struct<...>).
        for i in range(pf.metadata.num_row_groups):
            self._parquet_data.extend(pf.read_row_group(i).to_pylist())

        # Apply row slice if specified
        if self.row_slice is not None:
            start = self.row_slice.start or 0
            stop = self.row_slice.stop or len(self._parquet_data)
            self._parquet_data = self._parquet_data[start:stop]

        self._total_lines = len(self._parquet_data)
        logger.info(f"Parquet loaded: {self._total_lines} rows")

    def __len__(self) -> int:
        """Return total number of lines/rows."""
        if self._total_lines is None:
            self._build_index()
        return self._total_lines

    def __getitem__(self, idx: int) -> dict:
        """Read a single line/row by index.

        Args:
            idx: Line index (0-based)

        Returns:
            Parsed JSON object or parquet row
        """
        if self._total_lines is None:
            self._build_index()

        if idx < 0 or idx >= self._total_lines:
            raise IndexError(f"Index {idx} out of range [0, {self._total_lines})")

        if self._is_parquet:
            return self._parquet_data[idx]

        # Read from JSONL using line offset
        offset = self._line_offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8").strip()
            return json.loads(line)

    def iter_batch(self, indices: list[int]) -> Iterator[dict]:
        """Iterate over multiple indices efficiently.

        For JSONL, this sorts indices to minimize seeking.
        For Parquet, this is a simple iteration.

        Args:
            indices: List of indices to read

        Yields:
            (index, data) tuples
        """
        if self._total_lines is None:
            self._build_index()

        if self._is_parquet:
            for idx in indices:
                yield idx, self._parquet_data[idx]
            return

        # Sort indices for sequential reading (more efficient for HDD)
        sorted_pairs = sorted(enumerate(indices), key=lambda x: self._line_offsets[x[1]])
        results = [None] * len(indices)

        with open(self.path, "rb") as f:
            for original_pos, idx in sorted_pairs:
                offset = self._line_offsets[idx]
                f.seek(offset)
                line = f.readline().decode("utf-8").strip()
                results[original_pos] = (idx, json.loads(line))

        yield from results


class CompositeStreamingReader:
    """Compose multiple single-file StreamingReader instances into one logical
    reader with optional slicing over the concatenated sample stream."""

    def __init__(self, paths: list[str], row_slice: Optional[slice] = None):
        if not paths:
            raise ValueError("paths must not be empty")

        self.paths = paths
        self.row_slice = row_slice
        self.readers = [StreamingReader(path) for path in paths]
        self._cumulative_lengths: list[int] = []

        total = 0
        for reader in self.readers:
            total += len(reader)
            self._cumulative_lengths.append(total)

        self._base_total_lines = total
        self._slice_spec = None if row_slice is None else row_slice.indices(total)
        if self._slice_spec is None:
            self._total_lines = total
        else:
            start, stop, step = self._slice_spec
            self._total_lines = len(range(start, stop, step))

    def __len__(self) -> int:
        return self._total_lines

    def _external_to_global_index(self, idx: int) -> int:
        if idx < 0 or idx >= self._total_lines:
            raise IndexError(f"Index {idx} out of range [0, {self._total_lines})")

        if self._slice_spec is None:
            return idx

        start, _, step = self._slice_spec
        return start + idx * step

    def _locate_reader(self, global_idx: int) -> tuple[int, int]:
        reader_idx = bisect_right(self._cumulative_lengths, global_idx)
        prev_total = 0 if reader_idx == 0 else self._cumulative_lengths[reader_idx - 1]
        return reader_idx, global_idx - prev_total

    def __getitem__(self, idx: int) -> dict:
        global_idx = self._external_to_global_index(idx)
        reader_idx, local_idx = self._locate_reader(global_idx)
        return self.readers[reader_idx][local_idx]

    def iter_batch(self, indices: list[int]) -> Iterator[tuple[int, dict]]:
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for original_pos, idx in enumerate(indices):
            global_idx = self._external_to_global_index(idx)
            reader_idx, local_idx = self._locate_reader(global_idx)
            groups.setdefault(reader_idx, []).append((original_pos, idx, local_idx))

        results: list[Optional[tuple[int, dict]]] = [None] * len(indices)
        for reader_idx, entries in groups.items():
            local_indices = [local_idx for _, _, local_idx in entries]
            fetched = list(self.readers[reader_idx].iter_batch(local_indices))
            for (original_pos, external_idx, _), (_, data) in zip(entries, fetched, strict=True):
                results[original_pos] = (external_idx, data)

        for result in results:
            yield result


class SampleBuffer:
    """LRU cache for processed samples.

    Caches recently accessed samples to avoid re-processing them. Uses
    OrderedDict for O(1) access and LRU eviction.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of samples to cache
        """
        self.cache: OrderedDict[int, Sample] = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, idx: int) -> Optional[Sample]:
        """Get a cached sample.

        Args:
            idx: Sample index

        Returns:
            Cached sample or None if not in cache
        """
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(idx)
            self._hits += 1
            return self.cache[idx]
        self._misses += 1
        return None

    def put(self, idx: int, sample: Sample) -> None:
        """Cache a sample.

        Args:
            idx: Sample index
            sample: Sample to cache
        """
        if idx in self.cache:
            self.cache.move_to_end(idx)
            self.cache[idx] = sample
            return

        # Evict oldest if at capacity
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[idx] = sample

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self.cache)


class IndexManager:
    """Manages shuffle indices and epoch transitions.

    Generates reproducible shuffle permutations based on epoch ID and seed.
    Tracks current position within the epoch.
    """

    def __init__(self, total_size: int, seed: int = 42):
        """Initialize the index manager.

        Args:
            total_size: Total number of samples
            seed: Random seed for reproducible shuffling
        """
        self.total_size = total_size
        self.seed = seed
        self.current_epoch = -1
        self.indices: Optional[list[int]] = None
        self.position = 0

    def shuffle(self, epoch_id: int) -> None:
        """Generate shuffle permutation for a new epoch.

        Args:
            epoch_id: Epoch identifier (used with seed for reproducibility)
        """
        if epoch_id == self.current_epoch:
            return

        random.seed(self.seed + epoch_id)
        self.indices = list(range(self.total_size))
        random.shuffle(self.indices)
        self.current_epoch = epoch_id
        self.position = 0

        logger.info(f"Shuffled dataset for epoch {epoch_id}")

    def get_next_indices(self, n: int) -> tuple[list[int], bool]:
        """Get next n indices from the current epoch.

        If we reach the end of the epoch, wraps around and increments epoch.

        Args:
            n: Number of indices to get

        Returns:
            (indices, crossed_epoch): List of indices and whether we crossed an epoch boundary
        """
        if self.indices is None:
            self.shuffle(0)

        indices = []
        crossed_epoch = False

        remaining = n
        while remaining > 0:
            available = self.total_size - self.position
            take = min(remaining, available)

            indices.extend(self.indices[self.position : self.position + take])
            self.position += take
            remaining -= take

            if self.position >= self.total_size:
                # Epoch boundary reached
                crossed_epoch = True
                new_epoch = self.current_epoch + 1
                self.shuffle(new_epoch)

        return indices, crossed_epoch

    def reset(self, position: int = 0, epoch_id: int = 0) -> None:
        """Reset to a specific position and epoch.

        Args:
            position: Position within the epoch
            epoch_id: Epoch to set
        """
        self.shuffle(epoch_id)
        self.position = position

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "epoch_id": self.current_epoch,
            "position": self.position,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        epoch_id = state.get("epoch_id", 0)
        position = state.get("position", 0)
        self.reset(position=position, epoch_id=epoch_id)


class StreamingDataset(BaseDataset):
    """Memory-efficient streaming dataset with on-demand loading.

    Inherits from BaseDataset and implements lazy loading with LRU caching.

    Features:
    - Lazy loading: Only loads data when accessed
    - LRU caching: Caches recently accessed samples
    - Shuffle support: Epoch-based reproducible shuffling
    - Filter support: Length filtering done at access time

    Example:
        dataset = StreamingDataset(
            path="data.jsonl",
            tokenizer=tokenizer,
            processor=processor,
            max_length=2048,
        )

        samples, crossed = dataset.get_batch(32)
    """

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        processor: Any,
        max_length: Optional[int],
        *,
        prompt_key: str = "text",
        multimodal_keys: Optional[dict] = None,
        label_key: Optional[str] = None,
        tool_key: Optional[str] = None,
        metadata_key: str = "metadata",
        system_prompt: Optional[str] = None,
        seed: int = 42,
        apply_chat_template: bool = False,
        apply_chat_template_kwargs: Optional[dict] = None,
        use_audio_in_video: bool = False,
        buffer_size: int = 10000,
        prefetch_size: int = 100,
        multimodal_config: MultimodalConfig = None,
    ):
        """Initialize the streaming dataset.

        Args:
            path: Path to data file (JSONL or Parquet)
            tokenizer: Tokenizer for length checking and chat template
            processor: Processor for multimodal inputs
            max_length: Maximum prompt length (samples exceeding this are filtered)
            prompt_key: Key for prompt in data
            multimodal_keys: Mapping of multimodal types to data keys
            label_key: Key for labels in data
            tool_key: Key for tools in data
            metadata_key: Key for metadata in data
            system_prompt: System prompt key
            seed: Random seed for shuffling
            apply_chat_template: Whether to apply chat template
            apply_chat_template_kwargs: Additional kwargs for chat template
            use_audio_in_video: Whether to extract audio from video files for multimodal processing
            buffer_size: Maximum samples to cache
            prefetch_size: Number of samples to prefetch (not implemented yet)
        """
        # Initialize base class
        super().__init__(
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            prompt_key=prompt_key,
            multimodal_keys=multimodal_keys,
            label_key=label_key,
            tool_key=tool_key,
            metadata_key=metadata_key,
            system_prompt=system_prompt,
            seed=seed,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            use_audio_in_video=use_audio_in_video,
            multimodal_config=multimodal_config,
        )

        # Streaming-specific components
        paths, row_slice = resolve_path_plan(path)
        if len(paths) == 1 and row_slice is None:
            self.reader = StreamingReader(paths[0])
        else:
            self.reader = CompositeStreamingReader(paths, row_slice)
        self.buffer = SampleBuffer(max_size=buffer_size)
        self.index_manager = IndexManager(len(self.reader), seed=seed)

        self._filter_count = 0
        self._total_processed = 0

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.reader)

    def shuffle(self, epoch_id: int) -> None:
        """Shuffle the dataset for a new epoch.

        Args:
            epoch_id: Epoch identifier
        """
        self.index_manager.shuffle(epoch_id)
        self.epoch_id = epoch_id

    def _process_raw_data(self, data: dict) -> Optional[Sample]:
        """Process raw data into a Sample.

        Uses the shared _process_data from BaseDataset.

        Args:
            data: Raw data dict from file

        Returns:
            Sample or None if filtered out
        """
        self._total_processed += 1

        try:
            sample = self._process_data(data)

            # Filter by length if max_length is set
            if self.max_length is not None:
                if not check_sample_length(sample, self.tokenizer, self.processor, self.max_length):
                    self._filter_count += 1
                    return None

            return sample

        except Exception as e:
            logger.warning(f"Error processing data: {e}")
            return None

    def get_batch(self, n: int) -> tuple[list[Sample], bool]:
        """Get a batch of n valid samples.

        Automatically skips filtered samples and handles epoch boundaries.

        Args:
            n: Number of samples to get

        Returns:
            (samples, crossed_epoch): List of samples and whether an epoch boundary was crossed
        """
        samples = []
        crossed_epoch = False
        max_attempts = n * 10  # Prevent infinite loop if too many filtered
        attempts = 0

        while len(samples) < n and attempts < max_attempts:
            # Calculate how many more we need (with some buffer for filtered samples)
            need = n - len(samples)
            fetch_size = min(need * 2, 100)  # Fetch extra to account for filtering

            indices, epoch_crossed = self.index_manager.get_next_indices(fetch_size)
            crossed_epoch = crossed_epoch or epoch_crossed

            for idx in indices:
                if len(samples) >= n:
                    # Put back unused indices
                    break

                attempts += 1

                # Check cache first
                sample = self.buffer.get(idx)

                if sample is None:
                    # Load and process
                    raw_data = self.reader[idx]
                    sample = self._process_raw_data(raw_data)

                    if sample is not None:
                        self.buffer.put(idx, sample)

                if sample is not None:
                    samples.append(sample)

        if len(samples) < n:
            logger.warning(
                f"Could only get {len(samples)}/{n} samples after {attempts} attempts. "
                f"Filter rate: {self._filter_count}/{self._total_processed}"
            )

        return samples, crossed_epoch

    def __getitem__(self, idx: int) -> Optional[Sample]:
        """Get a single sample by index.

        Args:
            idx: Sample index

        Returns:
            Sample or None if filtered
        """
        sample = self.buffer.get(idx)
        if sample is not None:
            return sample

        raw_data = self.reader[idx]
        sample = self._process_raw_data(raw_data)

        if sample is not None:
            self.buffer.put(idx, sample)

        return sample

    @property
    def samples(self) -> "StreamingDatasetSamplesProxy":
        """Provide a proxy for samples access.

        This allows code that uses `dataset.samples[start:end]` to work, though
        it's less efficient than using get_batch().
        """
        return StreamingDatasetSamplesProxy(self)

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            **self.index_manager.get_state(),
            "filter_count": self._filter_count,
            "total_processed": self._total_processed,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.index_manager.load_state(state)
        self._filter_count = state.get("filter_count", 0)
        self._total_processed = state.get("total_processed", 0)

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "total_size": len(self),
            "buffer_size": len(self.buffer),
            "buffer_hit_rate": self.buffer.hit_rate,
            "filter_count": self._filter_count,
            "total_processed": self._total_processed,
            "current_epoch": self.index_manager.current_epoch,
            "current_position": self.index_manager.position,
        }


class StreamingDatasetSamplesProxy:
    """Proxy class to support `dataset.samples[start:end]` access pattern.

    This provides compatibility with code that expects the original Dataset
    interface, but with streaming behavior.
    """

    def __init__(self, dataset: StreamingDataset):
        self.dataset = dataset

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or len(self.dataset)
            step = key.step or 1

            indices = list(range(start, stop, step))
            samples = []

            for idx in indices:
                sample = self.dataset[idx]
                if sample is not None:
                    samples.append(sample)

            return samples
        else:
            return self.dataset[key]

    def __len__(self):
        return len(self.dataset)
