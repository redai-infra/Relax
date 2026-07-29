#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the format-aware reward router and registry.

Covers:
  - Registry: register, get, list, duplicate registration
  - Router:  metadata priority, CLI fallback, conflict detection, empty path
  - Integration: RewardWorker compute via registry, mixed-batch routing
  - Fallback:  unknown type → None + warning, missing type → zero reward
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Import strategy
# ---------------------------------------------------------------------------
# The relax.engine.rewards package __init__.py requires Ray.  When Ray is
# absent we load registry.py and router.py directly from their file paths.
# ---------------------------------------------------------------------------

_REWARDS_DIR = Path(__file__).resolve().parents[3] / "relax" / "engine" / "rewards"

_HAS_FULL_PIPELINE = False

# -- always-available: registry + router (loaded directly from files) ---------


def _load_module_from_file(name: str, path: Path):
    """Import a module directly from a file path, bypassing package init."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load registry module
_registry_mod = _load_module_from_file(
    "relax.engine.rewards.registry",
    _REWARDS_DIR / "registry.py",
)
REWARD_REGISTRY = _registry_mod.REWARD_REGISTRY
get_reward = _registry_mod.get_reward
register_reward = _registry_mod.register_reward
list_registered = _registry_mod.list_registered

# Load router module (depends on registry already being in sys.modules)
_router_mod = _load_module_from_file(
    "relax.engine.rewards.router",
    _REWARDS_DIR / "router.py",
)
resolve_rm_type = _router_mod.resolve_rm_type

# Sample from relax.utils.types requires torch+numpy.  Use a lightweight
# stand-in for router/registry tests; the full pipeline tests (which need
# Ray anyway) get the real Sample when Ray is available.
class _SampleStub:
    """Minimal Sample stand-in for router tests (no ML deps)."""

    __slots__ = ("response", "label", "metadata")

    def __init__(self, response="", label=None, metadata=None):
        self.response = response
        self.label = label
        self.metadata = metadata if metadata is not None else {}

Sample = _SampleStub  # used by router/registry tests

_FULL_SAMPLE = None  # set to the real Sample when Ray pipeline is available

# -- optional: full pipeline (requires Ray) -----------------------------------

try:
    import ray

    if not ray.is_initialized():
        try:
            ray.init(num_cpus=8, ignore_reinit_error=True)
        except ValueError as e:
            if "When connecting to an existing cluster" in str(e):
                ray.init(ignore_reinit_error=True)
            else:
                raise

    from relax.engine.rewards import RewardExecutor, RewardWorker, async_rm, batched_async_rm
    from relax.utils.types import Sample as _RealSample

    Sample = _RealSample  # replace stub with real Sample for integration tests
    _HAS_FULL_PIPELINE = True
except ImportError:
    pass


requires_full_pipeline = pytest.mark.skipif(
    not _HAS_FULL_PIPELINE,
    reason="Full pipeline requires Ray",
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_sample(response: str = "", label=None, metadata: dict | None = None):
    """Create a Sample with the given fields."""
    return Sample(response=response, label=label, metadata=metadata or {})


def _make_args(**overrides):
    """Create a mock args namespace."""
    defaults = {
        "rm_type": None,
        "custom_rm_path": None,
        "reward_max_concurrency": 64,
        "reward_num_workers": 4,
        "rm_url": None,
        "reward_key": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _kill_executor_workers():
    """Clean up Ray actors from the singleton executor."""
    inst = RewardExecutor._instance
    if inst is None:
        return
    for w in inst._workers:
        try:
            ray.kill(w)
        except Exception:
            pass
    inst._workers = []


# ===========================================================================
# Registry tests
# ===========================================================================


class TestRegistry:
    """Tests for the reward function registry."""

    def teardown_method(self):
        """Clean up test entries from the global registry."""
        for key in list(REWARD_REGISTRY.keys()):
            if key.startswith("test_"):
                del REWARD_REGISTRY[key]

    def test_register_and_retrieve(self):
        @register_reward("test_register")
        def fn(response, label, metadata=None):
            return 42

        assert get_reward("test_register") is fn
        assert fn(response="", label="") == 42

    def test_get_unknown_returns_none(self):
        assert get_reward("test_nonexistent_xyz") is None

    def test_list_registered(self):
        @register_reward("test_list_a")
        def a(response, label, metadata=None):
            return 1

        @register_reward("test_list_b")
        def b(response, label, metadata=None):
            return 2

        names = list_registered()
        assert "test_list_a" in names
        assert "test_list_b" in names

    def test_duplicate_registration_overwrites(self):
        @register_reward("test_dup")
        def first(response, label, metadata=None):
            return 1

        @register_reward("test_dup")
        def second(response, label, metadata=None):
            return 2

        # Last registration wins
        assert get_reward("test_dup") is second
        assert REWARD_REGISTRY["test_dup"](response="", label="") == 2

    def test_registry_custom_metadata_keys(self):
        """A reward function in the registry receives metadata if passed."""
        received_meta = []

        @register_reward("test_meta")
        def fn(response, label, metadata=None):
            received_meta.append(metadata)
            return len(metadata) if metadata else 0

        result = fn(response="x", label="y", metadata={"key": "val"})
        assert result == 1
        assert received_meta[0] == {"key": "val"}


# ===========================================================================
# Router tests
# ===========================================================================


class TestRouterMetadataPriority:
    """Tests for rm_type resolution via sample metadata."""

    def test_rm_type_from_metadata(self):
        s = _make_sample(metadata={"rm_type": "math"})
        assert resolve_rm_type(s) == "math"

    def test_task_type_from_metadata(self):
        s = _make_sample(metadata={"task_type": "multiple_choice"})
        assert resolve_rm_type(s) == "multiple_choice"

    def test_label_type_from_metadata(self):
        s = _make_sample(metadata={"label_type": "biology"})
        assert resolve_rm_type(s) == "biology"

    def test_rm_type_beats_task_type(self):
        """rm_type has higher priority than task_type."""
        s = _make_sample(metadata={"rm_type": "math", "task_type": "dapo"})
        assert resolve_rm_type(s) == "math"

    def test_task_type_beats_label_type(self):
        """task_type has higher priority than label_type."""
        s = _make_sample(metadata={"task_type": "math", "label_type": "dapo"})
        assert resolve_rm_type(s) == "math"

    def test_metadata_beats_cli_default(self):
        """Sample metadata overrides CLI --rm-type."""
        s = _make_sample(metadata={"rm_type": "multiple_choice"})
        assert resolve_rm_type(s, default_rm_type="deepscaler") == "multiple_choice"

    def test_strips_whitespace(self):
        s = _make_sample(metadata={"rm_type": "  math  "})
        assert resolve_rm_type(s) == "math"


class TestRouterFallback:
    """Tests for fallback behavior when metadata is missing or empty."""

    def test_cli_fallback_when_no_metadata(self):
        s = _make_sample()
        assert resolve_rm_type(s, default_rm_type="deepscaler") == "deepscaler"

    def test_none_when_no_type_at_all(self):
        s = _make_sample()
        assert resolve_rm_type(s) is None

    def test_none_when_empty_metadata_dict(self):
        s = _make_sample(metadata={})
        assert resolve_rm_type(s) is None

    def test_none_when_default_is_none(self):
        s = _make_sample(metadata={"rm_type": "math"})
        assert resolve_rm_type(s, default_rm_type=None) == "math"

    def test_sample_without_metadata_attribute(self):
        """Graceful handling of objects without a metadata attribute."""

        class BareSample:
            pass

        assert resolve_rm_type(BareSample(), default_rm_type="math") == "math"
        assert resolve_rm_type(BareSample()) is None

    def test_metadata_is_not_a_dict(self):
        """Graceful handling when metadata is not a dict (e.g. None or list)."""
        s = _make_sample()
        s.metadata = None
        assert resolve_rm_type(s, default_rm_type="math") == "math"

        s.metadata = ["not", "a", "dict"]
        assert resolve_rm_type(s, default_rm_type="math") == "math"


class TestRouterConflictDetection:
    """Tests for conflict logging when multiple sources disagree."""

    def test_metadata_vs_cli_conflict(self):
        """Metadata takes priority, conflict is logged."""
        s = _make_sample(metadata={"rm_type": "math"})
        result = resolve_rm_type(s, default_rm_type="deepscaler")
        assert result == "math"  # metadata wins

    def test_duplicate_same_value_no_conflict(self):
        """Same value from multiple sources should not conflict."""
        s = _make_sample(metadata={"rm_type": "math", "task_type": "math"})
        result = resolve_rm_type(s)
        assert result == "math"


# ===========================================================================
# Integration tests (require Ray)
# ===========================================================================


@requires_full_pipeline
class TestRewardWorkerRegistry:
    """Test RewardWorker.compute() dispatch via the format-aware registry."""

    @pytest.fixture(autouse=True)
    def _worker(self):
        self.worker = RewardWorker.remote()
        yield
        ray.kill(self.worker)

    def test_math_via_registry_correct(self):
        result = ray.get(self.worker.compute.remote("math", "\\boxed{7}", "7"))
        assert result == 1

    def test_math_via_registry_wrong(self):
        result = ray.get(self.worker.compute.remote("math", "\\boxed{8}", "7"))
        assert result == 0

    def test_multiple_choice_via_registry_correct(self):
        result = ray.get(
            self.worker.compute.remote(
                "multiple_choice",
                "<answer>A</answer>",
                "<answer>A</answer>",
            )
        )
        assert result == 1.0

    def test_multiple_choice_via_registry_wrong(self):
        result = ray.get(
            self.worker.compute.remote(
                "multiple_choice",
                "<answer>B</answer>",
                "<answer>A</answer>",
            )
        )
        assert result == 0.0

    def test_random_via_registry(self):
        result = ray.get(self.worker.compute.remote("random", "", ""))
        assert result in (0, 1)

    def test_unknown_type_raises(self):
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(self.worker.compute.remote("nonexistent_rm_type_xyz", "foo", "bar"))


@requires_full_pipeline
class TestRewardExecutorRouter:
    """Test RewardExecutor.execute() with the format-aware router."""

    @pytest.fixture(autouse=True)
    def _reset_executor(self):
        _kill_executor_workers()
        RewardExecutor._instance = None
        yield
        _kill_executor_workers()
        RewardExecutor._instance = None

    @pytest.mark.asyncio
    async def test_metadata_rm_type_math(self):
        args = _make_args()
        sample = _make_sample(
            response="\\boxed{42}",
            label="42",
            metadata={"rm_type": "math"},
        )
        result = await async_rm(args, sample)
        assert result == 1

    @pytest.mark.asyncio
    async def test_metadata_rm_type_multiple_choice(self):
        args = _make_args()
        sample = _make_sample(
            response="<answer>C</answer>",
            label="<answer>C</answer>",
            metadata={"rm_type": "multiple_choice"},
        )
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_metadata_overrides_cli(self):
        """Per-sample rm_type in metadata should override CLI --rm-type."""
        args = _make_args(rm_type="deepscaler")
        sample = _make_sample(
            response="\\boxed{7}",
            label="7",
            metadata={"rm_type": "openr1mm"},
        )
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_task_type_metadata(self):
        args = _make_args()
        sample = _make_sample(
            response="\\boxed{42}",
            label="42",
            metadata={"task_type": "openr1mm"},
        )
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_unknown_type_fallback_zero(self):
        """Unknown/missing rm_type should fall back to 0.0 with a warning."""
        args = _make_args()
        sample = _make_sample(
            response="foo",
            label="bar",
            metadata={"rm_type": "completely_unknown_type_xyz"},
        )
        result = await async_rm(args, sample)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_missing_type_fallback_zero(self):
        """Missing rm_type entirely should fall back to 0.0."""
        args = _make_args()  # rm_type=None
        sample = _make_sample(response="foo", label="bar", metadata={})
        result = await async_rm(args, sample)
        assert result == 0.0


@requires_full_pipeline
class TestMixedBatchRouting:
    """Test that a mixed batch routes each sample to the correct reward."""

    @pytest.fixture(autouse=True)
    def _reset_executor(self):
        _kill_executor_workers()
        RewardExecutor._instance = None
        yield
        _kill_executor_workers()
        RewardExecutor._instance = None

    @pytest.mark.asyncio
    async def test_mixed_math_and_multiple_choice(self):
        """Each sample in a mixed batch should hit the correct reward function."""
        args = _make_args()
        samples = [
            _make_sample(response="\\boxed{7}", label="7", metadata={"rm_type": "math"}),
            _make_sample(response="\\boxed{99}", label="7", metadata={"rm_type": "math"}),
            _make_sample(response="<answer>A</answer>", label="<answer>A</answer>",
                         metadata={"rm_type": "multiple_choice"}),
            _make_sample(response="<answer>B</answer>", label="<answer>A</answer>",
                         metadata={"rm_type": "multiple_choice"}),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards == [1, 0, 1.0, 0.0]

    @pytest.mark.asyncio
    async def test_mixed_with_unknown_type(self):
        """Unknown types in a mixed batch should get 0.0 reward."""
        args = _make_args()
        samples = [
            _make_sample(response="\\boxed{7}", label="7", metadata={"rm_type": "math"}),
            _make_sample(response="x", label="y", metadata={"rm_type": "unknown_xyz_type"}),
            _make_sample(response="<answer>A</answer>", label="<answer>A</answer>",
                         metadata={"rm_type": "multiple_choice"}),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards[0] == 1       # math: correct
        assert rewards[1] == 0.0     # unknown: fallback
        assert rewards[2] == 1.0     # mc: correct

    @pytest.mark.asyncio
    async def test_mixed_via_task_type_and_rm_type(self):
        """Samples using different metadata keys should all route correctly."""
        args = _make_args()
        samples = [
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"task_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"label_type": "openr1mm"}),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards == [1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_mixed_defaults_to_cli(self):
        """Samples without metadata should use the CLI default."""
        args = _make_args(rm_type="openr1mm")
        samples = [
            _make_sample(response="\\boxed{7}", label="7", metadata={"rm_type": "math"}),
            _make_sample(response="\\boxed{42}", label="42"),  # no metadata → CLI
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards[0] == 1     # math via metadata
        assert rewards[1] == 1.0   # openr1mm via CLI fallback

    @pytest.mark.asyncio
    async def test_mixed_empty_batch(self):
        args = _make_args()
        rewards = await batched_async_rm(args, [])
        assert rewards == []


# ===========================================================================
# Backward-compatibility tests
# ===========================================================================


@requires_full_pipeline
class TestBackwardCompatibility:
    """Verify existing reward types still work end-to-end."""

    @pytest.fixture(autouse=True)
    def _reset_executor(self):
        _kill_executor_workers()
        RewardExecutor._instance = None
        yield
        _kill_executor_workers()
        RewardExecutor._instance = None

    @pytest.mark.asyncio
    async def test_explicit_rm_type_via_args(self):
        """Using --rm-type on CLI should still work (backward compat)."""
        args = _make_args(rm_type="openr1mm")
        sample = _make_sample(response="\\boxed{7}", label="7")
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_boxed_prefix_still_works(self):
        """The boxed_ prefix is handled by the executor before registry lookup."""
        args = _make_args()
        sample = _make_sample(
            response="The final answer is \\boxed{42} extra text",
            label="42",
            metadata={"rm_type": "boxed_openr1mm"},
        )
        result = await async_rm(args, sample)
        assert result == 1.0
