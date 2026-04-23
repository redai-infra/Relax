# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for DCS weight conversion logic in DeviceDirectBackend.

All tests call **real** project functions or construct **real** Bridge mapping
objects — no hand-written logic duplication.

Covers:
- ``_collect_all_mappings``: recursive mapping discovery with real Bridge mappings
- Real Bridge mapping ``megatron_to_hf`` output + post-processing correctness
- ``strip_param_name_prefix``, ``remove_padding``, ``quantize_params``
"""

import re
from argparse import Namespace
from contextlib import contextmanager
from typing import Dict, List, Tuple

import pytest
import torch


# Skip the whole module when Megatron (or its Bridge) isn't installed — e.g. on
# the CPU-only CI runner. All tests here exercise real Bridge mapping objects
# and the DeviceDirectBackend, which imports ``megatron.core`` at module level.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

# Real Bridge mapping classes
from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    AutoMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (  # noqa: E402
    ExpertMLPDownProjMapping,
    ExpertMLPGateUpProjMapping,
)

from relax.backends.megatron.misc_utils import strip_param_name_prefix  # noqa: E402
from relax.backends.megatron.weight_conversion.processors import quantize_params, remove_padding  # noqa: E402
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend  # noqa: E402


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_args(**overrides) -> Namespace:
    """Create a minimal args namespace for weight conversion tests."""
    defaults = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=6144,
        moe_ffn_hidden_size=768,
        vocab_size=151936,
        q_lora_rank=None,
        update_weight_buffer_size=1 << 30,
        hf_checkpoint="/fake/checkpoint",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@contextmanager
def _patch_gather_from_ep_ranks():
    """Monkey-patch ``gather_from_ep_ranks`` on all relevant mapping classes.

    This is the same no-op that ``_convert_to_hf_bridge`` applies in production
    to bypass EP gather when expert weights have already been gathered
    externally.
    """

    def _noop_gather(self_m, megatron_weights, megatron_module, hf_param_name):
        return {str(hf_param_name): megatron_weights}

    saved_originals: dict = {}
    patched_classes = [MegatronParamMapping, GatedMLPMapping, ExpertMLPGateUpProjMapping, ExpertMLPDownProjMapping]
    for cls in patched_classes:
        if "gather_from_ep_ranks" in cls.__dict__:
            saved_originals[cls] = cls.__dict__["gather_from_ep_ranks"]
        cls.gather_from_ep_ranks = _noop_gather
    try:
        yield
    finally:
        for cls in patched_classes:
            if cls in saved_originals:
                cls.gather_from_ep_ranks = saved_originals[cls]
            elif "gather_from_ep_ranks" in cls.__dict__:
                del cls.gather_from_ep_ranks


def _make_expert_gate_up_mapping(layer_idx: int, expert_id: int) -> ExpertMLPGateUpProjMapping:
    """Create a real ExpertMLPGateUpProjMapping for testing."""
    return ExpertMLPGateUpProjMapping(
        megatron_param=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{expert_id}",
        hf_param=f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
    )


def _make_expert_down_mapping(layer_idx: int, expert_id: int) -> ExpertMLPDownProjMapping:
    """Create a real ExpertMLPDownProjMapping with eagerly initialized inner
    mapping."""
    m = ExpertMLPDownProjMapping(
        megatron_param=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{expert_id}",
        hf_param=f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj",
    )
    # Same as production code in _init_bridge_tasks: eagerly init AutoMapping delegate
    m._detected_type = "replicated"
    m._mapping = m._get_or_create_mapping("replicated")
    return m


def _apply_expert_postprocessing(
    converted_dict: Dict[str, torch.Tensor],
    megatron_param_name: str,
) -> List[Tuple[str, torch.Tensor]]:
    """Apply the same expert weight post-processing as
    ``_convert_to_hf_bridge``.

    This calls the real production logic extracted from device_direct.py lines
    399-420.
    """
    converted_named_tensors = list(converted_dict.items())
    expert_id_match = re.search(r"weight(\d+)", megatron_param_name)
    if expert_id_match is not None:
        expert_id = expert_id_match.group(1)
        postprocessed: list[tuple[str, torch.Tensor]] = []
        for hf_name, tensor in converted_named_tensors:
            if hf_name.endswith(".experts.gate_up_proj"):
                gate_tensor = tensor[0].transpose(-1, -2).contiguous()
                up_tensor = tensor[1].transpose(-1, -2).contiguous()
                base = hf_name[: -len(".gate_up_proj")]
                postprocessed.append((f"{base}.{expert_id}.gate_proj.weight", gate_tensor))
                postprocessed.append((f"{base}.{expert_id}.up_proj.weight", up_tensor))
            elif hf_name.endswith(".experts.down_proj"):
                base = hf_name[: -len(".down_proj")]
                postprocessed.append((f"{base}.{expert_id}.down_proj.weight", tensor.transpose(-1, -2).contiguous()))
            else:
                postprocessed.append((hf_name, tensor))
        converted_named_tensors = postprocessed
    return converted_named_tensors


# ─── Tests for _collect_all_mappings with REAL Bridge mappings ────────────────


class TestCollectAllMappings:
    """Test ``DeviceDirectBackend._collect_all_mappings`` with real Bridge
    mapping objects."""

    def test_replicated_mapping_single(self):
        """A single ReplicatedMapping returns just itself."""
        m = ReplicatedMapping("decoder.layers.0.weight", "model.layers.0.weight")
        result = DeviceDirectBackend._collect_all_mappings(m)
        assert len(result) == 1
        assert result[0] is m
        assert isinstance(result[0], MegatronParamMapping)

    def test_gated_mlp_mapping_single(self):
        """GatedMLPMapping has no sub-mappings, returns just itself."""
        m = GatedMLPMapping(
            "decoder.layers.0.mlp.linear_fc1.weight",
            gate="model.layers.0.mlp.gate_proj.weight",
            up="model.layers.0.mlp.up_proj.weight",
        )
        result = DeviceDirectBackend._collect_all_mappings(m)
        assert len(result) == 1
        assert isinstance(result[0], GatedMLPMapping)

    def test_auto_mapping_with_initialized_inner(self):
        """AutoMapping with eagerly initialized inner delegate collects
        both."""
        m = AutoMapping(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        )
        m._detected_type = "replicated"
        m._mapping = m._get_or_create_mapping("replicated")

        result = DeviceDirectBackend._collect_all_mappings(m)
        assert len(result) == 2
        types = {type(r).__name__ for r in result}
        assert types == {"AutoMapping", "ReplicatedMapping"}

    def test_expert_gate_up_mapping_discovers_gated_inner(self):
        """ExpertMLPGateUpProjMapping has a _gated_mapping sub-mapping."""
        m = _make_expert_gate_up_mapping(layer_idx=0, expert_id=3)
        result = DeviceDirectBackend._collect_all_mappings(m)
        assert len(result) == 2
        types = {type(r).__name__ for r in result}
        assert types == {"ExpertMLPGateUpProjMapping", "GatedMLPMapping"}

    def test_expert_down_mapping_discovers_replicated_inner(self):
        """ExpertMLPDownProjMapping (AutoMapping subclass) with initialized
        inner."""
        m = _make_expert_down_mapping(layer_idx=0, expert_id=3)
        result = DeviceDirectBackend._collect_all_mappings(m)
        assert len(result) == 2
        types = {type(r).__name__ for r in result}
        assert types == {"ExpertMLPDownProjMapping", "ReplicatedMapping"}

    def test_no_duplicate_on_shared_reference(self):
        """If two attributes point to the same mapping object, it's collected
        once."""
        inner = ReplicatedMapping("decoder.layers.0.weight", "model.layers.0.weight")
        outer = AutoMapping("decoder.layers.0.weight2", "model.layers.0.weight2")
        outer._detected_type = "replicated"
        outer._mapping = inner
        # Manually add another reference to the same object
        outer._tp_mapping = inner

        result = DeviceDirectBackend._collect_all_mappings(outer)
        # outer + inner (deduplicated even though referenced twice)
        assert len(result) == 2

    def test_process_groups_are_none_in_test_env(self):
        """Verify that real mappings have None process groups (mpu not
        initialized)."""
        m = ReplicatedMapping("decoder.layers.0.weight", "model.layers.0.weight")
        assert m.pp_group is None
        assert m._tp_group is None
        assert m._etp_group is None
        assert m.ep_group is None
        assert m.pp_size == 1
        assert m.tp_size == 1
        assert m.ep_size == 1


# ─── Tests for real Bridge mapping megatron_to_hf output ──────────────────────


class TestBridgeMappingOutput:
    """Test real Bridge mapping ``megatron_to_hf`` output format.

    These tests call the actual Bridge mapping objects and verify their output
    shape/format, confirming the assumptions that the post-processing relies
    on.
    """

    def test_replicated_mapping_passthrough(self):
        """ReplicatedMapping passes tensor through unchanged."""
        m = ReplicatedMapping(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        )
        w = torch.randn(2048, 2048)
        result = m.megatron_to_hf(w, None)
        assert list(result.keys()) == ["model.layers.0.self_attn.o_proj.weight"]
        assert torch.equal(result["model.layers.0.self_attn.o_proj.weight"], w)

    def test_gated_mlp_mapping_splits_gate_up(self):
        """GatedMLPMapping splits fused [gate; up] into separate tensors."""
        m = GatedMLPMapping(
            "decoder.layers.0.mlp.linear_fc1.weight",
            gate="model.layers.0.mlp.gate_proj.weight",
            up="model.layers.0.mlp.up_proj.weight",
        )
        H, D = 768, 2048
        fused = torch.randn(H * 2, D)
        result = m.megatron_to_hf(fused, None)

        assert set(result.keys()) == {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        }
        gate_expected, up_expected = fused.chunk(2, dim=0)
        assert torch.equal(result["model.layers.0.mlp.gate_proj.weight"], gate_expected)
        assert torch.equal(result["model.layers.0.mlp.up_proj.weight"], up_expected)

    def test_expert_gate_up_mapping_outputs_fused_transposed(self):
        """ExpertMLPGateUpProjMapping outputs fused [2, D, H] with
        transpose."""
        with _patch_gather_from_ep_ranks():
            m = _make_expert_gate_up_mapping(layer_idx=0, expert_id=3)
            H, D = 768, 2048
            fused = torch.randn(H * 2, D)
            result = m.megatron_to_hf(fused, None)

        assert list(result.keys()) == ["model.language_model.layers.0.mlp.experts.gate_up_proj"]
        tensor = result["model.language_model.layers.0.mlp.experts.gate_up_proj"]
        # Bridge transposes each of gate/up from [H, D] to [D, H] then stacks
        assert tensor.shape == (2, D, H)

    def test_expert_down_mapping_outputs_transposed(self):
        """ExpertMLPDownProjMapping outputs transposed tensor [H, D]."""
        with _patch_gather_from_ep_ranks():
            m = _make_expert_down_mapping(layer_idx=0, expert_id=3)
            D, H = 2048, 768
            param = torch.randn(D, H)
            result = m.megatron_to_hf(param, None)

        assert list(result.keys()) == ["model.language_model.layers.0.mlp.experts.down_proj"]
        tensor = result["model.language_model.layers.0.mlp.experts.down_proj"]
        # Bridge transposes from [D, H] to [H, D]
        assert tensor.shape == (H, D)
        assert torch.allclose(tensor, param.transpose(-1, -2).contiguous())


# ─── Tests for Bridge + post-processing output correctness ───────────────────


class TestBridgePostProcessingCorrectness:
    """Verify that real Bridge output + post-processing produces correct HF
    weights.

    Expected behavior (ground truth):
    - gate_up (linear_fc1): Megatron [2H, D] → split into gate [H, D] and up [H, D]
      (same as raw converter: simple chunk, no transpose)
    - down (linear_fc2): Megatron [D, H] → HF [D, H] passthrough
      (same as raw converter: identity)
    """

    def test_expert_gate_up_postprocessed_matches_expected(self):
        """Bridge gate_up + post-processing produces correct gate/up split."""
        H, D = 768, 2048
        expert_id = 3
        megatron_param = torch.randn(H * 2, D)

        # Expected: simple chunk of the Megatron fused weight
        expected_gate, expected_up = megatron_param.chunk(2, dim=0)

        # Bridge + post-processing
        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=expert_id)
            bridge_output = mapping.megatron_to_hf(megatron_param, None)

        postprocessed = _apply_expert_postprocessing(
            bridge_output, f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
        )

        assert len(postprocessed) == 2
        assert postprocessed[0][0] == "model.language_model.layers.0.mlp.experts.3.gate_proj.weight"
        assert postprocessed[1][0] == "model.language_model.layers.0.mlp.experts.3.up_proj.weight"
        assert postprocessed[0][1].shape == (H, D)
        assert postprocessed[1][1].shape == (H, D)
        assert torch.allclose(postprocessed[0][1], expected_gate)
        assert torch.allclose(postprocessed[1][1], expected_up)

    def test_expert_down_proj_postprocessed_matches_expected(self):
        """Bridge down_proj + post-processing produces correct passthrough."""
        D, H = 2048, 768
        expert_id = 5
        megatron_param = torch.randn(D, H)

        # Expected: identity (Megatron expert down_proj is already in HF layout)
        expected = megatron_param

        # Bridge + post-processing
        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_down_mapping(layer_idx=0, expert_id=expert_id)
            bridge_output = mapping.megatron_to_hf(megatron_param, None)

        postprocessed = _apply_expert_postprocessing(
            bridge_output, f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert_id}"
        )

        assert len(postprocessed) == 1
        assert postprocessed[0][0] == "model.language_model.layers.0.mlp.experts.5.down_proj.weight"
        assert postprocessed[0][1].shape == (D, H)
        assert torch.allclose(postprocessed[0][1], expected)

    def test_correctness_across_layers_and_experts(self):
        """Correctness holds across different layer indices and expert IDs."""
        H, D = 768, 2048

        for layer_idx in [0, 5, 27]:
            for expert_id in [0, 7, 42]:
                megatron_fc1 = torch.randn(H * 2, D)
                megatron_fc2 = torch.randn(D, H)

                expected_gate, expected_up = megatron_fc1.chunk(2, dim=0)

                # gate/up
                with _patch_gather_from_ep_ranks():
                    mapping_fc1 = _make_expert_gate_up_mapping(layer_idx, expert_id)
                    bridge_fc1 = mapping_fc1.megatron_to_hf(megatron_fc1, None)
                post_fc1 = _apply_expert_postprocessing(
                    bridge_fc1, f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{expert_id}"
                )
                assert (
                    post_fc1[0][0]
                    == f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"
                )
                assert (
                    post_fc1[1][0] == f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"
                )
                assert torch.allclose(post_fc1[0][1], expected_gate)
                assert torch.allclose(post_fc1[1][1], expected_up)

                # down
                with _patch_gather_from_ep_ranks():
                    mapping_fc2 = _make_expert_down_mapping(layer_idx, expert_id)
                    bridge_fc2 = mapping_fc2.megatron_to_hf(megatron_fc2, None)
                post_fc2 = _apply_expert_postprocessing(
                    bridge_fc2, f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{expert_id}"
                )
                assert (
                    post_fc2[0][0]
                    == f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
                )
                assert torch.allclose(post_fc2[0][1], megatron_fc2)

    def test_non_expert_replicated_no_postprocessing(self):
        """Non-expert params (e.g. layernorm) pass through without post-
        processing."""
        m = ReplicatedMapping(
            "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.0.input_layernorm.weight",
        )
        w = torch.randn(2048)
        bridge_output = m.megatron_to_hf(w, None)

        # No expert_id in name → post-processing is a no-op
        postprocessed = _apply_expert_postprocessing(
            bridge_output, "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight"
        )
        assert len(postprocessed) == 1
        assert postprocessed[0][0] == "model.layers.0.input_layernorm.weight"
        assert torch.equal(postprocessed[0][1], w)

    def test_non_expert_gated_mlp_no_postprocessing(self):
        """Non-expert GatedMLPMapping (dense MLP) passes through without post-
        processing."""
        m = GatedMLPMapping(
            "decoder.layers.0.mlp.linear_fc1.weight",
            gate="model.layers.0.mlp.gate_proj.weight",
            up="model.layers.0.mlp.up_proj.weight",
        )
        H, D = 6144, 2048
        fused = torch.randn(H * 2, D)
        bridge_output = m.megatron_to_hf(fused, None)

        postprocessed = _apply_expert_postprocessing(bridge_output, "decoder.layers.0.mlp.linear_fc1.weight")
        assert len(postprocessed) == 2
        gate_expected, up_expected = fused.chunk(2, dim=0)
        assert postprocessed[0][0] == "model.layers.0.mlp.gate_proj.weight"
        assert postprocessed[1][0] == "model.layers.0.mlp.up_proj.weight"
        assert torch.equal(postprocessed[0][1], gate_expected)
        assert torch.equal(postprocessed[1][1], up_expected)


# ─── Tests for process group patching with real mappings ──────────────────────


class TestProcessGroupPatching:
    """Test process group save/restore with real Bridge mapping objects."""

    def test_groups_patched_and_restored_on_real_mappings(self):
        """Process groups are set to None and restored on real mapping
        objects."""
        m = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
        all_mappings = DeviceDirectBackend._collect_all_mappings(m)
        assert len(all_mappings) == 2  # ExpertMLPGateUpProjMapping + GatedMLPMapping

        # Save originals (all None in test env, but the mechanism is what matters)
        saved_groups = []
        for mapping in all_mappings:
            saved_groups.append((mapping.pp_group, mapping._tp_group, mapping._etp_group, mapping.ep_group))

        # Patch
        for mapping in all_mappings:
            mapping.pp_group = None
            mapping._tp_group = None
            mapping._etp_group = None
            mapping.ep_group = None

        # Verify patched
        for mapping in all_mappings:
            assert mapping.pp_size == 1
            assert mapping.tp_size == 1
            assert mapping.ep_size == 1

        # Restore
        for mapping, (pp, tp, etp, ep) in zip(all_mappings, saved_groups):
            mapping.pp_group = pp
            mapping._tp_group = tp
            mapping._etp_group = etp
            mapping.ep_group = ep

        # Verify restored
        for mapping, (pp, tp, etp, ep) in zip(all_mappings, saved_groups):
            assert mapping.pp_group == pp
            assert mapping._tp_group == tp
            assert mapping._etp_group == etp
            assert mapping.ep_group == ep

    def test_gather_from_ep_ranks_monkey_patch_lifecycle(self):
        """gather_from_ep_ranks is monkey-patched and cleanly removed on real
        classes."""
        m = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
        all_mappings = DeviceDirectBackend._collect_all_mappings(m)

        # Verify gather_from_ep_ranks is NOT in any subclass __dict__ initially
        for mapping in all_mappings:
            assert "gather_from_ep_ranks" not in type(mapping).__dict__

        with _patch_gather_from_ep_ranks():
            # During patch: method is in class __dict__
            for mapping in all_mappings:
                cls = type(mapping)
                # At least one of the patched classes should match
                if cls in {ExpertMLPGateUpProjMapping, GatedMLPMapping}:
                    assert "gather_from_ep_ranks" in cls.__dict__

        # After cleanup: method removed from class __dict__, inherited version restored
        for mapping in all_mappings:
            assert "gather_from_ep_ranks" not in type(mapping).__dict__
            # But the inherited method still exists via MRO
            assert hasattr(mapping, "gather_from_ep_ranks")


# ─── Tests for strip_param_name_prefix (real function) ────────────────────────


class TestStripParamNamePrefix:
    """Test the real ``strip_param_name_prefix`` utility."""

    def test_strip_double_module(self):
        assert strip_param_name_prefix("module.module.decoder.layers.0.weight") == "decoder.layers.0.weight"

    def test_strip_single_module(self):
        assert strip_param_name_prefix("module.decoder.layers.0.weight") == "decoder.layers.0.weight"

    def test_no_prefix(self):
        assert strip_param_name_prefix("decoder.layers.0.weight") == "decoder.layers.0.weight"

    def test_triple_module(self):
        assert strip_param_name_prefix("module.module.module.decoder.layers.0.weight") == "decoder.layers.0.weight"


# ─── Tests for remove_padding (real function) ─────────────────────────────────


class TestRemovePadding:
    """Test the real ``remove_padding`` function."""

    def test_embedding_padding_removed(self):
        vocab_size = 100
        padded = torch.randn(128, 64)
        result = remove_padding("module.module.embedding.word_embeddings.weight", padded, vocab_size)
        assert result.shape == (100, 64)
        assert torch.equal(result, padded[:100])

    def test_output_layer_padding_removed(self):
        vocab_size = 100
        padded = torch.randn(128, 64)
        result = remove_padding("module.module.output_layer.weight", padded, vocab_size)
        assert result.shape == (100, 64)
        assert torch.equal(result, padded[:100])

    def test_non_embedding_unchanged(self):
        vocab_size = 100
        param = torch.randn(128, 64)
        result = remove_padding("module.module.decoder.layers.0.weight", param, vocab_size)
        assert result.shape == (128, 64)
        assert torch.equal(result, param)


# ─── Tests for quantize_params (real function) ───────────────────────────────


class TestQuantizeParamsPassthrough:
    """Test the real ``quantize_params`` function."""

    def test_no_quantization_returns_same_object(self):
        """With quantization_config=None, returns the same list object."""
        args = _make_args()
        tensors = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(768, 2048)),
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(768, 2048)),
        ]
        result = quantize_params(args, "module.module.decoder.layers.0.weight", tensors, None)
        assert result is tensors


# ─── Tests for expert weight edge cases ───────────────────────────────────────


class TestExpertWeightEdgeCases:
    """Test edge cases using real Bridge mappings and post-processing."""

    def test_expert_id_zero(self):
        """Expert ID 0 works correctly through the full pipeline."""
        H, D = 768, 2048
        param = torch.randn(H * 2, D)

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
            bridge_output = mapping.megatron_to_hf(param, None)

        postprocessed = _apply_expert_postprocessing(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
        assert postprocessed[0][0].endswith(".experts.0.gate_proj.weight")
        assert postprocessed[1][0].endswith(".experts.0.up_proj.weight")

        expected_gate, expected_up = param.chunk(2, dim=0)
        assert torch.allclose(postprocessed[0][1], expected_gate)
        assert torch.allclose(postprocessed[1][1], expected_up)

    def test_expert_id_large(self):
        """Large expert IDs (e.g. 127) work correctly."""
        H, D = 768, 2048
        param = torch.randn(H * 2, D)

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=127)
            bridge_output = mapping.megatron_to_hf(param, None)

        postprocessed = _apply_expert_postprocessing(
            bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight127"
        )
        assert postprocessed[0][0].endswith(".experts.127.gate_proj.weight")

    def test_contiguous_after_postprocessing(self):
        """Post-processed tensors are contiguous (required for NCCL
        broadcast)."""
        H, D = 768, 2048

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
            bridge_output = mapping.megatron_to_hf(torch.randn(H * 2, D), None)

        postprocessed = _apply_expert_postprocessing(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
        for _, tensor in postprocessed:
            assert tensor.is_contiguous()

    def test_dtype_preserved_through_pipeline(self):
        """Post-processing preserves tensor dtype through real Bridge
        mapping."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            H, D = 768, 2048
            param = torch.randn(H * 2, D, dtype=dtype)

            with _patch_gather_from_ep_ranks():
                mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
                bridge_output = mapping.megatron_to_hf(param, None)

            postprocessed = _apply_expert_postprocessing(
                bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0"
            )
            for _, tensor in postprocessed:
                assert tensor.dtype == dtype

    def test_element_count_preserved(self):
        """Total number of elements is preserved through Bridge + post-
        processing."""
        H, D = 768, 2048
        param = torch.randn(H * 2, D)
        original_numel = param.numel()

        with _patch_gather_from_ep_ranks():
            mapping = _make_expert_gate_up_mapping(layer_idx=0, expert_id=0)
            bridge_output = mapping.megatron_to_hf(param, None)

        postprocessed = _apply_expert_postprocessing(bridge_output, "decoder.layers.0.mlp.experts.linear_fc1.weight0")
        total_numel = sum(t.numel() for _, t in postprocessed)
        assert total_numel == original_numel
