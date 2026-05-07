# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses

from relax.utils import megatron_bridge_utils
from relax.utils.misc import chunk_named_params_by_size

from ..misc_utils import strip_param_name_prefix
from ..weight_conversion import postprocess_hf_param
from ..weight_conversion.processors import quantize_params
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            def iter_quantized_named_weights():
                hf_to_megatron_mapping = None

                for item in named_weights:
                    # Compatibility shim: old megatron-bridge yields 3-tuples
                    # ``(hf_param_name, weight, megatron_param_name)`` while
                    # the official bridge yields 2-tuples ``(hf_param_name, weight)``.
                    # Dispatch per-item so the same code path supports both.
                    if len(item) == 3:
                        hf_param_name, weight, megatron_param_name = item
                    elif len(item) == 2:
                        hf_param_name, weight = item
                        if hf_to_megatron_mapping is None:
                            hf_to_megatron_mapping = _build_hf_to_megatron_mapping(conversion_tasks)
                        # With PP > 1, export_hf_weights yields params from ALL
                        # PP ranks (via internal PP broadcast), but
                        # hf_to_megatron_mapping only contains params from this
                        # rank's conversion tasks.  For remote PP rank params
                        # we fall back to hf_param_name — this is safe because
                        # remove_padding checks megatron-style names and
                        # quantize_params_fp8 regex won't match HF-style names.
                        megatron_param_name = hf_to_megatron_mapping.get(hf_param_name, hf_param_name)
                    else:
                        raise ValueError(
                            f"Unexpected named_weights tuple length {len(item)} from "
                            f"megatron-bridge.export_hf_weights(); expected 2 (new) or 3 (old). "
                            f"Item: {item!r}"
                        )

                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )

                    converted_named_params = [(hf_param_name, processed_weight)]

                    quantized_batch = quantize_params(
                        args=self.args,
                        megatron_name=megatron_param_name,
                        converted_named_params=converted_named_params,
                        quantization_config=self.quantization_config,
                    )

                    yield from quantized_batch

            yield from chunk_named_params_by_size(
                iter_quantized_named_weights(),
                chunk_size=self.args.update_weight_buffer_size,
            )


def _build_hf_to_megatron_mapping(conversion_tasks):
    """Build a mapping from HF parameter names to megatron parameter names.

    Only relevant for the official megatron-bridge whose ``export_hf_weights``
    yields 2-tuples ``(hf_name, weight)`` and no longer carries the megatron
    name in the tuple.  We reconstruct the mapping by reading
    ``task.mapping.hf_param`` — a pure metadata attribute that requires NO
    collective communication.  This is critical for PP > 1 where different
    ranks hold different parameter subsets; calling ``megatron_to_hf()`` (which
    contains PP broadcast / TP gather) with inconsistent tasks across ranks
    would deadlock.

    ``mapping.hf_param`` is either:
    - ``str``: simple 1-to-1 mappings (AutoMapping, DirectMapping, …)
    - ``dict``: multi-output mappings (QKVMapping ``{"q","k","v"}``,
      GatedMLPMapping ``{"gate","up"}``)

    This mirrors the approach shown in the official ``get_conversion_tasks``
    docstring of megatron-bridge's ``AutoBridge``.

    Note: with PP > 1, each rank only holds a subset of conversion tasks, so
    the returned mapping is **incomplete** — it covers only the params that
    belong to this PP rank.  ``export_hf_weights`` yields params from ALL PP
    ranks (via internal PP broadcast), so callers must handle missing keys
    gracefully (e.g. fall back to the HF param name).
    """
    hf_to_megatron_mapping = {}

    for task in conversion_tasks:
        megatron_param_name = task.param_name
        hf_param = task.mapping.hf_param

        if isinstance(hf_param, str):
            hf_to_megatron_mapping[hf_param] = megatron_param_name
        elif isinstance(hf_param, dict):
            for hf_name in hf_param.values():
                hf_to_megatron_mapping[hf_name] = megatron_param_name
        else:
            raise TypeError(
                f"Unexpected mapping.hf_param type {type(hf_param).__name__} "
                f"for megatron param '{megatron_param_name}': {hf_param!r}"
            )

    return hf_to_megatron_mapping


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    """Replace param_weight in each conversion task with the latest trained
    weights.

    build_conversion_tasks() returns ``List[None | WeightConversionTask]``
    where None entries correspond to global params that have no mapping.  We
    filter them out here so that downstream consumers never see None.
    """

    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert weight_dict_key in new_weight_dict, (
            f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"
        )

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    # Filter out None tasks (params with no mapping in build_conversion_tasks)
    valid_tasks = [t for t in vanilla_conversion_tasks if t is not None]
    return _MapWithLen(_handle_one, valid_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
