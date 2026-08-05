from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

from relax.utils import device as device_utils


_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]

_NON_BLOCKING = device_utils.use_non_blocking_copy()
_PIN_MEMORY = device_utils.use_pinned_host_memory()

# Megatron attaches these as plain Python attributes on live nn.Parameter
# objects (not part of the tensor's storage), so torch.empty_like() alone
# drops them. Consumers that TP-gather from a backed-up snapshot instead of
# the live model (e.g. DCS's all_gather_param) need them preserved.
_MEGATRON_TP_ATTRS = ("tensor_model_parallel", "partition_dim", "partition_stride", "parallel_mode")


def _attach_tp_attrs(snapshot: torch.Tensor, param: torch.Tensor, backfill_defaults) -> None:
    for attr in _MEGATRON_TP_ATTRS:
        if hasattr(param, attr):
            setattr(snapshot, attr, getattr(param, attr))
    # Some params (e.g. this model's word_embeddings under bridge mode) aren't
    # explicitly marked at model-construction time and only get backfilled
    # with Megatron's own "not parallel" defaults later (see
    # set_defaults_if_not_set_tensor_model_parallel_attributes callers) —
    # apply the same backfill here so downstream TP-gather (e.g. DCS's
    # all_gather_param) can rely on the attributes always being present, same
    # contract Megatron itself guarantees for the live model.
    backfill_defaults(snapshot)


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag):
        if single_tag is None:
            return _TensorBackuperNormal(source_getter=source_getter)
        else:
            return _TensorBackuperNoop(source_getter=source_getter, single_tag=single_tag)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str, *, on_device: bool = False):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str, *, on_device: bool = False) -> None:
        """Snapshot the source tensors into ``tag``.

        By default the snapshot lives on host-pinned memory. Passing
        ``on_device=True`` keeps it on the source tensor's own device instead
        (a D2D copy, avoiding the D2H/H2D PCIe round trip), at the cost of
        holding a full extra device-resident copy of every tensor in ``tag``
        for as long as the tag exists.
        """
        from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes

        backup_dict = self._backups[tag]
        missing = [(name, param) for name, param in self._source_getter() if name not in backup_dict]
        if on_device and missing:
            # One cudaMalloc per parameter (hundreds for a real model) leaves the
            # CUDA caching allocator holding hundreds of small, permanently
            # non-reusable segments, fragmenting the segment table that later
            # dynamic-batch forward passes search when allocating activations.
            # Allocate one contiguous buffer per dtype instead, and give every
            # parameter a shape/stride view into it; each view is still a
            # distinct Tensor object so it can carry its own TP attributes.
            by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = defaultdict(list)
            for name, param in missing:
                by_dtype[param.dtype].append((name, param))
            for dtype, entries in by_dtype.items():
                device = entries[0][1].device
                flat_buffer = torch.empty(sum(p.numel() for _, p in entries), dtype=dtype, device=device)
                offset = 0
                for name, param in entries:
                    numel = param.numel()
                    snapshot = flat_buffer[offset : offset + numel].view(param.shape)
                    offset += numel
                    _attach_tp_attrs(snapshot, param, set_defaults_if_not_set_tensor_model_parallel_attributes)
                    backup_dict[name] = snapshot
        else:
            for name, param in missing:
                snapshot = torch.empty_like(param, device=torch.device("cpu"), pin_memory=_PIN_MEMORY)
                _attach_tp_attrs(snapshot, param, set_defaults_if_not_set_tensor_model_parallel_attributes)
                backup_dict[name] = snapshot

        for name, param in self._source_getter():
            backup_dict[name].copy_(param.detach(), non_blocking=_NON_BLOCKING)
        device_utils.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=_NON_BLOCKING)
        device_utils.synchronize()


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag
        # Sanity check for safety
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str, *, on_device: bool = False) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        device_utils.synchronize()

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        device_utils.synchronize()


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    # Not a real/good hash, but pretty fast
    x = x.contiguous()
    x = x.view(-1)
    x = x.view(torch.uint32)
    x = x.sum()
    return x.item()
