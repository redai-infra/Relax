from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

from relax.utils import device as device_utils


_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]

_NON_BLOCKING = device_utils.use_non_blocking_copy()
_PIN_MEMORY = device_utils.use_pinned_host_memory()


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
    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def ema(self, *, source_tag: str, target_tag: str, alpha: float) -> None:
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
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=_PIN_MEMORY)
            backup_dict[name].copy_(param.detach(), non_blocking=_NON_BLOCKING)
        device_utils.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def ema(self, *, source_tag: str, target_tag: str, alpha: float) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}.")
        source = self._backups[source_tag]
        target = self._backups[target_tag]
        if source.keys() != target.keys():
            raise ValueError("EMA source and target snapshots must have identical keys.")
        for name, source_tensor in source.items():
            target_tensor = target[name]
            if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
                raise ValueError(f"EMA snapshot mismatch for {name}.")
            if alpha == 1:
                target_tensor.copy_(source_tensor)
            elif not (source_tensor.is_floating_point() or source_tensor.is_complex()):
                target_tensor.copy_(source_tensor)
            else:
                target_tensor.mul_(1 - alpha).add_(source_tensor, alpha=alpha)

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

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        device_utils.synchronize()

    def ema(self, *, source_tag: str, target_tag: str, alpha: float) -> None:
        raise RuntimeError("SDPO EMA requires --enable-weights-backuper.")

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
