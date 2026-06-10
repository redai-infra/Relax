# Copyright (c) 2026 Relax Authors. All Rights Reserved.

try:
    from enum import StrEnum
except ImportError:
    # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from relax.components.actor import Actor
from relax.components.actor_fwd import ActorFwd
from relax.components.advantages import Advantages
from relax.components.rollout import Rollout
from relax.components.sft import SFT


# NOTE(dev): Use StrEnum and keep visiting order with definition order
class ROLES(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"
    actor_fwd: str = "actor_fwd"
    sft: str = "sft"


class ROLES_TRAIN_ONLY(StrEnum):
    actor: str = "actor"


class ROLES_ROLLOUT_ONLY(StrEnum):
    rollout: str = "rollout"


class ROLES_COLOCATE(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"


class ROLES_SFT_ONLY(StrEnum):
    sft: str = "sft"
    actor: str = "actor"


class ROLES_FULLY_ASYNC_ON_POLICY(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"


ALGOS = {
    "grpo": {
        ROLES.rollout: Rollout,
        ROLES.actor: Actor,
        ROLES.advantages: Advantages,
        ROLES.reference: ActorFwd,
        ROLES.actor_fwd: ActorFwd,
    },
    "gspo": {
        ROLES.rollout: Rollout,
        ROLES.actor: Actor,
        ROLES.advantages: Advantages,
        ROLES.reference: ActorFwd,
        ROLES.actor_fwd: ActorFwd,
    },
    "sapo": {
        ROLES.rollout: Rollout,
        ROLES.actor: Actor,
        ROLES.advantages: Advantages,
        ROLES.reference: ActorFwd,
        ROLES.actor_fwd: ActorFwd,
    },
    "sft": {
        ROLES.sft: SFT,
        ROLES.actor: Actor,
    },
}


def process_role(config):
    if config.debug_rollout_only:
        return ROLES_ROLLOUT_ONLY
    if config.debug_train_only:
        return ROLES_TRAIN_ONLY
    if getattr(config, "loss_type", None) == "sft":
        return ROLES_SFT_ONLY
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        if getattr(config, "true_on_policy_mode", False):
            # actor_fwd's log_probs equal the train forward's log_probs in this regime
            # (same weights, deterministic Megatron forward), so we recompute inline.
            return ROLES_FULLY_ASYNC_ON_POLICY
        return ROLES
    else:
        return ROLES_COLOCATE
