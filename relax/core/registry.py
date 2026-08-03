# Copyright (c) 2026 Relax Authors. All Rights Reserved.

try:
    from enum import StrEnum
except ImportError:
    # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


from relax.components.actor import Actor
from relax.components.actor_fwd import ActorFwd
from relax.components.advantages import Advantages
from relax.components.critic import Critic
from relax.components.rollout import Rollout
from relax.components.sft import SFT
from relax.core.service_plan import (
    MODE_COLOCATE,
    MODE_FULLY_ASYNC,
    MODE_FULLY_ASYNC_ON_POLICY,
    MODE_PPO_COLOCATE,
    MODE_PPO_FULLY_ASYNC,
    MODE_PPO_FULLY_ASYNC_ON_POLICY,
    MODE_ROLLOUT_ONLY,
    MODE_SFT,
    MODE_TRAIN_ONLY,
    resolve_role_mode,
)


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


class ROLES_PPO_COLOCATE(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"


class ROLES_PPO_FULLY_ASYNC(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"
    actor_fwd: str = "actor_fwd"


class ROLES_PPO_FULLY_ASYNC_ON_POLICY(StrEnum):
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
    "cispo": {
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
    "ppo": {
        ROLES.rollout: Rollout,
        ROLES.actor: Actor,
        ROLES.critic: Critic,
        ROLES.advantages: Advantages,
        ROLES.reference: ActorFwd,
        ROLES.actor_fwd: ActorFwd,
    },
}


def process_role(config):
    role_sets = {
        MODE_TRAIN_ONLY: ROLES_TRAIN_ONLY,
        MODE_ROLLOUT_ONLY: ROLES_ROLLOUT_ONLY,
        MODE_COLOCATE: ROLES_COLOCATE,
        MODE_SFT: ROLES_SFT_ONLY,
        MODE_FULLY_ASYNC: ROLES,
        MODE_FULLY_ASYNC_ON_POLICY: ROLES_FULLY_ASYNC_ON_POLICY,
        MODE_PPO_COLOCATE: ROLES_PPO_COLOCATE,
        MODE_PPO_FULLY_ASYNC: ROLES_PPO_FULLY_ASYNC,
        MODE_PPO_FULLY_ASYNC_ON_POLICY: ROLES_PPO_FULLY_ASYNC_ON_POLICY,
    }
    return role_sets[resolve_role_mode(config)]
