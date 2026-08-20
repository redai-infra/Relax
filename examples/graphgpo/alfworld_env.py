# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-environment ALFWorld text adapter for the GraphGPO recipe.

The ALFWorld dependency is imported only when a real environment is created.
Tests and downstream recipes can inject an already constructed environment or
an environment factory without installing ALFWorld.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.graphgpo.state import TrackerState


_ITEM_STATE_ACTION = re.compile(r"^(heat|cool|clean|slice)\s+([\w\s\d]+?)(?:\s+with\s+[\w\s\d]+)?$")
_GO_TO_ACTION = re.compile(r"^go to\s+(.+)$")
_TAKE_ACTION = re.compile(r"^take\s+(.+?)(?:\s+from\s+.+)?$")
_DROP_ACTION = re.compile(r"^drop\s+(.+)$")
_PUT_ACTION = re.compile(r"^(put|place)\s+(.+?)\s+(?:in|on)\s+.+$")
_MOVE_ACTION = re.compile(r"^move\s+(.+?)(?:\s+to\s+.+)?$")


@dataclass(frozen=True)
class AlfWorldSnapshot:
    """The observable state after one ALFWorld reset or step."""

    raw_observation: str
    admissible_commands: tuple[str, ...]
    won: bool
    done: bool
    gamefile: str | None
    tracker: TrackerState


EnvironmentFactory = Callable[
    [str | Path, int, str, str | Path | None],
    Any,
]


def _canonicalize_game_files(base_env: Any) -> None:
    """Make ALFWorld's seed-to-task mapping independent of ``os.walk``
    order."""

    game_files = getattr(base_env, "game_files", None)
    if isinstance(game_files, (str, bytes)) or not isinstance(game_files, Sequence):
        raise TypeError("ALFWorld base environment must expose game_files")

    sortable: list[tuple[str, Any]] = []
    for game_file in game_files:
        try:
            normalized = os.fspath(game_file).replace("\\", "/")
        except TypeError as exc:
            raise TypeError("every ALFWorld game file must be path-like") from exc
        sortable.append((normalized, game_file))

    normalized_paths = [normalized for normalized, _ in sortable]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("ALFWorld game_files contains duplicate normalized paths")

    base_env.game_files = [game_file for _, game_file in sorted(sortable, key=lambda item: item[0])]
    if hasattr(base_env, "num_games"):
        base_env.num_games = len(base_env.game_files)


def _select_game_file(
    base_env: Any,
    selected_game_file: str | Path,
    *,
    data_root: str | Path | None,
) -> None:
    selected = os.fspath(selected_game_file).replace("\\", "/")
    if not selected or ".." in Path(selected).parts:
        raise ValueError("selected ALFWorld game file must be a safe path")

    root = None
    if data_root is not None:
        root = os.fspath(data_root).replace("\\", "/").rstrip("/")
        if not root:
            raise ValueError("ALFWORLD_DATA must not be empty")

    matches: list[Any] = []
    for game_file in base_env.game_files:
        normalized = os.fspath(game_file).replace("\\", "/")
        relative = None
        if root is not None and normalized.startswith(f"{root}/"):
            relative = normalized[len(root) + 1 :]
        if normalized == selected or relative == selected:
            matches.append(game_file)
    if len(matches) != 1:
        raise ValueError(f"selected ALFWorld game file must match exactly one discovered game; matched {len(matches)}")
    base_env.game_files = matches
    if hasattr(base_env, "num_games"):
        base_env.num_games = 1


def _default_environment_factory(
    config_path: str | Path,
    seed: int,
    train_eval: str,
    game_file: str | Path | None,
) -> Any:
    try:
        import yaml
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld is required for a real GraphGPO rollout; inject env= or env_factory= for dependency-free tests."
        ) from exc

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"ALFWorld config does not exist: {path}")
    with path.open(encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    if not isinstance(config, Mapping):
        raise ValueError("ALFWorld config must contain a mapping")
    try:
        env_type = config["env"]["type"]
    except (KeyError, TypeError) as exc:
        raise ValueError("ALFWorld config must define env.type") from exc

    base_env = get_environment(env_type)(config, train_eval=train_eval)
    _canonicalize_game_files(base_env)
    if game_file is not None:
        _select_game_file(
            base_env,
            game_file,
            data_root=os.environ.get("ALFWORLD_DATA"),
        )
    env = base_env.init_env(batch_size=1)
    seed_method = getattr(env, "seed", None)
    if not callable(seed_method):
        raise TypeError("ALFWorld text environment must provide seed(seed)")
    seed_method(seed)
    return env


def _single_batch_item(value: Any, *, field: str) -> Any:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field} must be a one-element batch")
    if not isinstance(value, Sequence) and not (hasattr(value, "__len__") and hasattr(value, "__getitem__")):
        raise TypeError(f"{field} must be a one-element batch")
    if len(value) != 1:
        raise ValueError(f"{field} must contain exactly one environment value")
    return value[0]


def _normalize_info(infos: Any) -> dict[str, Any]:
    if isinstance(infos, Mapping):
        return {str(key): _single_batch_item(value, field=f"infos[{key!r}]") for key, value in infos.items()}
    info = _single_batch_item(infos, field="infos")
    if not isinstance(info, Mapping):
        raise TypeError("the single environment info must be a mapping")
    return dict(info)


def _normalize_commands(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("info['admissible_commands'] must be a sequence of strings")
    commands: list[str] = []
    for command in value:
        if not isinstance(command, str):
            raise TypeError("every admissible command must be a string")
        commands.append(command)
    return tuple(commands)


class AlfWorldTextEnv:
    """Adapt ALFWorld's list-shaped text API to one explicit environment."""

    def __init__(
        self,
        *,
        env: Any | None = None,
        config_path: str | Path | None = None,
        seed: int = 0,
        train_eval: str = "train",
        game_file: str | Path | None = None,
        env_factory: EnvironmentFactory | None = None,
    ) -> None:
        if env is not None and (config_path is not None or game_file is not None or env_factory is not None):
            raise ValueError("env cannot be combined with config_path, game_file, or env_factory")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(train_eval, str) or not train_eval:
            raise ValueError("train_eval must be a non-empty string")
        if env is None and config_path is None:
            raise ValueError("config_path is required when env is not injected")

        self._env = env
        self._config_path = config_path
        self._seed = seed
        self._train_eval = train_eval
        self._game_file = game_file
        self._env_factory = env_factory or _default_environment_factory
        self._closed = False
        self._done = False
        self._location = "middle of a room"
        self._holding = "nothing"
        self._history_items: dict[str, dict[str, bool]] = {}
        self._item_location: dict[str, dict[str, str]] = {}
        self._history_list: list[tuple[str, ...]] = []

    def _ensure_env(self) -> Any:
        if self._closed:
            raise RuntimeError("ALFWorld environment is already closed")
        if self._env is None:
            if self._config_path is None:
                raise RuntimeError("ALFWorld config path is unavailable")
            self._env = self._env_factory(
                self._config_path,
                self._seed,
                self._train_eval,
                self._game_file,
            )
        return self._env

    def _tracker(self) -> TrackerState:
        return TrackerState.from_mapping(
            {
                "location": self._location,
                "holding": self._holding,
                "history_items": copy.deepcopy(self._history_items),
                "item_location": copy.deepcopy(self._item_location),
            }
        )

    def _snapshot(
        self,
        *,
        raw_observation: Any,
        info: Mapping[str, Any],
        done: bool,
    ) -> AlfWorldSnapshot:
        if not isinstance(raw_observation, str):
            raise TypeError("ALFWorld observation must be a string")
        if "admissible_commands" not in info:
            raise KeyError("ALFWorld info is missing 'admissible_commands'")
        gamefile = info.get("extra.gamefile")
        if gamefile is not None and not isinstance(gamefile, str):
            raise TypeError("info['extra.gamefile'] must be a string or null")
        return AlfWorldSnapshot(
            raw_observation=raw_observation,
            admissible_commands=_normalize_commands(info["admissible_commands"]),
            won=bool(info.get("won", False)),
            done=done,
            gamefile=gamefile,
            tracker=self._tracker(),
        )

    def reset(self) -> AlfWorldSnapshot:
        env = self._ensure_env()
        result = env.reset()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("ALFWorld reset() must return (observations, infos)")
        observations, infos = result
        raw_observation = _single_batch_item(observations, field="observations")
        info = _normalize_info(infos)

        self._history_items = {}
        self._location = "middle of a room"
        self._holding = "nothing"
        self._item_location = {}
        self._history_list = []
        self._done = False
        return self._snapshot(
            raw_observation=raw_observation,
            info=info,
            done=False,
        )

    def _update_item_location(self, old_holding: str) -> None:
        obj = old_holding if old_holding != "nothing" else self._holding
        if obj == "nothing":
            raise AssertionError("a holding transition must identify an object")
        if obj not in self._item_location:
            if self._holding != obj:
                raise AssertionError("a new item location requires holding that item")
            self._item_location[obj] = {
                "old_location": self._location,
                "new_location": self._location,
            }
        elif old_holding == obj:
            self._item_location[obj]["new_location"] = self._location
        elif self._holding == obj:
            self._item_location[obj]["new_location"] = self._item_location[obj]["old_location"]

    def _update_item_state(self, action: str, observation: str) -> None:
        match = _ITEM_STATE_ACTION.match(action.lower().strip())
        if match is None:
            return
        verb, obj = match.groups()
        obj = obj.strip()
        if verb not in observation:
            return
        if obj not in self._history_items:
            # ``slice`` versus ``sliced`` below intentionally preserves the
            # frozen AlfworldWorker tracker schema used by the reference run.
            self._history_items[obj] = {
                "heated": False,
                "cooled": False,
                "cleaned": False,
                "slice": False,
            }
        if verb == "heat":
            self._history_items[obj]["heated"] = True
        elif verb == "cool":
            self._history_items[obj]["cooled"] = True
        elif verb == "clean":
            self._history_items[obj]["cleaned"] = True
        elif verb == "slice":
            self._history_items[obj]["sliced"] = True

    def _update_position(self, action: str, observation: str) -> None:
        match = _GO_TO_ACTION.match(action.lower().strip())
        if match is None:
            return
        target_location = match.group(1).strip()
        if target_location not in observation:
            raise AssertionError(f"target location {target_location!r} is absent from observation")
        self._location = target_location

    def _update_held_items(self, action: str, observation: str) -> None:
        action = action.lower().strip()
        old_holding = self._holding

        match_take = _TAKE_ACTION.match(action)
        if match_take is not None:
            obj = match_take.group(1).strip()
            if self._holding != "nothing":
                raise AssertionError("take action requires empty hands")
            if obj in observation:
                self._holding = obj

        match_drop = _DROP_ACTION.match(action)
        if match_drop is not None:
            obj = match_drop.group(1).strip()
            if obj in observation:
                self._holding = "nothing"

        match_put = _PUT_ACTION.match(action)
        if match_put is not None:
            obj = match_put.group(2).strip()
            if obj in observation:
                self._holding = "nothing"

        match_move = _MOVE_ACTION.match(action)
        if match_move is not None:
            obj = match_move.group(1).strip()
            if self._holding != obj:
                raise AssertionError("move action requires holding the moved object")
            if obj in observation:
                self._holding = "nothing"

        if old_holding != self._holding:
            self._update_item_location(old_holding)

    def step(self, action: str) -> AlfWorldSnapshot:
        if not isinstance(action, str):
            raise TypeError("action must be a string")
        if self._done:
            raise RuntimeError("cannot step a completed ALFWorld environment")
        env = self._ensure_env()
        result = env.step([action])
        if not isinstance(result, tuple) or len(result) != 4:
            raise TypeError("ALFWorld step() must return (observations, scores, dones, infos)")
        observations, scores, dones, infos = result
        raw_observation = _single_batch_item(observations, field="observations")
        _single_batch_item(scores, field="scores")
        done = bool(_single_batch_item(dones, field="dones"))
        info = _normalize_info(infos)

        self._history_list.append(
            (
                f"action:{action}",
                f"obs:{observations}",
                f"self.location: {self._location}",
                f"self.holding: {self._holding}",
                f"self.history_items: {self._history_items}",
                f"self.item_location: {self._item_location}",
            )
        )
        if "Nothing happens" not in raw_observation and not self._done:
            self._update_item_state(action, raw_observation)
            self._update_position(action, raw_observation)
            self._update_held_items(action, raw_observation)
        self._done = done
        return self._snapshot(
            raw_observation=raw_observation,
            info=info,
            done=done,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        env = self._env
        self._env = None
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()
