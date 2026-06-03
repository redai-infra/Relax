# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Terminal UI for browsing Relax ``rollout_result/*.jsonl`` files.

Adapted from ``redaccel/verl/tools/reward_viewer_v2.py`` (RedAccel
Authors). Trimmed to match Relax's per-sample summary schema: drops the
agent-trace / token-stats / data-source filters that depend on extra
sqlite dumps RedAccel produces and Relax does not. Adds a ``dataset``
filter that activates when eval JSONL files are loaded.

Requires the optional dependencies ``textual`` and ``rich``::

    pip install textual rich
"""

from __future__ import annotations

import json
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional


_INDEX_KEY = "__IDX"
_FILE_SUFFIX = ".jsonl"
_DEFAULT_MASK_STR = r"<\|image_pad\|>|<\|imgpad\|>|<\|audio_comp_pad\|>"


def _require_textual():
    """Import textual / rich lazily; raise a friendly error if missing."""
    try:
        import rich  # noqa: F401
        import textual  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TUI mode needs the optional 'textual' and 'rich' packages. Install them with: pip install textual rich"
        ) from e


def _load_path(p: Path, mask_str: str) -> list[dict]:
    samples: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in list(d.keys()):
                if isinstance(d[k], str):
                    if mask_str:
                        d[k] = re.sub(mask_str, "*", d[k])
                else:
                    d[k] = json.dumps(d[k], ensure_ascii=False, indent=4)
            d[_INDEX_KEY] = len(samples)
            samples.append(d)
    return samples


def _build_app(step_num: int, data: dict, file_idx_map: dict):
    """Build the textual App instance.

    Lazy-imported textual stays scoped here.
    """
    from rich.highlighter import ReprHighlighter
    from rich.table import Table
    from rich.text import Text
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, Input, Select, SelectionList, Static

    class _Highlighter(ReprHighlighter):
        highlights = ReprHighlighter.highlights + [
            r"(?P<tag_name>[][\<\>{}()\|（）【】\[\]=`])",
            r"\<\|(?P<tag_name>[\w\W]*?)\|\>",
        ]

    def _center(word: str, total: int, char: str = "=") -> str:
        if len(word) > total:
            return word
        pad = total - len(word)
        return char * (pad // 2) + " " + word + " " + char * ((pad + 1) // 2)

    def _highlight_kw(content: str, keyword: Optional[str]):
        if not keyword:
            return Text(content)
        text = Text()
        parts = content.split(keyword)
        for i, part in enumerate(parts):
            text.append(part)
            if i < len(parts) - 1:
                text.append(keyword, style="on #8f51b5")
        return text

    class JsonLineViewer(App):
        BINDINGS = [
            ("left", "focus_previous", "Focus Previous"),
            ("right", "focus_next", "Focus Next"),
            ("s", "switch_render", "switch render"),
            Binding("n", "next_sample", "Next Sample", key_display="n"),
            Binding("N", "next_step", "Next Step", key_display="N"),
            Binding("p", "previous_sample", "Previous Sample", key_display="p"),
            Binding("P", "previous_step", "Previous Step", key_display="P"),
            ("r", "refresh_page", "Refresh"),
            ("f", "toggle_search", "Find"),
            ("enter", "next_search", "Find next"),
            ("escape", "cancel_search", "Cancel find"),
            ("j", "page_down", "page down"),
            ("k", "page_up", "page up"),
            ("h", "page_left", "page left"),
            ("l", "page_right", "page right"),
            Binding("g", "page_home", "top", key_display="g"),
            Binding("G", "page_end", "bottom", key_display="G"),
        ]
        TITLE = "Relax Rollout Result Viewer (TUI)"
        CSS = """
        Select:focus > SelectCurrent { border: tall #8f51b5; }
        Select.-expanded > SelectCurrent { border: tall #8f51b5; }
        #select-container { width: 22%; height: 100%; align: center top; }
        #search-container { height: 3; align: center top; }
        #search-box { width: 80%; }
        #scroll-view { border: round #444; }
        """

        def __init__(self) -> None:
            super().__init__()
            self.step_num = step_num
            self.file_idx_map = file_idx_map
            self.data = data
            self.render_table = False
            self.selected_step_index = 0
            self.selected_sample_index = 0
            self.matches: list[dict] = []
            self.current_match_index = 0
            self.highlighter = _Highlighter()

            first_samples = data[next(iter(data))]["samples"]
            self.filter_fields = [(f, f, True) for f in first_samples[0].keys()] if first_samples else []
            self.sample_num = len(first_samples)

            if first_samples and "dataset" in first_samples[0]:
                self.datasets = sorted({s.get("dataset", "") for s in first_samples})
            else:
                self.datasets = []
            self.datasets.insert(0, "all datasets")

            self.sort_mode = 0
            self.ds_index = 0

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="search-container"):
                yield Input(placeholder="find...", id="search-box")
                yield Static("", id="search-status")
            with Horizontal():
                with Vertical(id="select-container"):
                    yield Select(
                        id="step-select",
                        value=0,
                        prompt="step",
                        options=[("step: 1", 0)],
                        allow_blank=False,
                    )
                    yield Select(
                        id="sample-select",
                        value=0,
                        prompt="sample",
                        options=[("sample: 1", 0)],
                        allow_blank=False,
                    )
                    yield Select(
                        id="ds-select",
                        value=0,
                        prompt="dataset",
                        options=[("all datasets", 0)],
                        allow_blank=False,
                    )
                    yield Select(
                        id="sample-sort",
                        value=0,
                        prompt="sort",
                        options=[
                            ("no sort", 0),
                            ("reward asc", 1),
                            ("reward desc", 2),
                            ("resp len asc", 3),
                            ("resp len desc", 4),
                        ],
                        allow_blank=False,
                    )
                    yield SelectionList[int](("Select ALL", 1, True), id="fields-select-all")
                    with VerticalScroll(id="scroll-view2"):
                        yield SelectionList[str](*self.filter_fields, id="fields-select")
                with VerticalScroll(id="scroll-view"):
                    yield Static("Loading...", id="content", markup=False)
            yield Footer()

        async def on_mount(self) -> None:
            self.step_select = self.query_one("#step-select", Select)
            self.sample_select = self.query_one("#sample-select", Select)
            self.ds_select = self.query_one("#ds-select", Select)
            self.sample_sort = self.query_one("#sample-sort", Select)
            self.content_display = self.query_one("#content", Static)
            self.search_box = self.query_one("#search-box", Input)
            self.scroll_view = self.query_one("#scroll-view", VerticalScroll)
            self.search_status = self.query_one("#search-status", Static)
            self.fields_select = self.query_one("#fields-select", SelectionList)
            self.fields_select.border_title = "field filter"

            if self.data:
                self.step_select.set_options([(f"step: {i + 1}", i) for i in range(self.step_num)])
                self.sample_select.set_options([(f"sample: {i + 1}", i) for i in range(self.sample_num)])
                self.ds_select.set_options([(f"{ds}", i) for i, ds in enumerate(self.datasets)])
                self.step_select.focus()
                await self.update_content()

        async def update_options(self, sort_mode: int, ds_index: int, offset: int = 0) -> None:
            if self.selected_step_index not in self.data:
                self.selected_sample_index = offset
                return
            samples = list(self.data[self.selected_step_index].get("samples", []))
            if not samples:
                self.selected_sample_index = offset
                return

            if ds_index > 0:
                want = self.datasets[ds_index]
                samples = [s for s in samples if s.get("dataset") == want]

            def _resp_len(x):
                try:
                    return int(x.get("response_length", 0))
                except (TypeError, ValueError):
                    return 0

            def _reward(x):
                try:
                    return float(x.get("reward", 0))
                except (TypeError, ValueError):
                    return 0.0

            if sort_mode == 1:
                samples.sort(key=_reward)
            elif sort_mode == 2:
                samples.sort(key=_reward, reverse=True)
            elif sort_mode == 3:
                samples.sort(key=_resp_len)
            elif sort_mode == 4:
                samples.sort(key=_resp_len, reverse=True)
            else:
                samples.sort(key=lambda x: x[_INDEX_KEY])

            options = [(f"sample: {r[_INDEX_KEY] + 1}", r[_INDEX_KEY]) for r in samples]
            self.sample_select.set_options(options or [("(empty)", 0)])
            self.sample_num = len(samples)
            self.selected_sample_index = offset
            self.sort_mode = sort_mode
            self.sample_sort.value = sort_mode
            self.ds_index = ds_index
            self.ds_select.value = ds_index

        async def update_content(self, search_keyword: Optional[str] = None) -> None:
            try:
                samples = self.data[self.selected_step_index].get("samples", [])
                options = self.sample_select._options
                if not options or not samples:
                    self.content_display.update("No samples.")
                    return
                content_dict = samples[options[self.selected_sample_index][1]]
                content_dict = {k: v for k, v in content_dict.items() if k in self.fields_select.selected}
                if self.render_table:
                    content = Table("key", "value", show_lines=True)
                    for k, v in content_dict.items():
                        content.add_row(k, self.highlighter(_highlight_kw(f"{v}", search_keyword)))
                else:
                    text = Text()
                    for k, v in content_dict.items():
                        text.append(_highlight_kw(_center(k, 64) + f"\n{v}\n", search_keyword))
                    content = self.highlighter(text)
            except KeyError:
                content = f"Loading data asynchronously: {len(self.data)}/{self.step_num} step"
            except Exception:
                content = self.highlighter(traceback.format_exc())
            self.content_display.update(content)

        @on(Select.Changed, "#step-select")
        async def _step_changed(self, event):
            self.selected_step_index = event.value
            await self.update_options(self.sort_mode, self.ds_index)
            await self.update_content()

        @on(Select.Changed, "#sample-select")
        async def _sample_changed(self, event):
            for i, (_, sample_id) in enumerate(self.sample_select._options):
                if sample_id == event.value:
                    self.selected_sample_index = i
                    break
            await self._clear_search()
            await self.update_content()

        @on(Select.Changed, "#sample-sort")
        async def _sort_changed(self, event):
            await self.update_options(sort_mode=event.value, ds_index=self.ds_index)
            await self.update_content()

        @on(Select.Changed, "#ds-select")
        async def _ds_changed(self, event):
            await self.update_options(sort_mode=self.sort_mode, ds_index=event.value)
            await self.update_content()

        @on(SelectionList.SelectedChanged, "#fields-select")
        async def _fields_changed(self, event):
            await self.update_content()

        @on(SelectionList.SelectedChanged, "#fields-select-all")
        async def _fields_all_changed(self, event):
            s = self.query_one("#fields-select-all", SelectionList)
            if s.selected:
                self.fields_select.select_all()
            else:
                self.fields_select.deselect_all()

        def action_focus_previous(self):
            self.screen.focus_previous()

        def action_focus_next(self):
            self.screen.focus_next()

        async def action_next_step(self) -> None:
            self.selected_step_index = (self.selected_step_index + 1) % self.step_num
            self.step_select.value = self.selected_step_index
            await self.update_options(self.sort_mode, self.ds_index)
            await self.update_content()

        async def action_previous_step(self) -> None:
            self.selected_step_index = (self.selected_step_index - 1) % self.step_num
            self.step_select.value = self.selected_step_index
            await self.update_options(self.sort_mode, self.ds_index)
            await self.update_content()

        async def action_next_sample(self) -> None:
            if not self.sample_num:
                return
            self.selected_sample_index = (self.selected_sample_index + 1) % self.sample_num
            self.sample_select.value = self.sample_select._options[self.selected_sample_index][1]
            await self._clear_search()
            await self.update_content()

        async def action_previous_sample(self) -> None:
            if not self.sample_num:
                return
            self.selected_sample_index = (self.selected_sample_index - 1) % self.sample_num
            self.sample_select.value = self.sample_select._options[self.selected_sample_index][1]
            await self._clear_search()
            await self.update_content()

        async def action_refresh_page(self) -> None:
            await self.update_content()

        async def action_switch_render(self) -> None:
            self.render_table = not self.render_table
            await self.update_content()

        def action_toggle_search(self) -> None:
            self.search_box.focus()

        async def action_cancel_search(self) -> None:
            self.search_box.value = ""
            await self._clear_search()
            await self.update_content()

        async def _clear_search(self) -> None:
            self.matches = []
            self.search_status.update("")
            self.current_match_index = 0

        @on(Input.Submitted, "#search-box")
        async def _on_search(self, event: "Input.Submitted") -> None:
            self.matches = []
            self.current_match_index = 0
            if not event.value:
                return
            await self.update_content(event.value)
            renderable = self.content_display.render()
            if isinstance(renderable, Table):
                return
            console = self.content_display._console
            lines = renderable.wrap(console, self.scroll_view.container_size.width)
            seen = set()
            for line_idx, line in enumerate(lines):
                if line_idx in seen:
                    continue
                if event.value in line:
                    self.matches.append({"line": line_idx, "word": event.value})
                    seen.add(line_idx)
            self.scroll_view.focus()
            await self.action_next_search()

        async def action_next_search(self) -> None:
            if not self.matches or self.current_match_index >= len(self.matches):
                return
            target_line = self.matches[self.current_match_index]["line"]
            self.scroll_view.scroll_to(x=0, y=target_line, animate=False)
            self.current_match_index = (self.current_match_index + 1) % len(self.matches)
            self.search_status.update(
                Text(f"Find: {self.current_match_index + 1}/{len(self.matches)}", style="bold on #8f51b5")
            )

        async def action_page_up(self):
            self.scroll_view.scroll_page_up(animate=False)

        async def action_page_down(self):
            self.scroll_view.scroll_page_down(animate=False)

        async def action_page_left(self):
            self.scroll_view.scroll_left(animate=False)

        async def action_page_right(self):
            self.scroll_view.scroll_right(animate=False)

        def action_page_home(self):
            self.scroll_view.scroll_home(animate=False)

        def action_page_end(self):
            self.scroll_view.scroll_end(animate=False)

    return JsonLineViewer()


def _stem_key(p: Path) -> int:
    return int(p.stem) if p.stem.isdigit() else 0


def run(data_dir: str, mask_str: str = _DEFAULT_MASK_STR) -> None:
    """Launch the TUI viewer on ``data_dir``.

    If ``data_dir`` contains a ``train/`` subdir it is used; otherwise
    ``eval/``; otherwise ``data_dir`` itself. Point at the subdir explicitly to
    override.

    Loading is synchronous and finishes before textual takes over the
    terminal, so any error during data loading prints normally rather than
    being hidden by the alt-screen. Errors that occur inside the textual
    app (compose, mount, callbacks) are re-raised after the terminal is
    restored.
    """
    _require_textual()
    path = Path(data_dir).resolve()
    if not path.exists():
        raise ValueError(f"Data directory does not exist: {path}")
    if (path / "train").is_dir():
        path = path / "train"
    elif (path / "eval").is_dir():
        path = path / "eval"

    paths = sorted(path.glob(f"*{_FILE_SUFFIX}"), key=_stem_key)
    if not paths:
        raise ValueError(
            f"No {_FILE_SUFFIX} files found under {path}. "
            f"Point DATA_DIR at a directory that contains '{{step}}.jsonl' files "
            f"(or a parent with train/ or eval/ subdirs)."
        )

    print(f"TUI using: {path} ({len(paths)} jsonl file(s))")

    # Sync-load step 0 so the UI has something to render the moment it
    # mounts; everything else streams in via a daemon thread.
    data: dict = {0: {"samples": _load_path(paths[0], mask_str)}}
    if not data[0]["samples"]:
        raise ValueError(
            f"First file {paths[0]} contained no parseable JSON lines. "
            f"Check that it is not empty and follows the rollout_result schema."
        )
    print(f"  loaded [1/{len(paths)}] {paths[0].name}: {len(data[0]['samples'])} samples (foreground)")

    bg_log = Path("/tmp/relax-tui-bg.log")
    bg_log.write_text("")  # truncate

    def _bg_loader():
        # Load remaining files newest-first so the latest step is ready next.
        remaining = [(i, p) for i, p in enumerate(paths) if i != 0]
        remaining.sort(key=lambda t: _stem_key(t[1]), reverse=True)
        for idx, p in remaining:
            try:
                data[idx] = {"samples": _load_path(p, mask_str)}
            except Exception:
                # Don't crash the TUI; record what went wrong so the user
                # can grep the log if a step shows up missing.
                with bg_log.open("a") as f:
                    f.write(f"failed: {p}\n")
                    traceback.print_exc(file=f)

    if len(paths) > 1:
        print(f"  loading remaining {len(paths) - 1} file(s) in background (progress logged to {bg_log})")
        threading.Thread(target=_bg_loader, daemon=True).start()

    file_idx_map = {i: p.stem for i, p in enumerate(paths)}
    app = _build_app(step_num=len(paths), data=data, file_idx_map=file_idx_map)

    # textual restores the terminal on exit. Re-raise after `app.run()` so
    # any internal traceback lands in the user's normal shell instead of
    # being hidden by the alt-screen.
    try:
        app.run()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
