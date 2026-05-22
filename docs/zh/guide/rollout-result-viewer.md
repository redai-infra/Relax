# Rollout 结果可视化

一个轻量级的网页工具，用于浏览 Relax 训练和评测过程中按 step 产出的 JSONL 文件。

![Relax Rollout Result Viewer](/relax-viewer.png)

## 数据来源

只要设置了 `--save`，Relax 会自动写出每个 rollout 的紧凑汇总：

| 子目录 | 数据来源 | 写入函数 |
|---|---|---|
| `<save>/rollout_result/train/{rollout_id}.jsonl` | 训练 rollout | `save_rollout_result_jsonl` |
| `<save>/rollout_result/eval/{rollout_id}.jsonl`  | 评测 rollout | `save_eval_summary_jsonl` |

每一行代表一个 sample，包含 `prompt`、`response`、`reward`、`response_length`、`total_length`、`status`、`group_index`，评测数据还会带上 `dataset`。多模态输入只会写出摘要而不会保存原始张量。

## 启动

```bash
python -m relax.entrypoints.visualize <save>/rollout_result --port 8080
# 然后在浏览器打开 http://localhost:8080
```

数据目录是必填的位置参数。`python -m relax.utils.visualize ...` 也可以用，参数完全一样 —— 入口脚本只是个薄壳，放在 `entrypoints/` 下是为了和 `train.py` 等其他入口保持一致。

服务启动时会自动探测数据目录下的 `train/` 和 `eval/` 子目录：

- 两个都存在 → 页面顶部显示 `[ train | eval ]` 切换按钮。
- 都不存在 → 把数据目录本身当作扁平目录处理，你也可以直接把它指向一个装着 `{step}.jsonl` 的文件夹。

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR`（位置参数） | 必填 | 要可视化的目录。 |
| `--port` | `8080` | HTTP 端口。 |
| `--host` | `0.0.0.0` | 绑定地址。 |
| `--cache-memory` | `4096`（MB） | 进程内 LRU 缓存的总内存上限。如果单个 `train/*.jsonl` 较大，可以调大。 |
| `--cache-entries` | `20` | 缓存中最多保留的文件数。 |
| `--base-path` | `""` | 反向代理下的 URL 前缀，例如 `--base-path /absproxy/8080`。 |

## 页面功能

- **Step 下拉框** — 列出当前子目录下的所有 `{step}.jsonl`，按 step 排序。
- **样本翻页** — `← Previous` / `Next →` 按钮、左右方向键，以及右下角悬浮的快速跳转。
- **Sample Info 卡片** — 展示标量字段（`rollout_id`、`sample_index`、`reward`、`response_length`、`total_length`、`status`、`group_index`、`dataset`）。
- **Prompt / Response / Label** — 带 chat template、tool-call、`<think>` 高亮的格式化文本。长文本就地滚动，默认滚到末尾，方便查看模型最新生成的内容。
- **排序** — 在 step 内按 `sample_index`、`reward`、`response_length` 升/降序重排。
- **缓存管理** — `GET /api/cache/stats`、`POST /api/cache/clear`。

## 终端 UI（`--tui`）

在不方便开浏览器的 SSH 会话里，同一个命令可以渲染成基于 textual 的终端界面：

```bash
python -m relax.entrypoints.visualize <save>/rollout_result --tui
```

依赖额外的 `textual` 和 `rich` 包（`pip install textual rich`），没有写进 `requirements.txt`，只用 Web 的同学不会被影响。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tui` | 关 | 切换到终端 UI。 |
| `--mask-str` | 匹配 `<\|image_pad\|>`、`<\|imgpad\|>`、`<\|audio_comp_pad\|>` 的正则 | 将多模态 pad token 替换成 `*`，让 prompt 更易读。传 `--mask-str ""` 可关闭。 |

TUI 一次只看一个扁平目录里的 `{step}.jsonl`。当数据目录同时存在 `train/` 和 `eval/` 时，TUI 默认进入 `train/`；想看 eval 直接把数据目录指到 `<save>/rollout_result/eval` 即可。

### 快捷键

| 按键 | 作用 |
|---|---|
| `n` / `p` | 下一个 / 上一个 sample |
| `N` / `P` | 下一个 / 上一个 step |
| `f` 输入关键词, `enter` | 查找；再按 `enter` 跳到下一个匹配 |
| `esc` | 清除搜索 |
| `s` | 在纯文本 / 表格渲染之间切换 |
| `r` | 刷新当前视图 |
| `j` / `k` | 下 / 上翻页 |
| `h` / `l` | 左 / 右翻页 |
| `g` / `G` | 跳到顶 / 底 |
| `tab` / `←` / `→` | 切换焦点 |

左侧栏有 step / sample / dataset（仅当 eval 数据加载时显示）/ 排序模式（`reward asc/desc`、`response_length asc/desc`）的下拉框，下方还有按字段展示/隐藏的勾选列表。

## 使用注意

- 仅适合本机或可信内网使用。CORS 完全开放、无鉴权，和其他 dump 检查工具一致。
- 解析后的 JSONL 文件会缓存在进程内存。dump 很大时可调大 `--cache-memory`，或重启进程释放内存。
- 服务信任磁盘上的 JSON 内容；解析失败的行会被跳过并打印 warning。
