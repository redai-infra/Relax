# Format-aware reward router 实验报告

> 状态：本地功能与回归测试完成；多节点训练验证待具备 Ray 集群地址和模型/数据路径后执行。

## Issue 固定信息

### 1. 仓库、基线与分支

- 仓库：Relax（当前 checkout 的 Git remote）
- 基线 commit：`30f219a1c963137129570dd5485a9c3d82390548`
- 基线分支：`main`（实验开始时本地比 `origin/main` 落后 1 个提交）
- 开发分支：`feat/format-aware-reward-router`
- 实验开始前已有且不属于本任务的工作区内容：
  `scripts/entrypoint/local.sh`、`contributor-program/`、`reward_curve.png`、
  `scripts/models/qwen3-0.6B.sh`。本任务不修改这些内容。

### 2. 运行环境

| 项目 | 实测值 |
| --- | --- |
| OS | Linux 5.15.0-139-generic x86_64，glibc 2.39 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu129 |
| CUDA toolkit | 12.9.86；容器 `CUDA_VERSION=12.9.1` |
| NVIDIA driver | 580.126.09 |
| Ray | 2.56.0 |
| SGLang | 0.5.12.post1 |
| math-verify | 0.8.0 |
| GPU | 4 × NVIDIA RTX 6000 Ada Generation，49140 MiB/卡 |
| CPU | 2 × Intel Xeon Platinum 8570，112 核/224 线程 |
| 内存 | 881 GiB；采集时 available 839 GiB |

本功能的本地验收只运行 CPU/Ray Actor 测试，没有启动模型训练或占用 GPU。

### 3. 启动脚本与完整命令

本任务没有修改训练脚本和 CLI 参数；现有 `--rm-type` 同时承担旧行为和 fallback：

- 样本存在有效的格式提示时，逐样本路由；
- 样本提示未知、缺失或冲突时，回退到显式 `--rm-type`；
- 没有有效 fallback 时返回 `0.0` 并记录 warning；
- 没有样本提示时，显式 `--rm-type` 保持原来的固定 scorer 行为。

本地复现命令：

```bash
git switch feat/format-aware-reward-router
git rev-parse HEAD
pytest -q tests/engine/rewards/test_reward_worker.py
pre-commit run --all-files
```

数据侧通过现有 metadata 字段声明格式，不需要新增启动参数：

```json
{"label": "42", "extra_info": {"format": "math"}}
{"label": "<answer>A</answer>", "extra_info": {"format": "multiple-choice"}}
```

若需要容忍坏数据，可继续在原训练命令中配置 fallback，例如：

```bash
--rm-type openr1mm
```

未提交远端训练任务：当前请求没有提供 `RAY_ADDRESS`、`MODEL_DIR` 和指定训练脚本。按开发流程，不猜测集群配置，也不自行启动远端任务。获得这些信息后的提交形式为：

```bash
ray serve shutdown -y
RAY_NO_WAIT=1 WORKING_DIR=./ RAY_ADDRESS=<ray-address> MODEL_DIR=<model-dir> \
  bash -x scripts/entrypoint/ray-job.sh <scripts/training/.../run-script.sh>
```

### 4. 文件、核心函数、范围与非目标

改动文件：

- `relax/engine/rewards/__init__.py`
  - `REWARD_REGISTRY` / `register_reward()`：统一注册同步与异步 scorer；
  - `resolve_rm_type()`：处理 metadata、结构化 label、格式推断、冲突与 fallback；
  - `RewardWorker.compute()`：由分支链改为 registry 查询；
  - `RewardExecutor.execute()`：逐样本解析路由，并安全处理零分降级。
- `tests/engine/rewards/test_reward_worker.py`
  - 覆盖混合 batch、路由优先级、label 推断、fallback、warning、registry 扩展和旧 CLI 行为。
- `docs/draft/format-aware-reward-router-experiment.md`
  - 本实验记录。

明确不做：

- 不修改 `relax/utils/arguments.py`，不新增 CLI 参数；
- 不修改 Controller、Service、Launcher；
- 不新增依赖，不删除或重命名公开 API；
- 不改变 custom reward、`boxed_`、远端 RM 和 GenRM 的调用方式；
- 不运行需要多节点 GPU 的集成训练。

### 5. 基线、指标与目标

| 指标 | 基线 | 目标 | 当前结果 |
| --- | ---: | ---: | ---: |
| reward 测试 | 40 passed / 48.10 s | 相关测试全部通过 | 51 passed（最终结果见下） |
| math + multiple-choice 混合 batch | 仅 metadata override 的弱覆盖 | 逐样本正确/错误结果均精确匹配 | `[1, 0, 1.0, 0.0]` |
| 未知/缺失/冲突类型 | 抛异常并使 batch 失败 | fallback 或 0，且 warning | 已覆盖 |
| registry 扩展 | worker 中修改 `if/elif` | 新 scorer 仅增加注册声明，不改路由 | 已覆盖 |
| `--rm-type` 兼容 | 固定 scorer；metadata.rm_type 可覆盖 | 保留 | 已覆盖 |

统计口径：pytest 用例数以 pytest 最终汇总为准；路由性能为同一进程连续
3 个窗口，每窗口 100000 次 `resolve_rm_type()`，样本按 math 和
multiple-choice 1:1 交替。

## 设计与兼容性

路由提示键为 `rm_type`、`reward_type`、`task_type`、`format`。
`multiple-choice`、`multiple choice`、`mcq` 统一映射到
`multiple_choice`。metadata 与结构化 label 中多个非空提示不一致时视为冲突。

优先级：

1. metadata/结构化 label 中一致、已注册的样本级提示；
2. 未设置全局 `--rm-type` 时，对明确的字符串 label 格式进行保守推断；
3. 现有 `args.rm_type` fallback；
4. warning + `0.0`。

为了保持兼容，显式设置 `--rm-type` 且样本没有格式提示时，不根据普通字符串
label 改写 scorer。动态修改 driver 进程中的 registry 不保证传播到已经启动的
Ray Actor；内置 reward 必须在模块导入时注册。

## 实验结果

### 正确性

基线：

```text
40 passed in 48.10s
```

实现后首次完整运行发现并修复了 `ans_2` 被宽松数字正则误判为 math 的回归。
修复后相关失败用例单独通过，完整 reward 测试通过。最终测试和 pre-commit
结果应以交付前最后一次复跑记录为准。

warning 测试不记录 prompt 或 label 内容，只验证未知、缺失和冲突路径确实发出
warning。测试输出未出现未解释异常、NaN/Inf 或数据丢失。

### 路由性能

| 窗口 | 样本数 | 耗时 | 吞吐 | 平均延迟 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 100000 | 0.144276 s | 693118.20 samples/s | 1.443 µs/sample |
| 2 | 100000 | 0.139798 s | 715316.83 samples/s | 1.398 µs/sample |
| 3 | 100000 | 0.141100 s | 708716.17 samples/s | 1.411 µs/sample |
| 均值 | 100000 | 0.141725 s | 705717.07 samples/s | 1.417 µs/sample |

该微基准只衡量 CPU 路由开销，不包含实际 reward 函数、Ray 调度或模型推理。
GPU 利用率和峰值显存为 N/A（未执行 GPU 工作负载），不能据此推断训练吞吐。
正确性护栏是同一提交下完整 reward 测试和 mixed-batch 精确结果。
