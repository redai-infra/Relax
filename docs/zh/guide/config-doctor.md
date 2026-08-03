# Config Doctor 与 Dry-run

`relax.entrypoints.doctor` 用于在训练启动前检查配置。它不会启动 Ray、Ray Serve、SGLang、GPU worker 或训练循环，只做参数解析、规则检查、角色拓扑推导、资源需求预览和启动命令预览。

## 基本用法

将训练参数放在 `--` 后面：

```bash
python -m relax.entrypoints.doctor -- \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --global-batch-size 32 \
  --num-rollout 1 \
  --colocate
```

输出包含以下核心信息：

- `config_state`：`validated`、`partial` 或 `unavailable`，用于区分完整配置与回退结果。
- `Final merged config`：解析和归一化后的训练配置。
- `Role topology`：算法、候选角色、必需角色、实际规划角色、每个 role 的资源计划和 placement group 关系。
- `resource_summary`：按 colocate / fully-async / hybrid 规则推导出的 GPU 需求。
- `Expected launch command`：doctor 预览的训练入口命令。

JSON 输出可直接接入 CI：

```bash
python -m relax.entrypoints.doctor --format json -- \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --global-batch-size 32 \
  --num-rollout 1 \
  --colocate
```

正确配置返回退出码 `0`。存在 `error` 级诊断时返回非零退出码。`--strict-warnings` 会将 warning 也作为 CI 失败处理。

## 跳过远端 HF 配置读取

如果只需要本地静态检查，且当前环境不能访问 HuggingFace，可以使用：

```bash
python -m relax.entrypoints.doctor --doctor-skip-hf-validate -- <training args>
```

该选项会把 `--skip-hf-validate` 追加到训练参数中，避免解析阶段读取远端 HF config。

## 校验回退与定向诊断

doctor 首先执行与训练入口相同的完整参数解析和校验。完整校验失败时，它会再执行一次无校验解析，只构造标记为 `partial` 的参数 Namespace，不读取远端 HF 配置、不派生资源参数、不执行 backend 校验或 TransferQueue 版本检查。

partial 配置只运行明确声明为 partial-safe 的规则，并且不生成角色拓扑或 GPU 需求。报告始终保留原始 `CONFIG_PARSE_ERROR`，即使其他定向规则也命中，因此不会隐藏完整校验的真实失败。任何规则内部异常都会转换为结构化的 `DOCTOR_RULE_EXECUTION_ERROR`，不会让 CLI 直接 traceback。

doctor 会汇总 Relax、Megatron、SGLang 和 teacher parser 实际注册的选项。未被任何 parser 注册的参数返回 `CONFIG_UNKNOWN_ARGUMENT`，例如拼错的 `--does-not-exist` 不会再被静默忽略。

## 拓扑与资源语义

`candidate_roles` 表示算法注册表可能创建的角色，`required_roles` 表示当前配置必须提供资源的角色，`roles` 表示结合 `--resource` 后实际进入预览的角色。Fully-async 模式仅在启用 `--use-kl-loss` 或设置非零 `--kl-coef` 时要求 `reference`；满足 true-on-policy 条件时不要求 `actor_fwd`。

角色选择、optional role、managed teacher、placement group 和 GPU 汇总统一由 `relax/core/service_plan.py` 推导，Controller 与 doctor 使用同一结果。同步 colocate 模式下，actor、rollout 及符合条件的 critic 共享 placement group，GPU 需求取共享角色中的最大值；Hybrid actor 与 rollout 使用独立 placement group，资源总量按各 role 求和。

`advantages` 和 `sft` 是允许 `[1, 0]` 的 CPU role。actor、rollout、critic、reference、actor_fwd、genrm 和 managed teacher 等模型 role 必须配置正数 GPU。

## 数据路径检查

`--prompt-data` 支持单文件、目录、文件列表和 `@[start:end]` 切片语法。doctor 会先解析广义路径，再逐个检查实际文件或目录，例如 `[a.jsonl,b.jsonl]@[0:100]` 会分别检查 `a.jsonl` 和 `b.jsonl`，不会把整个表达式当作文件名。

## 敏感信息脱敏

Text 和 JSON 报告会统一脱敏 `--agent-env`、`--train-env-vars` 内的敏感字段、`--wandb-key`、API key、token、password、credential、private key 和通知 URL。脱敏覆盖原始参数、预计启动命令、最终合并配置、解析错误以及诊断详情，敏感值统一显示为 `<redacted>`。

## 已覆盖的错误类型

doctor 以 rule id 输出诊断结果。当前规则覆盖：

- `CONFIG_RESOURCE_REQUIRED`：缺少 `--resource`。
- `CONFIG_RESOURCE_SHAPE`：resource 结构非法、`num_serves != 1` 或模型 role 使用 0 GPU。
- `CONFIG_UNKNOWN_ARGUMENT`：参数未被任何运行时 parser 注册。
- `CONFIG_ALGORITHM_SUPPORTED`：算法 key 未注册。
- `CONFIG_REQUIRED_ROLES`：当前模式需要的 role 未写入 resource。
- `CONFIG_MODE_CONFLICT`：`--fully-async` 与 `--colocate` 直接组合。
- `CONFIG_DEBUG_MODE_CONFLICT`：两个 debug-only 模式同时启用。
- `CONFIG_PPO_TOPOLOGY`：PPO 缺 critic、使用不支持的异步模式或 staleness。
- `CONFIG_SFT_REQUIREMENTS`：SFT 缺数据源、动态 batch 或 predict 依赖。
- `CONFIG_BATCH_SIZE`：batch size 缺失或互相矛盾。
- `CONFIG_ROLLOUT_COUNT`：缺少 rollout 次数边界。
- `CONFIG_OVERSAMPLING`：过采样 batch 小于 rollout batch。
- `CONFIG_DYNAMIC_BATCH`：动态 batch 缺 token budget。
- `CONFIG_CONTEXT_LENGTH`：上下文长度超过 per-GPU token budget。
- `CONFIG_SGLANG_PARALLEL`：SGLang PP/DP 参数冲突。
- `CONFIG_EVAL`：评测配置缺失或与 SFT/RL 模式不匹配。
- `CONFIG_SAVE`：保存间隔缺少保存路径。
- `CONFIG_GENRM_COLOCATE`：GenRM colocate GPU 分配不合法。
- `CONFIG_PATHS`：本地路径不存在。
- `CONFIG_LORA`：LoRA merge / adapter 配置冲突。
- `CONFIG_QKV_FORMAT`：`bshd` 与动态 batch 或非 Megatron 后端冲突。
- `CONFIG_ROTATE_CKPT`：checkpoint 轮换缺少必要参数。

## 扩展规则

新增规则放在 `relax/utils/doctor/rules.py`，使用 `@diagnostic_rule(rule_id, title)` 注册。规则只读取 `DoctorContext`，返回 `DiagnosticResult` 列表，不启动外部进程，不调用 Ray，不分配 GPU。

新增算法、role 或 backend 时，应更新 `relax/core/service_plan.py` 的共享规划逻辑，并添加 Controller/doctor 共用的计划测试。错误样例位于 `tests/doctor/fixtures/error_cases.json`。
