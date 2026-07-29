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

输出包含四类核心信息：

- `Final merged config`：解析和归一化后的训练配置。
- `Role topology`：算法、基础角色、可选角色、每个 role 的资源计划和 placement group 关系。
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

## 已覆盖的错误类型

doctor 以 rule id 输出诊断结果。当前规则覆盖：

- `CONFIG_RESOURCE_REQUIRED`：缺少 `--resource`。
- `CONFIG_RESOURCE_SHAPE`：resource 结构非法或 `num_serves != 1`。
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

新增算法或 backend 时，应同步更新 `relax/utils/doctor/topology.py` 中的纯拓扑映射，并添加对应错误样例。错误样例位于 `tests/doctor/fixtures/error_cases.json`。
