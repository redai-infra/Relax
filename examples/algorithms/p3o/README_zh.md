# P3O A100×4 配置

English version: [README.md](README.md)。

这些启动脚本会在相同的 on-policy 与可控 rollout-mismatch 场景下比较 P3O 和
GRPO。它们面向 4 张 colocate GPU，并通过 Ray Jobs 提交训练驱动。

## 所需环境

在非 dry-run 前设置以下路径：

```bash
export P3O_MODEL_DIR=/path/to/model
export P3O_TRAIN_DATA=/path/to/train.jsonl
export P3O_EVAL_DATA=/path/to/eval.jsonl
export P3O_OUTPUT_ROOT=/path/to/output
export P3O_MEGATRON_DIR=/path/to/Megatron-LM
export P3O_RAY_DASHBOARD=http://ray-dashboard-host:8265
```

`formal` 模式需要 `P3O_EVAL_DATA`；`smoke` 模式可不设置。模型、训练数据和
Megatron 路径必须在提交 Ray job 前存在。每次运行都会在 `P3O_OUTPUT_ROOT` 下
保存解析后的参数、命令、Git 身份、日志、Ray 状态、退出码和逐步 rollout JSONL。
只有外部证据布局需要不同的原始 rollout 目录时，才设置
`P3O_ROLLOUT_RESULT_DIR`；最终路径会记录在 `run_identity.env`。

Ray job runtime 会显式禁用继承的 HTTP proxy。SGLang 通过节点本地 IP 做健康检查
和 worker 注册；将宿主机 proxy 变量传入 Ray worker 可能使健康引擎无法通过启动
屏障。

`formal` 模式默认使用 DeepScaleR 的 `problem`/`answer` 字段；`smoke` 模式默认
使用常见的 `question`/`answer` schema。若数据集使用其他字段，显式设置
`P3O_INPUT_KEY` 和 `P3O_LABEL_KEY`；最终取值会记录在 `run_identity.env`。

`formal` 模式默认使用 `deepscaler` 规则 verifier，读取 Qwen-Thinking 的
`</think>` 后缀和最终 `\\boxed{...}` 答案。`smoke` 模式保留面向旧 GSM8K
资产的 `mopd` 默认值。若 smoke 使用 DeepScaleR 或其他奖励契约，请显式设置
`P3O_RM_TYPE`；最终 reward type 会记录在 `run_identity.env`。

正式评估默认使用 `deepscaler` 数据集名称、每个 prompt 16 个样本、4096 token
response 上限、温度 1.0 和 top-p 0.95。受限资源实验可设置 `P3O_EVAL_NAME`、
`P3O_EVAL_N_SAMPLES`、`P3O_EVAL_MAX_RESPONSE_LEN`、`P3O_EVAL_TEMPERATURE` 和
`P3O_EVAL_TOP_P`。这些值只影响评估，且必须在配对算法间保持一致。

默认 `P3O_ROLLOUT_SHUFFLE=1` 保持普通训练行为。只有在使用预先物化的固定 prompt
调度进行配对证据时才设置为 `0`；记录该设置，以免将 shuffled run 误认为固定比较。

当配对实验需要 P3O 和 GRPO 共享每个样本的采样 seed 时，设置
`P3O_DETERMINISTIC_INFERENCE=1`。解析后的开关会记录在 run identity。该开关只
控制采样随机性；首个更新之后，即使 seed 相同，不同策略权重也应产生不同响应。

`formal` 模式加载 `scripts/models/qwen3-4B.sh`，目标为
Qwen3-4B-Thinking-2507；`smoke` 模式加载 `scripts/models/qwen3-0.6B.sh`。仅在
有意验证另一兼容模型时设置 `P3O_MODEL_CONFIG`，解析后的路径会记录在
`run_identity.env`。正式启动器会将通用 4B 脚本的 RoPE base 覆盖为 `5000000`，
以匹配该 checkpoint 的 `config.json`；smoke 保持 `1000000`。可通过
`P3O_MODEL_ROTARY_BASE` 有意设置兼容 override，取值同样会记录。

## 当前 P3O 契约

正式 P3O 路径使用 `--p3o-ess-scope micro-batch`、
`--p3o-kl-mode proxy_safe` 和 `--clip-low/--clip-high 0.2` 监控边距。
`proxy_safe` 的前向值与 FeynRL 兼容的 sampled-token proxy 相同，只修正极端负
log-ratio 的梯度。`exact` 仅能通过纯全词表验证辅助函数调用，并不是命令行模式，
因为 rollout 数据保存的是已选 token 的行为策略 log-probability，而非完整行为
logits。

P3O 使用专用 policy-loss dispatch，并与 `--use-opd` 互斥。OPD teacher loss、
OPD advantage replacement 或 OPD-only reward 都会形成未经验证的混合目标。
reward/verifier 名称 `P3O_RM_TYPE=mopd` 与 `--use-opd` 训练功能无关，兼容数据集
仍可使用。

当 `--context-parallel-size > 1` 时，P3O 会自动为每个 THD TransformerLayer、最终
normalization 和 LM head 重建全序列，完成计算后再把输出切回各 CP rank。该严格路径对
`micro-batch` 和 `step` 两种 ESS scope 都生效，使 forward/backward kernel 看到与 CP1
相同的 token 顺序和形状。它会在每个 CP rank 上重复全序列计算和 activation，因此
峰值显存和计算量都高于原生 CP；长上下文建议开启整层 activation
recomputation。当前支持的合同是标准 zig-zag THD 且 tensor parallel size 为 1。
如果全序列路径发生 OOM，P3O 会终止并拒绝 CP>1，不会静默回退到数值不等价的
原生 CP kernel order。

严格分区不变性还要求 Megatron `--deterministic-mode` 和
`--batch-invariant-mode`。示例脚本会为配对的 P3O/GRPO run 同时启用两者，并在模型
构建前向 Ray runtime 注入 `NCCL_ALGO=Ring`、
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` 和
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。缺少任一模式时 P3O 会 fail closed。P3O 在该模式下
还会关闭 fused weight-gradient accumulation，因为
当前随附的 batch-invariant TE GEMM 无法遵守跨多个 micro-batch 的 `main_grad`
累积合同。该绕行使用稳定的 TE/DDP 累积路径，可能降低吞吐或增加瞬时梯度显存。

正式默认值为 G=16、global batch 64、micro-batch 1、rollout batch 4、response
length 4096 和 30 个 optimizer step（`--num-rollout 30`）。计划配对 seed 为 42、
123 和 2026。smoke 使用 G=4、global batch 16、response length 128 和 1 个
optimizer step。

以下环境变量可在不修改场景脚本的前提下公开这些对齐设置：

```bash
export P3O_ESS_SCOPE=micro-batch  # step 仅用于 capability/replay 验证
export P3O_KL_MODE=proxy_safe     # proxy 用于 golden parity
export P3O_CLIP_LOW=0.2
export P3O_CLIP_HIGH=0.2
export P3O_SEED=42
export P3O_RM_TYPE=deepscaler # smoke 与 DeepScaleR 配对时必需
```

Ray worker 默认继承正常 proxy 设置。若注入的 outbound proxy 拦截 SGLang 的节点
本地 readiness probe，可设置 `P3O_CLEAR_RUNTIME_PROXIES=1` 以在 job runtime 内
清除 proxy 变量。它是 opt-in，且会记录在 `run_identity.env`，因为它会同时禁用
该 job 内所有 worker 的 proxy 访问。

若 A100-40GB 容量无法进行 4B pilot，应先降低 pilot response length，同时保持
micro-batch size 为 1 并记录偏差。不要把缩小的 smoke run 视为正式证据，也不要
默默缩减三 seed 对比。需要保持 response 的资源回退时，设置
`P3O_ACTIVATION_RECOMPUTE=1` 启用整层统一 activation recomputation，并把
`P3O_LOG_PROBS_CHUNK_SIZE` 设为正 token 数以对 log-probability 与 entropy
reduction 分块。这两个设置对 P3O 和 GRPO 完全一致，都会记录在 run identity；
默认 `0`/`-1` 保留原始路径。

## 场景

| 场景                       | 更新间隔 | 温度 override | 含义                                   |
| -------------------------- | -------: | ------------: | -------------------------------------- |
| `on_policy`                |        1 |          关闭 | 以正常采样配置在每个 rollout 后同步。  |
| `periodic_sync_interval_3` |        3 |          关闭 | 只引入周期性的 rollout-policy 陈旧性。 |
| `temperature_0p6`          |        1 |           0.6 | 只改变行为策略温度。                   |
| `temperature_1p2`          |        1 |           1.2 | 只改变行为策略温度。                   |

同一场景中的 P3O 和 GRPO 启动器共享所有非算法配置。温度场景会保持 `top_p`、
`top_k`、response 限制和评估采样设置不变。

## 运行

```bash
bash examples/algorithms/p3o/run_p3o_on_policy_a100x4.sh
bash examples/algorithms/p3o/run_grpo_on_policy_a100x4.sh
bash examples/algorithms/p3o/run_p3o_periodic_sync_interval_3_a100x4.sh
bash examples/algorithms/p3o/run_p3o_temperature_0p6_a100x4.sh
```

一轮 rollout 检查可通过 smoke wrapper 选择任意场景：

```bash
bash examples/algorithms/p3o/run_p3o_smoke.sh p3o_temperature_1p2
```

设置 `P3O_DRY_RUN=1` 可打印解析后的训练参数，而不检查资产或提交 Ray job。

## Policy-age 指标

`train/p3o/rollout_policy_age_rollouts` 衡量当前 rollout ID 与生成当前 batch 的
rollout-policy snapshot ID 的差值。单位是 rollout，而不是 optimizer step。周期
刷新影响下一个 rollout；刷新边界 batch 的指标仍描述生成该 batch 的 snapshot。
