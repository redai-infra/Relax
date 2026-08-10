# Mixture-of-LoRA RL 训练

Mixture-of-LoRA 冻结基座模型，在每个目标投影层训练多个 LoRA expert。可训练的 token-level router 为每个 token 选择 `K` 个 expert，再用归一化后的 Top-K 权重组合这些 expert 的输出。

当单个 LoRA adapter 容量不足时可以使用这条路径。`--lora-num-experts 1` 或不传该参数时，Relax 继续使用现有单 LoRA 实现。

## 参数配置

`--lora-rank` 大于零且 `--lora-num-experts` 大于一时启用 Mixture 路径。

```bash
--lora-rank 16
--lora-alpha 32
--lora-target-modules linear_qkv linear_proj
--lora-dropout 0.0
--lora-num-experts 4
--lora-router-top-k 2
--lora-router-temperature 1.0
--lora-router-aux-loss-coef 0.01
```

| 参数 | 含义 |
| --- | --- |
| `--lora-num-experts` | 每个目标投影层的 LoRA expert 数量。值大于一时启用 Mixture-of-LoRA。 |
| `--lora-router-top-k` | 每个 token 选中的 expert 数量，必须满足 `1 <= K <= N`。 |
| `--lora-router-temperature` | router softmax 使用的温度。 |
| `--lora-router-aux-loss-coef` | 逐 site balance loss 的系数。设为 `0` 时不产生这部分梯度，但仍输出路由指标。 |

`N > 1` 时必须明确提供三个 router 参数。Mixture 模式不使用 `--lora-merge-mode` 或 `--lora-adapter-mode`。

## 支持范围

首版支持：

- Qwen3 dense 模型；
- attention 的 `linear_qkv` 和 `linear_proj`；
- Megatron data、tensor、sequence、pipeline 和静态 context parallel；
- 同步 colocate 训练；
- 多个独立 SGLang engine，每个 engine 内部使用 TP=1、DP=1；
- Megatron 原生 distributed checkpoint。

启动时会拒绝 fully async、动态 context parallel、VLM、MoE 基座、MLP target，以及内部 TP 或 DP 大于一的 SGLang engine。

## 路由与 Balance Loss

router 为每个 token 使用 FP32 计算 `N` 个 expert 的概率，选出 Top-K 后重新归一化，使选中权重之和为一。训练端和 SGLang rollout 端调用同一套路由函数并使用等价的 dense expert 计算；Megatron 执行器还负责 TP/SP collective。

每个 routed site 单独计算 balance objective，再对所有 site 取平均：

```text
L_balance = N * sum_e(F_e * P_e)
F_e = selection_count_e / (valid_response_tokens * K)
P_e = expert e 在 Top-K 前的平均 router 概率
```

Prompt、padding 和 dummy token 不参与 balance loss 与路由指标。静态 context parallel 下会先在 CP group 中汇总入选次数、概率和与有效 token 数，再计算完整序列的 objective。

## 路由指标

指标名称使用 `molora/<site_id>/...` 和 `molora/global/...`：

- `expert_<id>_pre_topk_mean_prob`；
- `expert_<id>_post_topk_mean_weight`；
- `expert_<id>_selection_share`；
- `expert_<id>_top1_fraction`；
- `pre_topk_normalized_entropy` 和 `post_topk_normalized_entropy`；
- 每个 site 的 `balance_loss` 与全局 `molora/aux_loss`。

判断路由塌缩时应查看每个 site 的 Top-K 后平均权重、选择份额和熵。全局指标只用于汇总，可能掩盖个别层的塌缩。

## Rollout 权重更新

启用 Mixture 后，Relax 会自动启动 Qwen3 SGLang external model。第一次 colocate 更新发送冻结的基座参数以及全部 expert/router 参数；后续更新只发送当前全部 expert/router 参数。最后一组 routed tensor 加载成功后才发布新的 weight version。

更新失败时会恢复 generation、保留原 weight version，并报告错误，不会静默使用只加载了一部分的新策略。

## Checkpoint

Expert 和 router 是 Megatron 原生 distributed checkpoint 中的普通模型参数。Optimizer、scheduler、iteration 和 RNG 状态沿用同一个 checkpoint 恢复。Mixture 模式不会额外导出一份 HF PEFT adapter。随附的 recipe 启用了 Megatron 的 fully reshardable distributed optimizer 格式，因此保持 TP 和 PP 不变时，可以修改 DP size 后继续训练。

恢复训练时使用同一 recipe，并让 `--load` 和 `--save` 指向已有输出目录。加载 tensor 前会检查 checkpoint 中的 expert 数、rank、Top-K、temperature、coefficient、alpha、target module、dtype 和 site 维度是否与当前配置一致。

## Qwen3-4B DAPO Recipe

参考脚本使用八张 colocate GPU，让 Qwen3-4B 在 DAPO math 上运行 200 个 rollout 的 GRPO：

```bash
MODEL_PATH=/path/to/Qwen3-4B \
PROMPT_DATA=/path/to/dapo-math-17k.jsonl \
OUTPUT_DIR=/path/to/qwen3-4b-mixture-lora \
bash scripts/training/text/run-qwen3-4B-mixture-lora-8xgpu.sh
```

Actor 使用 TP=2 和 sequence parallel。Rollout 资源会建立八个独立的单卡 SGLang engine。可以通过 `NUM_ROLLOUT`、`LORA_NUM_EXPERTS`、`LORA_RANK`、`LORA_ROUTER_TOP_K` 等环境变量覆盖 recipe 中的值。这份 recipe 使用 BF16，也可以在命令末尾继续追加 Relax 参数。
