# MTP RL 训练

本指南介绍如何在 Relax 的 RL 后训练阶段,将模型的 Multi-Token Prediction (MTP) 头与策略联合训练。MTP 头作为辅助目标与主 GRPO loss 一起优化——这与 slime 的实现方式一致。

阅读前请先完成[安装](./installation.md),并至少跑通一次基线 GRPO 任务(参考[快速上手](./quick-start.md))。

## 何时启用 MTP RL 训练

当你希望在 RL 阶段保持 MTP 头与策略同步演化,使得训练完成后的同一个 checkpoint 直接可用于推理加速(SGLang/vLLM 的 EAGLE/NEXTN speculative decoding),无需再单独做一次蒸馏时,启用此功能。

如果只关注主策略且下游没有 speculative decoding 计划,不建议启用——辅助 loss 会带来少量额外计算和显存开销。

## 前置条件

- HF checkpoint 必须已经包含 MTP 权重,即 `config.json` 中 `num_nextn_predict_layers >= 1`。目前覆盖 **Qwen3.5**、**Qwen3-next**、**MiMo-7B-RL**、**DeepSeek-V3 / V3.1**、**GLM-4.7-MoE**。
- 使用 Megatron 后端(也是 Relax 唯一的训练后端)。MTP 依赖 [`docker/patch/megatron/20260506-85bced0ae.patch`](../../../docker/patch/megatron/20260506-85bced0ae.patch) 中的 Megatron MTP 补丁;Relax 官方镜像已自动应用。
- 不能开启 combined 1F1B 流水线调度——与 MTP 不兼容,在 [`relax/backends/megatron/model.py:493`](../../../relax/backends/megatron/model.py) 处会被断言挡掉。

## 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--enable-mtp-training` | 关闭 | 启用 MTP forward 注入和 MTP 辅助 loss |
| `--mtp-num-layers` | 启用时必填 | 模型中 MTP 层数,必须与 HF checkpoint 一致 |
| `--mtp-loss-scaling-factor` | `0.1`(RL 推荐) | MTP loss 加到主 loss 前的缩放系数 |

前两项在 [`relax/utils/arguments.py:2765-2766`](../../../relax/utils/arguments.py) 处联合校验。三个参数都由 Megatron parser 提供,所有 Relax 启动脚本无需改代码即可使用。

## 启动脚本

| 脚本 | 模型 | 资源 | 备注 |
| --- | --- | --- | --- |
| [`run-qwen35-9B-mtp-8xgpu.sh`](../../../scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh) | `Qwen3.5-9B` | 8 GPU colocate | dense 模型,扩规模前的冒烟验证 |
| [`run-qwen35-35B-A3B-mtp-16xgpu.sh`](../../../scripts/training/text/run-qwen35-35B-A3B-mtp-16xgpu.sh) | `Qwen3.5-35B-A3B` | 16 GPU(2 节点)colocate | 生产目标,镜像基线 GRPO 脚本 + `MTP_ARGS` |

两个脚本都支持环境变量覆盖:

```bash
MTP_NUM_LAYERS=1 MTP_LOSS_SCALING_FACTOR=0.1 \
  bash scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh
```


## scaling factor 调优

默认 `0.1` 对 RL 而言比较保守,因为主 GRPO loss 自带来自 advantage 估计的梯度噪声。若观察到:

- `train/mtp_loss` 早早 plateau 而 `train/loss` 正常 → 提到 `0.2`–`0.3`
- 启用 MTP 后 `train/loss` 不稳定 → 降到 `0.05`
- MTP 梯度主导(对比基线观察 `train/grad_norm`)→ 降低缩放

作为参考,SFT MTP 脚本使用 `0.2`,因为 SFT 梯度更干净。slime 默认值也是 `0.2`;RL 调低是 Relax 的专门建议。

## 日志观察要点

健康的训练会在每个 train step 打出:

```text
train/loss              # 主 GRPO loss,与基线同量级
train/grad_norm         # 与基线同量级(≤2×)
train/mtp_loss          # 有界,逐步下降
```

若 `train/mtp_loss` 从未出现,说明 MTP block 没构建——通常是 checkpoint 不匹配(HF ckpt 没有 MTP 权重)。可通过下面命令验证:

```bash
python -c "from transformers import AutoConfig; \
  print(AutoConfig.from_pretrained('/path/to/ckpt').num_nextn_predict_layers)"
```


## 相关文档

- [SFT 训练](./sft-training.md) — MTP 头也可在 SFT 阶段训练,参见 `run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh`。
- [权重更新流水线优化](./update-weights-pipeline.md) — Megatron→SGLang 权重同步如何处理 MTP 层。
