# On-Policy Distillation Examples

本目录包含 On-Policy Distillation (OPD) 的示例启动脚本。

## 概述

On-Policy Distillation (OPD) 让学生模型在自己的 rollout 数据上训练，同时匹配教师模型的 token 级 log-probability，从而实现从大模型到小模型的知识传递。

## 选择建议

- **SGLang teacher（推荐异构蒸馏）**：teacher 和 student 可以不同架构（例如 32B -> 8B）。
- **Megatron teacher（同构蒸馏）**：teacher 与 student 必须结构一致（见下方硬性要求）。

## Megatron teacher 的硬性要求（重要）

当 `--opd-type megatron` 时，`--opd-teacher-load` 指向的 teacher checkpoint 必须与 student 模型结构一致（hidden size、层数、attention heads、词表相关形状等）。

> 换句话说：
>
> - ✅ 可行：8B student + 8B teacher（或 32B student + 32B teacher）
> - ❌ 不可行：8B student + 32B teacher（会触发参数 shape mismatch）

原因是当前 Megatron OPD teacher 通过同一 Megatron 模型图加载并切换权重实现，权重 shape 必须一一对应。

## 快速开始

### SGLang teacher 模式（支持异构 teacher）

适用于教师与学生架构不同，或教师模型较大不适合同图加载的场景。

```bash
# 1) 下载模型和数据
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2) 启动 SGLang teacher 服务（新终端）
python -m sglang.launch_server \
    --model /root/Qwen3-32B \
    --tp 8 \
    --port 30010

# 3) 使用 Megatron-Bridge 直接加载 HF student 并启动 OPD 训练
cd /root/Relax
bash examples/on_policy_distillation/run-qwen3-8B-sglang-opd.sh
```

### Megatron teacher 模式（仅支持同构 teacher）

适用于 teacher/student 结构一致的场景。可通过 Megatron-Bridge 直接加载 HF checkpoint，无需先转换成 torch_dist。

```bash
# 1) 下载模型和数据（示例：8B 对 8B）
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2) 使用 Megatron-Bridge 直接从 HF 路径加载 student/teacher
#    典型配置：
#    --megatron-to-hf-mode bridge
#    --hf-checkpoint /root/Qwen3-8B
#    --opd-teacher-load /root/Qwen3-8B
#    （teacher 路径必须与 student 结构一致）
cd /root/Relax
bash examples/on_policy_distillation/run-qwen3-8B-megatron-opd.sh
```

## 关键参数说明

| 参数                      | 说明                                                                           |
| ------------------------- | ------------------------------------------------------------------------------ |
| `--use-opd`               | 启用 OPD                                                                       |
| `--opd-type`              | 教师类型：`sglang` 或 `megatron`                                               |
| `--opd-kl-coef`           | OPD KL 系数（默认 1.0）                                                        |
| `--opd-teacher-load`      | teacher 模型路径（`--opd-type megatron` 时必需；可配合 bridge 直接填 HF 路径） |
| `--opd-teacher-timeout-s` | SGLang 模式下 OPD teacher HTTP 请求超时（秒），默认 `30`                       |
| `--opd-log-prob-top-k`    | teacher/student top-k 候选集合大小（设为 `0` 可关闭，默认 `0`）                |
| `--opd-only-reward`       | 仅保留 OPD reward 信号（将 base reward 置零，仅注入 OPD KL）                   |
| `--rm-url`                | SGLang teacher 服务地址（`--opd-type sglang` 时必需）                          |

> Note:
>
> 1. OPD `sglang` 模式不占用 `--custom-rm-path` 与 `--custom-reward-post-process-path`，可与自定义奖励并存。
> 2. `--opd-only-reward` 需要配合 `--use-opd` 使用。
> 3. 当 `--opd-log-prob-top-k > 0` 时，框架会在 SGLang teacher 请求中启用 top-k 采集，并尝试提取 teacher 的 top-k token ids / log-probs。
> 4. 若 teacher 响应缺失 top-k 字段或请求失败，会自动回退到安全路径（rollout log-probs + 占位 top-k），不打断 rollout 主流程。

## 动态指标

启用 top-k 采集后，OPD 可以在线监控 student 与 teacher 候选空间的一致性。

定义 $S_t^{(p)} = \\text{TopK}(p_t, k)$、$S_t^{(q)} = \\text{TopK}(q_t, k)$，分别表示 token 步 $t$ 上 student/teacher 的 top-$k$ 集合。

### Overlap Ratio（重叠率）

$$
\\mathcal{M}\_{\\text{overlap}} \\triangleq \\mathbb{E}\_t \\left\[ \\frac{|S_t^{(p)} \\cap S_t^{(q)}|}{k} \\right\]
$$

解释：

- 重叠率低：student 与 teacher 候选空间偏离较大。
- 重叠率高：student 策略逐步靠近 teacher 支撑区域。

## 后端支持

| 后端     | SGLang teacher | Megatron teacher                    |
| -------- | -------------- | ----------------------------------- |
| Megatron | ✅             | ✅（要求 teacher/student 结构一致） |

## 参考文献

- [On-Policy Distillation - Slime Docs](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/on-policy-distillation.md)
- [Tinker Cookbook - On-Policy Distillation](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py)
