# 在线策略蒸馏 (OPD)

在线策略蒸馏 (OPD) 通过在学生模型自身的回滚数据上训练学生，同时匹配教师的词元级对数概率，实现从大型教师模型到小型学生模型的知识迁移。OPD 与优势估计器正交——它作为 KL 惩罚项，可以与任何估计器（GRPO、GSPO、SAPO，以及实验性的 PPO 和 REINFORCE++）结合使用。

## 关键参数

| 参数                      | 描述                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--use-opd`               | 启用在线策略蒸馏。使用 OPD 时需要此标志。                                                                        |
| `--opd-type`              | OPD 类型：`sglang` 或 `megatron`。启用 `--use-opd` 时必须设置。                                                  |
| `--opd-kl-coef`           | OPD KL 惩罚系数（默认：1.0）。控制蒸馏信号相对于 RL 优势的权重。                                                 |
| `--opd-teacher-load`      | 教师模型路径。当 `--opd-type=megatron` 时**必须**设置，当 `--opd-type=sglang` 时**不能**设置。                   |
| `--opd-teacher-ckpt-step` | 教师模型的可选检查点步骤。                                                                                       |
| `--opd-teacher-timeout-s` | SGLang 模式下 OPD teacher HTTP 请求超时（秒），默认 `30`。                                                      |
| `--opd-log-prob-top-k`    | 用于 OPD 动态指标的 teacher/student top-k 候选集合大小（设为 `0` 可关闭，默认 `0`）。                          |
| `--opd-only-reward`       | 仅保留 OPD 奖励信号（将基础 RL reward 置零，只使用 OPD KL 项）。需配合 `--use-opd`。                            |

## 工作原理

OPD 通过减去 KL 惩罚项来修改优势计算，鼓励学生匹配教师的输出分布：

$$\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t$$

其中 $A_t$ 是来自基础估计器（例如 GRPO）的原始优势，$\lambda_{\text{opd}}$ 是 `--opd-kl-coef`，$D_{\text{KL}}$ 是词元级反向 KL 散度。

因此，OPD 可以与任何优势估计器结合，包括 GRPO、GSPO、SAPO，以及实验性的 PPO 和 REINFORCE++。

## 两种教师模式

### SGLang 模式 (`--opd-type sglang`)

教师模型在外部 SGLang 服务器上运行，教师的对数概率在回滚阶段获得。

**使用场景**：教师架构与学生不同，或教师模型太大无法与训练模型一起加载。

**工作流程**：

1. 外部 SGLang 服务器运行教师模型。
2. 在回滚期间，每个样本的奖励计算完成后，框架自动将样本发送到教师服务器以获取词元级对数概率，并将其存储在 `sample.teacher_log_probs` 中。
3. 当 `--opd-log-prob-top-k > 0` 时，框架还会请求并提取 teacher 的 top-k 候选信息（通过 SGLang 请求字段），并存储：
	- `sample.teacher_topk_token_ids`
	- `sample.teacher_topk_log_probs`（若 teacher 返回）
4. 当 teacher 请求失败或响应缺失 top-k 字段时，OPD 会进入安全回退路径（使用 rollout log-probs 与占位 top-k 数据），避免打断 rollout 主流程。
5. 在训练期间，从存储的教师对数概率计算 KL 惩罚并应用于优势。

> **注意**：OPD sglang 模式不再占用 `--custom-rm-path` 和 `--custom-reward-post-process-path`。用户可以自由地同时使用自定义奖励函数和 OPD，两者互不冲突。

**配置**：

```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--opd-teacher-timeout-s 30
--opd-log-prob-top-k 10
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
```

## 动态指标

启用 top-k 采集后，OPD 可以在线监控 student 与 teacher 候选空间的一致性。

定义 $S_t^{(p)} = \text{TopK}(p_t, k)$、$S_t^{(q)} = \text{TopK}(q_t, k)$，分别表示 token 步 $t$ 上 student/teacher 的 top-$k$ 集合。

### Overlap Ratio（重叠率）

$$
\mathcal{M}_{\text{overlap}} \triangleq \mathbb{E}_t \left[ \frac{|S_t^{(p)} \cap S_t^{(q)}|}{k} \right]
$$

解释：

- 重叠率低：student 与 teacher 候选空间偏离较大。
- 重叠率高：student 策略逐步靠近 teacher 支撑区域。

### Megatron 模式 (`--opd-type megatron`)

教师模型通过 `--opd-teacher-load` 直接加载到 Megatron 中，教师的对数概率在训练前向传递期间计算。

**使用场景**：教师和学生/参考模型具有相同的架构，可以一起放入 GPU 内存中。

**硬性要求（重要）**：Megatron teacher 必须与 student 结构一致（例如 hidden size、层数、attention heads、词表相关参数形状等）。

- ✅ 可行：8B student + 8B teacher，或 32B student + 32B teacher
- ❌ 不可行：8B student + 32B teacher（会触发参数 shape mismatch）

**工作流程**：

1. 教师模型在初始化期间作为额外的 Megatron 模型加载。
2. 在训练前向传递期间，教师模型为每个样本计算对数概率。
3. KL 惩罚被内联计算并应用于优势。

**配置**：

```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_model
```

> **注意**：
>
> 1. 使用 `--megatron-to-hf-mode bridge` 时，可直接从 HuggingFace 路径加载模型，无需预先转换为 `torch_dist`。
> 2. 如果不使用 bridge（例如 `raw` 模式），则仍需提供 Megatron 格式检查点（`torch_dist` 或 `torch`）。

## 运行示例

完整的示例脚本位于 `examples/on_policy_distillation/`

## 初步结果

使用 Qwen3-8B-Base 模型，在 [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) 数据集的一部分上执行 SFT，然后在剩余数据上应用在线策略蒸馏（使用 Qwen3-32B 教师），Math500 评估结果如下：

| 模型                               | Pass@1 |
| ---------------------------------- | ------ |
| Qwen3-8B-Base + SFT                | 76%    |
| Qwen3-8B-Base + SFT + 在线策略蒸馏 | 94%    |

## 后端支持

| 后端     | SGLang 教师 | Megatron 教师 |
| -------- | ----------- | ------------- |
| Megatron | ✅          | ✅（要求 teacher/student 结构一致） |
