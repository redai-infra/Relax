# GDPO 示例：correctness + format 双奖励

本目录是 GDPO（[arXiv 2601.05242](https://arxiv.org/abs/2601.05242)）的最小可运行示例，用 Qwen3-0.6B 单卡在 GSM8K 上训练。

## 为什么需要 GDPO

奖励函数 `reward_gdpo.py` 返回两个分量：

- `correctness` —— `<answer>` 标签里的答案是否正确（0 或 1）
- `format` —— 输出是否同时带 `<think>` 与 `<answer>`（0、0.5 或 1）

这两个分量会**不同步**：模型可能答对但没按格式输出，也可能格式完美但答错。

GRPO 把它们**加起来**再做一次组内归一化。这一步会丢掉两类信息。

**一、组间的相对强度。** 组内标准化把每一组的 advantage 都拉到单位方差，于是「只有 correctness 在变」的组和「两个分量都在变」的组，得到完全相同的 advantage。GDPO 对每个分量各自标准化后相加，两个分量都起作用时幅度自然是两倍；而第三步的 batch 白化是**跨组**的，所以这个差异会一路保留到最终 advantage：

|                  | 组 A（只有 correctness 变化） | 组 B（两个分量都变化） |
| ---------------- | ----------------------------- | ---------------------- |
| GRPO             | ±0.707                        | ±0.707（分不出）       |
| GDPO（含第三步） | ±0.548                        | ±1.095（B 强一倍）     |

**二、分量之间的尺度差异。** `correctness ∈ {0,1}` 与一个取值上百的 `format`（或论文实验里的响应长度）相加时，和的方差几乎全部来自大尺度那一维，GRPO 的方向就由它单独决定。GDPO 先让每个分量单位方差，权重才真正表达「相对重要性」而不是量纲。极端一点：`correctness=[1,1,0,0]`、`format=[0,100,200,300]`（两者排序相反）时，GRPO 给答错的长响应最高 advantage，GDPO 不会。

## GDPO 不能做什么

**各分量在组内恰好加和为常数时，GDPO 也救不回来。** 若 `correctness + format ≡ C`，则 `format = C − correctness`，两者标准化后恒有 `z_format = −z_correctness`，**等权重下完全抵消为零**——与 GRPO 得到同样的结果。

这一点本文档此前写反了，说这正是 GDPO 的优势场景。它不是：那是一个数学上不可能被等权重 GDPO 区分的情形。只有不等权重（如 `--gdpo-reward-weights 2.0 1.0`）能在这类组上拿到信号。

（另外，若两个分量在组内**都**恒定，GDPO 与 GRPO 一样返回零，不会无中生有。）

## 运行

```bash
export MODEL_DIR=/path/to/models      # 需含 Qwen3-0.6B
export DATA_DIR=/path/to/data         # 需含 gsm8k/train.jsonl
export EXP_DIR=/path/to/experiments

bash examples/gdpo/run-qwen3-0.6B-1xgpu-gdpo.sh
```

### 数据要求

每条 prompt 需要**自带**格式要求，否则基座模型不会产出 `<think>`/`<answer>` 标签，`format` 分量会恒为 0（组内塌缩），GDPO 就退化成只看 `correctness`。准备数据时给 question 追加一句即可：

```python
instruction = (
    "\n\nThink step by step inside <think> </think> tags, then give only the "
    "final number inside <answer> </answer> tags."
)
df["question"] = df["question"] + instruction
```

**不要用 `--system-prompt` 代替**：`relax/utils/data/data_utils.py:181` 把 system message 的 content 构造成多模态 list（`content: [{"type": "text", ...}]`），Qwen3-0.6B 这类纯文本 chat template 渲染时会报
`TypeError: can only concatenate str (not "list") to str`。这是既有的框架限制，与 GDPO 无关。

## 参数说明

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format   # 独立归一化的分量，至少两个
   --gdpo-reward-weights 1.0 1.0           # 可省略，默认全 1
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score                      # 必填：metrics 与 raw_reward 用的标量
   --n-samples-per-prompt 8                # 必须 ≥ 2
)
```

`--gdpo-reward-weights` 乘的是**归一化之后**的 advantage，不是原始 reward。经过第一步各分量已经是单位方差，所以权重表达的是相对重要性，而不是分量的量纲——把 `format` 的取值范围从 `[0,1]` 改成 `[0,100]` 基本不改变训练结果。

说「基本」而不是「完全」，是因为第一步除的是 `std + 1e-4` 而不是 `std`。当某分量的组内标准差本身就落到 `1e-4` 量级时，这个加性 epsilon 会随量纲变化而改变阻尼比例，缩放就不再是严格等价的。二值奖励（std≈0.4）离这个区间很远，连续奖励（如响应长度）则可能撞上。

## 换成自己的奖励

改 `reward_gdpo.py` 的 `compute_gdpo_reward`，返回的 dict 需要包含 `--gdpo-reward-keys` 列出的全部 key，外加 `--reward-key` 指定的那个标量。

分量缺失、非数值、bool、NaN/Inf 都会直接报错。这是有意的：静默填 0 会让一个坏掉的奖励函数看起来像是「这一维恰好塌缩了」，训练照跑，问题要很久以后才暴露。

## 已知偏差

1. **第三步的 batch 边界（已正确处理）**。调用方为效率会先合并多个训练批再调用 advantage，但 `_whiten_by_segment` 用 `mini_batch_sizes` 把它们切回**每个 optimizer 训练批各自白化**，因此 `num_rollout_minis > 1` 时仍对齐论文 Eq. 6，**不要求** `rollout_batch_size × n_samples_per_prompt == global_batch_size`。本脚本把 `4 × 8` 与 `--global-batch-size 32` 设成相等只是让例子最简单，并非必需。跨 DP 的 all-reduce 保证统计量覆盖全部 rank。**`--fully-async` 会在参数校验阶段被拒绝**——那条路径的切片可能小到只有一个样本，白化输出恒为 0。

2. **单个奖励时 GDPO 不等于 GRPO**。step1 除以 `std_g + 1e-4`、GRPO 除以 `std_g + 1e-6`，各组 `std_g` 不同 → 尺度因子逐组不同，step3 还会再做一次 batch 白化，所以不是「差一个正标量」那么简单。要 GRPO 语义就用 `--advantage-estimator grpo`。

3. **`--n-samples-per-prompt 2` 时幅度信息丢失**：任意两个不同值标准化后恒为 ±0.7071。示例用 8 就是为了避开这一点。

4. **恒和分量在大量级下会产生数值假信号**。若两个分量恰好满足 `r₂ = C − r₁`，数学上应完全抵消为零（见上文「GDPO 不能做什么」）。但奖励在进入归一化前会被 cast 成 float32，而 `C − r` 这种值不一定能被 float32 精确表示——两者相加仍舍回 `C`（看起来恒和），各自却已偏离，标准化后留下约 `1e-4` 的残差。第三步再用同量级的 batch 标准差去除它，输出就变成 O(1) 的 advantage：**一个本无信号的组拿到了方向由舍入决定的梯度**。实测一组和为 `308.95172119140625` 的奖励，最终 advantage 为 `[-0.4344, 0.5251, -0.0908]`。

   GRPO 没有这个问题：它归一化的是**和**，而和在 float32 下确实恒定，塌缩检测会直接置零。

   触发需要「恒和 + 分量绝对值远大于其组内差异」同时成立，实践中少见；彻底解决要么把奖励全链路加宽到 float64，要么在第三步引入噪声下限，两者都超出本 PR 范围。若你的奖励设计天然满足恒和，请直接用不等权重（此时不再抵消，也就不存在这个残差被放大的问题）。

## 冲突项

`--normalize-advantages`、`--custom-reward-post-process-path`、`--agentic-custom-advantage-path` 和 `--fully-async` 都不能与 GDPO 同用，参数校验阶段会直接报错。第一个会造成双重白化；第二、三个都会赶在归一化器之前从 `post_process_rewards` 返回，导致 GDPO 的前两步被静默跳过（这两个由 `AlgorithmSpec.allows_reward_post_process_hooks` 一起把守）；第四个的统计窗口可能小到只剩一个样本。

配 `--dynamic-sampling-filter-path` 时**不冲突**：内置的 `check_reward_nonzero_std` 会直接算出 GDPO 前两步的组合结果、按其是否非零判组，与训练实际拿到的信号一致（权重把某分量静音、或两个分量抵消，它都能算出来）。只有指向**自定义** filter 时才会警告——那种 filter 若只看 `--reward-key` 标量，就会丢掉只在其它分量里有信号的组。
