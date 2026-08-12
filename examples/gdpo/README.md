# GDPO 示例：correctness + format 双奖励

本目录是 GDPO（[arXiv 2601.05242](https://arxiv.org/abs/2601.05242)）的最小可运行示例，用 Qwen3-0.6B 单卡在 GSM8K 上训练。

## 为什么需要 GDPO

奖励函数 `reward_gdpo.py` 返回两个分量：

- `correctness` —— `<answer>` 标签里的答案是否正确（0 或 1）
- `format` —— 输出是否同时带 `<think>` 与 `<answer>`（0、0.5 或 1）

这两个分量会**不同步**：模型可能答对但没按格式输出，也可能格式完美但答错。

GRPO 把它们**加起来**再做一次组内归一化。问题是不同的分量组合可能得到**相同的总和**：一组 rollout 里，有的是「答对但格式差」`(correctness=1, format=0)`、有的是「答错但格式好」`(0, 1)`，总奖励都等于 1——GRPO 看到组内总奖励全相同，整组 advantage 归零、样本白采，可两个分量各自明明都有信号。

GDPO 分别对每个分量做组内标准化再合并，就能区分这两类样本，`correctness` 与 `format` 各自的差异都会转成梯度信号。（反过来，若两个分量在组内**都**恒定，GDPO 与 GRPO 一样返回零，不会无中生有。）

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

`--gdpo-reward-weights` 乘的是**归一化之后**的 advantage，不是原始 reward。经过第一步各分量已经是单位方差，所以权重表达的是相对重要性，与分量本身的量纲无关——把 `format` 的取值范围从 `[0,1]` 改成 `[0,100]` 不会改变训练结果。

## 换成自己的奖励

改 `reward_gdpo.py` 的 `compute_gdpo_reward`，返回的 dict 需要包含 `--gdpo-reward-keys` 列出的全部 key，外加 `--reward-key` 指定的那个标量。

分量缺失、非数值、bool、NaN/Inf 都会直接报错。这是有意的：静默填 0 会让一个坏掉的奖励函数看起来像是「这一维恰好塌缩了」，训练照跑，问题要很久以后才暴露。

## 已知偏差

1. **第三步的 batch 边界（已正确处理）**。调用方为效率会先合并多个训练批再调用 advantage，但 `_whiten_by_segment` 用 `mini_batch_sizes` 把它们切回**每个 optimizer 训练批各自白化**，因此 `num_rollout_minis > 1` 时仍对齐论文 Eq. 6，**不要求** `rollout_batch_size × n_samples_per_prompt == global_batch_size`。本脚本把 `4 × 8` 与 `--global-batch-size 32` 设成相等只是让例子最简单，并非必需。跨 DP 的 all-reduce 保证统计量覆盖全部 rank。**`--fully-async` 会在参数校验阶段被拒绝**——那条路径的切片可能小到只有一个样本，白化输出恒为 0。
2. **单个奖励时 GDPO 不等于 GRPO**。step1 除以 `std_g + 1e-4`、GRPO 除以 `std_g + 1e-6`，各组 `std_g` 不同 → 尺度因子逐组不同，step3 还会再做一次 batch 白化，所以不是「差一个正标量」那么简单。要 GRPO 语义就用 `--advantage-estimator grpo`。
3. **`--n-samples-per-prompt 2` 时幅度信息丢失**：任意两个不同值标准化后恒为 ±0.7071。示例用 8 就是为了避开这一点。

## 冲突项

`--normalize-advantages`、`--custom-reward-post-process-path`、`--agentic-custom-advantage-path` 和 `--fully-async` 都不能与 GDPO 同用，参数校验阶段会直接报错。第一个会造成双重白化；第二、三个都会赶在归一化器之前从 `post_process_rewards` 返回，导致 GDPO 的前两步被静默跳过（这两个由 `AlgorithmSpec.allows_reward_post_process_hooks` 一起把守）；第四个的统计窗口可能小到只剩一个样本。

配 `--dynamic-sampling-filter-path` 时只警告不报错：内置过滤器按 `--reward-key` 的单个标量判组，可能丢掉只在其它分量里有信号的组。
