# Relax Paper - Page 1 Cursor Extraction

Source: `/data/models/minimax/Relax/amd/paper/relax.pdf`

Extraction method: Cursor `ReadFile` PDF text extraction.

## Raw Extracted Text

```text
Relax: An Asynchronous Reinforcement Learning Engine
for Omni-Modal Post-Training at Scale
Liujie Zhang1 Benzhe Ning1 Rui Yang1 Xiaoyan Yu1 Jiaxing Li1 Lumeng Wu2 Jia Liu3
Minghao Li1 Weihang Chen1 Weiqi Hu1 Lei Zhang1
1AI Platform, Xiaohongshu Inc
2The University of Hong Kong 3University of Science and Technology of China
April 15, 2026
ABSTRACT
Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection,
and tool-use capabilities in large language models. As models extend to omni-modal inputs and
agentic multi-turn workflows, RL training systems face three interdependent challenges: hetero-
geneous data flows, operational robustness at scale, and the staleness–throughput tradeoff. We
present Relax (Reinforcement Engine Leveraging Agentic X-modality), an open-source RL training
engine that addresses these challenges through three co-designed architectural layers. First, an
omni-native architecture builds multimodal support into the full stack—from data preprocessing and
modality-aware parallelism to inference generation—rather than retrofitting it onto a text-centric
pipeline. Second, each RL role runs as an independent, fault-isolated service that can be scaled,
recovered, and upgraded without global coordination. Third, service-level decoupling enables asyn-
chronous training via the TransferQueue data bus, where a single staleness parameter smoothly
interpolates among on-policy, near-on-policy, and fully asynchronous execution. Relax achieves
a 1.20× end-to-end speedup over veRL on Qwen3-4B on-policy training. Its fully async mode
delivers a 1.76× speedup over colocate on Qwen3-4B and a 2.00× speedup on Qwen3-Omni-30B,
while all modes converge to the same reward level. Relax supports R3 (Rollout Routing Replay) [ 1]
for MoE models with only 1.9% overhead, compared to 32% degradation in veRL under the same
configuration. It further demonstrates stable omni-modal RL convergence on Qwen3-Omni across
image, text, and audio, sustaining over 2,000 steps on video without degradation. Relax is available
at https://github.com/redai-infra/Relax.
1 Introduction
1.1 From Text-Only to Omni-Modal and Agentic RL
Reinforcement learning has become a critical post-training technique for improving reasoning in large language
models. Landmark systems such as DeepSeek-R1 [2 ] and the OpenAI o1/o3 series have demonstrated that RL can
unlock capabilities that supervised fine-tuning alone cannot produce, including self-reflection, step-by-step verification,
and complex multi-hop reasoning. In parallel, the algorithmic toolkit has expanded rapidly: from PPO [ 3] and
its memory-efficient variant GRPO [4 ], to the decoupled clip and dynamic sampling strategies of DAPO [ 5], and
further to reward-model-free paradigms such as RLVR, RL post-training has matured from research exploration to
production deployment. This momentum has spurred a wave of open-source RL training frameworks, including
veRL [ 6], OpenRLHF [7], AReaL [8], AsyncFlow [ 9], ROLL [10 ], ProRL [11 ], and Slime [12 ], each advancing the
state of the art along complementary dimensions such as hybrid control flow, asynchronous scheduling, and multi-turn
agentic rollouts.
While algorithmic and systems progress continues, the training paradigm itself is undergoing a fundamental shift
along two axes. First, models are becoming omni-modal: from vision-language models such as Qwen3-VL to unified
omni-modal architectures such as Qwen3-Omni and Qwen3.5-Omni, models increasingly consume images, video,
and audio alongside text, and their RL training must handle all of these modalities from the data pipeline through the
parallel execution strategy to the inference backend. Second, training is becoming agentic: rather than single-turn
prompt–response pairs, RL workflows now involve multi-turn reasoning, tool calling, and search-augmented generation,
where the model interacts with external environments across many steps before a reward signal is produced. These
two trends interact: omni-modal data makes workloads more heterogeneous and failure-prone, while agentic rollouts
introduce variable-length, multi-step trajectories. Addressing them effectively requires rethinking the training system as
a whole rather than patching individual components.
```
