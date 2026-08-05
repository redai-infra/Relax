# 可复现实验清单

Relax 会为每次训练启动写入一份小型、已脱敏的 experiment manifest。它记录命令、解析后的参数、代码
版本、软硬件环境、输入标识、训练并行拓扑以及 Ray/Slurm 运行形态，用于检查或复跑实验。采集采用
best-effort 语义：工具缺失、超时或输出目录不可写时只记录告警，不会阻塞训练主流程。

## 输出位置

`relax.entrypoints.train` 会在 Ray 启动前记录初始来源信息，再将初始化后的 Ray 拓扑更新到同一个文件；
仅 rank 0 写入。默认路径为：

```text
<tensorboard-dir>/manifest_<run-id>.json
```

未配置 `tensorboard_dir` 时使用启动工作目录。文件名包含时间戳和随机后缀，不会覆盖已有运行。也可以
手工生成独立清单：

```bash
python -m relax.entrypoints.reproducibility generate --output manifest.json
```

## 检查、比较与复跑

请在原训练使用的 Relax checkout 目录下执行：

```bash
# 报告代码、依赖、硬件、配置和运行时差异。
python -m relax.entrypoints.reproducibility check manifest.json

# 比较两次运行并输出 Markdown 表格。
python -m relax.entrypoints.reproducibility diff old.json new.json

# 仅打印待审阅的环境重建脚本，不执行。
python -m relax.entrypoints.reproducibility rerun manifest.json --dry-run

# 校验当前环境后，只执行清单中记录的 argv。
python -m relax.entrypoints.reproducibility rerun manifest.json --confirm
```

`check` 在完全匹配或兼容性告警时返回 `0`，不兼容差异时返回 `1`。原 Ray 集群在当前检查进程中不可
观测，或只有兼容的 patch 版本变化时，结果为告警。`rerun --confirm` 遇到不兼容环境会拒绝执行，也
不会执行含脱敏参数的命令。dry-run 输出可能包含 checkout 和依赖重建命令，把它当作 shell 脚本前必须
人工审阅；confirmed 模式本身不会安装依赖或修改 Git 状态。

## Schema 与兼容策略

权威 v1 JSON Schema 位于
[`docs/schema/experiment-manifest-v1.schema.json`](../../schema/experiment-manifest-v1.schema.json)。所有新清单
写入 `schema_version: "1.0"`。读取端接受后续 `1.x` 文档、保留未知字段，并把旧版 `cli_args` 归一化到
`command.argv`；其他 major 版本会被拒绝。minor 版本只能新增可选字段，删除字段或改变既有字段语义
必须升级 major 版本。

主要分区如下：

| 分区 | 内容 |
| --- | --- |
| `command` | 已脱敏 argv 和归一化后的启动工作目录（`.`） |
| `code` | Relax/Megatron commit、分支和 dirty 状态 |
| `config` | 解析后的参数、选定环境变量、runtime env 和配置哈希 |
| `environment` | Python、CUDA/NCCL 和关键依赖版本 |
| `hardware` | CPU、内存、NUMA、GPU 型号/数量/显存和驱动 |
| `runtime` | 本地、单机 Ray 或多节点 Ray 模式，节点角色/资源、Slurm 与并行拓扑 |
| `inputs` | 已脱敏的模型、tokenizer、数据标识以及有界的大小/哈希信息 |
| `training` | 算法、batch size 与 TP/PP/CP/EP/DP 拓扑 |

各 collector 相互独立且都是可选项。因此某个 collector 超时后，对应分区可以缺失，清单仍然是一份
合法的 best-effort 记录。

## 隐私与安全边界

Relax 会递归移除敏感字段、Authorization header、URL 密码、私网 IP、内部主机名、home 目录用户名和
基础设施 endpoint。环境变量只有命中前缀白名单才会采集，复跑时采用更小的恢复白名单。Ray 节点身份
和节点地址资源不会落盘。

不超过 1 MiB 的小输入文件可以记录 SHA-256；模型权重和数据集不会被复制进 bundle。包含脱敏参数的
命令会被刻意禁止复跑。清单定位为可分享的运维元数据，但应用自定义的自由文本无法从理论上保证绝不
含敏感内容，对外发布前仍应人工检查。

## 最小 CPU 示例

下面三条命令演示手工生成、环境检查和 dry-run；不会启动训练：

```bash
python -m relax.entrypoints.reproducibility generate --output manifest.json
python -m relax.entrypoints.reproducibility check manifest.json
python -m relax.entrypoints.reproducibility rerun manifest.json --dry-run
```

真实复跑应使用原训练自动生成的 manifest，并只在理解 `check` 结果和 dry-run 输出后执行 `--confirm`。
