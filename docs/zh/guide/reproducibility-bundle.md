# 可复现实验清单

Relax 会为每次训练自动生成带版本号的 experiment manifest。它记录代码、最终配置、软件环境、硬件、Ray 拓扑、模型与数据输入以及原始命令，并默认排除凭据和内部地址。

## 输出位置

默认路径为：

```text
relax_runs/<实验名称>/<唯一运行 ID>/experiment-manifest.json
```

默认运行 ID 包含 UTC 时间、进程 ID 和随机后缀，因此重复运行不会覆盖之前的 manifest。可通过 `RELAX_MANIFEST_PATH` 指定固定文件。生成 manifest 采用 best-effort 语义：收集或写入失败只输出告警，不会阻塞训练。

## Schema v1

当前 `schema_version` 为 `1.0`。读取器接受整数形式的 v1 和后续 v1 次版本，忽略未知字段，并为空缺的可选分区提供空值。不同主版本会被拒绝。

| 分区 | 内容 |
| --- | --- |
| `code` | 仓库名、commit、分支与 dirty 状态 |
| `command` | 参数数组和可移植工作目录 |
| `config` | 最终参数与 Ray runtime environment |
| `environment` | Python、系统、关键包、CUDA 和容器镜像版本 |
| `hardware` | CPU、内存、GPU 型号、驱动和显存 |
| `runtime` | 本地、单机 Ray 或多节点 Ray 模式；集群及逐节点资源摘要；角色与并行拓扑 |
| `inputs` | 模型、checkpoint 和数据集标识，以及低开销的本地文件元数据 |

收集过程不会递归扫描目录、计算模型 hash、访问网络或执行 `pip freeze`。`collection_duration_ms` 字段可用于衡量启动开销。

## 默认脱敏

名称类似 `token`、`password`、`api_key`、`authorization` 和地址的字段会替换成 `<redacted>`。相同规则会递归应用到最终参数、runtime environment 和命令行参数。Ray 节点 ID、主机名、IP 地址及包含节点地址的资源标签不会写入文件。

环境变量仅按小型白名单收集。若命令中存在被脱敏的参数，系统会拒绝复跑；凭据应通过运行环境提供。

## 检查与复跑

比较当前代码、Python/依赖版本、系统、CPU 和 GPU：

```bash
python -m relax.entrypoints.reproducibility check path/to/experiment-manifest.json
```

环境一致返回 `0`，发现差异返回 `1`，manifest 无效返回 `2`。每项差异都会给出修复建议，适合接入 CI。

预览记录的命令：

```bash
python -m relax.entrypoints.reproducibility rerun path/to/experiment-manifest.json
```

环境检查通过后，在当前目录执行：

```bash
python -m relax.entrypoints.reproducibility rerun path/to/experiment-manifest.json --execute
```

只有确认差异符合预期时才使用 `--allow-drift`。复跑直接向子进程传递参数数组，不会调用 shell。
