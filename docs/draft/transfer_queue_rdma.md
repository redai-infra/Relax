# TransferQueue RDMA 数据面使用与运维指南

## 概述

Relax 的数据面（rollout ↔ train 之间的样本传输）默认走 TransferQueue 的 SimpleStorage/ZMQ。本特性把 TransferQueue 已有的 MooncakeStore 后端接出来，使数据面可以走 RDMA，并在能力不足时安全回退。

首期只做配置接入、能力探测与一致回退，**不改变 payload 形状与数据分发语义**。默认参数下行为与接入前完全一致。

## 配置入口

只暴露四个表达使用意图的参数，Mooncake 底层参数（endpoint、buffer、segment、timeout、master 策略）不做 CLI，走内部默认与部署环境。

| 参数 | 取值 | 说明 |
|---|---|---|
| `--tq-storage-backend` | `simple`（默认）/ `mooncake` | `simple` 等价于接入前行为 |
| `--tq-rdma-mode` | `off`（默认）/ `auto` / `required` | `off` 即使有硬件也不用 RDMA；`auto` 探测失败自动降级；`required` 探测失败直接报错退出 |
| `--tq-rdma-device` | 设备名，如 `mlx5_bond_0`；空为自动 | 多网卡机器上自动选择可能选错，跨节点时建议显式指定 |
| `--tq-use-gdr` | 默认关 | **实验性**，见下文 |

`--tq-rdma-mode=required` 只覆盖**传输层**（MooncakeStore + RDMA 可用性与 segment 容量），不覆盖 GDR。

### GDR 为实验性

GDR 的可用性无法在启动时探测：探测跑在独立的 Ray task 里，该进程没有初始化 CUDA context，`torch.cuda.is_initialized()` 恒为 False，若在此判定会让 GDR 永远不可达。真实判定发生在每个 worker 的 TQ 客户端内部（`mooncake_client.py`），没有 CUDA context 时**静默回退到 host RDMA**并打 WARNING。

因此首期：`--tq-use-gdr` 标记为实验性，`required` 不对 GDR 做 fail-fast。要把 GDR 纳入分级降级，需要补一条 worker → driver 的能力回报通道，属于后续工作。

## 启动流程与降级

driver 在**第一次 `tq.init` 之前**完成探测并生成 job 级唯一的 effective config，其余组件（actor / critic / rollout / sft / advantages / actor_fwd）都读同一份，不各自决策。

1. 校验参数组合（例如 `simple` + `rdma-mode` 会被拒绝）
2. `probe_cluster_nodes()` 通过 Ray 把探测任务绑定到每个**存活且有 GPU** 的节点，各自读本机 `/sys` 与 mooncake 状态；超时或崩溃的节点转为退化结果，不静默丢弃
3. `reduce_results()` 做 AND 归约：整个作业只能跑在最低共同能力上
4. `required` 模式下若发生任何回退，直接抛异常并打印每个节点的探测明细

降级阶梯：

```
GDR → host RDMA        （worker 运行时判定，静默回退 + WARNING）
RDMA → Mooncake/TCP    （任一节点无 RDMA 能力，或指定设备缺失）
Mooncake → SimpleStorage（任一节点 mooncake 不可导入，或 segment 容量不足）
```

## 启动日志怎么读

正常启动会打三段，排障时先看这三段：

```
[dataplane] requested: backend=mooncake rdma_mode=auto device=mlx5_bond_0 gdr=False
[dataplane] probe result:
[probe:<ip>] protocol=rdma device=mlx5_bond_0 gdr=True
  [ok] mooncake_import: version=0.3.10.post2
  [ok] rdma_devices: mlx5_bond_0, ...
  [ok] port_state: ACTIVE
  [ok] gid: ...
  [ok] memlock: unlimited
[dataplane] backend=MooncakeStore protocol=rdma device=mlx5_bond_0 gdr=off
```

第三段带 `fallback=...` 就说明发生了降级，原因直接写在里面（例如 `fallback=mooncake_unavailable:<node-id>`）。

## Mooncake master 生命周期

`auto_init` 固定为 `false`：**Relax 既不启动也不停止 master**，master 由部署环境管理。这样做是因为 TQ 的 `auto_init=true` 路径会执行 `pkill -f "[m]ooncake_master"`，在共享集群上会杀掉其他人的进程。

启动 master（部署侧，节点上执行一次）：

```bash
setsid mooncake_master -rpc_port=50051 -metrics_port=9004 > /var/log/mooncake_master.log 2>&1 < /dev/null &
```

然后给作业设置 `MC_MASTER_ADDRESS=<master-host>:50051`。未设置时内部默认 `localhost:50051`，仅适用于单节点开发。

三种情形下的行为：

| 情形 | 表现 | 处理 |
|---|---|---|
| **初始化失败**（master 不可达） | TQ 客户端 `setup` 返回 `-1`，抛 `Mooncake store setup failed with error code: -1`。`required` 模式下作业直接失败；`auto` 模式下这一步已过了探测，不会自动回退——探测只验证本机能力，不验证 master 连通性 | 先确认 master 进程与 `MC_MASTER_ADDRESS`，再重启作业 |
| **正常退出**（作业结束或全局重启） | Relax 在拆数据面时调用 `close_tq_and_unmount()`：先 `tq.close()`（它内部还要用 store 做 `remove_all()`），再显式 `storage_client.close()` 卸载 segment 并从 master 注销。master 本身不动 | 无需操作 |
| **异常退出**（worker 被 kill / OOM / 节点掉线） | Python 层不执行，segment 仍在 master 注册。master 要等 `client_ttl`（默认 30 s）才判定客户端过期，期间新作业的 put 会打到死端点并报 `Failed to open segment ... Connection refused` | 等 30 s 后重启，或部署侧调小 `-client_ttl` |

## 资源所有权与安全清理

首期按**单任务独占 Ray 集群**实现，**不承诺同一节点上多个 Relax job 并发**：多 job 并发、端口租约、master 共享机制都不在首期范围内。

清理只动本作业拥有的资源：

- 不使用任何 `pkill` / `killall`
- `tq.init` 之前会检查已存在的 `TransferQueueController` 命名 actor：**只有取不到 config（半初始化）或 actor 已死时才回收**，健康的 controller 保持不动并正常 attach
- master 进程始终不被 Relax 触碰

## 排障表

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 启动日志 `backend=SimpleStorage fallback=mooncake_unavailable:<node>` | 该节点上 `import mooncake` 失败 | 检查该节点的 `mooncake-transfer-engine` 安装；镜像是否一致 |
| `protocol=tcp fallback=...`，但机器有 RDMA 卡 | 端口非 ACTIVE、GID 取不到、`memlock` 过低，或指定的 `--tq-rdma-device` 在部分节点不存在 | 看 `probe result` 里哪一项 FAIL；`memlock` 需要 unlimited |
| `setup failed with error code: -1` | master 不可达 | 检查 master 进程与 `MC_MASTER_ADDRESS` |
| `Failed to open segment ... Connection refused` | 上一轮客户端异常退出，死 segment 仍在 master 注册 | 等 `client_ttl`（30 s）过期后重试 |
| `batch_get_into failed ... error codes [-800, ...]` | 会话内切换协议（0.3.10 上更敏感），或对端不可达 | 每个协议单独进程跑；确认对端存活 |
| 多网卡机器跨节点建连失败 | 自动选卡选到了不通的网卡 | 显式 `--tq-rdma-device`；必要时用 `MC_TCP_BIND_ADDRESS` 指定 TCP 侧绑定地址 |
| 训练卡在启动、无日志推进 | 半初始化的 controller（TQ 的 `_init_from_existing` 会无限轮询 config） | 本特性已加自动回收；若仍出现，确认 `[dataplane] ... reaping it` 是否打出 |

### 选卡核验

指定设备前先确认端口状态与 GID：

```bash
ls /sys/class/infiniband/                                   # 有哪些设备
cat /sys/class/infiniband/mlx5_bond_0/ports/1/state          # 需要 ACTIVE
cat /sys/class/infiniband/mlx5_bond_0/ports/1/rate
ulimit -l                                                    # 需要 unlimited
```

判定数据面是否真的走了 RDMA（get 前后取差值）：

```bash
cat /sys/class/infiniband/mlx5_bond_0/ports/1/counters/port_rcv_data   # RDMA，单位是 4 字节字
cat /sys/class/net/bond0/statistics/rx_bytes                           # TCP
```

RDMA 生效时前者按 payload 增长、后者基本不动；反之则说明落在 TCP。

## 已知限制

- 写侧（put）收益明显，读侧（get）收益有限：get 每次调用都会注册/注销 MR，且 key 粒度是 `样本 × 字段`，碎片化开销盖过了传输收益。MR 常驻注册与读路径零拷贝成型不在首期范围。
- 跨节点 RDMA vs TCP 的收益在多轮之间波动较大，验收结论应基于多轮分布而非单轮数据。
- 消费端节点在 get 过程中中途死亡的端到端行为需要双节点真机验证，未做成自动化测试。
- Mooncake 传输层自身的超时参数不由 Relax 控制。
