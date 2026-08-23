# TransferQueue RDMA 数据面使用与运维指南

## 概述

Relax 的数据面（rollout ↔ train 之间的样本传输）默认走 TransferQueue 的 SimpleStorage/ZMQ。本特性把 TransferQueue 已有的 MooncakeStore 后端接出来，使数据面可以走 RDMA，并在能力不足时安全回退。

首期只做配置接入、能力探测与一致回退，**不改变 payload 形状与数据分发语义**。默认参数仍使用 SimpleStorage 及原有 controller 所有权模型；同时所有 worker attach（包括 SimpleStorage）新增默认 60 秒 deadline，半初始化 controller 会被回收，Controller 构造失败时会关闭本进程已经完成的 legacy `tq.init`。

## 配置入口

只暴露三个表达使用意图的参数，Mooncake 底层参数（endpoint、buffer、segment、timeout、master 策略）不做 CLI，走内部默认与部署环境。

| 参数 | 取值 | 说明 |
|---|---|---|
| `--tq-storage-backend` | `simple`（默认）/ `mooncake` | `simple` 保留接入前的存储与 controller 所有权语义，并共享新增的有界 attach/失败清理 |
| `--tq-rdma-mode` | `off`（默认）/ `auto` / `required` | `off` 即使有硬件也不用 RDMA；`auto` 探测失败自动降级；`required` 探测失败直接报错退出 |
| `--tq-rdma-device` | 设备名，如 `mlx5_bond_0`；空为自动 | 多网卡机器上自动选择可能选错，跨节点时建议显式指定 |

`--tq-rdma-mode=required` 覆盖的是**传输层**：MooncakeStore + RDMA 可用性与 segment 容量。

### 首期只交付 host RDMA

数据面 payload 先经过主机内存再通过 RDMA 跨节点传输。GDR（GPU Direct RDMA）不在本期范围内：它不是任务书要求，可用性也无法由 driver 的启动探测代表（探测进程没有初始化 CUDA context），因此 Relax 把 Mooncake 的 `use_gdr` 固定为 `false`，不提供开关。如后续出现真实需求，GDR 应由独立 PR 实现并单独验证。

## 启动流程与降级

driver 在**第一次 `tq.init` 之前**完成探测并生成 job 级唯一的 effective config，其余组件（actor / critic / rollout / sft / advantages / actor_fwd）都读同一份，不各自决策。

1. 校验参数组合（例如 `simple` + `rdma-mode` 会被拒绝）
2. `probe_cluster_nodes()` 通过 Ray 把探测任务绑定到每个**存活且有 GPU** 的节点，并额外探测 driver（driver 也会创建 Mooncake owner client）；各节点读取本机 `/sys`、mooncake 状态，并在 2 秒上限内检查外部 master 的 TCP 可达性；超时或崩溃的节点转为退化结果，不静默丢弃
3. `reduce_results()` 做 AND 归约：整个作业只能跑在最低共同能力上
4. Mooncake 生效前，driver 在**每个存活节点**（不限 GPU，因为 Serve replica 与 0-CPU actor 没有 placement 绑定）用真实配置各跑一次**有界 attach 握手**并立即 detach；`auto` 下任一节点失败则统一关闭 Mooncake 状态、全作业收敛到 SimpleStorage，`off`/`required` 下启动失败并列出失败节点
5. `required` 模式下若发生任何回退，直接抛异常并打印每个节点的探测明细

第 4 步不是轻量探针：每个节点都会创建真实 Mooncake client，并按配置请求挂载/注册完整 client segment（默认 `global_segment_size=4 GiB`，另有默认 1 GiB local buffer），完成后立即释放。具体物理 RSS、锁页与注册方式取决于 Mooncake 实现，但启动阶段会出现节点级瞬时内存/注册资源尖峰；CPU-only head 也在覆盖范围内。节点内存或 `memlock` 不足会表现为 attach 握手失败：`auto` 下整个作业统一回退 SimpleStorage，`off`/`required` 下启动失败。调大 `RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB` 时必须把这份每节点启动资源足迹一并纳入容量规划。

降级阶梯：

```
RDMA → Mooncake/TCP    （任一节点无 RDMA 能力，或指定设备缺失）
Mooncake → SimpleStorage（任一节点 mooncake/master 不可用、运行时正确性契约不满足，或 segment 预检不足）
```

## 启动日志怎么读

正常启动会打三段，排障时先看这三段：

```
[dataplane] requested: backend=mooncake rdma_mode=auto device=mlx5_bond_0
[dataplane] probe result:
[probe:<ip>] protocol=rdma device=mlx5_bond_0
  [ok] mooncake_import: version=0.3.10.post2
  [ok] rdma_devices: mlx5_bond_0, ...
  [ok] port_state: ACTIVE
  [ok] gid: ...
  [ok] memlock: unlimited
[dataplane] backend=MooncakeStore protocol=rdma device=mlx5_bond_0
```

第三段带 `fallback=...` 就说明发生了降级，原因直接写在里面（例如 `fallback=mooncake_unavailable:<node-id>`）。

## Mooncake master 生命周期

`auto_init` 固定为 `false`：**Relax 既不启动也不停止 master**，master 由部署环境管理。这样做是因为 TQ 的 `auto_init=true` 路径会执行 `pkill -f "[m]ooncake_master"`，在共享集群上会杀掉其他人的进程。

启动 master（部署侧，节点上执行一次）：

```bash
setsid mooncake_master -rpc_port=50051 -metrics_port=9004 > /var/log/mooncake_master.log 2>&1 < /dev/null &
```

然后给**每个节点**的作业环境设置 `MC_MASTER_ADDRESS=<master-host>:50051`。该变量是必填项：未设置时启动直接失败，Relax 不会假定 loopback 端点（多节点作业里每个节点都把自己的 localhost 当 master 会导致误降级或误中止）。

Mooncake/TCP 长会话还应在 driver 与所有 Ray worker 中统一设置 `MC_TCP_ENABLE_CONNECTION_POOL=1`。未启用连接池时，大批量反复传输可能耗尽临时 TCP 端口并报 `Cannot assign requested address`；该变量必须通过作业运行时环境传播到所有节点，不能只在提交命令所在的 shell 中设置。RDMA transport 不依赖此选项。

启动前置条件：部署侧必须先启动 master，所有 GPU 节点和 driver 都能解析并连接 `MC_MASTER_ADDRESS`，防火墙允许 master RPC 端口；作业镜像中的 TQ 必须包含本文“正确性依赖”所列修复。Relax 不负责拉起、重启或终止 master。

三种情形下的行为：

| 情形 | 表现 | 处理 |
|---|---|---|
| **master 在探测时不可达** | `auto` 统一降级到 SimpleStorage；`required` 启动失败并列出失败节点 | 先确认 master、DNS/路由和 `MC_MASTER_ADDRESS` |
| **master 探测通过、但 `tq.init` 时失败/超时** | 第一次初始化在独立 owner actor 中执行，driver 最多等待 60 秒。失败后回收该 actor 及其拥有的半初始化 controller；`auto` 只重试一次 SimpleStorage，`required` 清理后抛出原始错误 | 查看 `mooncake_init_failed:*`、master 日志和 owner 清理日志 |
| **attach 握手在某节点失败/超时** | driver 汇总各节点结果：`auto` 关闭 Mooncake 状态并统一回退 SimpleStorage（日志 `attach_handshake_failed:*`）；`off`/`required` 启动失败并列出节点。握手使用一次性 Ray worker，超时的 `tq.init` watchdog thread 不会污染后续任务 | 检查失败节点到 master 的连通性、RDMA 状态、可用内存与 `memlock`；握手会瞬时创建完整 client segment，CPU-only head 也会执行 |
| **worker attach 卡住**（controller 半初始化 / mooncake setup 挂起） | 每个 worker 的 attach 有统一 deadline（默认 60 s，`RELAX_TQ_ATTACH_TIMEOUT_SECONDS` 可调）：先有界等待 controller 提供配置，再在 watchdog 线程里跑 `tq.init`；超时该 worker 立刻失败而不是无限挂起 | 看 `TqAttachTimeout` 报错中的阶段描述 |
| **正常退出**（作业结束或全局重启） | 只有 owner actor 调用全局 `tq.close()`，随后显式 `storage_client.close()` 卸载 segment；附加 worker 只关闭本地 client，不能删除全局数据或 controller。master 本身不动 | 无需操作 |
| **异常退出**（worker 被 kill / OOM / 节点掉线） | Python 层不执行，segment 仍在 master 注册。master 要等 `client_ttl`（默认 30 s）才判定客户端过期，期间新作业的 put 会打到死端点并报 `Failed to open segment ... Connection refused` | 等 30 s 后重启，或部署侧调小 `-client_ttl` |

## 资源所有权与安全清理

首期按**单任务独占 Ray 集群**实现，**不承诺同一节点上多个 Relax job 并发**：多 job 并发、端口租约、master 共享机制都不在首期范围内。

清理只动本作业拥有的资源：

- 不使用任何 `pkill` / `killall`
- `tq.init` 之前会检查已存在的 `TransferQueueController` 命名 actor：**只有取不到 config（半初始化）或 actor 已死时才回收**，健康的 controller 保持不动并正常 attach
- 首次初始化在专用 owner actor 中执行，config 内保存随机 owner token；清理只有在 token 匹配或 controller 确认不可用时才回收全局 actor，避免初始化竞争中的附加者误杀所有者
- 全局 `tq.close()` 只能由 owner actor 调用；actor、critic、rollout 等附加 worker 只能做本地 detach
- master 进程始终不被 Relax 触碰

## 容量不足与正确性依赖

Mooncake 配置固定 `hard_pin=true`，不会为了腾空间静默驱逐已经生产但尚未消费的数据。启动前按 **token 预算推导最坏情况 payload** 检查 client segment（默认 4 GiB）：文本按 `seq_length × 32 B`，多模态样本另加 `seq_length × 784 像素/token × 12 B`（ViT patch 14、merge 2、float32 RGB，例如 8k 序列约 77 MiB/样本），再乘 rollout batch、采样数与 `staleness+1`；`auto` 预检不足时回退 SimpleStorage，`required`/`off` 直接失败。segment 大小可用 `RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB` 调整，容量校验与客户端配置读取同一个值。该上界是保守推导，不能替代运行时错误处理。

运行时依赖固定到 TransferQueue commit `58054a33834aadbcf76aacd6b1e32e25c030f2c9`，并在 Mooncake 启动前检查以下能力：

- `batch_upsert_from` / `batch_get_into` 对每个 key 的返回码做有限次数重试，耗尽后抛异常，不能把失败当成功或无限重试；
- `KVStorageManager.put_data` 必须先等待 storage put 成功，之后才能更新 production-ready 状态；写入失败时不通知消费者；
- Relax 的契约测试用失败 store 验证“写失败、production 状态不更新”，并用隔离 master 的真机故障注入验证物理容量溢出在 30 秒内显式失败。

因此，上游曾出现的“返回码未检查导致静默丢数据”不是已知限制，而是 Mooncake 启用的硬门槛：`auto` 在契约不满足时禁用 Mooncake 并回退，`required` 拒绝启动。Docker 镜像固定上述修复 commit，运行时检查用于防止环境被旧包覆盖。

正确性守卫强制 `MC_STORE_MEMCPY=0` 且 **fail-closed**：mooncake 0.3.10 在 TCP-only 环境会自动启用 memcpy 快拷贝路径，该路径存在已确认的静默截断缺陷（现象与处置见排障表）；RDMA 会话本就自动禁用 memcpy，不受影响。由于缺陷在当前 pin 上已实证，显式导出 `MC_STORE_MEMCPY=1` 会在启动时被直接拒绝，待 pin 升级到修复版本后再按版本重新放开。另外，针对 pin 版本剩余缺口的运行时补丁（逐次重试返回码校验、严格 production-status ACK）已拆分到独立的版本门控分支 `feat/tq-mooncake-loss-guards`（`relax/utils/tq_mooncake_patches.py`），本 PR 只保留只读的能力校验。

真机容量故障注入会故意创建 64 MiB segment 并写入 96 MiB，仅允许在独立、可丢弃的 master 上运行：

```bash
MC_MASTER_ADDRESS=<isolated-master>:50051 \
RELAX_RUN_REAL_MOONCAKE_CAPACITY_TEST=1 \
pytest -q tests/utils/test_tq_failure_paths.py \
  -k real_mooncake_capacity_overflow
```

## 验收分层

Mock/本机测试和真实双节点 RDMA 测试必须分别报告，前者不能替代后者。

| 层级 | 验证内容 | 通过标准 |
|---|---|---|
| CI/mock | 参数矩阵、节点 AND 归约、master 不可达、owner 超时/清理/token、auto/required、有限重试、写失败不发布状态 | `tests/utils/test_rdma_probe.py` 与 `tests/utils/test_tq_failure_paths.py` 全部通过；真机项允许明确 skip |
| 本机 TQ | SimpleStorage 全链路 put/get、容量 backpressure、空读、清理、字节一致性；`multimodal_train_inputs` 以生产容器（`list[dict]` / NonTensorStack，存储层非张量路径）全链路逐叶子 SHA-256 一致 | `tests/utils/test_tq_dataplane_behavior.py` 通过（含 `TestRealMultimodalFullLink`） |
| 真实 Mooncake | TCP/RDMA direct-client：混合 dtype 稠密张量逐字节一致；`list[dict]` 非张量 msgpack 慢路径逐叶子一致（每协议独立 spawn 子进程，规避 0.3.10 会话内协议切换问题） | `TestMooncakeByteExact` 的 TCP/RDMA 各两档均通过，不得 skip |
| 真实多模态载荷 | 真实数据集图像走完整生产预处理链（`build_messages` → `apply_chat_template` → `process_vision_info` → HF processor → `remap_mm_train_inputs`）生成 fixture；上述两级多模态用例检测到 fixture 后自动升级为真实载荷档 | fixture 存在时以 `[real]` 档通过；无 fixture 环境回退 `[synthetic]`（生产同构状，CI 兜底）；交付报告须注明真实档在何处跑过 |
| 真实双节点 | 同一拓扑的 SimpleStorage、Mooncake/TCP、Mooncake/RDMA；synthetic、production-shaped multimodal、real-multimodal 三种 profile；256/1024/2048/4096 MiB；每档 warmup + 至少 5 轮 | 每次 get 的逐字段 SHA-256 全部 PASS；`--require-wire-proof` 证明 RDMA 档 IB counter 增长且 TCP 档网络 counter 增长；CSV 留档并报告均值、median、stddev |

真实多模态 fixture 生成（需要本地数据集 parquet 与 Qwen-VL 模型目录；产物写入 `tests/fixtures/`，已 gitignore，不入库）：

```bash
PYTHONPATH=. python scripts/benchmarks/make_multimodal_fixture.py \
  --dataset <path>/<multimodal-dataset>.parquet \
  --model <path>/<qwen-vl-model-dir> \
  --num-prompts 6 --n-samples-per-prompt 2 \
  --output tests/fixtures/tq_multimodal_fixture.pt \
  --manifest-json tests/fixtures/tq_multimodal_fixture.manifest.json
```

生成时自动做 processor 双跑字节一致性校验（以真实数据验证 F4 组共享假设）；fixture 载入时按叶子清单自校验，损坏即报错。测试与 bench 通过 `RELAX_MM_FIXTURE`（或默认路径 `tests/fixtures/tq_multimodal_fixture.pt`）发现 fixture。逐叶子指纹的规范实现在 `relax/utils/payload_digest.py`（哈希原始存储字节，对 NaN 也成立，严格强于 `torch.equal`）。

双节点验收命令（master 与 Ray 集群需由部署侧预先准备）：

```bash
PYTHONPATH=. python -u scripts/benchmarks/tq_cross_node_bench.py \
  --master <master-host>:50051 \
  --nodeb-ip <consumer-node-ip> \
  --device <rdma-device> \
  --payload-profiles synthetic multimodal real-multimodal \
  --payload-mib 256 1024 2048 4096 \
  --repeats 5 --require-wire-proof \
  --csv tq_cross_node_acceptance.csv
```

`real-multimodal` profile 按目标档位循环平铺 fixture 样本，`multimodal_train_inputs` 列以 NonTensorStack 走存储层非张量路径（SimpleStorage pickle / Mooncake msgpack），字节校验用行多重集指纹（采样器可重排行序；张量行按 dtype+字节比较以兼容后端间标量行 `()` 与 `[1]` 的表示差异，dict 叶子仍全形状校验）。

若验收环境没有两个 RDMA 节点，交付结论必须写成“真机验收未执行”，不能用 mock 通过推导真机已经通过。

### 参考吞吐区间

2 节点 × 8 GPU、mooncake 0.3.10.post2 上按上述命令完成过一次全矩阵验收（36/36 测量点逐字节 PASS、wire-proof 全部成立），量级供容量规划参考：

- get 均值：C2 RDMA 2.1~4.6 GB/s（real-multimodal 最高 8.4），C1 TCP（守卫后）0.8~1.1 GB/s，C0 SimpleStorage 1.2~3.2 GB/s；C2 相对 C1 增益 2.1×~8.4×，全档位满足 ≥20% 门槛
- put 均值：C2 4.1~12.9 GB/s，C1/C0 1.4~2.6 GB/s——写侧收益显著大于读侧，与“已知限制”第一条一致
- 大档位（4G）下 C2 吞吐明显上扬：大批量摊薄了每 key 的 MR 注册开销

逐档明细、逐轮分布与原始 CSV 属于交付验收材料，随验收报告存档，不在本文档维护。

## 排障表

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 启动日志 `backend=SimpleStorage fallback=mooncake_unavailable:<node>` | 该节点上 `import mooncake` 失败 | 检查该节点的 `mooncake-transfer-engine` 安装；镜像是否一致 |
| 启动日志 `MooncakeStore capacity fallback to SimpleStorage` | 保守的最坏情况容量上界超过 client segment；多模态按“每个 token 都可能是 vision token”估算，8k 序列约 77 MiB/样本，再乘 batch、采样数与 `staleness+1`，很容易超过默认 4 GiB | 按容量日志核对参数；减少 batch / `n_samples_per_prompt` / `max_staleness`，或调大 `RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB` 并同步规划每节点 attach 的瞬时资源足迹 |
| `protocol=tcp fallback=...`，但机器有 RDMA 卡 | 端口非 ACTIVE、GID 取不到、`memlock` 过低，或指定的 `--tq-rdma-device` 在部分节点不存在 | 看 `probe result` 里哪一项 FAIL；`memlock` 需要 unlimited |
| `setup failed with error code: -1` | master 不可达 | 检查 master 进程与 `MC_MASTER_ADDRESS` |
| `Failed to open segment ... Connection refused` | 上一轮客户端异常退出，死 segment 仍在 master 注册 | 等 `client_ttl`（30 s）过期后重试 |
| `batch_get_into failed ... error codes [-800, ...]` | 会话内切换协议（0.3.10 上更敏感），或对端不可达 | 每个协议单独进程跑；确认对端存活 |
| Mooncake/TCP 档 get 数据尾部全零，但批量返回码全部成功（逐字节校验 FAIL） | mooncake 0.3.10 memcpy 快拷贝路径缺陷：TCP-only 环境被自动启用后，跨节点 get 会静默截断（坏行自 64 KiB 对齐偏移起全零） | 正确性守卫已强制 `MC_STORE_MEMCPY=0`（见“容量不足与正确性依赖”）；显式设 `1` 会被启动拒绝，unset 即可 |
| Mooncake/TCP 档在单机回环下原生 SIGSEGV | 与上一行同源（memcpy 路径），回环下表现为崩溃而非静默截断 | 同上 |
| 长时间反复起停会话后 `batch_upsert_from ... error codes [-800, ...]`，重试耗尽（响亮失败，非静默） | master 长期吸收异常退出的客户端后状态劣化，metrics 仍报 serving | 重启 mooncake master；长跑验收前先起新 master |
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
- Mooncake/TCP（C1）在 0.3.10 上依赖 `MC_STORE_MEMCPY=0` 守卫保证字节正确，且守卫后 get 吞吐低于 SimpleStorage：TCP 档定位为 RDMA 不可用时的正确性兜底，不是性能选项。
- 消费端节点在 get 过程中中途死亡的端到端行为需要双节点真机验证，未做成自动化测试。
- Mooncake 传输层自身的超时参数不由 Relax 控制。
