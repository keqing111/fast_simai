# SimAI/NS-3 All-to-All Baseline 设计

- 日期：2026-10-12
- 状态：已批准
- 规范实现目录：`/home/julius/fast/SimAI`

## 1. 背景与目标

本项目需要在 SimAI 的 ns-3 后端上建立可复现的 all-to-all 调度基线，用同一份流量矩阵比较不同调度器的完成时间、链路行为和流级表现。第一阶段只使用合成流量，包含均衡矩阵和 FAST 原生分布生成器；真实负载在用户提供数据集后接入。

FAST 原生仿真器不是逐流反馈的事件驱动调度器。它先根据流量矩阵和硬件参数计算 balance、intrinsic、inter、dispatch 各阶段的解析耗时，再把阶段串并行关系展开为时间轴。因此，本设计把 FAST 第一版定义为离线 open-loop 调度：在 ns-3 启动前生成全部流和绝对开始时间，然后一次性注入 SimAI。ns-3 负责网络竞争和实际完成时间，不负责触发后续 FAST 阶段。

目标如下：

1. 提供统一的调度器接口，使所有算法接收相同的流量矩阵与硬件描述，输出同一种调度表示。
2. 将已经通过原生 C++ 交叉验证的 FAST Python 移植作为唯一 FAST 实现来源，避免继续使用行为不一致的独立重写版。
3. 在 SimAI 中以 MICRO/通信模式运行 schedule，计算部分设为零，使用 ns-3 得到实际网络结果。
4. 支持均衡、FAST uniform random、FAST Zipf 三类可复现合成矩阵，并记录倾斜度而不把 Gini 系数作为生成器输入。
5. 建立从输入、调度、仿真到汇总的可追踪实验产物和验收测试。

## 2. 第一阶段边界

第一阶段包含：

- 调度器：`direct-concurrent`、`spreadout`、`roundrobin`、`fast`。
- 流量：balanced、FAST uniform random、FAST Zipf。
- 后端：SimAI + ns-3。
- 拓扑：SimAI/ns-3 已能正确表达的无阻塞与可配置超卖网络；具体实验拓扑由配置文件给出。
- 指标：collective makespan、算法带宽、物理字节数、流完成时间、调度生成开销，以及 FAST 阶段执行偏差。

第一阶段不包含：

- NCCL-like 的真实性建模；当前“所有点对在零时刻发送”只称为 `direct-concurrent`。
- ring/hybrid cube 等需要 GPU 中继转发才能忠实表达的拓扑。
- 真实 trace、训练计算与通信重叠、设备性能随机波动。
- 根据 ns-3 实际完成事件动态释放下一阶段流的闭环 FAST 变体。

## 3. 代码与版本边界

`/home/julius/fast/SimAI` 是后续实现、测试和文档的规范仓库。它当前记录了 `SimCCL`、`aicb`、`ns-3-alibabacloud` 的固定 submodule commit，但工作区内尚未初始化这些 submodule；实施阶段应按仓库记录的 SHA 初始化，而不是使用浮动版本。

`/home/julius/simAI/SimAI` 中已有外部 schedule 注入和实验代码，但同时含有不相关的本地修改。它只用于阅读和选择性迁移，不能整体复制或作为实验结果的规范来源。

`/home/julius/fast/poc_simai/fast_decomp_py/fast_decomp.py` 是已与 FAST C++ 原生模拟器完成逐项交叉验证的 Python 实现。实施时将其整理进规范仓库，并保留交叉验证入口。

## 4. 总体数据流

```text
TrafficGenerator
      |
      v
TrafficMatrix + MatrixManifest
      |
      v
AllToAllScheduler + HardwareProfile
      |
      v
Schedule + ScheduleManifest
      |
      v
SimAI external-schedule adapter
      |
      v
ns-3 execution
      |
      v
Raw events + RunManifest -> Analyzer -> Summary
```

生成器、调度器和 ns-3 执行器之间通过落盘产物解耦。每一步均可单独测试，且相同矩阵可以被多个调度器复用。

## 5. 数据模型

### 5.1 TrafficMatrix

矩阵使用 NumPy `.npy` 文件保存，类型为非负 `int64`，单位为 byte，形状为 `N x N`，主对角线必须为零。`matrix_manifest.json` 至少包含：

- `schema_version`
- `generator` 与生成器参数
- `seed`
- `num_devices`
- `total_bytes`
- `matrix_sha256`
- `cell_gini`
- `source_gini`
- `destination_gini`
- `server_pair_gini`
- `source_bottleneck_ratio`
- `destination_bottleneck_ratio`

其中 cell Gini 衡量非对角矩阵元素，source/destination Gini 衡量行和与列和，server-pair Gini 在给定 `gpus_per_server` 后聚合服务器对流量。Gini 仅作为结果元数据和分层分析依据，不参与 FAST 调度决策。

### 5.2 HardwareProfile

硬件配置使用 JSON 保存，至少包含：

- `num_servers`
- `gpus_per_server`
- `intra_alpha_ns`、`intra_beta_ns_per_byte`
- `inter_alpha_ns`、`inter_beta_ns_per_byte`
- `topology_id`
- `link_bandwidth_gbps`
- `oversubscription`

FAST 解析模型只读取它需要的 alpha/beta 与设备布局；SimAI/ns-3 使用拓扑和链路字段。二者来自同一个 profile，以避免调度器预测环境和实际仿真环境无意失配。无法精确等价的参数必须在 run manifest 中明确记录映射规则。

### 5.3 Schedule

调度器公共接口为：

```python
class AllToAllScheduler(Protocol):
    def schedule(
        self,
        matrix: TrafficMatrix,
        hardware: HardwareProfile,
        *,
        seed: int,
    ) -> Schedule: ...
```

每条 `ScheduledFlow` 包含：

- `flow_id`
- `src`
- `dst`
- `size_bytes`
- `start_ns`
- `phase`：`direct`、`balance`、`intrinsic`、`inter` 或 `dispatch`
- `stage_id`

`schedule.csv` 不带表头，固定为 `src,dst,size_bytes,start_ns` 四列，供 SimAI C++ 适配器读取。`flow_id` 定义为该文件中的零基记录序号；`schedule_metadata.csv` 按 `flow_id` 保存 `phase,stage_id` 等分析字段。这样既能稳定关联 ns-3 结果，又不依赖 C++ 解析器忽略额外列的偶然行为。

`schedule_manifest.json` 至少记录：算法名及参数、输入矩阵哈希、硬件配置哈希、seed、调度生成耗时、逻辑字节数、物理字节数、预测 makespan、输出文件哈希和代码 commit。

`start_ns` 为相对一次 collective 起点的整数纳秒。相同起点的流允许并发；排序只用于生成确定性文件，不能被执行器解释为隐含依赖。

## 6. 流量生成器

所有生成器接收显式 seed，并在同样参数和代码版本下产生字节级一致的矩阵。

### 6.1 balanced

所有 `i != j` 的元素相等，主对角线为零。参数为每个有向点对的消息大小。

### 6.2 fast-uniform

先按 FAST 原生测试程序的语义，对每个矩阵单元独立采样 `[0, 9999]` 的离散均匀整数权重，再清零主对角线，并以 `bytes_per_unit` 转成字节。Python 生成器采用项目固定的 PRNG 和 seed；分布语义与原生代码一致，但不承诺复现不同 libc 的 `rand()` 比特流。

### 6.3 fast-zipf

对每个矩阵单元独立采样 FAST 使用的截断 Zipf 权重，支持原生实验中的 alpha 范围，清零主对角线后以 `bytes_per_unit` 转成字节。生成器实现与 FAST 原生逆 CDF 公式建立统计和固定样例测试。

第一阶段保留 FAST 原生的“固定 `bytes_per_unit`”语义，不对不同 alpha 的总字节量做隐式归一化。调度器之间只比较同一个矩阵。跨 alpha 展示结果时必须同时报告 `total_bytes`；如果后续需要隔离纯倾斜效应，另加显式的 `fixed-total` 实验组，不能覆盖原生组。

## 7. 调度器语义

### 7.1 direct-concurrent

对每个非零 `T[src,dst]` 生成一条同大小的流，全部在 `start_ns = 0` 启动。它是简单的直接发送下界/竞争基线，不称为 NCCL。

### 7.2 spreadout

按 FAST 原生 baseline 的循环置换集合生成发送顺序。调度器使用硬件 alpha/beta 模型为各轮分配离线开始时间，并以固定的设备/轮次遍历规则消除容器迭代顺序带来的随机性。

### 7.3 roundrobin

按 FAST 原生 round-robin baseline 的配对和轮次规则生成 schedule，使用相同硬件开销模型计算离线开始时间。任意随机化只允许来自显式 seed，并写入 manifest。

### 7.4 fast

`fast` 必须调用通过 C++ 交叉验证的 faithful port，不再调用现有独立重写的 `fast_scheduler.py`。

FAST pipeline2 的时间关系为：

```text
balance
   |
   v
slot 0: inter[0] || intrinsic
   |
   v
slot 1: inter[1] || dispatch[0]
   |
  ...
   |
   v
slot k: inter[k] || dispatch[k-1]
   |
   v
final dispatch
```

每个 slot 的解析时长是其中并行分支预测耗时的最大值；下一个 slot 的绝对开始时间由这个解析时长累加得到。balance 在 slot 0 前完成，最后一个 dispatch 在全部 inter slot 后运行。

调度输出只表达这条原生解析时间轴。即使 ns-3 中前一阶段尚未完成，后续流仍按预计算的 `start_ns` 释放。这是有意的 open-loop 语义，不是实现缺陷。分析器必须计算阶段偏差：

```text
stage_violation_ns = max(0, actual_previous_slot_end_ns - scheduled_slot_start_ns)
```

并汇总存在 violation 的 stage 比例和最大/总 violation。未来若实现基于实际完成事件的闭环版本，算法名必须与 `fast` 区分，例如 `fast-feedback`。

## 8. SimAI/ns-3 适配

SimAI 增加显式 external-schedule 执行路径：

1. 启动前读取并校验整个 `schedule.csv`。
2. 将相对 `start_ns` 转换为 SimAI/ns-3 事件时间。
3. 为每条记录创建对应的有向流，并用零基记录序号保留可关联到 `flow_id` 的结果标识。
4. 不根据 CSV 行顺序串行化相同时间的流。
5. 不引入计算事件；通信基线使用 MICRO workload 和零计算时长。
6. 当 schedule 的设备数、端点范围、字节数或时间非法时快速失败，不静默截断。

第一阶段仍是“流量生成/调度”和“网络执行”解耦。适配器不在运行时调用 scheduler，也不尝试动态新增未知流。

## 9. 结果与指标

每次运行至少输出：

- `collective_makespan_ns`：最后一条 flow 完成时间减去 collective 起点。
- `logical_bytes` 与 `physical_bytes`。
- `algorithm_bandwidth_GBps = logical_bytes / collective_makespan`。
- flow completion time 的 mean、p50、p95、p99、max。
- scheduler wall-clock 开销。
- ns-3 可获得时的链路利用率、队列峰值、丢包/PFC 指标。
- FAST 的 stage violation 数量、比例、最大值与总值。

每次运行生成 `run_manifest.json`，记录 SimAI、ns-3、调度器 commit，输入和 schedule 哈希，完整命令、环境配置和结果文件哈希。所有脚本使用相对于仓库根目录的路径或命令行参数，禁止硬编码个人绝对路径。

## 10. 正确性验证

### 10.1 调度器级

- 输入矩阵形状、类型、非负性和对角线检查。
- 每种调度器固定 seed 的快照/哈希重现测试。
- `direct-concurrent` 的每个非零矩阵元素与输出流一一对应。
- 每个算法验证逻辑字节守恒；若算法引入中间转发，额外核对并报告物理字节。
- FAST Python 与原生 C++ 的现有 13 组交叉验证必须继续 13/13 通过，包括预测时间、stage 数量、物理字节和 flow multiset。
- FAST phase/stage 的开始时间满足 pipeline2 的解析关系。

### 10.2 SimAI 适配器级

- 解析合法 schedule，并拒绝越界端点、负字节、负时间和重复 `flow_id`。
- 两条同起点流被注册为并发事件。
- 一个无竞争小例子中，实际完成时间与链路 alpha/beta 预期在可说明的误差内。
- flow 结果可以无歧义关联回 `flow_id`。

### 10.3 端到端 smoke test

在一个小规模、快速完成的 ns-3 拓扑上，对四种调度器分别运行 balanced 和一个 skewed 矩阵。验收要求：

- 全部进程成功退出且无漏流。
- 同一矩阵下所有调度器的逻辑字节相同。
- 相同配置重复运行的 schedule 哈希和关键结果一致。
- 结果汇总能明确区分预测 makespan 与 ns-3 实际 makespan。
- FAST 输出 stage violation 指标，即使其值为零。

## 11. 初始实验组织

实验配置采用显式笛卡尔积，但第一轮仅运行足够小的 smoke matrix，确认语义后再扩展规模：

- scheduler：4 种。
- generator：balanced、fast-uniform、fast-zipf。
- message scale：至少一个小消息和一个大消息档位。
- hardware profile：至少一个单机/高速域和一个多机网络域。
- topology：先使用已验证的非 ring 配置，再加入无阻塞与超卖对照。

runner 必须先生成并缓存矩阵，再让所有 scheduler 引用同一矩阵哈希。失败的 run 独立记录，重跑不得覆盖原始 manifest。

## 12. 验收定义

第一阶段完成需同时满足：

1. 统一 scheduler API 和四个实现可由同一个 CLI 调用。
2. 三类生成器可复现，且倾斜度与总流量元数据完整。
3. faithful FAST 交叉验证保持 13/13 通过。
4. SimAI/ns-3 能预载并执行带绝对开始时间的 schedule。
5. 端到端 smoke test 覆盖四个 scheduler，并通过字节守恒、无漏流和确定性检查。
6. 汇总结果同时给出网络实际指标、FAST 预测指标和 stage violation。
7. 所有运行不依赖工作区外的代码或硬编码个人路径。
