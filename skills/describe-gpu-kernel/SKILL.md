---
name: describe-gpu-kernel
description: Specify a GPU kernel as an explicit Python function with an executable reference and 20 reproducible input cases, then describe its active launch, per-warp tasks, arithmetic, memory traffic, and dependencies.
---

# GPU kernel 描述规范

适用于定义新的 kernel case、解释实现、优化前建模、代码审查和构建 warp/task 依赖图。每种 kernel 先提供 Python 接口、参考实现和输入集，再描述具体实现的结构；静态结构、测得时间和模型推断分别标注。

使用 [描述模板](references/kernel-description.md) 组织输出。任务只需要文字时使用 Markdown 表格和依赖边；需要浏览大图时再生成可缩放的 HTML。无需依赖某一种 agent 或可视化工具。

## 每种 kernel 的接口、参考实现与 20 组输入

这里的 kernel 是一个可替换的计算单元，可对应单个 device kernel，也可对应多个 kernel 拼接成的模块。先固定计算范围，不以 launch 数定义其语义。

每种 kernel 在 `kernels/<kernel-name>/` 中提供：

- 可直接调用的 Python function。所有 tensor 输入、权重、初始状态、scalar 配置以及输出缓冲区都显式出现在函数参数中；注明 input/output/inout。不要以隐式全局配置、内部随机输入或 opaque workload dict 代替接口。
- 可执行的 reference；允许由多个现有 kernel 或标准 tensor 算子拼接。给出来源、数学语义、精度/舍入边界和依赖。candidate 使用相同接口；reference 的作用是定义结果，不要求和 candidate 有相同的融合结构。
- 暂定恰好 **20 组合法 inputs**，逐项记录 ID、shape、dtype、stride/layout、数值分布/范围、seed、状态和 scalar 参数。按 kernel 的实际支持域覆盖不同 shape、dtype、边界与数值情形，不强迫每个 kernel 支持任意 dtype 或 shape。
- 确定性的输入生成函数：按 case ID 生成完整函数实参，输出由调用方预分配；保存配方，不把大型 tensor 二进制提交到仓库。输入集合不能只改 ID 而生成等价数据。
- 接口/输入表、运行方法和已执行/未执行范围。大输入按需生成，不能用缩小输入的运行结果宣称原 shape 已验证。

具体字段与交付格式见 [Kernel case 规范](references/kernel-case.md)。KDA 完整模块的实例放在仓库 `kernels/kda_megakernel/`；技能迁移时无需依赖该实例。生成代码的注释使用英文。

## 先展开实际实现

从宿主入口追踪 dispatch、模板实参、宏、编译选项、环境和 launch。记录每个相关分支为 active、inactive 或 unreachable；宏已定义但值为 0 与未定义可能具有不同语义。不能凭函数名推断使用了某条矩阵指令。

报告 grid/block 公式及具体数值、warp/warpgroup 数、各角色映射、shared allocation、寄存器资源和 cooperative/residency 限制。资源上限、编译属性、occupancy 查询和实测驻留分别标明来源；`launch_bounds` 不能直接当作实际驻留数量。

对于 persistent kernel，区分 block 数、逻辑 task 数、task 分区和动态分配。说明 task 编解码、队列所有者、claim/完成/终止条件及跨 block 的依赖。

## 细化任务和依赖

每个有意义的任务说明负责的 warp/warpgroup、计算公式、操作数位置、输出位置、有效参与者、重复次数和源码文件:行号/符号。细分标量浮点/SFU、矩阵运算、global load/store、shared load/store及控制/同步。寄存器地址构造和 issue bookkeeping 可以解释，但不要伪装成有效浮点运算。

对异步操作分开描述 issue、数据 ready、结果完成和最后 reader。普通 shared load/store 由发出它的 warp 执行，不是自动独立于该 warp 的复制流水线。

依赖边记录“生产事件 → 消费事件；对象/切片/槽位/代次；同步机制；源码依据”。覆盖数据 RAW、覆盖前的 reader 退休、对象生命周期、任务控制和必要的 collective 条件。不要从绘图相邻位置、角色排列或粗粒度阶段名称添加全局 barrier 或虚假依赖。

屏障发布一次共享 ticket，并不保证所有 warp 都已读完它。终止或切换角色时，检查 leader 下一次覆盖与全部读者完成之间的关系；operand drain 与控制字生命周期可能是不同的边。

## 工作量与流量

先定义符号，包括 shape、tile、元素字节数、有效 lane、循环次数和 tail。先给 per task/per block 的公式，再给具体配置数值与全 grid 合计；动态 task 归属不能假定每个 block 平分。

矩阵乘加按乘法和加法各一次计算，常规 GEMM 的逻辑工作为 `2MNK`。区分逻辑工作与包括 padding、变换、merge 在内的执行工作。集体矩阵指令的 tile 已覆盖参与线程，不再乘 warp/warpgroup 的线程数。

标量运算按 `+`、`*`、FMA、除法、SFU、转换和整数/地址操作分别计数，结合实际向量宽度与 active lanes。不同运算的次数不能直接相加成等价吞吐率；源代码计数需核对实际编译路径。

分别列 global/shared load/store 的请求字节公式。global→shared 传输有两个端点，不能计成两次 DRAM 传输。分清 allocation、逻辑 payload、实际请求、L2/DRAM transaction、shared bank/wavefront；global 请求可能由 cache 服务，隐式矩阵 shared 访问也可能不体现在显式 load opcode 中。

## 展示和交付

在开头放符号表与并行配置，随后给任务表、细粒度依赖、per block/全 kernel 汇总和证据限制。公式使用一致数学符号并提供具体代入值。保留源码 revision；行号变动时能通过符号重新定位。

图的横轴可表示 warp/warpgroup，纵轴表示逻辑先后。没有真机时间证据时将其标为结构示意，不给出虚构持续时间或硬件 busy 区间。交互 HTML 应支持缩放、平移、任务详情和图例，便于大 kernel 检查；离线交付时说明公式渲染等资源依赖。
