# Kernel 描述模板

每个数值标注 static-derived、measured 或 modeled；缺少数据时写 unknown。源码锚点需要文件、行号/符号和 revision。

## Python 契约与参考实现

- 完整 Python function 签名；逐参数 input/output/inout、shape/dtype/layout、alias 和 scalar 语义。
- 可执行 reference 的位置、依赖、拼接顺序、数学语义与精度边界。
- 20 个输入 case 的 ID、维度、dtype、分布、seed、状态与覆盖目的；关联输入生成器和单例运行命令。
- reference/candidate 支持域及已执行、metadata-only 和未执行的输入范围。

详细字段见 [Kernel case 规范](kernel-case.md)。

## 身份和并行配置

- 入口 → dispatch → template/macro → 实际 device kernel；源码/config/binary 身份。
- 符号表：张量维度、tile 大小、dtype 字节数、tail、重复次数和有效参与者。
- `grid = ...`、`block = ...`：符号公式及具体代入。
- 每 CTA warp/warpgroup 数与角色；静态/动态 shared、compiled registers/local、launch 限制。
- 每 SM 驻留上限的约束与实际查询/测量；cooperative kernel 的全 grid 驻留条件。
- persistent task 总数、分区、task ID 编解码和运行时分配方式。

宏分支表：

| 文件:行号/符号 | 条件及解析值 | active / inactive / unreachable | 影响 |
|---|---|---|---|

## 任务

| task ID | warp/WG/CTA | 操作与公式 | 读取对象/位置 | 写入对象/位置 | 次数/active lanes | 源码锚点 |
|---|---|---|---|---|---|---|

类别至少区分矩阵浮点、标量浮点/SFU、global load/store、shared load/store、issue/control 与同步。异步 issue task 与其 ready/completion event 可以拆开表示。

## 依赖

| from event | to event | 对象/切片/slot/generation | 依赖原因 | 实现机制与参与者 | 源码依据 |
|---|---|---|---|---|---|

每个复用对象检查：init → publish → ready → readers → all-readers-retired → release → overwrite/next-generation → drain。需要显式重新初始化的对象另记 invalidate/init 协议。多个对象不强制共用一个阶段屏障。

角色交接例：leader 写 terminal ticket → 发布 → 全部旧角色读取 → 读者退休 → leader 写新队列 ticket。发布和退休是不同的条件；实际代码可能通过 collective 或其他顺序保证，不应预设机制。

## 工作量

| 指标 | per task/block 公式 | 特化数值 | 全 grid 累计方法 | 证据/限制 |
|---|---|---|---|---|
| Tensor logical FLOPs | 有效数学问题的工作量 | 按实际 shape | 逻辑 task 集合 | 与执行工作分开 |
| Tensor executed FLOPs | 指令 tile × 动态次数，含 padding | 按实际实例 | 已验证动态覆盖 | collective 不乘线程数 |
| scalar + / * / FMA / SFU | active lanes × 指令/操作数 × 次数 | 分项列出 | 分项累计 | FMA=2，向量宽度另计 |
| global load bytes | 请求元素 × dtype bytes × 次数 | 分项列出 | 按实际 task | 不等于 DRAM |
| global store bytes | 同上 | 分项列出 | 同上 | 记录 masked tail |
| shared load bytes | 显式与隐式 operand 服务分别计 | 分项列出 | 同上 | 区别请求与 wavefront |
| shared store bytes | 写入宽度 × active lanes × 次数 | 分项列出 | 同上 | 记录异步传输端点 |

动态队列不能用“单 CTA 工作 × grid”推算总量，除非确实均分或该单 CTA 值是经过定义的平均值。对计数器也核对 per-thread/per-warp/per-SM/device 的统计范围。

## 可视化与限制

图例说明角色、资源类别、事件、依赖边和时间基准。每个节点能定位公式、数据位置、工作/流量与源码。同步画成对应事件或依赖，不用横跨全部角色的大条代替细粒度关系。

静态图只表达可行偏序与潜在重叠；没有动态测量时不要给资源占用比例或精确持续时间。实测图标明采样、插桩、缺失事件和与原始普通 binary 的区别。
