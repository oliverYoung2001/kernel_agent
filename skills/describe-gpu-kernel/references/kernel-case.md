# Kernel case 规范

## 文件职责

```text
kernels/<kernel-name>/
├── README.md       # 接口、语义、输入表、依赖、运行与验证范围
├── reference.py    # 显式 Python function 和可执行参考实现
├── cases.json      # 20 个输入配方及参数；也可使用等价 Python 数据定义
├── inputs.py       # 由配方生成全部 function arguments
└── run.py          # 枚举/运行单个或多个 case 的简洁入口
```

文件名可以遵循已有仓库约定。reference 可以调用多个 device kernel；不需要为了形式强行融合。仅供 agent 使用的 reference 自检脚本、日志和结果放在 `tmp/`，不放入 kernel 交付目录。

## 函数接口

```python
def reference(x, weight, initial_state, out, final_state, *, scale):
    """Read inputs and write caller-owned output buffers."""
```

上例仅说明显式参数风格。真实函数列出全部输入、所有输出缓冲区和影响结果的 scalar 参数。每项说明 shape 公式、dtype、device、layout、读写方向及 alias 规则。返回值可以是 None 或已写入的输出，但不能隐藏未在接口声明的结果。

生成器可返回与函数参数名一一对应的字典，调用方式为 `reference(**arguments)`；计算函数本身保持显式签名。输入创建不计入 kernel 的计算边界。

## 20 组 inputs

每组至少记录：

| 字段 | 要求 |
|---|---|
| case ID | 唯一、稳定，可单独选择 |
| dimensions | 足以推导每个 tensor shape 的全部维度 |
| dtypes | 各类 tensor 的 dtype，不能只写一个含糊的全局 dtype |
| layout | strides、contiguity 和需要的 alignment |
| values | 各输入的分布、范围/尺度、状态初始化和特殊值配方 |
| randomness | seed、生成算法/版本与 device 约定；不污染调用者 RNG |
| scalars | eps、scale、mask 等影响计算的参数 |
| purpose | 覆盖的 shape/精度/数值/边界特性 |
| cost | 大内存或长运行 case 的按需执行方式 |

同一 seed 下改变分布或 shape 也可形成不同输入；明确配方，而不是只声明“随机”。负例与 unsupported case 单独记录，不占这 20 组合法输入名额。

## 验证与证据

先核对 20 组参数能绑定真实函数、shape/dtype 一致、生成可复现且无未声明输出。再执行可承受的 reference case，检查有限性、输入/状态不被意外修改、输出被完整写回，并对核心数学关系使用独立小例核验。

每个优化版本需要对全部输入分别比较 reference 的所有输出，按自定义阈值返回逐输入 pass/fail。若 reference 是数学 FP32/FP64 oracle，而历史实现有 chunk 求解/低精度中间舍入，应明确区别，再按任务确定比较门限，不能宣称 bitwise 等价。

对仅做 metadata 检查的生产大 shape，标明“输入规格已检查，数值未执行”。提供真实执行命令，禁止用小 shape 冒充生产 case。
