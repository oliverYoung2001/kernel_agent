# Verification 实现方法

## 最小职责

```text
verify_one(candidate, reference, fixed_input, output_names, thresholds)
    -> "pass" or "fail"

verify_all(candidate, reference, input_cases, output_names, thresholds)
    -> {input_id: "pass" or "fail"}
```

这是职责示意，不要求使用固定的框架、函数名或 harness API。实现时复用 kernel 的 Python 接口、reference 和输入生成器。

## 确定输入与全部输出

同一个输入 case 只生成一次，再为两条执行路径复制 tensor。每次均从相同初始状态、相同 scalar 参数、相同 inout 初值开始；output buffer 的形状和 dtype 按接口分配。不能分别推进两个 RNG 来冒充共享输入。

reference 通过输出参数写回时，比较这些输出缓冲区；通过返回值输出时，按接口提取全部命名输出。inout 的最终值也是输出。KDA 当前接口写回 `out` 和 `final_state`，两者都必须通过，才能给该输入返回 pass。

只比较 reference 接口声明的输出，不额外要求导出 kernel 内部中间变量。输出必须完整检查，不能用抽样元素或单个最终输出替代全部输出。

异步执行后等待结果完成再读取或比较。对 shape/dtype 按接口核对；不通过广播、隐式 reshape 或降低比较精度掩盖错误。候选/reference 输出或误差归约出现 NaN/Inf 时，该输入 fail。

## 自定义阈值

以下仅是设置方式示例，具体值由用户或任务定义：

```python
thresholds = {
    "out": {"atol": 1e-2, "rtol": 1e-2},
    "final_state": {"scaled_max": 1e-2, "rmse_ratio": 1e-2},
}
```

每个输出都必须被一项显式规则或统一默认规则覆盖，不能把没有门限的输出当作通过。阈值应与当前版本的验证请求一起确定；用户可以调整配置后重新验证，不在看到误差后偷偷放宽门限。

设候选为 a，参考为 r，d_i=|a_i-r_i|：

| Gate | 通过条件 |
|---|---|
| atol/rtol | 每个元素满足 `d_i <= atol + rtol * abs(r_i)` |
| max_abs | `max(d_i) <= threshold` |
| scaled_max | `max(d_i) / max(max(abs(r_i)), eps_scale) <= threshold` |
| rmse_ratio | `sqrt(sum(d_i**2) / max(sum(r_i**2), eps_energy)) <= threshold` |

epsilon、比较/归约精度和 dtype 策略随配置固定。常见的 scale/energy floor 示例为 1e-12 / 1e-24，不是精度门限。所有门限必须有限且非负；零 reference 的处理不能依赖除零或 NaN 比较行为。

大输出可分块归约，但保留完整元素覆盖和全局 max/sum；不能平均各块的 ratio 代替全局指标。整数、索引或布尔输出按接口采用精确比较。

## 遍历与返回

```text
results = {}
for input_case in all_declared_input_cases:
    try:
        shared_values = materialize_once(input_case)
        expected = run_reference_with_independent_buffers(shared_values)
        actual = run_current_version_with_independent_buffers(shared_values)
        results[input_case.id] = compare_all_outputs(expected, actual, thresholds)
    except execution_or_comparison_error:
        results[input_case.id] = "fail"
        save_optional_diagnostics_under_tmp()
return results
```

compare_all_outputs 仅返回 pass/fail，且必须把所有输出的所有 gate 做逻辑 AND。不同输入的结果分别保留；某组 fail 后继续其他输入。最后核对结果 ID 集合与声明输入集合相同，不允许遗漏或空集合通过。

主要输出为 [输入 ID → pass/fail 的 JSON 映射](../templates/verification-result.json)。辅助误差或失败原因放 `tmp/`；reference 自检、性能测量和同步工具报告不属于这个返回值。
