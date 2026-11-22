"""Render a CP accuracy report only from a complete, source-matched test grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .cp_alignment import NAMES, cases, upstream, write


def generate(experiment, log):
    experiment = Path(experiment).resolve()
    data = experiment / "verification/kcp_alignment"
    identity = json.loads((data / "identity.json").read_text())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=identity["fla_root"], text=True).strip() == identity["fla_revision"]
    assert subprocess.check_output(["git", "status", "--short"], cwd=identity["fla_root"], text=True) == identity["fla_status"]
    for name, digest in identity["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, f"Source changed: {name}"
    grid = cases(upstream(Path(identity["fla_root"])))
    summary = json.loads((data / "summary.json").read_text())
    assert summary["complete"] and summary["all_pass"]
    assert {case["id"] for case in grid} == {case["id"] for case in summary["cases"]}
    records = []
    ranks = []
    adapters = []
    for case in grid:
        for rank in range(case["world_size"]):
            record = json.loads((data / case["id"] / f"rank{rank}.json").read_text())
            assert record["pass"] and record["case"] == case
            assert all(item["finite"] and item["exact"] and item["max_abs"] == 0
                       for item in record["core_vs_upstream"].values())
            ranks.append(record)
            if record["module_adapter_vs_upstream"] is not None:
                assert all(item["finite"] and item["exact"] and item["max_abs"] == 0
                           for item in record["module_adapter_vs_upstream"].values())
                adapters.append(record)
            if rank == 0:
                assert all(item["finite"] and item["pass"] for item in record["vs_recurrent_oracle"].values())
                records.append(record)
    log_text = Path(log).read_text()
    assert "KDA_MODULE_DIST_CHECKS_COMPLETE world_size=2 gpu=True" in log_text
    module_checks = [json.loads(line) for line in log_text.splitlines() if line.startswith('{"case": "small_')]
    assert len(module_checks) == 12, f"Missing module checks: {len(module_checks)}"
    assert all(item["finite"] and item["scaled_max"] <= .01 and item["rmse_ratio"] <= .01 for item in module_checks)
    aggregate = {}
    for name in NAMES:
        items = [record["vs_recurrent_oracle"][name] for record in records]
        aggregate[name] = dict(max_rmse_ratio=max(item["rmse_ratio"] for item in items),
                               max_upstream_ratio=max(item["upstream_ratio"] for item in items),
                               max_abs=max(item["max_abs"] for item in items),
                               warning_only=items[0]["warning_only"],
                               above_upstream_threshold=sum(item["upstream_ratio"] >= .008 for item in items))
    report = experiment / "reports"
    job_ids = sorted({record["slurm_job_id"] for record in ranks})
    environment_files = [experiment / "verification" / f"gpu-environment-{job}.txt" for job in job_ids]
    environments = {path.name: path.read_text() for path in environment_files if path.exists()}
    write(report / "kcp_alignment_metrics.json", dict(identity=identity, cases=records,
          rank_cases=len(ranks), adapter_rank_cases=len(adapters), oracle=aggregate,
          module_checks=module_checks, gpu_environments=environments, log=str(Path(log).resolve())))
    lines = ["# KCP 与 FLA CP 精度对照", "",
             f"FLA revision：`{identity['fla_revision']}`。目标为 `tests/cp/test_kda.py` 的全部 {len(grid)} 组参数，"
             "H=12、K=V=128、BF16，包含 2/4/8 卡、contiguous/zigzag、packed 边界、pre-gated/raw gate、"
             "safe gate、V-first、关闭 recompute 和 legacy precision=True。", "",
             f"设备：{', '.join(sorted({record['device'] for record in ranks}))}；"
             f"节点：{', '.join(sorted({record['hostname'] for record in ranks}))}；Slurm job：{', '.join(job_ids)}。"
             f"PyTorch `{identity['torch']}`（CUDA build tag `{identity['cuda_build']}`）。"
             "完整驱动、Triton、NCCL 和 GPU UUID 见 `../verification/gpu-environment-<job>.txt`。", "",
             "本地 `kcp.chunk_kda/build_cp_context` 直接导出 FLA 公开函数；module 的 KCP 适配层已改成"
             "目标测试的内核内 Q/K norm、融合 gate+cumsum 和 activation dtype beta sigmoid。"
             "分区、64-token fragment chunk、仿射状态 all-gather/merge、数值类型和 autograd 均由同一上游实现提供。"
             "独立数值证据来自下述逐 token recurrence reference。", "",
             f"全部 {len(grid)} 组、{len(ranks)} 个 rank-case 通过：两入口的输出及 dq/dk/dv/dg/dbeta "
             f"共 {len(ranks)*6} 个 tensor 逐元素一致，最大绝对误差为 0。另有 {len(adapters)} 个 rank-case 的 "
             "module recurrence adapter 使用非连续 QKV 视图及原始 beta logits，与显式原生调用的输出/梯度逐元素一致。", "",
             "独立数值对照复用目标测试逐序列 FP32 naive recurrence 及反向图。output/dq/dk/dv 严格使用上游"
             " 8e-3 ratio 门限（绝对误差 ≤1e-6 也通过）；dg/dbeta 沿用上游 warning-only，超阈值数量明确列出。"
             "下面统计涵盖所有元素；FP64 RMSE 是额外统计，上游 ratio 保留其原 dtype 算术。", "",
             "65,536-token reference 用 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 卸载保存张量，"
             "计算仍使用原 GPU 算子、dtype 和反向图；序列长度、输入及门限均未改变。首次全 GPU 保存图的尝试"
             "在 H20 显存接近上限后停止推进，已终止并保留在 `../verification/kcp_alignment_gpu_graph_attempt0/`。", "",
             "| 输出 | 最大上游 ratio | 最大 FP64 RMSE ratio | 最大绝对误差 | ≥8e-3 的 case 数 | 判定 |",
             "|---|---:|---:|---:|---:|---|"]
    for name, item in aggregate.items():
        lines.append(f"| {name} | {item['max_upstream_ratio']:.8g} | {item['max_rmse_ratio']:.8g} | "
                     f"{item['max_abs']:.8g} | {item['above_upstream_threshold']} | "
                     f"{'warning-only' if item['warning_only'] else 'strict pass'} |")
    lines += ["", "逐 case 上游 ratio：", "",
              "| case | P | layout | out | dq | dk | dv | dg | dbeta |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        case = record["case"]
        values = " | ".join(f"{record['vs_recurrent_oracle'][name]['upstream_ratio']:.6g}" for name in NAMES)
        lines.append(f"| {case['id']} | {case['world_size']} | {case['layout']} | {values} |")
    lines += ["", "完整 module 另在两张 H20 上通过 12 项回归：TP/Ulysses/KCP 各覆盖 dense、packed GVA、"
              "弱衰减跨 rank state 和零输出投影残差，同时检查输入/权重不变。"
              f"最大 scaled_max={max(item['scaled_max'] for item in module_checks):.8g}，"
              f"最大 RMSE ratio={max(item['rmse_ratio'] for item in module_checks):.8g}，门限均为 0.01。", "",
              "原始逐 rank JSON 与源码身份位于 `../verification/kcp_alignment/`；完整启动/编译/回归日志："
              f"`{Path(log).resolve()}`。汇总数据为 [kcp_alignment_metrics.json](kcp_alignment_metrics.json)。"
              "这些是精度结果，不包含本次修改后的生产性能计时；experiment0 的旧性能结果保持独立。", "",
              "复现：", "", "```bash", "source kernels/kda_module_dist/tools/env.sh",
              "sbatch --wait --output=kernels/kda_module_dist/experiment1/verification/slurm-%j.log \\",
              "  kernels/kda_module_dist/verification/run_cp_alignment.sh",
              "python -m kernels.kda_module_dist.verification.report_cp_alignment \\",
              f"  --experiment kernels/kda_module_dist/experiment1 --log {Path(log).resolve()}", "```", ""]
    report.mkdir(parents=True, exist_ok=True)
    (report / "kcp_alignment.md").write_text("\n".join(lines))
    print(report / "kcp_alignment.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    generate(args.experiment, args.log)


if __name__ == "__main__":
    main()
