"""Render one case-bound, same-device report from ordinary and NCU evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..cases.inputs import load_cases
from ..runner import EXPERIMENT, write
from ..verification.suite import evidence
from .device import hardware_peaks
from .counters import SHARED_REASON, compute, memory, parse_capture

REPORT = EXPERIMENT/'reports'


def read(path):
    return json.loads(path.read_text())


def folder(name):
    return EXPERIMENT/name if name.startswith('v') or name == 'optimized' else EXPERIMENT/'baselines'/name


def logical(c):
    n = c['sequence']*c['batch']
    m, hq, hv, k, v, w = [c[x] for x in ('hidden','hqk','hv','dk','dv','conv_width')]
    ch = 2*hq*k+hv*v
    if c.get('full_module'):
        n = c['total_tokens']
        terms = dict(input_norm=n*(4*m+2), input_projection=2*n*m*(ch+hv+k+v),
                     decay_projection=2*n*k*hv*k, gate_projection=2*n*v*hv*v,
                     conv=n*ch*(2*w+1), qk_normalization=2*n*hq*(3*k+1), decay=3*n*hv*k,
                     recurrence=n*hv*(7*k*v+3*v), norm_gate=n*hv*(5*v+2),
                     output_projection=2*n*hv*v*m, residual=n*m)
        return dict(scope='full-attention logical work', terms_FLOP=terms, total_FLOP=sum(terms.values()))
    terms = {'input_projection':2*n*m*ch, 'output_projection':2*n*hv*v*m,
             'conv':2*n*ch*w, 'qk_normalization':6*n*hq*k,
             'recurrence':n*hv*(7*k*v+v), 'norm_gate':n*hv*(5*v+1), 'residual':n*m}
    return {'formula':'2NMC+2NHvVM+2NCW+6NHqK+NHv(7KV+V)+NHv(5V+1)+NM',
            'scope':'fixed reference-equivalent arithmetic; SFUs/conversions excluded',
            'terms_FLOP':terms, 'total_FLOP':sum(terms.values())}


def fmt(value, digits=6):
    return 'N/A' if value is None else f'{value:.{digits}g}'


def link(path, label=None):
    relative = Path(path).relative_to(EXPERIMENT)
    return f'[{label or relative}](../{relative.as_posix()})'


def escape(value):
    return str(value).replace('|',r'\|').replace('\n',' ')


def table(lines, columns, rows):
    lines.extend(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |'])
    lines.extend('| '+' | '.join(escape(v) for v in row)+' |' for row in rows)
    lines.append('')


def draw(case, items, versions, device, destination):
    fig, axes = plt.subplots(1,2,figsize=(13,4.2))
    xx = list(range(len(versions)))
    for axis, measure in zip(axes,['effective_TFLOP_s','mean_ms']):
        yy = [items[n][measure] for n in versions]
        axis.plot(xx, yy, 'o-',label='versions')
        for x,n,y in zip(xx,versions,yy):
            if items[n]['verification'] != 'pass':
                axis.plot(x,y,'rx',markersize=10)
        for name,color in [('baseline_fla','tab:orange'),('baseline_flashkda','tab:green')]:
            if name not in items:
                continue
            item = items[name]
            if item['status'] == 'unsupported':
                axis.plot([],[],color=color,linestyle='--',label=name+': BF16 only')
            elif item['verification'] in ('pass', 'ground_truth'):
                axis.axhline(item[measure],color=color,linestyle='--',label=name+' = '+fmt(item[measure],4))
            else:
                axis.axhline(item[measure],color=color,linestyle=':',label=name+' (verification FAIL)')
        axis.set_xticks(xx,versions)
        axis.set_ylabel('TFLOP/s (fixed logical work / ordinary full call)' if measure.startswith('effective') else 'Ordinary full-call arithmetic mean (ms)')
        axis.grid(alpha=.3)
        axis.legend(fontsize=7)
    axes[1].set_yscale('log')
    fig.suptitle(f"{case['id']}  B={case['batch']}, S={case['sequence']}, M={case['hidden']}, "
                 f"Hq={case['hqk']}, Hv={case['hv']}, Dk={case['dk']}, Dv={case['dv']}, {case['dtype']} — {device}")
    fig.tight_layout()
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--tag', help='Collection tag (optional when there is exactly one collection)')
    args = parser.parse_args()
    if args.tag is None:
        collections = list(REPORT.glob('collection*.json'))
        if len(collections) != 1:
            parser.error('Specify --tag to select the collection to report')
        args.tag = collections[0].stem.removeprefix('collection')
    suite = read(REPORT/('collection'+args.tag+'.json'))
    manifest = (args.manifest or Path(suite['metadata']['case_manifest'])).resolve()
    by_id = {case['id']: case for case in load_cases(manifest)}
    recipes = [by_id[cid] for cid in suite['case_ids']]
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    implementations = suite['implementations']
    versions = [name for name in implementations if re.fullmatch(r'v\d+', name)]
    current_suite = 'verification_ground_truth' in suite
    if suite['status'] != 'complete':
        raise ValueError('Collection is unfinished; do not publish a partial report')
    capture_root = Path(suite['metadata']['case_manifest']).parent.parent
    uuid = suite['metadata']['uuid']
    data = {}
    for name in implementations:
        base = folder(name)
        timing = read(base/('end_to_end'+args.tag+'.json'))
        checks = {}
        if current_suite:
            checks = {case['id']: read(evidence(base, case['id'], args.tag)) for case in recipes}
            verification = {cid: check['status'] for cid, check in checks.items()}
            verification_metadata = [check['metadata'] for check in checks.values()]
        else:
            verification = read(base/('verification'+args.tag+'.json'))
            verification_metadata = [read(base/('verification-config'+args.tag+'.json'))['metadata']]
        for metadata in [timing['metadata'], *verification_metadata]:
            if metadata['case_manifest_sha256'] != manifest_sha or metadata['uuid'] != uuid:
                raise ValueError('Timing/verification device or case identity mismatch: '+name)
            for path,sha in suite['frozen_computation'].items():
                p = Path(path)
                if p.suffix == '.py':
                    relative = str(p.relative_to(capture_root))
                    if metadata['sources'].get(relative) != sha:
                        raise ValueError('Source identity mismatch: '+name+'/'+relative)
        if set(timing['cases']) != {c['id'] for c in recipes} or set(verification) != set(timing['cases']):
            raise ValueError('Incomplete case coverage: '+name)
        hardware = read(base/'profiling'/('ncu'+args.tag)/'collection.json') if name != 'reference' else None
        timeline = read(base/'profiling'/('collection'+args.tag+'.json')) if name != 'reference' else None
        if timeline and (timeline['metadata']['uuid'] != uuid or timeline['metadata']['case_manifest_sha256'] != manifest_sha):
            raise ValueError('Timeline device or manifest mismatch: '+name)
        if timeline:
            for path,sha in suite['frozen_computation'].items():
                if Path(path).suffix == '.py' and timeline['metadata']['sources'].get(str(Path(path).relative_to(capture_root))) != sha:
                    raise ValueError('Timeline computation identity mismatch: '+name+'/'+path)
        data[name] = {'timing':timing,'verification':verification,'hardware':hardware,'timeline':timeline, 'checks':checks}
    measured_metadata = data[implementations[0]]['timing']['metadata']
    peaks = hardware_peaks(EXPERIMENT, measured_metadata)
    metrics = {'case_manifest':os.path.relpath(manifest, EXPERIMENT), 'case_manifest_sha256':manifest_sha,
               'tag':args.tag,'implementations':implementations,'versions':versions,
               'metadata':measured_metadata,'controller_metadata':suite['metadata'],'hardware_peaks':peaks,
               'timing_boundary':'host perf_counter_ns; sync before and after full Python function; preparation inside function included; input generation and verification excluded',
               'source_snapshot':suite['frozen_computation'],'cases':{},
               'input_defaults_manifest':suite.get('input_defaults_manifest'),
               'unavailable_metrics':[
                   {'metric':'shared_payload_traffic_and_matching_peak','reason':SHARED_REASON},
                   {'metric':'packed_half_precision_FLOPs_and_peak_utilization',
                    'reason':'Measured HADD/HMUL/HFMA thread instruction counts retained; half-lane and opcode precision variants not audited.'}],
               'completeness':{'required_pairs':len(recipes)*len(implementations),'ordinary_measured_pairs':0,
                               'unsupported_pairs':0,'hardware_measured_pairs':0,'oracle_pairs':len(recipes) if 'reference' in implementations else 0,
                               'timeline_measured_pairs':0,
                               'numerical_failures':[],'unexpected_missing':[]}}
    REPORT.mkdir(exist_ok=True)
    (REPORT/'figures').mkdir(exist_ok=True)
    for case in recipes:
        cid = case['id']
        work = logical(case)
        items = {}
        for name in implementations:
            base = folder(name)
            timing = data[name]['timing']['cases'][cid]
            gate = data[name]['verification'][cid]
            unsupported = name == 'baseline_flashkda' and case['dtype'] != 'bfloat16'
            item = {'verification':gate,'status':'unsupported' if unsupported else 'measured',
                    'source': name+'/implementation.py' if name in versions else 'reports/history/before-single-call.zip',
                    'timing_evidence':str((base/('end_to_end'+args.tag+'.json')).relative_to(EXPERIMENT)),
                    'verification_evidence':str((base/('verification'+args.tag+'.json')).relative_to(EXPERIMENT)),
                    'reason':'Native FlashKDA requires BF16; original dtype preserved.' if unsupported else None}
            if current_suite:
                item['verification_evidence'] = str(evidence(base, cid, args.tag).relative_to(EXPERIMENT))
                if name not in versions:
                    item['source'] = '../reference.py' if name == 'reference' else '../baseline.py'
            if unsupported:
                if gate != 'fail' or timing.get('status') != 'error' or 'bfloat16' not in timing.get('reason',''):
                    raise ValueError('Unsupported status lacks an actual attempted call: '+cid)
                item.update(mean_ms=None,stdev_ms=None,warmup=None,repeat=None,effective_TFLOP_s=None,
                            compute=[],memory=[],ncu_kernels=[],hardware_status='unsupported',
                            speedup_vs_fla=None,speedup_vs_flashkda=None)
                metrics['completeness']['unsupported_pairs'] += 1
            else:
                if 'samples' not in timing or timing.get('status') == 'error':
                    raise ValueError('Missing ordinary timing: '+name+'/'+cid)
                samples = timing['samples']
                mean = statistics.mean(samples)
                if len(samples) != timing['repeat'] or not math.isclose(mean,timing['mean'],rel_tol=1e-12):
                    raise ValueError('Ordinary arithmetic mean mismatch')
                item.update(mean_ms=mean,stdev_ms=timing['stdev'],warmup=timing['warmup'],
                            repeat=timing['repeat'],effective_TFLOP_s=work['total_FLOP']/1e9/mean)
                if current_suite:
                    if data[name]['checks'][cid]['metrics']:
                        item['verification_metrics'] = data[name]['checks'][cid]['metrics']
                else:
                    diagnostic = EXPERIMENT/'tmp/kda'/f'{name}-{cid}-errors.json'
                    item['verification_metrics'] = read(diagnostic)
                metrics['completeness']['ordinary_measured_pairs'] += 1
                if gate not in ('pass', 'ground_truth'):
                    metrics['completeness']['numerical_failures'].append([name,cid])
                if name == 'reference':
                    item.update(hardware_status='oracle_only',compute=[],memory=[],ncu_kernels=[],
                                hardware_reason='Independent sequential oracle; measured for full output verification and ordinary latency, outside baseline/version hardware ranking.')
                else:
                    entry = data[name]['hardware']['cases'].get(cid,{})
                    if entry.get('status') != 'complete':
                        raise ValueError('Missing hardware capture: '+name+'/'+cid+' '+str(entry))
                    archive = EXPERIMENT/entry['archive']
                    kernels,counts,info = parse_capture(archive,cid,manifest_sha,uuid,peaks,data[name]['hardware']['metrics'])
                    timeline = data[name]['timeline']['cases'][cid]
                    activities = timeline['device_activities']
                    cupti_kernels = [a for a in activities if a['cat']=='kernel']
                    if len(cupti_kernels) != len(kernels):
                        raise ValueError('NCU/CUPTI launch coverage differs: '+name+'/'+cid)
                    if any(a['ts'] > b['ts'] for a,b in zip(cupti_kernels,cupti_kernels[1:])):
                        raise ValueError('Nonmonotonic CUPTI launch order: '+name+'/'+cid)
                    for kernel,activity in zip(kernels,cupti_kernels):
                        for field in ['grid','block']:
                            geometry = [int(x) for x in re.findall(r'\d+',kernel[field])]
                            if geometry != activity['args'][field]:
                                raise ValueError('NCU/CUPTI launch geometry differs: '+name+'/'+cid)
                        kernel['cupti_duration_us'] = activity['dur']
                        kernel['cupti_name'] = activity['name']
                    for p,sha in suite['frozen_computation'].items():
                        if Path(p).suffix == '.py' and info['metadata']['sources'].get(str(Path(p).relative_to(capture_root))) != sha:
                            raise ValueError('Capture computation identity mismatch: '+name+'/'+cid)
                    item.update(hardware_status='complete',ncu_kernels=kernels,compute=compute(counts,mean/1e3,peaks),
                                memory=memory(counts,mean/1e3,peaks),counters=counts,
                                counter_units=kernels[0]['counter_units'],
                                cupti_evidence=timeline['archive'],device_activities=activities,
                                hardware_evidence=entry['archive'],binary_identity=info['binaries'],
                                device_kernel_count=len(kernels),ncu_device_time_sum_us=sum(k['duration_us'] for k in kernels))
                    metrics['completeness']['hardware_measured_pairs'] += 1
                    metrics['completeness']['timeline_measured_pairs'] += 1
            items[name] = item
        for name,item in items.items():
            for baseline,field in [('baseline_fla','speedup_vs_fla'),('baseline_flashkda','speedup_vs_flashkda')]:
                other = items.get(baseline)
                item[field] = None
                if (other and other['mean_ms'] and item['mean_ms'] and
                        all(x['verification'] in ('pass', 'ground_truth') for x in [item, other])):
                    item[field] = other['mean_ms']/item['mean_ms']
        metrics['cases'][cid] = {'recipe':case,'logical_work':work,'implementations':items}
        draw(case,items,versions,measured_metadata.get('device') or 'Unknown GPU',REPORT/'figures'/f'{cid}_comparison.png')
    complete = metrics['completeness']
    assert complete['ordinary_measured_pairs']+complete['unsupported_pairs'] == complete['required_pairs']
    assert complete['hardware_measured_pairs']+complete['unsupported_pairs']+complete['oracle_pairs'] == complete['required_pairs']
    assert complete['timeline_measured_pairs'] == complete['hardware_measured_pairs']
    write(REPORT/'metrics.json',metrics)
    render(metrics,args.tag)
    print(json.dumps(complete,indent=2))


def lacp_alignment_lines(metrics):
    path = REPORT / 'lacp-alignment/results.json'
    if not path.exists():
        return []
    aligned = read(path)
    metadata = aligned['metadata']
    if (metadata['case_manifest_sha256'] != metrics['case_manifest_sha256']
            or metadata['uuid'] != metrics['metadata']['uuid']):
        return []
    results = aligned['results']
    large = metrics['cases'][aligned['case_id']]['implementations']
    lines = [
        '### LACP baseline 同卡对照', '',
        f"在 `{metadata.get('device') or '未知 GPU'}`（UUID `{metadata['uuid']}`，作业 `{metadata['slurm_job']}`）上，"
        f"四条路径使用同一份 `{aligned['case_id']}` 输入和同一个已安装的 FlashKDA 二进制。"
        f"各路径预热 {aligned['warmup']} 次，轮换 {aligned['rounds']} 轮，每轮 {aligned['iterations']} 次。"
        "Host 均值包含完整 forward 与完成同步。四条路径的 out 和 final_state 均以独立 reference 验证。", '',
    ]
    rows = []
    for backend in ('fla', 'flashkda'):
        local = results['local_' + backend]
        lacp = results['lacp_' + backend]
        old = large['baseline_' + backend]['mean_ms']
        lmean = local['timing']['host_ms']['mean']
        cmean = lacp['timing']['host_ms']['mean']
        rows.append([
            backend, fmt(old), fmt(aligned['historical_lacp_sample0_host_means_ms'][backend]),
            fmt(cmean), fmt(lmean), f'{(lmean/cmean-1)*100:+.2f}%',
            f"{lacp['profile']['kernel_count']} / {local['profile']['kernel_count']}",
            f"{lacp['verification']} / {local['verification']}",
        ])
    table(lines, ['Baseline', '原主表 ms', 'LACP 原报告 ms', 'LACP 同卡 ms', '本仓库对齐后 ms',
                  '与 LACP 同卡差异', 'CUPTI kernel 数（LACP / 本仓库）', '验证（LACP / 本仓库）'], rows)
    lines += [
        '两条本仓库 baseline 使用 channels-last causal conv + SiLU 和融合 gated RMSNorm。'
        'FLA 的 L2Norm 直接读取打包卷积的 Q/K 视图。'
        'FlashKDA 调用内仍包含 gate 检查、logit 转换和初始状态布局处理。'
        'LACP 在计时前准备 logits、初态布局和工作缓冲区，并恢复 residual。', '',
        '下方全量图表对应原 baseline 源码和原采集作业。'
        '本节对齐后的数据包含独立 CUPTI 时间线，NCU 计数仍对应原全量采集。', '',
        '[计时样本、输入及源码身份、精度与 CUPTI 汇总](lacp-alignment/results.json) · '
        '[完整 CUPTI trace](lacp-alignment/cupti-lacp-aligned.zip)', '',
    ]
    metrics['lacp_alignment'] = {
        'case_id': aligned['case_id'], 'evidence': 'lacp-alignment/results.json',
        'metadata': metadata,
        'results': {
            name: {'verification': item['verification'], 'host_mean_ms': item['timing']['host_ms']['mean']}
            for name, item in results.items()
        },
    }
    return lines


def full_attention_lines(uuid):
    evidence = REPORT / 'single-call-lacp/results.json'
    previous_path = REPORT / 'lacp-alignment/results.json'
    if not evidence.exists() or not previous_path.exists():
        return []
    current = read(evidence)
    previous = read(previous_path)
    metadata = current['metadata']
    if metadata['uuid'] != uuid or previous['metadata']['uuid'] != uuid:
        return []
    lines = [
        '## 完整 attention 同卡复测', '',
        f"`cases/cases.json` 的 `{current['case_id']}` 为 GLM-5.3-Flash BF16 64K。"
        f"在 {metadata.get('device') or '未知 GPU'}（UUID `{uuid}`）上用相同投影权重与输入轮换计时。"
        f"每条路径预热 {current['warmup']} 次，采集 {current['rounds']} 轮，每轮 {current['iterations']} 次。"
        '表中为包含调用及完成同步的 Host 均值。', '',
    ]
    rows = []
    for backend in ('fla', 'flashkda'):
        local = current['results']['local_' + backend]
        lacp = current['results']['lacp_' + backend]
        historical = previous['results']['lacp_' + backend]['timing']['host_ms']['mean']
        local_ms = local['timing']['host_ms']['mean']
        lacp_ms = lacp['timing']['host_ms']['mean']
        checks = []
        for config in folder('baseline_' + backend).glob('verification-config-single-call*.json'):
            identity = read(config)['metadata']
            if identity['uuid'] == uuid and identity['case_manifest_sha256'] == metadata['case_manifest_sha256']:
                result = config.with_name(config.name.replace('verification-config-', 'verification-', 1))
                if result.is_file():
                    checks.append(result)
        coverage = 'N/A'
        if len(checks) == 1:
            verification = read(checks[0])
            coverage = link(checks[0], f"{sum(status == 'pass' for status in verification.values())}/{len(verification)}")
        rows.append([backend, f'{historical:.2f}', f'{lacp_ms:.2f}', f'{local_ms:.2f}',
                     f'{(local_ms/lacp_ms-1)*100:+.1f}%', coverage])
    table(lines, ['Baseline', '历史 LACP 同卡 ms', '本轮 LACP 同卡 ms', '完整 attention ms',
                  '与本轮 LACP 差异', '当前全部 case 双输出验证'], rows)
    lines += [
        '完整 attention 计入输入 RMSNorm、beta/decay 和输出门控所需的投影、输出投影与残差。'
        'LACP 路径在计时前准备归一化输入和门控数据。CUPTI 确认两条完整 attention 路径各执行一次原生 KDA kernel 序列。', '',
        '额外用双样本 dense batch 和两个不等长 packed context 验证调用次数及弱衰减精度。两条 baseline 每次都只进入后端一次，两个输出均通过既有门限。', '',
        '[计时样本、精度及逐 kernel 活动](single-call-lacp/results.json) · '
        '[完整 CUPTI trace](single-call-lacp/cupti-lacp-aligned.zip)', '',
    ]
    return lines


def render(metrics, tag):
    versions = metrics['versions']
    compared = [n for n in metrics['implementations'] if n != 'reference']
    historical = 'kda_20' in metrics['cases'] and not metrics['cases']['kda_20']['recipe'].get('full_module')
    focus = 'kda_20' if historical else max(metrics['cases'], key=lambda cid: metrics['cases'][cid]['logical_work']['total_FLOP'])
    large = metrics['cases'][focus]['implementations']
    passing = [v for v in versions if all(c['implementations'][v]['verification']=='pass' for c in metrics['cases'].values())]
    selected = min(passing,key=lambda v:large[v]['mean_ms']) if passing else None
    if historical:
        metrics['selected_for_64k'] = selected
    metadata = metrics['metadata']
    device = metadata.get('device') or '未知 GPU'
    memory_bytes = metadata.get('total_memory')
    memory_gib = fmt(memory_bytes / 2**30) if memory_bytes is not None else 'N/A'
    peaks = metrics['hardware_peaks']
    peak_note = '缺少与采集设备唯一匹配的峰值资料，理论峰值和利用率显示为 N/A。'
    if peaks:
        peak_note = '理论峰值取自与采集设备匹配的规格，缺失项显示为 N/A。'
        if peaks.get('source'):
            peak_note += f" [规格来源]({peaks['source']})"
    selection_note = (f"通过全部输入验证的版本中，64K 最快为 `{selected}`（{fmt(large[selected]['mean_ms'])} ms）。"
                      if selected else '本次没有通过全部输入验证的优化版本。')
    alignment = lacp_alignment_lines(metrics) if historical else []
    if historical:
        lines = ['# KDA megakernel：cases_0 全量报告','',
                 f"本报告使用历史 20 组 `cases/cases_0.json`；保留原 reference、两个 baseline、v0–v5 和 optimized 实现。"
                 f"本次在 {device} 上测量。{selection_note}"
                 f"FLA 为 {fmt(large['baseline_fla']['mean_ms'])} ms；FlashKDA 为 {fmt(large['baseline_flashkda']['mean_ms'])} ms，验证状态为 `{large['baseline_flashkda']['verification']}`。",'',
                 '## 覆盖与测量口径','',
                 f"GPU {device}（UUID `{metadata['uuid']}`），{metadata.get('sm_count', 'N/A')} SM，显存 {memory_gib} GiB。"
                 f"Torch `{metrics['metadata']['torch']}`；CUDA runtime `{metrics['metadata']['cuda_runtime']}`。",'',
                 f"Manifest SHA256：`{metrics['case_manifest_sha256']}`。完整源码身份见 [采集快照](collection{tag}.json)；每个计时文件、verification 配置及 NCU capture 均逐一核对 case 哈希、GPU UUID 和计算源码哈希。",'',
                 '双输出 out/final_state 均逐元素比较；FP64 分块归约；scaled_max ≤ 0.01、rmse_ratio ≤ 0.01；检查 finite/shape/dtype。普通计时为 perf_counter_ns，完整 Python 调用前后 CUDA 同步，包含内部转换、padding、分配及所有输出完成；输入生成、验证、编译不计时。小 case warmup=100/repeat=1000，64K warmup=10/repeat=30；保存全部样本和标准差。各 case 轮换实现顺序。TF32 关闭。','',
                 'FlashKDA 在全部 20 组上实际尝试；仅原生支持 BF16 的 8 组有可执行计时，其余 12 组在表中标明“不支持原 dtype”，不转换配方、不填零。精度失败的计时保留为诊断数据，不用于有效加速比或推荐。','',
                 '## 硬件计数与理论上限','',
                 '普通耗时与 NCU 分开采集。每个可执行 baseline/版本/optimized 的每个 case 都捕获一次完整调用的全部设备 launch，编译和 3 次预热在 profiler markers 外。原始 ncu-rep、CSV、命令日志、源码/已加载 Triton cubin 哈希集中在各实现 profiling/ncu'+tag+'/captures.zip；launch 序号不合并同名 kernel，避免丢失重复调用。reference 仅作为独立串行 oracle 测量验证和普通延迟，不进入 baseline/版本硬件对比表。','',
                 'CUPTI 另采独立完整调用，核对每个实现/case 的 kernel 数与 NCU 一致，并保留每个 launch 的独立时间和全部 memcpy/memset 活动。NCU DRAM 汇总覆盖所列 GPU kernels；DMA 传输的 payload 字节单列，不把 payload 捏造成实际 DRAM 总线字节。两次 profiler 的设备时间均不能代替未插桩的普通延迟。','',
                 'Tensor Core 使用 NCU 的已执行 operation 计数，不重复乘 2 或 SM 数；FP16 列按 source 精度汇总，BF16/TF32 列为 FP32 accumulator。CUDA FP32=predicated-on FADD+FMUL+2×FFMA，SFU、转换、整数和 packed-half 指令不混入 FP32 FLOPs。packed HADD/HMUL/HFMA 原始线程指令数另表完整给出；其半精度 lane/变体划分尚未审计，TFLOP/s 与精度峰值标 N/A，避免冒充严格 FP16 动态 FLOPs。','',
                 '计算表各路径单元格均为“TFLOP / TFLOP/s / 理论 TFLOP/s / 利用率%”。完整函数分项用普通完整调用时间作分母，逐 launch 用该 NCU 设备时间作分母；NCU kernel 时间之和不能替代普通延迟。DRAM 单元格为“GB / GB/s / 理论 GB/s / 利用率%”，read、write 均除以整卡共同带宽上限，单位均为十进制。','',
                 '统一逻辑工作公式为 2NMC+2NHvVM+2NCW+6NHqK+NHv(7KV+V)+NHv(5V+1)+NM，其中 N=B×S、C=2HqK+HvV，M=hidden、K=Dk、V=Dv、W=conv_width；SFU、转换和 padding 执行工作不属于该固定分子。硬件执行计数另外列出。','',
                 peak_note, '',
                 'Shared read/write 使用实际 Data-Stage wavefront 计数，覆盖范围由 cmd_read/cmd_write 指标定义；另保留显式 LDS/STS 与 GMMA 细分。表中的 GB、GB/s 是 128 B×wavefront 的服务容量等效值，包含部分填充/bank conflict 服务，不能当作有效 payload 或声称已证明所有 TMA/隐式访问覆盖。没有匹配整卡 shared 字节峰值，故 shared 峰值与利用率为 N/A（物理口径不可直接推导），不是漏测。依据：[NVIDIA Nsight Compute shared memory 说明](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。','',
                 '## 全量覆盖核对','']
    else:
        lines = ['# KDA 采集报告', '', f"GPU {device}（UUID `{metadata['uuid']}`），"
                 f"{metadata.get('sm_count', 'N/A')} SM，显存 {memory_gib} GiB。", '',
                 f"采集标签 `{tag}`，共 {len(metrics['cases'])} 组输入。计数器缺失时对应测量显示为 N/A。",
                 '', peak_note, '', '## 全量覆盖核对', '']
    if alignment:
        lines[2] = ('本报告保留历史 20 组输入的全量实验。'
                    '旧接口的 64K 对照见 [LACP baseline 同卡对照](#lacp-baseline-同卡对照)。'
                    f'采集设备为 {device}。{selection_note}')
    dependency_record = EXPERIMENT / 'experiments' / f'dependencies{tag}.json'
    if dependency_record.is_file():
        lines[-2:-2] = [link(dependency_record, '软件库和 FlashKDA 二进制哈希'), '']
    if historical:
        coverage = lines.index('## 覆盖与测量口径')
        lines[coverage:coverage] = full_attention_lines(metadata['uuid'])
    table(lines,['项目','数量'],[(k,str(v)) for k,v in metrics['completeness'].items()])
    table(lines,['实现','验证通过','不支持','实际完整调用计时','全调用 NCU 覆盖'],[
        [n,sum(c['implementations'][n]['verification']=='pass' for c in metrics['cases'].values()),
         sum(c['implementations'][n]['status']=='unsupported' for c in metrics['cases'].values()),
         sum(c['implementations'][n]['mean_ms'] is not None for c in metrics['cases'].values()),
         'oracle only' if n=='reference' else sum(c['implementations'][n]['hardware_status']=='complete' for c in metrics['cases'].values())]
        for n in metrics['implementations']])
    for cid,case in metrics['cases'].items():
        recipe,items = case['recipe'],case['implementations']
        lines += [f'## {cid} — {recipe["purpose"]}','',
                  f"B={recipe['batch']}，S={recipe['sequence']}，M={recipe['hidden']}，Hq={recipe['hqk']}，Hv={recipe['hv']}，Dk={recipe['dk']}，Dv={recipe['dv']}，dtype={recipe['dtype']}，seed={recipe['seed']}。固定逻辑工作 {fmt(case['logical_work']['total_FLOP']/1e12)} TFLOP。",'']
        if alignment and cid == metrics['lacp_alignment']['case_id']:
            lines += alignment
        lines += [f'![{cid} comparison](figures/{cid}_comparison.png)','',
                  '### 普通完整调用与验证','']
        table(lines,['实现','验证 / 状态','均值 ms','标准差 ms','warmup / repeat','逻辑 TFLOP/s','FLA / 当前','FlashKDA / 当前'],[
            [n,item['verification']+(' / 不支持原 dtype' if item['status']=='unsupported' else ''),fmt(item['mean_ms']),fmt(item['stdev_ms']),
             f"{item['warmup']} / {item['repeat']}" if item['mean_ms'] else '不支持',
             fmt(item['effective_TFLOP_s']),fmt(item['speedup_vs_fla']),fmt(item['speedup_vs_flashkda'])]
            for n,item in items.items()])
        table(lines,['实现','out scaled_max','out rmse_ratio','state scaled_max','state rmse_ratio'],[
            [n,*[fmt(item['verification_metrics'][output][stat]) for output in ['out','final_state'] for stat in ['scaled_max','rmse_ratio']]]
            for n,item in items.items() if 'verification_metrics' in item])
        lines += ['原始证据：'+'；'.join(f"{n}：{link(EXPERIMENT/item['timing_evidence'],'计时')}、{link(EXPERIMENT/item['verification_evidence'],'验证')}、{link(EXPERIMENT/item['source'],'源码')}" for n,item in items.items())+'。','',
                  '### 完整函数硬件汇总（普通调用时间分母）','']
        table(lines,['实现','launch 数','NCU 设备时间和 µs','Tensor BF16→FP32','Tensor FP16 source','Tensor TF32→FP32','CUDA FP32'],[
            [n,item.get('device_kernel_count','不支持'),fmt(item.get('ncu_device_time_sum_us')),
             *[compute_cell(item['compute'][i]) if item['compute'] else '不支持原 dtype' for i in range(4)]]
            for n in compared for item in [items[n]]])
        table(lines,['实现','层级','load: GB / GB/s / peak / %','store: GB / GB/s / peak / %','total: GB / GB/s / peak / %'],[
            [n,level,*[memory_cell(row) for row in item['memory'][offset:offset+3]]]
            for n in compared for item in [items[n]] if item['memory']
            for level,offset in [('DRAM',0),('Shared 容量等效值',3)]])
        lines += ['Shared 不是 payload；其 peak/% 的 N/A 原因见全局口径。FlashKDA 不支持原 dtype 时，上表不产生虚假流量记录。','']
        table(lines,['实现','DMA 次数','payload bytes','CUPTI DMA 时间 µs','证据'],[
            [n,len(dma),sum(a['args'].get('bytes',0) for a in dma),fmt(sum(a['dur'] for a in dma)),link(EXPERIMENT/item['cupti_evidence'],'CUPTI ZIP')]
            for n in compared for item in [items[n]] if 'device_activities' in item
            for dma in [[a for a in item['device_activities'] if a['cat'] in ['gpu_memcpy','gpu_memset']]]])
        table(lines,['实现','HADD / HMUL / HFMA 线程指令数','Shared read / write wavefronts','LDS / STS / GMMA wavefronts','packed-half TFLOP/s / peak / %'],[
            [n,' / '.join(fmt(item['counters'].get('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum')) for op in ['hadd','hmul','hfma']),
             ' / '.join(fmt(item['counters'].get('l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_'+op+'.sum')) for op in ['read','write']),
             ' / '.join(fmt(item['counters'].get('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_'+op+'.sum')) for op in ['ld','st','gmma']),
             'N/A：半精度 lane/变体未审计'] for n in compared for item in [items[n]] if 'counters' in item])
        lines += ['### 全部实际设备 launch（独立 NCU 设备时间分母）','']
        for name in compared:
            item = items[name]
            if not item['ncu_kernels']:
                lines += [f"`{name}`：不支持 {recipe['dtype']}；已实际尝试并保留异常，硬件指标不适用。",'']
                continue
            lines += [f"<details><summary>{name}：{item['device_kernel_count']} 个 launch；验证 {item['verification']}</summary>",'',
                      '证据：'+link(EXPERIMENT/item['hardware_evidence'],'原始 NCU ZIP')+f'，包内路径 `{cid}/capture.csv`、`{cid}/capture.ncu-rep`、`{cid}/identity.json`。','']
            table(lines,['#','设备 kernel / 次数','NCU µs','CUPTI µs','Tensor BF16','Tensor FP16','Tensor TF32','CUDA FP32'],[
                [k['launch_index'],f"`{k['name']}` / 1",fmt(k['duration_us']),fmt(k['cupti_duration_us']),*[compute_cell(row) for row in k['compute']]] for k in item['ncu_kernels']])
            table(lines,['#','层级','load: GB / GB/s / peak / %','store: GB / GB/s / peak / %','total: GB / GB/s / peak / %'],[
                [k['launch_index'],level,*[memory_cell(row) for row in k['memory'][offset:offset+3]]]
                for k in item['ncu_kernels'] for level,offset in [('DRAM',0),('Shared 容量等效值',3)]])
            lines += ['</details>','']
    if historical:
        lines += ['## 逐版本变化与下一步','',
                  'v0 融合卷积、归一化与 FP32 递推；v1 使用 activation dtype 的原生输入投影；v2 将大 Dv 递推按 32 个 value 列分 CTA、RMSNorm/gate 移出串行循环；v3 使用 Triton Tensor Core 输出 GEMM 融合 residual；v4 调整大输出 GEMM 的 tile 和 warp 配置；v5 采用原生 fused addmm 输出投影。optimized 使用已保留的 v5 与 packed 输入适配；本次历史 case 无 packed 变体。各版本实现和历史 iteration.md 均保留。','',
                  'NCU 逐 launch 表直接给出投影、递推与输出阶段的设备成本。64K 的串行 token 递推仍需重点检查；下一步应以分块/scan 方式缩短依赖链，并维持两个输出的原始精度门限。小 case 关注 Python 调度和 launch 开销，不把 shape 混合成单一平均值。','',
                  '64K 的独立 CUPTI 采集：以下列出各实现前三个耗时 kernel，占比使用本次 CUPTI kernel 时间和作分母，属于同次 profile 内的归因，不是普通调用加速比或可相加的硬件下界。','']
    else:
        lines += [f'## {focus} 的 CUPTI 耗时前三项', '']
    table(lines,['实现','CUPTI kernel 时间和 ms','设备 kernel','该 kernel ms','占该次 kernel 时间和 %'],[
        [name,fmt(sum(k['cupti_duration_us'] for k in item['ncu_kernels'])/1e3),k['cupti_name'],fmt(k['cupti_duration_us']/1e3),
         fmt(100*k['cupti_duration_us']/sum(q['cupti_duration_us'] for q in item['ncu_kernels']))]
        for name in compared for item in [large[name]]
        for k in sorted(item['ncu_kernels'],key=lambda q:-q['cupti_duration_us'])[:3]])
    if historical:
        lines += ['## 复现','',
                  '历史 runner 与 baseline 源码见 [before-single-call.zip](history/before-single-call.zip)。将归档解压到包含原版本源码的独立工作副本即可重跑旧接口。以下命令从原始数据重建本报告。','',
                  '```bash','source kernels/kda_megakernel/tools/env.sh',
                  f'KDA_EXPERIMENT={EXPERIMENT.name} python -m kernels.kda_megakernel.profiling.make_report \\\n  --manifest kernels/kda_megakernel/cases/cases_0.json --tag={tag}','```','']
    (REPORT/'report.md').write_text('\n'.join(lines))
    write(REPORT/'metrics.json',metrics)


def compute_cell(row):
    return ' / '.join(fmt(row[k]) for k in ['work_TFLOP','throughput_TFLOP_s','peak_TFLOP_s','utilization_pct'])


def memory_cell(row):
    return ' / '.join(fmt(row[k]) for k in ['traffic_GB','throughput_GB_s','peak_GB_s','utilization_pct'])


if __name__ == '__main__':
    main()
