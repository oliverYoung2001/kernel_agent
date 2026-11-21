"""NCU counter definitions and decoding for the profiling reports."""
import csv
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from .device import device_identity, hopper


NCU_METRICS = [
    'sm__ops_path_tensor_src_bf16_dst_fp32.sum',
    'sm__ops_path_tensor_src_fp16.sum',
    'sm__ops_path_tensor_src_tf32_dst_fp32.sum',
    'gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
    *['smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum' for op in ['fadd','fmul','ffma','hadd','hmul','hfma']],
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_write.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_gmma.sum',
]


EXTRA = ['sm__ops_path_tensor_src_fp64.sum'] + [
    f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum' for op in ['dadd','dmul','dfma']]
COUNTERS = NCU_METRICS + EXTRA
HALF_OPS = ('hadd', 'hmul', 'hfma')
HALF_SASS = {
    'HFMA2.MMA': ('FP16', 4),
    'HADD2.BF16_V2': ('BF16', 2),
}
SHARED_REASON = ('NCU exposes shared Data-Stage wavefronts, not complete useful bytes. '
                 '128 B/wavefront below is a service-capacity equivalent, including bank-conflict '
                 'service and partially filled transactions; it is not payload traffic. '
                 'A matching whole-device shared byte-bandwidth peak has not been established.')
COMPUTE_PATHS = [
    ('Tensor', 'BF16→FP32', 'sm__ops_path_tensor_src_bf16_dst_fp32.sum', 'tensor_fp16_bf16_dense_TFLOP_s'),
    ('Tensor', 'FP16 source', 'sm__ops_path_tensor_src_fp16.sum', 'tensor_fp16_bf16_dense_TFLOP_s'),
    ('Tensor', 'TF32→FP32', 'sm__ops_path_tensor_src_tf32_dst_fp32.sum', 'tensor_tf32_dense_TFLOP_s'),
    ('CUDA', 'FP32 FADD/FMUL/FFMA', None, 'cuda_fp32_TFLOP_s'),
]


DURATION = 'gpu__time_duration.sum'


def query_metrics(device, requested):
    """Ask NCU for collectable full metric names on the selected CUDA device."""
    bases = list(dict.fromkeys(key.rsplit('.', 1)[0] for key in requested))
    query = subprocess.run(
        ['ncu', '--devices', str(device['cuda_device_index']), '--query-metrics-mode', 'suffix',
         '--metrics', ','.join(bases), '--csv'], capture_output=True, text=True, check=True)
    rows = csv.DictReader(io.StringIO(query.stdout))
    available = {row['Metric name'] for row in rows if '1' in list(row.values())[4:]}
    selected = [key for key in requested if key in available]
    if DURATION not in selected:
        raise RuntimeError(f"{device['device']}: NCU did not expose {DURATION}")
    return selected, query.stdout


def counter_sum(counts, terms):
    """Keep incomplete counter groups unavailable, including across launches."""
    if any(counts.get(key) is None for key, _ in terms):
        return None
    return sum(counts[key] * factor for key, factor in terms)


def read_count(row, key):
    value = row[key].replace(',', '').strip()
    if value.lower() in ('', 'n/a', 'nan', 'not available'):
        if key == DURATION:
            raise ValueError('NCU launch duration unavailable')
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or (key == DURATION and number == 0):
        raise ValueError(f'Invalid NCU counter {key}: {value}')
    return number


def check_capture_device(row, metadata):
    # Newer NCU exports include the profiled device alongside each kernel.
    device = row.get('device__attribute_display_name')
    if device and device != metadata['device']:
        raise ValueError(f"NCU profiled {device}, expected {metadata['device']}")


def compute(counts, seconds, peaks=None):
    result = []
    for unit, precision, key, peak_key in COMPUTE_PATHS:
        peak = peaks.get(peak_key) if peaks else None
        amount = counts.get(key) if key else counter_sum(counts, [
            ('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum', factor)
            for op, factor in [('fadd', 1), ('fmul', 1), ('ffma', 2)]])
        work = amount/1e12 if amount is not None else None
        rate = work/seconds if work is not None else None
        result.append({'unit':unit,'precision':precision,'work_TFLOP':work,
                       'throughput_TFLOP_s':rate,'peak_TFLOP_s':peak,'utilization_pct':100*rate/peak if rate is not None and peak else None,
                       'counter':key or 'FADD + FMUL + 2*FFMA predicated-on thread instructions',
                       'time_s':seconds, 'scope':'executed arithmetic; excludes SFU, integer and conversion instructions'})
    return result


def memory(counts, seconds, peaks):
    result = []
    for level, prefix, load_key, store_key in [
        ('DRAM','bytes','dram__bytes_read.sum','dram__bytes_write.sum'),
        ('Shared Data-Stage capacity equivalent','wavefronts',
         'l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read.sum',
         'l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_write.sum')]:
        rd, wr = counts.get(load_key), counts.get(store_key)
        peak = peaks.get('dram_total_GB_s') if prefix == 'bytes' else None
        reason = None if peak else 'DRAM peak unavailable for this device.'
        if prefix == 'wavefronts':
            reason = SHARED_REASON
        for direction, value in [('load',rd),('store',wr),('total',counter_sum(counts, [(load_key, 1), (store_key, 1)]))]:
            traffic = None if value is None else value * (1 if prefix == 'bytes' else 128) / 1e9
            rate = traffic / seconds if traffic is not None else None
            result.append({'level':level,'direction':direction,'traffic_GB':traffic,
                           'throughput_GB_s':rate,'peak_GB_s':peak,
                           'utilization_pct':100*rate/peak if rate is not None and peak else None,
                           'wavefronts':value if prefix == 'wavefronts' else None,
                           'traffic_kind':'measured DRAM bytes' if prefix == 'bytes' else '128 B × measured wavefronts; service capacity, not payload bytes',
                           'reason':reason,
                           'counter_source':[load_key,store_key], 'time_s':seconds,
                           'peak_scope':'fraction of common read+write device bandwidth'})
    return result


def parse_capture(archive, cid, expected_manifest_sha, expected_uuid, peaks, metrics=NCU_METRICS):
    with zipfile.ZipFile(archive) as bundle:
        raw = bundle.read(cid+'/capture.csv').decode()
        info = json.loads(bundle.read(cid+'/identity.json'))
    if info['metadata']['case_manifest_sha256'] != expected_manifest_sha or info['metadata']['uuid'] != expected_uuid:
        raise ValueError(f'Capture identity differs: {archive}/{cid}')
    rows = list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])) )
    units, data = rows[0],rows[1:]
    launches = []
    for index, row in enumerate(data):
        check_capture_device(row, info['metadata'])
        counts = {key: read_count(row, key) for key in metrics}
        seconds = counts['gpu__time_duration.sum']*{'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1,'nsecond':1e-9,'usecond':1e-6,'msecond':1e-3,'second':1}[units['gpu__time_duration.sum']]
        launches.append({'launch_index':index, 'ncu_id':row['ID'], 'name':row['Kernel Name'],
                         'grid':row.get('Grid Size'), 'block':row.get('Block Size'),
                         'registers_per_thread':row.get('launch__registers_per_thread'),
                         'shared_mem_per_block':row.get('launch__shared_mem_per_block_allocated'),
                         'duration_us':seconds*1e6, 'count':1, 'counters':counts,
                         'counter_units':{key:units[key] for key in metrics},
                         'counter_normalization':'NCU --print-units base; operation counts are not multiplied by SMs or lanes',
                         'compute':compute(counts, seconds, peaks),'memory':memory(counts, seconds, peaks),
                         'time_scope':'independent NCU kernel replay; not ordinary wall-clock time'})
    if not launches:
        raise ValueError(f'Empty launch coverage: {archive}/{cid}')
    totals = {}
    for key in metrics:
        values = [launch['counters'][key] for launch in launches]
        totals[key] = sum(values) if None not in values else None
    return launches, totals, info


def half_operations(action, counts, metadata):
    """Attribute each nonzero SM90 thread-instruction bucket to its SASS variant."""
    flops = {'FP16': 0, 'BF16': 0}
    unavailable = {'FP16': None, 'BF16': None}
    executions = {op: counts.get(f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum') for op in HALF_OPS}
    if not hopper(metadata) or None in executions.values():
        return unavailable, {}
    if not any(executions.values()):
        return flops, {}

    functions = action.metric_by_name('launch__function_pcs')
    if functions is None:
        return unavailable, {}
    variants = {op: set() for op in HALF_OPS}
    for i in range(functions.num_instances()):
        pc = functions.value(i)
        if not action.sass_by_pc(pc):
            return unavailable, {}
        while instruction := action.sass_by_pc(pc):
            opcode = re.match(r'\s*(?:@!?\w+\s+)?(\w+(?:\.\w+)*)', instruction)
            if opcode:
                mnemonic = opcode[1]
                family = mnemonic[:4].lower()
                if family in variants:
                    variants[family].add(mnemonic)
            pc += 16  # Hopper SASS instructions occupy 128 bits.

    attributed = {}
    for op, count in executions.items():
        if not count:
            continue
        forms = variants[op]
        if len(forms) != 1 or not forms <= HALF_SASS.keys():
            return unavailable, {}
        mnemonic, = forms
        precision, weight = HALF_SASS[mnemonic]
        flops[precision] += count * weight
        attributed[mnemonic] = count
    return flops, attributed


def hardware(counts, sec, half_flops, peaks):
    comp=compute(counts,sec,peaks)
    for unit,prec,terms,peak in [('Tensor','FP64',[(EXTRA[0],1)],peaks.get('tensor_fp64_TFLOP_s')),
                                 ('CUDA','FP64',list(zip(EXTRA[1:],[1,1,2])),peaks.get('cuda_fp64_TFLOP_s'))]:
        operations = counter_sum(counts, terms)
        amount = operations/1e12 if operations is not None else None
        rate = amount/sec if amount is not None else None
        comp.append(dict(unit=unit,precision=prec,work_TFLOP=amount,throughput_TFLOP_s=rate,
                         peak_TFLOP_s=peak,utilization_pct=100*rate/peak if rate is not None and peak else None))
    mem=memory(counts,sec,peaks)
    half = [counts.get(f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum') for op in HALF_OPS]
    half_rows = {}
    for precision, operations in half_flops.items():
        work = operations / 1e12 if operations is not None else None
        rate = work / sec if work is not None else None
        peak = peaks.get('cuda_fp16_bf16_TFLOP_s')
        half_rows[precision] = [work, rate, peak, 100 * rate / peak if rate is not None and peak else None]
    return {'cuda_half': half_rows,
            'compute':[[c[k] for k in ('work_TFLOP','throughput_TFLOP_s','peak_TFLOP_s','utilization_pct')] for c in comp],
            'memory':[[r[k] for k in ('traffic_GB','throughput_GB_s','peak_GB_s','utilization_pct')] for r in mem],
            'half_instructions':half}


def csv_rows(file):
    text=file.read_text()
    rows=csv.DictReader(io.StringIO(text[text.index('"ID"'):]))
    return next(rows),list(rows)


def read_nvtx_capture(experiment, cid, names, peaks, *, collection='ncu'):
    d=experiment/'reports'/collection/cid
    meta=json.loads((d/'identity.json').read_bytes())
    units,rows=csv_rows(d/'counters.csv')
    _,owners=csv_rows(d/'owners.csv')
    assert len(rows)==len(owners) and rows
    groups={k:[] for k in names}
    scale={'nsecond':.001,'usecond':1.,'msecond':1000.,'ns':.001,'us':1.,'ms':1000.}[units['gpu__time_duration.sum']]
    # Library ranges may shadow the experiment name in the CLI display.
    try:
        import ncu_report
    except ImportError:
        ncu_bin = Path((experiment/'environment/report-ncu-path.txt').read_text().strip())
        sys.path.insert(0, str(ncu_bin.parent/'extras/python'))
        import ncu_report
    capture = ncu_report.load_report(str(d/'capture.ncu-rep'))
    assert capture.num_ranges()==1
    stream = capture.range_by_idx(0)
    assert stream.num_actions()==len(rows)
    metrics = [key for key in COUNTERS if key in units]
    for idx,(row, owner) in enumerate(zip(rows,owners)):
        check_capture_device(row, meta.get('metadata', meta))
        assert all(row[key] == owner[key] for key in ['ID','Process ID','Grid Size','Block Size',*metrics])
        action=stream.action_by_idx(idx)
        state=action.nvtx_state()
        ranges=[r for dom in state.domains() for r in state.domain_by_id(dom).push_pop_ranges()]
        assigned=[k for k in names if k in ranges]
        assert len(assigned)==1 and int(row['ID'])==idx
        name=assigned[0]
        counts = {k: read_count(row, k) for k in metrics}
        dur=counts['gpu__time_duration.sum']*scale
        assert math.isclose(action.metric_by_name('gpu__time_duration.sum').as_double(),dur*1000,rel_tol=1e-8)
        half_flops, half_opcodes = half_operations(action, counts, meta.get('metadata', meta))
        group=groups[name]
        group.append(dict(index=len(group)+1, id=row['ID'],name=row['Kernel Name'],count=1,nvtx=ranges,
                          grid=row['Grid Size'],block=row['Block Size'],duration_us=dur,raw=counts,
                          half_sass_threads=half_opcodes, half_flops=half_flops,
                          **hardware(counts,dur*1e-6,half_flops,peaks)))
    assert all(groups.values())
    return meta,units,groups


if __name__ == '__main__':
    device = device_identity()
    selected, _ = query_metrics(device, COUNTERS)
    print(','.join(selected))
