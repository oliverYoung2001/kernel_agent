"""Collect CUPTI device activities and preserve their raw traces."""
import gzip
import json
import os
import zipfile
import torch

from ..cases.inputs import run_case
from ..runner import write
from .measurement import measure_cases


def trace_events(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as source:
        return json.load(source)['traceEvents']


def kernel_events(path):
    return [event for event in trace_events(path) if event.get('cat') == 'kernel']


def capture_trace(path, fn, data, case):
    path.parent.mkdir(exist_ok=True, parents=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        run_case(fn, data, case)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = kernel_events(path)
    if not events:
        raise RuntimeError('CUPTI returned no kernel activity')
    return events


def record_kernel_times(folder, fn, data, case, results):
    cid = case['id']
    events = capture_trace(folder/'cupti'/(cid + '.json.gz'), fn, data, case)
    results[cid] = [(event['name'], event['dur']) for event in events]
    write(folder/'kernels-us.json', results)


def profile(args, case_by_id, experiment):
    def measure(name, fn, data, case, folder, warmup, meta):
        cid = case['id']
        target = folder/'profiling/cupti'
        trace_path = target/f'{cid}{args.tag}.json'
        kernels = capture_trace(trace_path, fn, data, case)
        events = [{'launch_index': i, 'name': e['name'], 'start_us': e['ts'],
                   'duration_us': e['dur'], 'args': e.get('args', {})}
                  for i, e in enumerate(kernels)]
        write(target/f'{cid}{args.tag}-kernels.json', {'metadata': meta, 'kernels': events})
        print(name, cid, 'profile launches', len(events), flush=True)
        return {'trace': str(trace_path), 'kernel_count': len(events)}

    functions, states, meta = measure_cases(
        args, case_by_id, experiment, f'profiling/collection{args.tag}.json', measure
    )
    for name, (_, folder) in functions.items():
        target = folder/'profiling/cupti'
        archive = folder/'profiling'/('cupti'+args.tag+'.zip')
        collection = states[name]
        with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as bundle:
            for cid in args.cases:
                entry = collection[cid]
                if entry.get('status') == 'error':
                    continue
                trace_path = target/f'{cid}{args.tag}.json'
                events = trace_events(trace_path)
                activities = [{key:e.get(key) for key in ['cat','name','ts','dur','args']}
                              for e in events if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
                members = []
                for suffix in ['.json','-kernels.json']:
                    source = target/f'{cid}{args.tag}{suffix}'
                    member = cid+'/'+source.name
                    bundle.write(source,member)
                    members.append(member)
                    source.unlink()
                collection[cid] = {'archive':os.path.relpath(archive, experiment), 'members':members,
                                   'kernel_count':entry['kernel_count'],'device_activities':activities}
        write(folder/f'profiling/collection{args.tag}.json',{'metadata':meta,'cases':collection})
