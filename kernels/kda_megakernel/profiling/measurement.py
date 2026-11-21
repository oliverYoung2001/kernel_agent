"""Warm up and measure implementations on the same materialized inputs."""
import gc
import json
import statistics
import time
import traceback

import torch

from ..cases.inputs import make_inputs, run_case
from ..runner import PROTOCOL, identity, load, write


def configure_measurement():
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def measure_cases(args, case_by_id, experiment, result_path, measure):
    """Warm the selected implementations and record one result per implementation and case."""
    diagnostic = args.diagnostic_dir or experiment/'tmp/kda'
    configure_measurement()
    meta = identity(args.manifest, experiment=experiment)
    write(experiment / 'tmp/kda' / 'current-run.json', meta)
    states = {}
    functions = {}
    for name in args.impl:
        fn, folder = load(name, experiment=experiment)
        if args.output_dir is not None:
            folder = args.output_dir
        functions[name] = (fn, folder)
    for name, (_, folder) in functions.items():
        path = folder / result_path
        previous = json.loads(path.read_text()) if path.exists() else {}
        states[name] = previous.get('cases', {})
    for cid in args.cases:
        print(f'{args.mode} {cid}', flush=True)
        case = case_by_id[cid]
        a = make_inputs(cid, device='cuda', manifest_path=args.manifest)
        # Rotate implementation order per case; paired confirmation uses a separate driver.
        shift = int(cid[-2:]) % len(args.impl)
        order = args.impl[shift:] + args.impl[:shift]
        for name in order:
            fn, folder = functions[name]
            try:
                large = case_by_id[cid].get('large', False)
                warmup = args.warmup if args.warmup is not None else PROTOCOL['warmup_large' if large else 'warmup_small']
                for _ in range(warmup):
                    run_case(fn, a, case)
                torch.cuda.synchronize()
                states[name][cid] = measure(name, fn, a, case, folder, warmup, meta)
            except Exception:
                message = traceback.format_exc()
                print(name, cid, message, flush=True)
                write(diagnostic / f'{name}-{cid}-{args.mode}-exception.json', {'exception': message})
                states[name][cid] = {'status': 'error', 'reason': message}
            write(folder / result_path,
                  {'metadata': meta, 'cases': states[name]})
        del a
        gc.collect()
        torch.cuda.empty_cache()
    return functions, states, meta


def timing(args, case_by_id, experiment):
    def measure(name, fn, a, case, folder, warmup, meta):
        large = case.get('large', False)
        repeat = args.repeat if args.repeat is not None else PROTOCOL['repeat_large' if large else 'repeat_small']
        samples = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            run_case(fn, a, case)
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - begin) / 1e6)
        result = {'warmup': warmup, 'repeat': repeat, 'unit': 'ms',
                  'samples': samples, 'mean': statistics.mean(samples),
                  'stdev': statistics.stdev(samples) if repeat > 1 else 0,
                  'median': statistics.median(samples)}
        print(name, case['id'], result['mean'], 'ms', flush=True)
        return result

    measure_cases(args, case_by_id, experiment, f'end_to_end{args.tag}.json', measure)
