"""Shared experiment selection, execution and source identity."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
import sys
from uuid import uuid4
import torch
from kernels.kda_megakernel.cases.inputs import MANIFEST_PATH, load_cases, make_inputs, run_case
from kernels.kda_megakernel.reference import reference
from .profiling.device import device_identity, hardware_peaks


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / 'profiling/protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
BASELINES = ('baseline_fla', 'baseline_flashkda')

EXPERIMENT_NAME = re.compile(r'experiment(0|[1-9][0-9]*)')


def experiment_numbers(root):
    for path in root.glob('experiment*'):
        match = EXPERIMENT_NAME.fullmatch(path.name)
        if match and path.is_dir():
            yield int(match[1])


def new_experiment(root):
    number = max(experiment_numbers(root), default=-1) + 1
    while True:
        path = root / f'experiment{number}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


experiment_name = os.getenv('KDA_EXPERIMENT', f'experiment{max(experiment_numbers(ROOT), default=0)}')
if not EXPERIMENT_NAME.fullmatch(experiment_name):
    raise ValueError('KDA_EXPERIMENT must be an experiment directory name such as experiment0')
EXPERIMENT = ROOT / experiment_name


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def identity(case_manifest=MANIFEST_PATH, *, experiment=EXPERIMENT):
    device = device_identity()
    return {**device, 'hardware_peaks': hardware_peaks(experiment, device),
            'case_manifest': str(Path(case_manifest).resolve()),
            'case_manifest_sha256': hashlib.sha256(Path(case_manifest).read_bytes()).hexdigest(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'torch_git': torch.version.git_version,
            'precision': {'matmul':torch.get_float32_matmul_precision(), 'cuda_tf32':torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32':torch.backends.cudnn.allow_tf32, 'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, 'fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction},
            'num_cpu_threads': torch.get_num_threads(), 'slurm_job': os.getenv('SLURM_JOB_ID'),
            'command': sys.argv, 'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for pattern in ('*.py', 'cases/*.py', 'verification/*.py', 'profiling/*.py', 'profiling/*.sh', f'{experiment.name}/v[0-9]*/implementation.py')
                        for p in ROOT.glob(pattern)}}


def load(name, *, experiment=EXPERIMENT):
    if name == 'reference':
        return reference, experiment / 'baselines/reference'
    if name in BASELINES:
        module = importlib.import_module('kernels.kda_megakernel.baseline')
        return getattr(module, name), experiment / 'baselines' / name
    if re.fullmatch(r'v[0-9]+', name):
        folder = experiment / name
        module = importlib.import_module(f'kernels.kda_megakernel.{experiment.name}.{name}.implementation')
        return module.implementation, folder
    raise ValueError(f'Unknown full-attention implementation: {name}')


def prepare_cases(args, parser, experiment):
    args.explicit_cases = bool(args.case or args.cases)
    args.impl = args.impl or ['reference']
    if args.case:
        if args.cases:
            parser.error('Use --case or --cases, not both')
        args.cases = None if args.case == 'all' else [args.case]
    case_specs = load_cases(args.manifest)
    case_by_id = {c['id']: c for c in case_specs}
    full_module = case_specs[0]['full_module']
    case_ids = set(case_by_id)
    if args.cases is None:
        args.cases = [c['id'] for c in case_specs]
    elif not set(args.cases) <= case_ids:
        parser.error('Requested IDs are absent from the selected manifest')
    if not full_module and (not args.list or args.validate_inputs):
        parser.error('Historical execution is archived in experiment0/reports/history/before-single-call.zip')
    if args.list or args.validate_inputs:
        if args.list:
            print(json.dumps(case_specs, indent=2))
        if args.validate_inputs:
            from .reference import validate_input_specs
            for recipe in case_specs:
                run_case(validate_input_specs, make_inputs(recipe['id'], device='meta', manifest_path=args.manifest), recipe)
            print(json.dumps({'input_specs_validated':len(case_specs),'numerical_execution':False}))
        return
    versions = {p.parent.name for p in experiment.glob('v*/implementation.py')
                if re.fullmatch(r'v[0-9]+', p.parent.name)}
    unknown = set(args.impl) - {'reference', *BASELINES, *versions}
    if unknown:
        parser.error('Unknown implementation(s) in ' + experiment.name + ': ' + ', '.join(sorted(unknown)))
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('KDA attention requires a CUDA device')
    return case_by_id


def execute(args, parser, case_by_id, experiment):
    device = torch.device(args.device)
    if not args.explicit_cases:
        print(json.dumps(list(case_by_id.values()), indent=2))
        return
    if any(case_by_id[cid].get('large',False) for cid in args.cases) and not args.include_large:
        parser.error('Large cases require --include-large')
    for cid in args.cases:
        data = make_inputs(cid, device=device, manifest_path=args.manifest)
        for name in args.impl:
            run_case(load(name, experiment=experiment)[0], data, case_by_id[cid])
            torch.cuda.synchronize(device)
            finite = all(bool(torch.isfinite(data[key]).all()) for key in ['out','final_state'])
            print(json.dumps({'case':cid,'implementation':name,'finite':finite}))
            if not finite:
                raise RuntimeError(f'{name}/{cid}: nonfinite output')
