"""Implement the verify mode for fixed inputs and the weak-decay case."""
import json
import re
import traceback

import torch

from ..cases.inputs import load_cases, make_inputs_from_recipe, run_case
from .. import runner
from . import PROTOCOL, copy_args, verify_candidate, weak_decay

GROUND_TRUTH = runner.BASELINES[0]


def evidence(folder, cid, tag=''):
    return folder / 'verification' / GROUND_TRUTH / f'{cid}{tag}.json'


def expected_outputs(fn, arguments, case):
    independent = copy_args(arguments)
    run_case(fn, independent, case)
    torch.cuda.synchronize()
    result = {}
    for name in ('out', 'final_state'):
        tensor = independent[name]
        if tensor.shape != arguments[name].shape or tensor.dtype != arguments[name].dtype:
            raise ValueError(f'{GROUND_TRUTH}/{name}: invalid output shape or dtype')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'{GROUND_TRUTH}/{name}: nonfinite output')
        result[name] = tensor.cpu()
    return result


def verify(args, parser, experiment):
    versions = sorted((p.name for p in experiment.iterdir()
                       if p.is_dir() and re.fullmatch(r'v[0-9]+', p.name)), key=lambda s: int(s[1:]))
    available = [*runner.BASELINES, 'reference', *versions]
    names = list(dict.fromkeys(args.impl or available))
    if set(names) - set(available):
        parser.error('Available implementations: ' + ', '.join(available))
    if args.output_dir and len(names) != 1:
        parser.error('--output-dir requires exactly one selected implementation')
    if args.case and args.cases:
        parser.error('Use --case or --cases, not both')
    recipes = load_cases(args.manifest)
    if not all(c['full_module'] for c in recipes):
        parser.error('Use the archived runner for historical manifests')
    cases = {c['id']: c for c in (*recipes, weak_decay.CASE)}
    ids = args.cases or ([args.case] if args.case and args.case != 'all' else list(cases))
    if set(ids) - cases.keys():
        parser.error('Unknown verification case; use --mode verify --list')
    if args.list or args.validate_inputs:
        if args.list:
            print(json.dumps(dict(ground_truth=GROUND_TRUTH,
                                  candidates=[n for n in names if n != GROUND_TRUTH],
                                  cases=ids), indent=2))
        if args.validate_inputs:
            from ..reference import validate_input_specs
            for cid in ids:
                run_case(validate_input_specs, make_inputs_from_recipe(cases[cid], device='meta'), cases[cid])
        return
    if torch.device(args.device).type != 'cuda':
        parser.error('KDA attention requires a CUDA device')
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    metadata = runner.identity(args.manifest, experiment=experiment)
    diagnostic = args.diagnostic_dir or experiment / 'tmp/kda/verification' / GROUND_TRUTH
    results = {name: {} for name in names}
    for cid in ids:
        case = cases[cid]
        expected = arguments = None
        truth_error = None
        try:
            arguments = weak_decay.make_inputs() if cid == weak_decay.CASE['id'] else make_inputs_from_recipe(case)
            truth, _ = runner.load(GROUND_TRUTH, experiment=experiment)
            expected = expected_outputs(truth, arguments, case)
        except Exception:
            truth_error = traceback.format_exc()
        for name in names:
            error = truth_error
            metrics = {}
            status = 'ground_truth' if name == GROUND_TRUTH else 'fail'
            if name != GROUND_TRUTH and error is None:
                try:
                    fn, _ = runner.load(name, experiment=experiment)
                    passed, metrics = verify_candidate(fn, copy_args(arguments), case, expected)
                    status = 'pass' if passed else 'fail'
                except Exception:
                    error = traceback.format_exc()
            if error:
                status = 'fail'
                runner.write(diagnostic / f'{name}-{cid}{args.tag}-exception.json', {'exception': error})
            folder = args.output_dir or (experiment / name if name in versions
                                         else experiment / 'baselines' / name)
            runner.write(evidence(folder, cid, args.tag),
                         dict(ground_truth=GROUND_TRUTH, implementation=name, case=cid, status=status,
                              metrics=metrics, thresholds=PROTOCOL['thresholds'], metadata=metadata))
            results[name][cid] = status
            print(name, cid, status, flush=True)
        del expected, arguments
    if any('fail' in states.values() for states in results.values()):
        raise RuntimeError(f'Numerical verification against {GROUND_TRUTH} failed; see per-case records')
    return results
