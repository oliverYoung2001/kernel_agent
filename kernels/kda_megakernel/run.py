"""Command-line entry point for KDA experiments."""
import argparse
import sys
from pathlib import Path

import torch

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels.kda_megakernel.cases.inputs import MANIFEST_PATH
from kernels.kda_megakernel.runner import (
    BASELINES, EXPERIMENT, ROOT, execute, identity, load, new_experiment, prepare_cases,
    reference, write,
)
from kernels.kda_megakernel.profiling.counters import NCU_METRICS
from kernels.kda_megakernel.profiling.cupti import profile
from kernels.kda_megakernel.profiling.measurement import timing
from kernels.kda_megakernel.profiling.ncu import capture_one
from kernels.kda_megakernel.profiling.suite import collect_suite
from kernels.kda_megakernel.verification.suite import verify


def main(argv=None, *, experiment=EXPERIMENT):
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-experiment', action='store_true', help='Create the next numbered experiment and print its name')
    parser.add_argument('--impl', nargs='+', help='Select implementations; verify defaults to all candidates')
    parser.add_argument('--mode', choices=['execute', 'verify', 'timing', 'profile', 'capture', 'suite'], default='execute')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--case', help='One case ID or all; alias for --cases')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--validate-inputs', action='store_true')
    parser.add_argument('--include-large', action='store_true')
    parser.add_argument('--hardware', action='store_true', help='Collect full-call NCU counters in suite mode')
    parser.add_argument('--resume', action='store_true', help='Explicitly resume an existing suite tag')
    parser.add_argument('--nvtx', action='store_true', help='Label each implementation in NCU capture mode')
    parser.add_argument('--capture-info', type=Path)
    parser.add_argument('--manifest', type=Path, default=MANIFEST_PATH)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--tag', default='')
    parser.add_argument('--output-dir', type=Path,
                        help='Fresh output directory for a single implementation')
    parser.add_argument('--diagnostic-dir', type=Path,
                        help='Directory for per-case verification or profiling diagnostics')
    args = parser.parse_args(argv)
    if args.new_experiment:
        print(new_experiment(ROOT).name)
        return
    if not experiment.is_dir():
        parser.error(f'Experiment directory does not exist: {experiment}')
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None and not (args.list or args.validate_inputs):
        torch.cuda.set_device(device)
    if args.mode == 'verify':
        return verify(args, parser, experiment)
    cases = prepare_cases(args, parser, experiment)
    if cases is None:
        return
    if args.mode == 'execute':
        return execute(args, parser, cases, experiment)
    if args.mode == 'suite':
        return collect_suite(args, experiment)
    if args.mode in ('timing', 'profile'):
        if args.output_dir is not None and len(args.impl) != 1:
            parser.error('--output-dir requires exactly one implementation')
        args.tag = args.tag or '-full-attention'
    handlers = {'timing': timing, 'profile': profile, 'capture': capture_one}
    return handlers[args.mode](args, cases, experiment)


if __name__ == '__main__':
    main()
