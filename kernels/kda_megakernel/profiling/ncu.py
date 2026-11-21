"""NCU capture boundaries and native binary metadata."""
import hashlib
import sys
import torch

from ..cases.inputs import make_inputs, run_case
from ..runner import identity, load, write
from .measurement import configure_measurement


def capture_one(args, cases, experiment):
    if len(args.cases) != 1 or (len(args.impl) != 1 and not args.nvtx):
        raise ValueError('Capture requires one case. Use --nvtx for multiple implementations.')
    configure_measurement()
    cid, = args.cases
    case = cases[cid]
    data = make_inputs(cid, device='cuda', manifest_path=args.manifest)
    for name in args.impl:
        fn, _ = load(name, experiment=experiment)
        for _ in range(args.warmup if args.warmup is not None else 3):
            run_case(fn, data, case)
        torch.cuda.synchronize()
        if args.nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            torch.cuda.cudart().cudaProfilerStart()
            try:
                run_case(fn, data, case)
                torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
        finally:
            if args.nvtx:
                torch.cuda.nvtx.range_pop()
    if args.capture_info:
        write(args.capture_info, {'metadata': identity(args.manifest, experiment=experiment),
                                  'binaries': compiled_binaries()})


def compiled_binaries():
    binaries = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(('kernels.kda_megakernel', 'fla.')) or module is None:
            continue
        for symbol, value in list(vars(module).items()):
            if not type(value).__module__.startswith('triton.runtime'):
                continue
            from triton.runtime.jit import JITFunction
            while not isinstance(value, JITFunction) and hasattr(value, 'fn'):
                value = value.fn
            if not isinstance(value, JITFunction):
                continue
            for caches in value.device_caches.values():
                for compiled in caches[0].values():
                    raw = getattr(compiled, 'asm', {}).get('cubin')
                    if raw:
                        digest = hashlib.sha256(raw).hexdigest()
                        binaries[digest] = {'symbol':module_name+'.'+symbol, 'sha256':digest,
                                            'n_regs':compiled.n_regs, 'n_spills':compiled.n_spills,
                                            'metadata':str(compiled.metadata)}
    return list(binaries.values())


def native_resources(fn):
    module = sys.modules[fn.__module__]
    binaries = []
    for obj in vars(module).values():
        for cache, *_ in getattr(obj, 'device_caches', {}).values():
            for kernel in cache.values():
                binaries.append(dict(name=kernel.name, cubin_sha256=hashlib.sha256(kernel.kernel).hexdigest(),
                                     registers=kernel.n_regs, spills=kernel.n_spills,
                                     shared=kernel.metadata.shared, warps=kernel.metadata.num_warps))
    return binaries
