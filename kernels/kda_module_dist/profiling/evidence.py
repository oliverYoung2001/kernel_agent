"""Source, device and environment identities shared by distributed collectors."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASELINES = ("tp", "ulysses", "kcp")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def command(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def identity(manifest):
    import fla
    import triton

    index = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(index)
    sources = [ROOT / "baseline.py", ROOT / "kcp.py", ROOT / "distributed.py", ROOT / "reference.py",
               ROOT / "cases/inputs.py", ROOT / "run.py", *sorted((ROOT / "profiling").glob("*.py")),
               Path(manifest), REPO / "kernels/kda_megakernel/reference.py"]
    source_hashes = {str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    return dict(
        timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        rank=dist.get_rank(), world_size=dist.get_world_size(), hostname=socket.gethostname(),
        cuda_index=index, device=prop.name, uuid=str(prop.uuid),
        total_memory=prop.total_memory, sm_count=prop.multi_processor_count,
        compute_capability=[prop.major, prop.minor], torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cuda_version_scope="torch.version.cuda build tag; not a runtime API query", nccl=list(torch.cuda.nccl.version()), triton=triton.__version__,
        python=sys.version, python_executable=sys.executable, argv=sys.argv,
        fla_path=fla.__file__, fla_revision=command(["git", "rev-parse", "HEAD"], Path(fla.__file__).parent),
        fla_status=command(["git", "status", "--short"], Path(fla.__file__).parent),
        repo_revision=command(["git", "rev-parse", "HEAD"], REPO), source_sha256=source_hashes,
        environment={key: value for key, value in os.environ.items()
                     if key.startswith(("NCCL_", "SLURM_")) or key in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "OMP_NUM_THREADS")},
        nvidia_smi=command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.memory,power.limit,power.draw,temperature.gpu", "--format=csv,noheader"]),
        topology=command(["nvidia-smi", "topo", "-m"]),
    )


def compiled_binaries():
    from triton.runtime.jit import JITFunction

    result = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(('kernels.kda_megakernel', 'fla.')):
            continue
        for symbol, value in list(vars(module).items()):
            if not type(value).__module__.startswith('triton.runtime'):
                continue
            while not isinstance(value, JITFunction) and hasattr(value, 'fn'):
                value = value.fn
            if not isinstance(value, JITFunction):
                continue
            for cache in value.device_caches.values():
                for compiled in cache[0].values():
                    raw = getattr(compiled, 'asm', {}).get('cubin')
                    if raw:
                        digest = hashlib.sha256(raw).hexdigest()
                        # n_regs/n_spills may lazily load a CUDA module and synchronize.
                        # Never initialize GPU handles while peer ranks enter a barrier.
                        result[digest] = dict(symbol=module_name + '.' + symbol,
                                              sha256=digest, metadata=str(compiled.metadata))
    return list(result.values())
