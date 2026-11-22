"""Node-local immutable CPU input cache outside all measured function calls."""
from __future__ import annotations

import fcntl
import gc
import hashlib
from itertools import accumulate
import json
import os
from pathlib import Path

import torch

from ..cases import inputs as recipes
from ..distributed import WEIGHT_NAMES, partition_weights, topology


def materialize(case, *, implementation, rank=None, device="cuda", group=None, cache_root=None):
    """Load bit-identical recipes, cloning every supplied tensor into owned storage."""
    if rank is None:
        rank = topology(group)[1]
    recipes.validate_recipe(case)
    source = Path(recipes.__file__)
    identity = dict(case=case, torch=torch.__version__, recipe_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    shape_helper_sha256=hashlib.sha256((source.parents[1] / 'distributed.py').read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_root or os.environ.get("KDA_REPORT_INPUT_CACHE", f"/tmp/kda-dist-report-{os.getuid()}-20261122")) / key
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    weights_path = root / 'weights.pt'
    x_path = root / f'input-rank-{rank:04d}.pt'
    with (root / 'weights.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not weights_path.exists():
            full = recipes.make_inputs_from_recipe(case, implementation='reference', device='cpu', rank=rank, group=group)
            temporary = root / f'weights.{os.getpid()}.tmp'
            torch.save({name: full[name] for name in WEIGHT_NAMES}, temporary)
            temporary.replace(weights_path)
            x_temporary = root / f'input-rank-{rank:04d}.{os.getpid()}.tmp'
            torch.save(full['inpx'], x_temporary)
            x_temporary.replace(x_path)
            (root / 'recipe.json').write_text(json.dumps(identity, indent=2) + '\n')
            del full
            gc.collect()
    with (root / f'input-rank-{rank:04d}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not x_path.exists():
            lengths = case.get('sequence_lengths', [case['sequence']] * case['batch'])
            total = sum(lengths)
            local = total // case['parallel_size']
            value = recipes._normal((total, case['hidden']), seed=case['seed'], stream=0,
                                    dtype=getattr(torch, case['dtype']), device=torch.device('cpu'),
                                    selection=(0, list(range(rank * local, (rank + 1) * local))))
            temporary = root / f'input-rank-{rank:04d}.{os.getpid()}.tmp'
            torch.save(value, temporary)
            temporary.replace(x_path)
    weights = torch.load(weights_path, weights_only=True, mmap=True)
    if implementation == 'tp':
        weights = partition_weights(weights, hq=case['hqk'], hv=case['hv'], dk=case['dk'], dv=case['dv'],
                                    rank=rank, world_size=case['parallel_size'])
    elif implementation not in ('reference', 'ulysses', 'kcp'):
        raise ValueError(implementation)
    values = torch.load(x_path, weights_only=True, mmap=True)
    x = values.to(device=device, copy=True)
    lengths = case.get('sequence_lengths', [case['sequence']] * case['batch'])
    return dict(inpx=x, out=torch.full_like(x, float('nan')),
                cu_seqlens=torch.tensor(list(accumulate(lengths, initial=0)), dtype=torch.int64, device=device),
                hq=case['hqk'], hv=case['hv'], dk=case['dk'], dv=case['dv'], group=group,
                **{name: value.to(device=device, copy=True) for name, value in weights.items()})
