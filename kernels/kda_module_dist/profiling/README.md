# Distributed baseline evidence

All modes enter through `python -m kernels.kda_module_dist.run`. Use the original
case ID and launch exactly `parallel_size` ranks. Production cases require
`--include-large`. The public baseline call still writes only `out`.

- `--mode suite --experiment <path> --warmup 10 --repeat 30 --resume` verifies
  against a fresh TP execution, times ordinary complete calls, then captures a
  separate full call with CUPTI on every rank. Each wall-clock sample is the
  maximum rank duration; the reported mean is the arithmetic sample mean.
- `--mode profile` captures a separate CUPTI call. Raw compressed Chrome traces
  retain memory activities as well as actual kernel launches.
- `--mode capture` provides the `cudaProfilerStart/Stop` boundary for NCU after
  warmup. It adds CPU Gloo barriers before collectives and device completion synchronization only in this profiling
  mode to drain excluded communication before per-kernel replay. Replay times
  are separate observations and must not replace ordinary distributed latency.

`inputs.py` caches immutable CPU recipe tensors in node-local `/tmp` to avoid
repeated large deterministic generation. The cache key includes the entire
recipe, Torch version, and generator/shape-helper source hashes. Every candidate
receives independent GPU storage and a fresh NaN output. Input preparation and
validation are outside timing. `evidence.py` records source and GPU identities;
compiled Triton CUBIN hashes use existing compiler metadata without lazily
initializing additional CUDA module handles.

Experiment-specific orchestration, counter selection, parsing, plotting and
report checks are in `experiment0/reports/`. Raw baseline evidence belongs in
`experiment0/baselines/{tp,ulysses,kcp}/`. No baseline is assigned a version
number. NCU is optional when driver counter permissions are unavailable; retain
the error log and explicit missing values instead of estimating executed work
or transaction bytes from source or activity duration.

A completed `--resume` record with a different collector/source identity is
preserved and rejected. Choose a fresh `--experiment` directory for a changed
implementation or collector instead of replacing existing ordinary evidence.
