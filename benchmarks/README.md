# Fair Python and Rust comparisons

The common entry point is Python. Both implementations use the same input
factories, validation gates, timing boundaries, and thread budget. The existing
benchmark suites in `benchmarks/benchmarks.py` remain available.

## Files

| File | Purpose |
| --- | --- |
| `benchmarks/_workloads.py` | Reproducible, offline data and immutable kernel inputs |
| `benchmarks/_implementations.py` | Implementation selection and output validation |
| `benchmarks/_environment.py` | Single-thread controls and local compiler-cache defaults |
| `benchmarks/bench_kernels.py` | ASV latency, process memory, throughput, and paired speedup |
| `benchmarks/_metrics.py` | Calibrated repeated timings for scalar rate/ratio metrics |
| `benchmarks/bench_models.py` | ASV comparisons of repeated real PyMC sampling workflows |
| `profile_workload.py` | The same workloads in fresh workers, with JSON metadata and optional profiles |
| `tests/test_comparisons.py` | Harness correctness and isolation checks |

Paths in this table are relative to this directory. Commands below start at the
repository root with your PyMC environment activated.

## Install and check

Install the benchmark dependencies into the environment containing this checkout:

```powershell
uv pip install -r benchmarks/requirements.txt
```

The optional existing Rust NUTS sampler can be installed using the project's
extra; it is separate from your custom Rust kernel:

```powershell
uv pip install -e ".[nutpie]"
```

The existing ASV configuration selects Python 3.14 and Conda for managed
environments. For local work, `--python=same` uses the active interpreter and
installed checkout. ASV's managed environments need their own optional extension
installation; a wheel installed only in your active environment is not sufficient.
Keep Python, PyMC, PyTensor, numerical dependencies, and extension versions fixed
within a comparison.

```powershell
python -m asv machine --yes --config benchmarks/asv.conf.json
python -m asv check --config benchmarks/asv.conf.json --python=same
python -m asv run --config benchmarks/asv.conf.json --python=same --bench "bench_kernels" --quick --show-stderr --dry-run
```

The machine command accepts detected defaults and stores ASV machine metadata
in its per-user machine file. Review those values before publishing results.
The quick run checks execution and skips; it is not a performance result.
For actual measurements, remove `--quick`. Use `--dry-run` for a local experiment
that should not enter ASV's commit history. For cross-commit measurements, use
managed environments and install the appropriate extension build for each commit.

## Make shortcuts

From the repository root:

```powershell
make bench-machine  # One-time ASV machine setup on a new machine
make bench-quick    # Smoke check: both implementations and all kernel metrics
make bench          # Repeated latency, memory, throughput, and speedup comparisons
```

Both benchmark targets run the `baseline` and `rust` parameters of the same
`bench_kernels` suite for resampling, logsumexp, and weights_at_beta across their
input sizes and cases.
They use the existing installed environment and show results in the terminal;
`--dry-run` keeps local working-tree experiments out of ASV history. Quick-mode
timings serve as execution checks.

Make selects `.venv/Scripts/python.exe`, `.venv/bin/python`, or `python`, in that
order. Override `PYTHON` to choose another interpreter. Install ASV and the release
Rust extension in that environment first, and rebuild
[rsmc](../rsmc/README.md) after editing Rust sources.

Select just the timing benchmark or pass extra ASV run options:

```powershell
make bench ASV_BENCH="bench_kernels.Resampling.time_resampling"
make bench ASV_ARGS="--attribute repeat=3"
```

`ASV_CONFIG` defaults to `benchmarks/asv.conf.json`; `ASV_BENCH` defaults to
`bench_kernels`; `ASV_ARGS` adds options to either benchmark command.

## Kernel metrics

Both Make benchmark targets now include these tables:

| ASV measurement | Meaning | Preferred direction |
| --- | --- | --- |
| `Resampling.time_resampling` | Seconds per complete resampling call, formatted as microseconds or milliseconds | Lower |
| `Resampling.peakmem_resampling` | Peak memory of the complete process in bytes, including setup and libraries | Lower |
| `ResamplingThroughput.track_million_particles_per_second` | Millions of particles processed per second: particle count divided by median seconds per call and by 1,000,000 | Higher |
| `ResamplingSpeedup.track_rust_speedup` | Python baseline seconds per call divided by Rust seconds per call | Higher |

A throughput of `40.0` means 40 million particles per second. A speedup of
`4.0` means Rust takes one quarter of the baseline time; a value below `1.0`
means Rust is slower. The speedup table has particle-count and distribution
parameters only, since each value already compares both implementations.

Throughput and speedup use separate warmed measurements through the same
validated adapters. Each callable is calibrated independently to batches lasting
at least 10 milliseconds, then measured in six batches. Results use median time
per call. Paired speedup alternates baseline-first and Rust-first batches, giving
each implementation three runs in each position. Input generation and reference
validation stay outside timing; bindings, kernel input checks, allocation, and
result disposal remain inside the measured batch.

The two derived metrics are rounded to three significant figures for readable
console tables; the original latency measurements retain their precision. These
are ASV scalar tracking metrics, so they do not have ASV timing error bars.
Their repeated measurement helper also runs in quick mode; the quick flag still
shortens ASV's original latency measurements. Because each table is independently
measured, throughput and speedup need not be exact reciprocals or ratios of the
other printed tables. Use `make bench` for the normal repeated comparison.

To display only a new metric:

```powershell
make bench ASV_BENCH="bench_kernels.ResamplingThroughput"
make bench ASV_BENCH="bench_kernels.ResamplingSpeedup"
```

These metrics describe the resampling operation. They do not measure total SMC
throughput, effective posterior samples per second, memory bandwidth, or allocation
rate. Those require an integrated sampler workload or different instrumentation.

## Logsumexp comparisons

The `LogSumExp` entries compare `scipy.special.logsumexp`, used by PyMC's SMC
weight update, with your Rust function exposed as `pymc_rust.logsumexp`.
SciPy already uses compiled numerical operations, so this compares two real
implementations through Python entry points.

```powershell
make rust-release
make bench-quick ASV_BENCH="bench_kernels.LogSumExp"
make bench ASV_BENCH="bench_kernels.LogSumExp"
```

The selector includes latency, whole-process peak memory, throughput in millions
of input elements per second, and paired SciPy/Rust speedup. It uses 100, 10,000,
and 100,000 elements with three cases: normal values, values near +1000, and
values near -1000. Each pair receives identical immutable contiguous float64
arrays. Input generation and SciPy reference checks are outside timing; Python
dispatch, input checks, FFI, temporary allocations, and scalar return costs are
included. The Rust binding supports nonempty finite inputs for this exercise.

For a JSON timing report or Python call profile, use the same standalone runner:

```powershell
.venv/Scripts/python.exe benchmarks/profile_workload.py --workload logsumexp --implementation baseline --elements 10000 --case large_positive --output benchmarks/results/logsumexp-baseline.json
.venv/Scripts/python.exe benchmarks/profile_workload.py --workload logsumexp --implementation rust --elements 10000 --case large_positive --output benchmarks/results/logsumexp-rust.json
```

Add `--profile cprofile` for a diagnostic profile, or choose `--case normal` or
`--case large_negative`. The default `make bench` and `make bench-quick` commands
include these entries alongside resampling. This is an isolated reduction
benchmark; total SMC performance needs a separate integrated measurement.

## Weights at a trial temperature

The `WeightsAtBeta` entries compare your second exercise with the NumPy/SciPy
expressions inside `SMC_KERNEL.update_beta_and_weights`. The baseline extracts
one trial temperature's calculation; it does not time the full temperature
search. Both implementations return `(log_weights, log_normalizer, ess)`.
Validation checks both floating-point outputs and requires the integer ESS to
match exactly, outside timing.

```powershell
make rust-release
make bench-quick ASV_BENCH="bench_kernels.WeightsAtBeta"
make bench ASV_BENCH="bench_kernels.WeightsAtBeta"
```

The four metrics are latency, whole-process peak memory, throughput in millions
of elements per second, and paired baseline/Rust speedup. ASV uses 100, 10,000,
and 100,000 finite likelihoods in the same normal and +/-1000 cases as logsumexp,
with `old_beta=0.25` and trial `beta=0.75`. The input hash includes both
temperatures. Python dispatch, Rust input checks, allocations, FFI, and returning
the log-weight array and scalars are included in timing.

The standalone runner also accepts custom temperatures, including trials above
one as used inside the search:

```powershell
.venv/Scripts/python.exe benchmarks/profile_workload.py --workload weights_at_beta --implementation baseline --elements 10000 --case large_negative --old-beta 0.25 --beta 1.5 --output benchmarks/results/weights-baseline.json
.venv/Scripts/python.exe benchmarks/profile_workload.py --workload weights_at_beta --implementation rust --elements 10000 --case large_negative --old-beta 0.25 --beta 1.5 --output benchmarks/results/weights-rust.json
```

For this exercise the binding requires nonempty finite contiguous float64
likelihoods, `0 <= old_beta < 1`, and `old_beta <= beta <= 2`. Scaled likelihoods
must remain finite. These benchmarks do not require exercise 3 or measure total
SMC performance. Add `--profile cprofile` for a diagnostic Python call profile.

## Connect your Rust kernel

The [rsmc starter crate](../rsmc/README.md) now provides the first native resampling
implementation. Follow its build steps to install it in release mode, then select
its importable module name:

```powershell
$env:PYMC_BENCH_RUST_MODULE = "pymc_rust"
```

The module must export:

```python
def systematic_resampling(weights, offset):
    ...
```

The contract is:

- `weights` is a read-only, C-contiguous, normalized float64 NumPy vector.
- `offset` is a pre-generated scalar in `[0, 1)`.
- Return a newly allocated NumPy vector with shape `(len(weights),)` and dtype
  `numpy.intp`.
- Match PyMC's systematic resampling: positions are
  `(offset + arange(n)) / n`, using the first cumulative weight greater than or
  equal to each position. Do not modify or retain an unsafe reference to inputs.

The baseline calls the actual `pymc.smc.kernels.systematic_resampling` function,
supplying the same pre-generated variate through a small RNG adapter. Random-number
generation and fixture creation are excluded for both; their kernel-specific
allocations, Python dispatch, required conversions, and FFI costs remain inside
the timed call. Outputs are checked against the baseline before warm measurements
and after standalone timed calls, outside the timer.

A missing extension raises an explicit ASV setup skip. The standalone runner
writes an unavailable result and exits with status 2. An installed extension
with a broken dependency, wrong output, or invalid API fails instead of silently
falling back. Python stand-ins are used only in harness tests, never as Rust
benchmark results. Record the extension's release build settings and source
revision alongside your report; the runner also records its module path and hash.

## Measure the same workload through either adapter

Each repeat launches a fresh interpreter. With warm measurements, each worker
prepares and validates the workload, performs the same number of warmup calls,
then measures individual calls:

```powershell
python benchmarks/profile_workload.py --implementation baseline --distribution skewed --output benchmarks/results/baseline.json
python benchmarks/profile_workload.py --implementation rust --distribution skewed --output benchmarks/results/rust.json
```

Compare reports only when input hashes, workload settings, phase, dependency
versions, and resource controls agree. The report retains individual measurements
and worker identities, not just a median. Repeat paired comparisons in alternating
order to check sensitivity to machine load and CPU temperature.

To measure a first call in each fresh worker:

```powershell
python benchmarks/profile_workload.py --implementation baseline --phase cold --repeats 5 --output benchmarks/results/baseline-cold.json
python benchmarks/profile_workload.py --implementation rust --phase cold --repeats 5 --output benchmarks/results/rust-cold.json
```

Cold measurements require one call per worker. They exclude interpreter startup
and report input/model preparation separately. They are process-cold, not
necessarily compiler-cache-cold: persistent caches are reused. The default caches
live under `benchmarks/results/cache/`; explicitly supplied `PYTENSOR_FLAGS`
compiler-cache settings and `NUMBA_CACHE_DIR` are preserved and recorded. For a
cache-cold experiment, give each comparison a separate, initially empty cache
directory. Do not mix warm-cache and empty-cache results.

The model workload uses a deterministic regression with explicit sampler
selection, Numba compilation, two chains, one active core, matching seeds, draws,
tuning, initialization requests, and target acceptance:

```powershell
python benchmarks/profile_workload.py --workload regression --implementation pymc --output benchmarks/results/pymc.json
python benchmarks/profile_workload.py --workload regression --implementation nutpie --output benchmarks/results/nutpie.json
```

These measure `pm.sample`, including compilation/cache lookup, tuning, draws, and
result conversion. A warm run repeats that API call; it is not isolated sampling
of a precompiled model. Samplers retain their own adaptation/initialization
semantics, so this comparison is not a direct translation benchmark. Validate
posterior agreement and effective samples per second over adequate runs before
drawing inference-performance conclusions. The harness only gates output shape,
coordinates, and finite draws for this model.

The benchmark package configures numerical-library and Rayon thread limits before
numerical imports, including with ASV's existing-environment mode. Model calls also
set `cores=1` and `blas_cores=1`. Run competing measurements sequentially.

## Profile without using profiler timings as speed measurements

Use the same command and inputs, adding the diagnostic option:

```powershell
python benchmarks/profile_workload.py --implementation baseline --profile cprofile --output benchmarks/results/baseline-profile.json
python benchmarks/profile_workload.py --implementation rust --profile cprofile --output benchmarks/results/rust-profile.json
```

Each worker writes a `.prof` file next to its JSON artifact. Profile reports are
marked `diagnostic_only` and omit the headline median. Warmups and correctness
validation are excluded from the recorded profile. cProfile exposes Python calls
and aggregate extension-call costs; it cannot explain Rust internals. Use an
OS-appropriate native sampling profiler with optimized Rust symbols to inspect
the same driver workload, including all child workers.

ASV also provides `peakmem_*` measurements for whole-process peak resident memory,
including native allocations. They include the live setup/reference buffers and
process baseline, not just incremental kernel allocations. Use identical setup
and fresh processes; do not compare Python-only allocation tracking against
whole-process Rust memory.

Generated artifacts stay under the already ignored `benchmarks/results/`.
ASV history uses `results/asv/`, the standalone runner defaults to `results/manual/`,
and compiler caches use `results/cache/`. Keep manual reports and caches outside
`results/asv/` so ASV never tries to read another tool's JSON as benchmark history.
Optional Cargo-only microbenchmarks belong in your crate's `benches/` directory
and should use the same fixture corpus. Label them as native-kernel diagnostics;
the shared Python comparison is the user-facing performance result.

## Check changes to the harness

```powershell
uv pip install -r benchmarks/requirements-dev.txt
python -m pytest -c benchmarks/pytest.ini benchmarks/tests
ruff check benchmarks/benchmarks/_metrics.py benchmarks/benchmarks/_environment.py benchmarks/benchmarks/_implementations.py benchmarks/benchmarks/_workloads.py benchmarks/benchmarks/bench_kernels.py benchmarks/benchmarks/bench_models.py benchmarks/profile_workload.py benchmarks/tests
python -m mypy --strict --follow-imports=silent --ignore-missing-imports --exclude "benchmarks.py$" benchmarks/benchmarks benchmarks/profile_workload.py benchmarks/tests
```

Use these focused checks in CI for harness changes. Existing legacy benchmarks
are discoverable but are not automatically part of the new comparison protocol.

ASV's [benchmark guide](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)
documents parameters and setup skips. Its
[configuration reference](https://asv.readthedocs.io/en/stable/asv.conf.json.html#matrix)
documents managed-environment thread variables.
