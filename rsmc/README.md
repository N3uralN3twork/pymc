# rsmc: Rust kernels for PyMC

This folder is a starting point for selective Rust implementations. The first
working kernel is systematic resampling, which plugs into the shared
[comparison harness](../benchmarks/README.md).

There are three names:

- `rsmc/`: the source directory and Cargo package.
- `pymc-rsmc`: the Python distribution built by Maturin.
- `pymc_rust`: the Python import used by the benchmark adapter.

The extension exposes a kernel for comparison. It does not yet replace
`pm.sample_smc` or provide a complete Rust SMC sampler.

For a guided second kernel, follow the
[temperature and weights exercise](exercises/temperature.md). Its binding is
ready, with numbered Rust checkpoints to work through. The learning checks are
run separately from the normal extension tests.

The completed `logsumexp` helper is also exposed as `pymc_rust.logsumexp(values)`
for nonempty finite contiguous float64 vectors. Compare it with SciPy using
`make bench-quick ASV_BENCH="bench_kernels.LogSumExp"` after `make rust-release`.
See the [logsumexp benchmark notes](../benchmarks/README.md#logsumexp-comparisons)
for input cases, metrics, and the standalone runner.

The second exercise is exposed as
`pymc_rust.weights_at_beta(likelihood_logp, old_beta, beta)`, returning normalized
log weights, the log normalizer, and integer ESS. After rebuilding, run
`make bench-quick ASV_BENCH="bench_kernels.WeightsAtBeta"` to compare it with the
NumPy/SciPy calculation. The [weight benchmark notes](../benchmarks/README.md#weights-at-a-trial-temperature)
describe its inputs and timing boundaries.

## 1. Check prerequisites

Run commands from the PyMC repository root, using PowerShell on Windows:

```powershell
rustc --version
cargo --version
uv --version
.venv/Scripts/python.exe --version
```

Use Rust 1.85 or newer and the existing Python 3.12-3.14 PyMC environment.
On Windows, Rust's MSVC toolchain also needs the Visual Studio C++ build tools
and Windows SDK. If the build reports a missing linker, install those components
through Visual Studio Installer. Rustup supplies `rustfmt` and `clippy`:

```powershell
rustup component add rustfmt clippy
```

On Linux or macOS, use the equivalent Python path, such as `.venv/bin/python`,
and the platform's C/C++ linker toolchain.

## 2. Install development tools and select the Python environment

```powershell
$rsmcPython = (Resolve-Path .venv/Scripts/python.exe).Path
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = $rsmcPython
uv pip install --python $rsmcPython -r rsmc/requirements-dev.txt
```

This uses the existing PyMC environment rather than creating a separate Python
environment inside this folder. The extension itself depends on NumPy; the
reference tests and comparisons also use the installed PyMC checkout.

## 3. Test Rust and build an optimized Python extension

After installing the development tools, build or rebuild and install the
release extension from the repository root with:

```powershell
make rust-release
```

This selects the repository's `.venv` automatically. Use
`make rust-release PYTHON="path/to/venv/python"` to choose another environment.
Run it again after editing Rust sources; Cargo reuses unchanged build output.
On Windows, the build helper clears read-only flags on this extension's installed
package directories so `uv` can replace them during a rebuild.

```powershell
cargo test --manifest-path rsmc/Cargo.toml --locked
& $rsmcPython -m maturin develop --uv --manifest-path rsmc/Cargo.toml --release --locked
& $rsmcPython -c "import pymc_rust; print(pymc_rust.__version__); print('debug assertions:', pymc_rust.__debug_assertions__)"
```

The last line should show `debug assertions: False` for the release build.
Maturin enables the Cargo `python` feature automatically. Plain `cargo test`
runs the independent Rust core without linking a Python interpreter.

Keep `Cargo.lock` with the source so `--locked` resolves the same dependency
versions. Rebuild after editing Rust files; an installed native extension does
not automatically change with the source. Use `--release` for comparisons:
`maturin develop` otherwise defaults to a debug build.
[The Maturin guide](https://www.maturin.rs/local_development) explains this workflow.

For a distributable wheel instead of a local development installation:

```powershell
& $rsmcPython -m maturin build --manifest-path rsmc/Cargo.toml --interpreter $rsmcPython --release --locked --out rsmc/dist
```

This builds locally without publishing. Python wheels are specific to the
interpreter and platform in this initial setup.

## 4. Check the Python boundary and PyMC reference behavior

```powershell
& $rsmcPython -m pytest rsmc/tests -q
& $rsmcPython -m pytest -c benchmarks/pytest.ini benchmarks/tests
```

Tests compare the actual native function against PyMC over several particle
counts, seeds, offsets, and cumulative-probability boundaries. They also check
input preservation, output independence, dtype/layout handling, and Python
exceptions for invalid inputs.

The public function is:

```python
import numpy as np
import pymc_rust

weights = np.array([0.25, 0.75], dtype=np.float64)
indices = pymc_rust.systematic_resampling(weights, 0.5)
assert indices.dtype == np.intp
```

Weights must be a nonempty, one-dimensional, C-contiguous float64 NumPy array
containing finite, nonnegative values with a sum within `1e-8` of one. The
offset is a supplied uniform variate in `[0, 1)`; no RNG runs inside this
kernel. Inputs are checked but never renormalized.

The resampling loop preserves PyMC's strict `position > cumulative_weight`
comparison and sequential accumulation. If floating-point cumulative mass
cannot reach a sampling position, the extension raises `ValueError` rather
than indexing past the buffer. Invalid-input exceptions are an interface choice
for this new extension, not a proposal to change PyMC's existing API.

The binding borrows inputs and holds the GIL during computation. Do not mutate
an alias of an input concurrently. Parallel execution or GIL detachment needs
an explicit ownership design and separate measurements; the initial benchmark
uses immutable inputs and one thread. Returned indices use a new Rust-owned
allocation transferred to NumPy.

## 5. Run equal Python/Rust comparisons

The module name matches the adapter default. If you previously selected another
module, set it explicitly for this shell:

```powershell
$env:PYMC_BENCH_RUST_MODULE = "pymc_rust"
uv pip install --python $rsmcPython -r benchmarks/requirements.txt
& $rsmcPython benchmarks/profile_workload.py --implementation baseline --distribution skewed --output benchmarks/results/rsmc-baseline.json
& $rsmcPython benchmarks/profile_workload.py --implementation rust --distribution skewed --output benchmarks/results/rsmc-rust.json
```

Both commands use the same fixtures, seed, validation, thread controls, and
Python entry point. Binding costs, input checks, and required output allocations
remain inside Rust's timed call. The reference implementation retains its own
normal costs. Do not compare a debug build or a Cargo-only timing against the
Python measurement.

For an ASV execution check, including timing and whole-process memory cases:

```powershell
& $rsmcPython -m asv machine --yes --config benchmarks/asv.conf.json
& $rsmcPython -m asv run --config benchmarks/asv.conf.json --python=same --bench "bench_kernels" --quick --show-stderr --dry-run
```

Both `baseline` and `rust` cases should now run instead of skipping Rust.
Quick ASV timings only verify execution. Remove `--quick` for repeated
measurements, and follow the [full protocol](../benchmarks/README.md) before
drawing speedup conclusions. A faster resampling kernel still needs an
end-to-end `pm.sample_smc` experiment to show a practical benefit.

## Files and where to work next

```text
rsmc/
  Cargo.toml             Rust package, optional bindings, release symbols
  Cargo.lock             Reproducible dependency resolution
  pyproject.toml         Maturin/Python packaging
  pymc_rust.pyi          Python type information
  src/
    lib.rs               Core module registration and optional binding feature
    resampling.rs        Pure Rust computation and Rust unit tests
    temperature.rs       Guided temperature/weights port with three blanks
    python.rs            Thin NumPy/PyO3 bindings
  tests/
    test_resampling.py   Actual extension versus PyMC reference
    test_temperature.py  Completed exercise binding and validation checks
  exercises/
    temperature.md       Step-by-step hints and checkpoint commands
    test_temperature_reference.py  Opt-in checks for your completed answers
```

For each development cycle:

1. Edit the numeric kernel in `src/resampling.rs`, keeping Python conversion in
   `src/python.rs`.
2. Run Rust checks, rebuild in release mode, and run the reference tests.
3. Run the shared comparison commands with matching settings.
4. Profile the complete SMC workflow to decide whether another kernel is useful.

```powershell
cargo fmt --manifest-path rsmc/Cargo.toml -- --check
cargo clippy --manifest-path rsmc/Cargo.toml --locked --all-targets --all-features -- -D warnings
```

To add a second kernel, add a Rust module under `src/`, expose it through
`lib.rs` and `python.rs`, and add a matching workload/adapter under
`benchmarks/benchmarks/`. Establish reference behavior and a profile before
choosing the next target. Weight updates or particle handling are candidates;
do not assume they dominate the real SMC workload.

The initial bindings use [PyO3](https://pyo3.rs/v0.30.1/module.html) and
[rust-numpy](https://docs.rs/numpy/0.30.0/numpy/). Keep their minor versions
compatible when updating dependencies, regenerate the lockfile deliberately,
and rerun both Rust and Python checks.

If Windows reports that a native DLL cannot be replaced, close Python processes
that have imported `pymc_rust` before rebuilding. Generated binaries, wheels,
and compiler output stay out of Git; source, tests, and `Cargo.lock` belong in
version control.
