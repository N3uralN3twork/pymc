# Temperature and weights: a guided Rust exercise

Start in [src/temperature.rs](../src/temperature.rs). Fill the three blocks
labelled `EXERCISE`, in order. Each block ends in an explicit
`ExerciseIncomplete` error; replace that block with your code. There is no
completed Rust answer hidden elsewhere in the extension.

The input is one log likelihood per SMC particle, the current inverse
temperature `old_beta`, and an ESS threshold. The output contains:

- `beta`: the next inverse temperature, capped at one.
- `weights`: one normalized importance weight per particle.
- `log_marginal_likelihood_increment`: an amount to **add** to the previous
  log marginal likelihood, not a replacement total.

The Python binding, validation, final normalization, and output ownership are
provided. This exercise is a standalone kernel: it is not connected to
`pm.sample_smc`, and no performance improvement has been established.

For this first exercise, likelihoods must be finite and the current temperature
must be below one. Real models can produce `-inf` likelihoods; their handling
is deliberately left for a later numerical-design exercise. Keep intermediate
calculations within float64 range.

## Rust concepts you will use

| Syntax | Meaning here |
| --- | --- |
| `&[f64]` | Borrow a slice of numbers without taking ownership or copying it. |
| `Vec<f64>` | Own a growable array, such as the calculated log weights. |
| `let mut low = ...` | Declare a local variable that you can reassign. |
| `.iter().map(...).collect::<Vec<f64>>()` | Transform each element into an owned vector. |
| `.sum::<f64>()`, `.exp()`, `.ln()` | Sum, exponential, and natural logarithm. |
| `Ok(value)` | Return a successful `Result`. |
| `some_function(...)?` | Extract its successful value or return its error immediately. |
| `*value` | Read a number through a reference when needed. |

Each temporary `let _ = ...` only suppresses unused-variable warnings while
the block is blank. Remove it when you implement that block.

Run the following commands from the PyMC repository root.

## 1. Stable logsumexp

Complete `logsumexp`. Directly summing `exp(1000)` overflows even though the
logarithm of the desired result is representable.

Hint: let `m` be the largest input. Compute:

```text
logsumexp(x) = m + ln(sum(exp(x_i - m)))
```

The largest exponential is then one. The helper receives nonempty finite
inputs, so you can focus on the calculation.

An iterator maximum can be expressed using `.copied().fold(...)` and
`f64::max`; `f64::NEG_INFINITY` is a useful initial value. Floating-point
numbers do not implement the total ordering required by a plain `.max()`.

Check your work:

```powershell
cargo test --manifest-path rsmc/Cargo.toml --locked exercise_1 -- --ignored
```

This checks ordinary inputs, very large positive and negative inputs, and a
single element. No Python build is needed yet.

## 2. Log weights and effective sample size

Complete `weights_at_beta`. The initial unnormalized log weights and their
normalizer are already calculated for you.

```text
u_i = (beta - old_beta) * log_likelihood_i
z = logsumexp(u)
log_weight_i = u_i - z
ESS = exp(-logsumexp(2 * log_weight))
```

Return `Ok(WeightStats { ... })` containing the normalized **log** weights,
`z`, and ESS truncated to `usize`. Preserve PyMC's integer ESS comparison;
rounding to the nearest integer would change the temperature search.

Hints: collect the normalized log weights into a `Vec<f64>`. Make another
vector containing twice those values, borrow it for `logsumexp`, and use `?`
to propagate errors. Optimize away temporary vectors only after the port is
correct.

```powershell
cargo test --manifest-path rsmc/Cargo.toml --locked exercise_2 -- --ignored
```

The checkpoint uses four unequal weights whose ESS is about 3.33. The expected
integer ESS is three.

## 3. Choose the next temperature

Complete `choose_next_beta`, using the actual
[Python reference](../../pymc/smc/kernels.py), method
`SMC_KERNEL.update_beta_and_weights`.

Use mutable lower and upper bounds, starting at `old_beta` and `2.0`.
The target is `(number_of_particles * threshold)` truncated to `usize`.
Repeatedly evaluate `weights_at_beta` at the midpoint:

- If integer ESS equals the target, stop.
- If ESS is too small, the candidate temperature is too high. Which bound
  should move to the midpoint?
- Otherwise, increase the lower bound.

Continue while the bracket width exceeds `1e-6`. Return the **last evaluated
midpoint**, capped at one. Preserve the upper bound of two during the search;
capping every trial at one changes the algorithm. The provided outer function
recomputes the weights at your final capped temperature.

```powershell
cargo test --manifest-path rsmc/Cargo.toml --locked exercise_3 -- --ignored
cargo test --manifest-path rsmc/Cargo.toml --locked temperature -- --include-ignored
```

The normal `cargo test` command leaves these learning checkpoints ignored so
the incomplete exercise does not break the existing resampling tests. Use
`--ignored` or `--include-ignored` explicitly to test your answers.

## 4. Rebuild and compare with Python

Once all three checkpoints pass, rebuild the extension:

```powershell
$rsmcPython = (Resolve-Path .venv/Scripts/python.exe).Path
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = $rsmcPython
& $rsmcPython -m maturin develop --uv --manifest-path rsmc/Cargo.toml --release --locked
& $rsmcPython -m pytest rsmc/tests -q
& $rsmcPython -m pytest rsmc/exercises/test_temperature_reference.py -q
```

The final command is an opt-in exercise suite. It **fails until the blanks are
filled**. It calls the actual Python method on controlled sampler state and
compares temperature, weights, and evidence increments. Cases include flat
likelihoods, large log likelihoods, an existing nonzero temperature, and
threshold boundaries. The ordinary `rsmc/tests` suite checks the completed
binding and validation independently.

If a reference check disagrees, inspect integer ESS and the chosen branch at
each midpoint. Floating-point reduction differences can cross an integer ESS
boundary; investigate that before changing tolerances or the algorithm.

After rebuilding, you can call the kernel with:

```python
import numpy as np
import pymc_rust

likelihoods = np.array([-10.0, -3.0, -1.0, 0.0], dtype=np.float64)
beta, weights, evidence_increment = pymc_rust.update_beta_and_weights(
    likelihoods, old_beta=0.0, threshold=0.5
)
```

While exercise 3 is blank, this call raises `NotImplementedError` with its
exercise number. Rust edits take effect in Python only after rebuilding.

Use `cargo fmt --manifest-path rsmc/Cargo.toml` to format your Rust code.
Once the reference checks pass, the next task is to profile and benchmark the
kernel through a real SMC workload before considering integration.
