# PyMC to Rust: selective computation port timeline

Prepared: 29 October 2026. Status: planning proposal; no performance measurements have been run for this document.

**Goal:** accelerate measured CPU hotspots with Rust while preserving Python model construction, sampling entry points, and inference results. Budget **16 weeks for a first release containing one selected computation path**, assuming one full-time engineer comfortable with Python and Rust, plus part-time numerical and PyMC review. This is an estimate, not a commitment to port every expensive operation. Week 1 starts when implementation begins; review and release scheduling can extend calendar time.

## Start from the existing computation stack

PyMC already separates model construction from numerical execution. The local [architecture guide](ARCHITECTURE.md) assigns automatic differentiation and tensor execution to PyTensor, and diagnostics and inference-data handling to ArviZ. The port should preserve those boundaries.

Rust sampling is also already available: [nutpie](https://github.com/pymc-devs/nutpie) wraps the Rust [nuts-rs](https://github.com/pymc-devs/nuts-rs) NUTS implementation. Its [PyMC integration](https://pymc-devs.github.io/nutpie/pymc-usage.html) provides an existing route from Python models to native sampling. Benchmark that route before developing another sampler.

The [PyMC 6 release announcement](https://www.pymc.io/blog/pymc_v6_ecosystem_updates.html) describes Numba as the default computation backend. The inspected local [sampler dispatcher](pymc/sampling/mcmc.py) also selects nutpie automatically when its eligibility conditions hold. Consequently, a Python-versus-Rust comparison must explicitly select its sampler and backend: the current default may already run Rust sampling and compiled model evaluations.

Repository basis: local HEAD `31aed52d360b0d415065b472829111f3b799cd95` and the current working tree. Reconfirm versions and backend behavior at kickoff; upstream and local changes can alter the opportunity.

## What to port

These are candidates to profile, not confirmed bottlenecks. Select one at the end of week 2.

| Area | Initial decision | Code to inspect | Reason and dependency |
| --- | --- | --- | --- |
| NUTS tree building, leapfrog integration, and adaptation | Reuse nutpie/nuts-rs first | [NUTS](pymc/step_methods/hmc/nuts.py), [integration](pymc/step_methods/hmc/integration.py), [mass matrix](pymc/step_methods/hmc/quadpotential.py) | New work is justified only by a measured gap in existing Rust support. A Python callback for every gradient can erase the benefit of a native sampling loop. |
| Sequential Monte Carlo (SMC) particle operations | First new-kernel candidate | [SMC kernels](pymc/smc/kernels.py), especially `update_beta_and_weights`, `systematic_resampling`, and mutation methods | Investigate repeated reductions, particle traversal, allocation, and per-particle calls. Existing NumPy/SciPy work is already native; gains must come from removing overhead, fusing work, or improving memory use. |
| Specialized likelihood or recurrence kernels | Conditional alternative to SMC | [Distribution mathematics](pymc/distributions/dist_math.py), [model evaluation](pymc/model/core.py) | Consider a coarse fused operation only if compiled graph profiling demonstrates a gap. Keep differentiation and compilation integrated with PyTensor. |
| ODE-based models | Later, separate investigation | [ODE operation](pymc/ode/ode.py) | Solver accuracy, stiffness, sensitivity equations, and native evaluation of the right-hand side make this a larger project. Existing solvers and alternatives need comparison before replacement. |
| Dense linear algebra and elementary distribution formulas | Keep existing implementations initially | PyTensor and its numerical dependencies | Expensive execution does not imply Python overhead. Replacing optimized compiled routines with Rust alone provides no evidence of improvement. |
| Modeling API, symbolic log-probability construction, plotting, and result containers | Keep in Python and existing dependencies | [Model](pymc/model/core.py), [log-probability graphs](pymc/logprob), [sampling](pymc/sampling) | Preserve the user-facing workflow and focus engineering effort on runtime computation. |

Variational inference and posterior prediction can later reuse a proven kernel when profiling shows they benefit. They are not separate rewrites in the first release.

## Timeline for the first release

The table is the critical path for **one** target. SMC, a specialized likelihood, and a sampler extension are alternatives, not simultaneous deliverables.

| Weeks | Phase | Work | Deliverable and completion gate |
| --- | --- | --- | --- |
| 1-2 | Establish the baseline and choose a target | Pin dependencies and hardware; profile representative supported models through normal PyMC compilation and execution. Separate compilation, warmup, sampling, and output conversion. Compare explicit native-PyMC and nutpie configurations; include relevant existing Numba/C/JAX alternatives. | Reproducible baseline, runtime breakdown, and one selected hotspot with a plausible end-to-end benefit. If existing backends solve the problem, finish with configuration guidance instead of adding code. |
| 3-4 | Prove the integration boundary | Create an optional Rust extension prototype with PyO3, rust-numpy, and maturin. Run one real batched operation through PyMC. Prototype any required PyTensor/backend integration now. Specify array ownership, dtype, shapes, error behavior, and RNG responsibilities. | A release-mode wheel, a working Python-to-Rust-to-result path, and measurements of crossing and copying costs. Stop or redesign if the intended backend needs slow Python callbacks in its hot loop. |
| 5-7 | Implement the selected computation | Port the agreed bounded kernel or loop. Preserve mathematical semantics and use established numerical routines where appropriate. Add focused reference comparisons while implementing. Start single-threaded; add parallelism only when measurements justify it. | A correct Rust implementation with kernel benchmarks, including FFI and allocation costs, and an initial end-to-end comparison. No additional targets enter this phase. |
| 8-10 | Integrate with the supported modeling workflow | Connect to the existing sampler/kernel or PyTensor operation. Implement required shape inference, derivative support, and backend lowering. Preserve the original path for unsupported models and unavailable extensions. Exercise data updates, multiprocessing, and result conversion. | A user model runs through the ordinary PyMC entry point with equivalent output structure and documented support limits. For a differentiated operation, compiled model logp and gradients agree with the reference. |
| 11-12 | Validate numerical and practical benefit | Compare representative models over multiple seeds and problem sizes. Check numerical ranges, memory use, compilation overhead, and sampler quality using the criteria below. Investigate regressions before adding optimizations. | A benchmark and correctness report that meets the agreed acceptance criteria. Revert or defer a target whose maintenance cost exceeds its measured benefit. |
| 13-14 | Package and test distribution | Build wheels for agreed Windows, Linux, and macOS configurations. Test fresh installation, supported Python/NumPy combinations, worker startup, and behavior without the extension. Establish the release-mode build and CI process. | Installable artifacts and a tested compatibility matrix. End users on supported wheel platforms can use the extension without installing Rust. |
| 15-16 | Pilot and prepare release | Run the target workload in a realistic environment; document installation, selection, support limits, reproducibility, and benchmark methodology. Review the final contribution at the correct repository layer. | A reviewed first-release candidate with a measured benefit, an explicit way to use the existing path, and a documented maintenance owner. Publishing and upstream acceptance have their own schedules. |

**Milestones:** week 2: target selected; week 4: integration risk resolved; week 7: kernel complete; week 10: PyMC integration complete; week 12: benefit demonstrated; week 16: release candidate ready.

Testing starts with the prototype and continues throughout. Weeks 11-12 consolidate evidence rather than defer correctness work until the end.

## Implementation boundaries

A proposed structure is a small optional extension with a Python adapter and an independent Rust computation core:

```text
Python: model API, graph construction, orchestration, result packaging
    |
    +-- existing PyTensor compiler and numerical backends
    |       +-- selected Rust Op only when differentiation/lowering is supported
    |
    +-- selected sampler or SMC adapter
            +-- batched Rust computation

Existing nutpie/nuts-rs remains the first choice for supported Rust NUTS sampling.
```

Use [PyO3](https://pyo3.rs/main/performance.html) for the Python interface and [rust-numpy](https://github.com/PyO3/rust-numpy) for array interoperability. Cross the boundary once per useful batch or substantial operation. Avoid converting arrays into Python lists and avoid per-element Python callbacks. Detach from the Python interpreter only while Rust operates on data whose ownership and synchronization make that safe; retaining a NumPy owner alone does not prevent concurrent mutation.

Define supported layouts and dtypes explicitly. Borrow input buffers only with a sound lifetime and mutation contract; otherwise make and measure an owned copy. Reuse allocations where practical, return clear Python errors, and keep panics from escaping a native ABI boundary. For random operations, decide whether Python supplies random variates or Rust owns reproducible per-chain streams before porting the loop.

For a new graph operation, implement the shape and differentiation contract described in [PyTensor's Op guide](https://pytensor.readthedocs.io/en/latest/extending/creating_an_op.html). A Python `perform` method that calls a Rust extension is only a prototype: it does not automatically enable native Numba execution or JAX compilation. [Backend-specific lowering](https://pytensor.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html) must be designed and tested. If needed, prototype a native ABI bridge for Numba in weeks 3-4 and account for library lifetimes and process startup. Preserve existing backend expressions when Rust lowering is unavailable.

Keep shared tensor/compiler changes in PyTensor and sampler changes in nutpie/nuts-rs when those projects own the behavior. A separate optional extension is the proposed starting point, avoiding an immediate change to PyMC's existing build system. [Maturin](https://www.maturin.rs/) supplies the extension build and wheel workflow; packaging design still needs to fit the selected host project.

## How to decide whether the port succeeds

Agree on the following criteria during weeks 1-2. The numerical thresholds below are proposed project targets, not measured results or universal PyMC contribution requirements.

- **Realistic benchmark set:** include a small overhead-sensitive model, a hierarchical regression, and a model dominated by the selected target. If targeting SMC, include a realistic multimodal model and vary particle count and dimensionality. Reuse useful cases from the existing [benchmark suite](benchmarks/benchmarks/benchmarks.py).
- **Fair timing:** record dependency versions, hardware, BLAS implementation/thread count, chains, seeds, data sizes, and model configuration. Measure cold start and compilation separately from warmed execution; also report total user-visible time. Use repeated measurements with uncertainty, not one favorable run. Build Rust in release mode.
- **Meaningful benefit:** aim for at least 1.25x faster total execution on the chosen target workloads, including relevant setup and output costs, with no unexplained regression greater than 5% on the agreed comparison set. Set a memory budget as well. A fast microbenchmark alone is insufficient; revise or reject the implementation if these targets are not practical.
- **Numerical correctness:** compare with the reference over realistic parameter ranges, tails, broadcasting, supported layouts, and constrained variables. Choose tolerances by operation and dtype. Check support boundaries and numerical stability without changing the intended model. Validate through normal compilation, transforms, Jacobians, and pre-gradient rewrites; raw symbolic differentiation alone does not establish PyMC behavior.
- **Inference correctness:** for MCMC, compare posterior summaries relative to Monte Carlo error, effective samples per second, R-hat, divergences, and adaptation behavior over multiple seeds. Do not require identical draws from different RNGs or sampler implementations. For SMC, compare posterior estimates, weight/temperature behavior, resampling properties, and marginal-likelihood estimates where available; MCMC diagnostics alone are insufficient.
- **Compatibility:** preserve variable names, dimensions, coordinates, sampler statistics, and the return-container contract of the pinned PyMC/ArviZ versions. Exercise unsupported models and extension absence through the existing path. Test concurrent workers with independent RNG streams and bounded numerical-library threads.

Use the measured runtime fraction to estimate the ceiling before porting. If a kernel occupies fraction `p` of runtime and Rust accelerates it by `s`, the ideal total speedup is:

```text
speedup = 1 / ((1 - p) + p / s)
```

For example, accelerating 30% of runtime by 5x yields only about 1.32x overall before additional integration costs. This is why the timeline begins with profiling, and why an already compiled matrix operation is not automatically a good port target.

## Expansion after the first release

Re-profile after shipping: the next bottleneck may differ. Budget an additional **4-8 engineering weeks per comparable, independent hotspot**, repeating the integration and acceptance gates. Treat an ODE solver replacement as a separate **8-16+ week feasibility and numerical-validation project**, subject to its requirements. These are rough planning allowances, not cumulative promises.

A second release may extend the first kernel to more shapes or workflows rather than introduce another Rust subsystem. A complete PyTensor compiler, autodiff engine, GPU backend, or standalone Rust modeling API would require a separate plan and is outside this selective port.
