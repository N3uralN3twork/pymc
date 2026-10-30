"""Run or profile a shared comparison workload in fresh, controlled workers."""

# ruff: noqa: T201 -- This CLI reports status and writes benchmark artifacts.

import argparse
import cProfile
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--workload",
        choices=["resampling", "logsumexp", "weights_at_beta", "regression"],
        default="resampling",
    )
    result.add_argument(
        "--implementation", required=True, choices=["baseline", "rust", "pymc", "nutpie"]
    )
    result.add_argument("--particles", type=int, default=10000)
    result.add_argument("--distribution", choices=["uniform", "skewed"], default="uniform")
    result.add_argument("--elements", type=int, default=10000)
    result.add_argument(
        "--case", choices=["normal", "large_positive", "large_negative"], default="normal"
    )
    result.add_argument("--old-beta", type=float, default=0.25)
    result.add_argument("--beta", type=float, default=0.75)
    result.add_argument("--observations", type=int, default=200)
    result.add_argument("--draws", type=int, default=250)
    result.add_argument("--tune", type=int, default=250)
    result.add_argument("--seed", type=int, default=20261029)
    result.add_argument("--phase", choices=["cold", "warm"], default="warm")
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument(
        "--iterations", type=int, help="Calls per worker; default 10 kernels or 1 model"
    )
    result.add_argument("--repeats", type=int, default=3, help="Independent worker processes")
    result.add_argument("--profile", choices=["none", "cprofile"], default="none")
    result.add_argument(
        "--output", type=Path, help="JSON report; generated .prof files live alongside it"
    )
    result.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return result


def environment_metadata() -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    packages: dict[str, str] = {}
    for name in ("pymc", "pytensor", "numpy", "scipy", "numba", "nutpie", "asv"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
        "threadpools": threadpool_info(),
        "pytensor_flags": os.environ.get("PYTENSOR_FLAGS", ""),
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR", ""),
    }


def worker(args: argparse.Namespace) -> int:
    from benchmarks._environment import configure_threads
    from benchmarks._implementations import ImplementationUnavailable, prepare
    from threadpoolctl import threadpool_limits

    controls = configure_threads()
    try:
        started = time.perf_counter()
        workload = prepare(
            args.workload,
            args.implementation,
            particles=args.particles,
            distribution=args.distribution,
            elements=args.elements,
            case=args.case,
            old_beta=args.old_beta,
            beta=args.beta,
            observations=args.observations,
            draws=args.draws,
            tune=args.tune,
            seed=args.seed,
        )
        setup_seconds = time.perf_counter() - started
    except ImplementationUnavailable as exc:
        args.worker_output.write_text(
            json.dumps({"status": "unavailable", "reason": str(exc)}), encoding="utf-8"
        )
        return 2

    timings: list[float] = []
    profile = cProfile.Profile() if args.profile == "cprofile" else None
    with threadpool_limits(limits=1):
        # A cold measurement cannot first run a correctness/warmup call.
        if args.phase == "warm":
            for _ in range(args.warmups):
                workload.validate(workload.run())
        if profile:
            profile.enable()
        for _ in range(args.iterations):
            started = time.perf_counter()
            value = workload.run()
            timings.append(time.perf_counter() - started)
            if profile:
                profile.disable()
            workload.validate(value)
            del value
            if profile:
                profile.enable()
        if profile:
            profile.disable()
        metadata = environment_metadata()
    record = {
        "status": "ok",
        "workload": workload.metadata,
        "environment": metadata,
        "thread_controls": controls,
        "setup_seconds": setup_seconds,
        "seconds_per_call": timings,
    }
    if profile:
        profile_path = args.worker_output.with_suffix(".prof")
        profile.dump_stats(str(profile_path))
        record["profile_file"] = str(profile_path)
    args.worker_output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    allowed = {
        "resampling": {"baseline", "rust"},
        "logsumexp": {"baseline", "rust"},
        "weights_at_beta": {"baseline", "rust"},
        "regression": {"pymc", "nutpie"},
    }
    if args.implementation not in allowed[args.workload]:
        cli.error(f"{args.implementation!r} is not an implementation of {args.workload}")
    if args.workload == "weights_at_beta" and not (
        0.0 <= args.old_beta < 1.0 and args.old_beta <= args.beta <= 2.0
    ):
        cli.error("weights_at_beta requires 0 <= old-beta < 1 and old-beta <= beta <= 2")
    if args.iterations is None:
        args.iterations = 1 if args.phase == "cold" or args.workload == "regression" else 10
    if (
        min(
            args.particles,
            args.elements,
            args.observations,
            args.draws,
            args.tune,
            args.repeats,
            args.iterations,
        )
        < 1
    ):
        cli.error("sizes, iterations, repeats, draws and tune must be positive")
    if args.warmups < 1:
        cli.error("warmups must be positive")
    if args.phase == "cold" and args.iterations != 1:
        cli.error("cold measurements require --iterations 1 so each call has a fresh process")
    if args.worker_output:
        return worker(args)

    from benchmarks._environment import configure_threads

    configure_threads()
    output = (
        args.output
        or ROOT
        / "results"
        / "manual"
        / f"{args.workload}-{args.implementation}-{time.time_ns()}.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    exit_code = 0
    for index in range(args.repeats):
        sample_path = output.with_name(f"{output.stem}.{time.time_ns()}.worker-{index}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-output",
            str(sample_path),
        ]
        completed = subprocess.run(command, check=False)
        if sample_path.exists():
            samples.append(json.loads(sample_path.read_text(encoding="utf-8")))
        else:
            samples.append({"status": "failed", "returncode": completed.returncode})
        if completed.returncode:
            exit_code = completed.returncode
            break

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    values = [value for sample in samples for value in sample.get("seconds_per_call", [])]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok" if exit_code == 0 else "failed",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_head": git.stdout.strip(),
        "git_status": git_status.stdout.splitlines(),
        "phase": args.phase,
        "profile": args.profile,
        "diagnostic_only": args.profile != "none",
        "warmups_per_worker": args.warmups if args.phase == "warm" else 0,
        "samples": samples,
    }
    if values and args.profile == "none" and exit_code == 0:
        report["median_seconds_per_call"] = statistics.median(values)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output}")
    if exit_code:
        print(
            "Implementation unavailable or measurement failed; no fallback was used.",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
