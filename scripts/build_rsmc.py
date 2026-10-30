"""Build and install the release extension into this interpreter's environment."""

import os
import stat
import subprocess
import sys
import sysconfig

from pathlib import Path


def prepare_windows_installation() -> None:
    """Allow uv to remove read-only rsmc package directories during reinstall.

    This checkout has Windows/OneDrive directories with the read-only attribute,
    including the wheel's sboms directory. Leave other installed packages alone.
    """
    if sys.platform != "win32":
        return

    site_packages = Path(sysconfig.get_path("platlib"))
    package_dirs = [
        site_packages / "pymc_rust",
        *site_packages.glob("pymc_rsmc-*.dist-info"),
    ]
    for package_dir in package_dirs:
        if not package_dir.is_dir():
            continue
        for directory in [package_dir, *package_dir.rglob("*")]:
            if directory.is_dir():
                directory.chmod(directory.stat().st_mode | stat.S_IWRITE)


def main() -> int:
    prepare_windows_installation()
    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = sys.prefix
    environment["PYO3_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "develop",
            "--uv",
            "--manifest-path",
            "rsmc/Cargo.toml",
            "--release",
            "--locked",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
