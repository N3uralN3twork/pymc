"""Select one or more test areas and run pytest with the current interpreter."""

# ruff: noqa: T201 -- This command-line picker writes its menu to stdout.

import subprocess
import sys

from pathlib import Path


def select_paths(areas: dict[str, list[str]]) -> list[str] | None:
    """Prompt for area names or numbers; return None when the user quits."""
    names = list(areas)
    print("Select test areas:")
    for number, name in enumerate(names, start=1):
        description = " (tests directly under tests/)" if name == "general" else ""
        print(f"  {number:2}. {name}{description}")
    print("   0. all (the full suite can take hours)")
    print("Enter names or numbers separated by spaces or commas; q to quit.")

    aliases = {str(number): name for number, name in enumerate(names, start=1)}
    while True:
        tokens = input("Areas: ").lower().replace(",", " ").split()
        if tokens in (["q"], ["quit"]):
            return None
        selected = [aliases.get(token, token) for token in tokens]
        invalid = [name for name in selected if name not in areas and name not in ("0", "all")]
        if not selected or invalid:
            print("Choose valid areas." if not invalid else f"Unknown areas: {', '.join(invalid)}")
            continue
        if "0" in selected or "all" in selected:
            return ["tests"]
        return list(dict.fromkeys(path for name in selected for path in areas[name]))


def main() -> int:
    """Discover test areas and preserve pytest's exit status."""
    root = Path(__file__).resolve().parents[1]
    tests = root / "tests"
    areas = {
        directory.name: [directory.relative_to(root).as_posix()]
        for directory in sorted(tests.iterdir())
        if directory.is_dir() and any(directory.rglob("test_*.py"))
    }
    general = [path.relative_to(root).as_posix() for path in sorted(tests.glob("test_*.py"))]
    if general:
        areas["general"] = general

    try:
        paths = select_paths(areas)
        if paths is None:
            return 0
        print(f"Running: {', '.join(paths)}", flush=True)
        return subprocess.call([sys.executable, "-m", "pytest", *paths, *sys.argv[1:]], cwd=root)
    except EOFError:
        print("\nNo selection received; tests were not run.")
        return 1
    except KeyboardInterrupt:
        print("\nTest run cancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
