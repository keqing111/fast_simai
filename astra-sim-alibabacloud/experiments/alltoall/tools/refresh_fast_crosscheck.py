#!/usr/bin/env python3
"""Regenerate native FAST fixture dumps using an explicitly supplied driver."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-driver", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]
    from tests import test_fast_crosscheck as cross

    if not args.cpp_driver.is_file():
        raise SystemExit(f"C++ driver does not exist: {args.cpp_driver}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cross.DRIVER = str(args.cpp_driver.resolve())
    cross.WORK = str(args.work_dir.resolve())
    fixtures = root / "tests" / "fixtures" / "fast_cpp"
    fixtures.mkdir(parents=True, exist_ok=True)

    failures = 0
    for case in cross.CASES:
        matrix, records, cpp_result, _ = cross.run_cpp(case)
        python_result, _ = cross.run_python(case, matrix)
        if case["mode"] == "core":
            problems = cross.check_core(case, records, python_result, matrix)
        else:
            problems = cross.check_pipeline(case, records, python_result)
            for name in ("t", "ratio", "baseline", "speedup"):
                ok, message = cross.cmp_float(
                    name,
                    python_result[name],
                    cpp_result[name],
                )
                if not ok:
                    problems.append(message)
        if problems:
            failures += 1
            print(f"{case['name']}: FAIL: {'; '.join(problems)}", file=sys.stderr)
            continue
        for suffix in (".mat", ".dump"):
            shutil.copy2(args.work_dir / f"{case['name']}{suffix}", fixtures)
        print(f"{case['name']}: refreshed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
