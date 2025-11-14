#!/usr/bin/env python3
"""
Lightweight test runner to execute simple test modules without pytest.

Usage:
  From the project root run:
    python run_tests_without_pytest.py

This script discovers modules under `src/tests` whose filenames start with
`test_` and executes all functions within them whose names start with
`test_`. It prints a simple summary and returns a non-zero exit code if any
test fails.
"""
import sys
import os
import importlib
import pkgutil
import inspect
import traceback


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    # Ensure project root is on sys.path so `src` package imports work
    if root not in sys.path:
        sys.path.insert(0, root)

    tests_pkg = "src.tests"
    try:
        pkg = importlib.import_module(tests_pkg)
    except Exception as e:
        print(f"Failed to import tests package '{tests_pkg}': {e}")
        print("Make sure you run this script from the project root so 'src' is importable.")
        return 2

    results = []

    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if not name.startswith("test_"):
            continue
        modname = f"{tests_pkg}.{name}"
        print(f"Running tests in {modname}")
        try:
            m = importlib.import_module(modname)
        except Exception:
            print(f"  ERROR: failed to import {modname}")
            traceback.print_exc()
            results.append(False)
            continue

        funcs = [obj for _, obj in inspect.getmembers(m, inspect.isfunction) if _.startswith("test_")]
        if not funcs:
            print("  (no test_ functions found)")
            continue

        for func in funcs:
            name = func.__name__
            print(f"  - {name}() ... ", end="", flush=True)
            try:
                func()
            except Exception:
                print("FAILED")
                traceback.print_exc()
                results.append(False)
            else:
                print("ok")
                results.append(True)

    if not results:
        print("No tests were discovered under src/tests. Nothing to run.")
        return 3

    failed = results.count(False)
    total = len(results)
    print(f"\nSummary: {total-failed}/{total} tests passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
