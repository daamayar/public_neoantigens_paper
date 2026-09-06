#!/usr/bin/env python3
"""
Orchestrator for the peptide-MD structural analyses (RMSD / RMSF).

Runs, in order:
    1. build_cache      -- reduce the 170 x 5 GB DCDs to a ~4 GB coordinate cache
    3. analyze_rmsd     -- Task 1: RMS distributions (11 panels)
    4. analyze_rmsf     -- Task 2: the two RMSF plots (all 17 peptides)
    6. make_report      -- assemble REPORT.md

Environment
-----------
    conda activate AmberTools25
    source $CONDA_PREFIX/amber.sh

Usage
-----
    python run_md_analyses.py                    # everything
    python run_md_analyses.py --step rmsd        # one step
    python run_md_analyses.py --skip-cache       # cache already built

Steps are individually idempotent: the cache skips replicas that are already
valid, and the analyses simply overwrite their own figures and CSVs.
"""
import sys
import argparse
import subprocess

STEPS = ["cache", "rmsd", "rmsf", "report"]

MODULES = {
    "cache":  "scripts.build_cache",
    "rmsd":   "scripts.analyze_rmsd",
    "rmsf":   "scripts.analyze_rmsf",
    "report": "scripts.make_report",
}


def run_step(step, workers=None):
    """Run one pipeline step as a subprocess and stream its output.

    Args:
        step (str): one of STEPS.
        workers (int | None): passed through as --workers to the cache step.

    Returns:
        int: the subprocess return code.
    """
    cmd = [sys.executable, "-m", MODULES[step]]
    if workers and step == "cache":
        cmd += ["--workers", str(workers)]

    print("\n" + "=" * 78)
    print(f"  STEP: {step}    ({' '.join(cmd)})")
    print("=" * 78)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(description="Run the peptide-MD structural analyses.")
    ap.add_argument("--step", choices=STEPS, action="append", default=None,
                    help="run only these steps (repeatable; default: all)")
    ap.add_argument("--skip-cache", action="store_true",
                    help="assume the trajectory cache is already built")
    ap.add_argument("--workers", type=int, default=10, help="parallel cpptraj jobs")
    args = ap.parse_args()

    steps = args.step or list(STEPS)
    if args.skip_cache and "cache" in steps:
        steps.remove("cache")

    for step in steps:
        rc = run_step(step, workers=args.workers)
        if rc != 0:
            print(f"\nSTEP '{step}' FAILED (rc={rc}) -- stopping.")
            sys.exit(rc)

    print("\n" + "=" * 78)
    print("  ALL STEPS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
