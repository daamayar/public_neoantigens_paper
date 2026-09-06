#!/usr/bin/env python3
"""
Phase 3: MMPBSA.py Execution
Run MM-PBSA binding free energy calculations for a single replica.

Supports both serial (MMPBSA.py) and MPI (MMPBSA.py.MPI) modes.
Computes enthalpy (ΔH) only at this stage. Entropy corrections (IE, QHA)
are computed as post-processing in Phase 4 (analyze_results.py).

Usage:
    python -m scripts.run_mmpbsa <complex_name> <rep_num> [--window 500ns|100ns|200ns|300ns|400ns|all]
                                                          [--use-mpi] [--mpi-cores N]
"""
import os
import time
import argparse

from . import config
from . import utils


def write_mmpbsa_input(filepath):
    """
    Write MMPBSA.py input file for PB calculation with CHARMM36m.

    Args:
        filepath: Output file path

    Raises:
        ValueError: If radiopt=0 but inp not in {1, 2} (required by MMPBSA.py
                    when PB radii are read from the prmtop instead of assigned
                    at runtime).
    """
    p = config.MMPBSA_PARAMS

    # Validate parameter consistency: radiopt=0 requires inp=1 or inp=2.
    # radiopt=0 reads PB radii from prmtop (assigned as mbondi_pb3 during
    # topology conversion). inp=0 is incompatible because it tries to
    # reassign radii internally, which fails with CHARMM atom types.
    if p.get("radiopt", 0) == 0 and p.get("inp", 1) not in (1, 2):
        raise ValueError(
            f"MMPBSA parameter conflict: radiopt={p['radiopt']} requires "
            f"inp in {{1, 2}}, but inp={p['inp']}. "
            f"See AMBER manual, pbsa inp parameter."
        )

    content = f"""\
MM-PBSA for peptide-HLA (CHARMM36m, PB enthalpy)
&general
  startframe=1,
  endframe=99999999,
  interval=1,
  verbose=2,
  keep_files=2,
  netcdf=1,
  use_sander={p['use_sander']},
/
&pb
  istrng={p['istrng']},
  fillratio={p['fillratio']},
  radiopt={p['radiopt']},
  inp={p['inp']},
  indi={p['indi']},
  exdi={p['exdi']},
  scale={p['scale']},
  linit={p['linit']},
  prbrad={p['prbrad']},
/
"""

    with open(filepath, "w") as f:
        f.write(content)


def _mmpbsa_output_is_complete(filepath):
    """Check that FINAL_RESULTS_MMPBSA.dat contains a completed result.

    A complete file has a 'DELTA' section with a 'TOTAL' line. A partial file
    (from a killed run) typically lacks this.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
        # The final section of a successful run contains "Differences" or "DELTA"
        # followed by a TOTAL line
        in_delta = False
        for line in content.splitlines():
            stripped = line.strip()
            if "Differences" in stripped or stripped.startswith("DELTA"):
                in_delta = True
            if in_delta and "TOTAL" in stripped:
                return True
        return False
    except Exception:
        return False


def run_mmpbsa(complex_name, rep_num, analysis_window="all",
               use_mpi=False, mpi_cores=10):
    """
    Run MMPBSA.py for a single replica.

    Args:
        complex_name: Name of the complex directory
        rep_num: Replica number (1-based)
        analysis_window: A window name (e.g. "500ns"), or "all" for all windows
        use_mpi: Whether to use MPI parallelization
        mpi_cores: Number of MPI cores (if use_mpi=True)

    Returns:
        dict: {window_name: output_file_path}
    """
    print(f"\n{'='*60}")
    print(f"Phase 3: MMPBSA.py for {complex_name} rep_{rep_num}")
    print(f"  Mode: {'MPI (' + str(mpi_cores) + ' cores)' if use_mpi else 'Serial'}")
    print(f"{'='*60}")

    if analysis_window == "all":
        windows = list(config.ANALYSIS_WINDOWS.keys())
    else:
        windows = [analysis_window]

    results = {}

    for window in windows:
        workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, window)
        os.makedirs(workdir, exist_ok=True)

        output_file = os.path.join(workdir, "FINAL_RESULTS_MMPBSA.dat")

        # Check if already done (validate completeness, not just existence)
        if utils.check_file_exists(output_file) and _mmpbsa_output_is_complete(output_file):
            print(f"  [{window}] Results already exist: FINAL_RESULTS_MMPBSA.dat")
            results[window] = output_file
            continue
        elif utils.check_file_exists(output_file):
            print(f"  [{window}] WARNING: Incomplete FINAL_RESULTS_MMPBSA.dat detected "
                  f"(missing DELTA TOTAL). Re-running MMPBSA.")
            os.remove(output_file)

        # Prevent concurrent execution on the same workdir
        lock_file = os.path.join(workdir, ".mmpbsa.lock")
        if os.path.exists(lock_file):
            try:
                with open(lock_file, "r") as lf:
                    lock_pid = int(lf.read().strip())
                # Check if the process is still running
                os.kill(lock_pid, 0)
                print(f"  [{window}] ERROR: Another MMPBSA process (PID {lock_pid}) is "
                      f"running in {workdir}. Skipping.")
                continue
            except (ValueError, ProcessLookupError, PermissionError):
                # Stale lock file — previous process is gone
                print(f"  [{window}] Removing stale lock file (PID no longer running).")
                os.remove(lock_file)

        # Create lock file
        with open(lock_file, "w") as lf:
            lf.write(str(os.getpid()))

        try:
            results[window] = _run_mmpbsa_window(
                complex_name, rep_num, window, workdir, output_file,
                use_mpi, mpi_cores
            )
        finally:
            # Remove lock file
            if os.path.exists(lock_file):
                os.remove(lock_file)

    print("  MMPBSA.py execution complete!")
    return results


def _run_mmpbsa_window(complex_name, rep_num, window, workdir, output_file,
                       use_mpi, mpi_cores):
    """Run MMPBSA.py for a single analysis window. Called with lock held."""
    # Check required files
    complex_prmtop = os.path.join(workdir, "complex.prmtop")
    receptor_prmtop = os.path.join(workdir, "receptor.prmtop")
    ligand_prmtop = os.path.join(workdir, "ligand.prmtop")
    traj_file = os.path.join(workdir, f"rep{rep_num}_dry_{window}.nc")

    for f in [complex_prmtop, receptor_prmtop, ligand_prmtop, traj_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Required file not found: {f}\n"
                f"Run topology conversion and trajectory preparation first."
            )

    # Write MMPBSA input file
    mmpbsa_input = os.path.join(workdir, "mmpbsa_pb.in")
    write_mmpbsa_input(mmpbsa_input)

    # Build command
    mmpbsa_log = os.path.join(workdir, "mmpbsa.log")

    if use_mpi:
        mmpbsa_bin = os.path.join(config.AMBERHOME, "bin", "MMPBSA.py.MPI")
        mpirun_bin = os.path.join(config.AMBERHOME, "bin", "mpirun")
        cmd = [
            mpirun_bin, "--oversubscribe", "-np", str(mpi_cores),
            mmpbsa_bin, "-O",
            "-i", mmpbsa_input,
            "-o", output_file,
            "-cp", complex_prmtop,
            "-rp", receptor_prmtop,
            "-lp", ligand_prmtop,
            "-y", traj_file,
        ]
    else:
        mmpbsa_bin = os.path.join(config.AMBERHOME, "bin", "MMPBSA.py")
        cmd = [
            mmpbsa_bin, "-O",
            "-i", mmpbsa_input,
            "-o", output_file,
            "-cp", complex_prmtop,
            "-rp", receptor_prmtop,
            "-lp", ligand_prmtop,
            "-y", traj_file,
        ]

    print(f"  [{window}] Running MMPBSA.py...")
    print(f"    Command: {' '.join(cmd)}")
    print(f"    Working directory: {workdir}")

    start_time = time.time()

    result = utils.run_command(
        cmd,
        cwd=workdir,
        log_file=mmpbsa_log,
        description=f"[{window}] MMPBSA.py running..."
    )

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

    if result.returncode != 0:
        print(f"    ERROR: MMPBSA.py failed (exit code {result.returncode})")
        print(f"    Elapsed: {time_str}")
        print(f"    Check log: {mmpbsa_log}")
        # Print last few lines of stderr for diagnostics
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-10:]:
            print(f"    STDERR: {line}")
        raise RuntimeError(f"MMPBSA.py failed for {complex_name} rep_{rep_num} {window}")

    print(f"    SUCCESS: {window} complete")
    print(f"    Elapsed: {time_str}")
    print(f"    Output: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run MMPBSA.py for a single replica")
    parser.add_argument("complex_name", help="Complex directory name")
    parser.add_argument("rep_num", type=int, help="Replica number (1-based)")
    parser.add_argument("--window", choices=config.ALL_WINDOW_NAMES + ["all"], default="all",
                        help="Analysis window (default: all)")
    parser.add_argument("--use-mpi", action="store_true", help="Use MPI parallelization")
    parser.add_argument("--mpi-cores", type=int, default=10,
                        help="Number of MPI cores (default: 10)")
    args = parser.parse_args()
    run_mmpbsa(args.complex_name, args.rep_num, args.window,
               args.use_mpi, args.mpi_cores)


if __name__ == "__main__":
    main()
