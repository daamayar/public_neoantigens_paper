#!/usr/bin/env python3
"""
Master orchestrator for the MM-PBSA binding free energy pipeline.

Processes peptide-HLA complexes through 5 phases:
  1. Topology conversion (CHARMM PSF -> AMBER prmtop)
  2. Trajectory preparation (cpptraj strip + frame selection)
  3. MMPBSA.py execution
  4. Results analysis, convergence, validation
  5. Report generation (Markdown)

Usage:
    # Full pipeline for all complexes in list_dirs.txt
    python run_amber_mmpbsa.py list_dirs.txt

    # Only topology conversion
    python run_amber_mmpbsa.py list_dirs.txt --step topology

    # Only MMPBSA computation (after topology + trajectory are done)
    python run_amber_mmpbsa.py list_dirs.txt --step mmpbsa

    # Only analysis + reports (after MMPBSA is done)
    python run_amber_mmpbsa.py list_dirs.txt --step analyze

    # Generate OAR job submission scripts
    python run_amber_mmpbsa.py list_dirs.txt --generate-oar-scripts

    # Process specific replicas
    python run_amber_mmpbsa.py list_dirs.txt --replicas 1,2,3

    # Process specific complex (without list file)
    python run_amber_mmpbsa.py --complex pik3ca_e545k_neo_a_1101_strdplseitk

    # Use MPI mode
    python run_amber_mmpbsa.py list_dirs.txt --use-mpi --mpi-cores 10

    # Frequent use case
    python run_amber_mmpbsa.py list_dirs.txt --replicas 1,2,3,4,5,6,7,8,9,10 --use-mpi
"""
import os
import sys
import argparse
import time

# Ensure the scripts package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import config
from scripts import utils
from scripts.convert_topology import convert_topology
from scripts.prepare_trajectory import prepare_trajectory
from scripts.run_mmpbsa import run_mmpbsa
from scripts.analyze_results import analyze_results
from scripts.generate_reports import generate_reports


def process_single_replica(complex_name, rep_num, step="all",
                           analysis_window="all", use_mpi=False,
                           mpi_cores=10):
    """
    Process a single replica through the pipeline phases.

    Args:
        complex_name: Complex directory name
        rep_num: Replica number (1-based)
        step: Which step(s) to run
        analysis_window: A window name (e.g. "500ns"), or "all" for all windows
        use_mpi: Use MPI parallelization
        mpi_cores: Number of MPI cores
    """
    rep_dir = utils.get_replica_dir(complex_name, rep_num)
    if not os.path.isdir(rep_dir):
        print(f"  WARNING: Replica directory not found: {rep_dir}")
        return False

    try:
        # Phase 1: Topology conversion
        if step in ("all", "topology", "topology+trajectory"):
            convert_topology(complex_name, rep_num)

        # Phase 2: Trajectory preparation
        if step in ("all", "trajectory", "topology+trajectory"):
            prepare_trajectory(complex_name, rep_num, analysis_window)

        # Phase 3: MMPBSA execution
        if step in ("all", "mmpbsa"):
            run_mmpbsa(complex_name, rep_num, analysis_window,
                       use_mpi, mpi_cores)

        return True
    except Exception as e:
        print(f"  ERROR processing {complex_name} rep_{rep_num}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_complex(complex_name, replicas=None, step="all",
                    analysis_window="all", use_mpi=False,
                    mpi_cores=10):
    """
    Process all replicas of a complex.

    Args:
        complex_name: Complex directory name
        replicas: List of replica numbers to process (None = all)
        step: Which step(s) to run
        analysis_window: A window name (e.g. "500ns"), or "all" for all windows
        use_mpi: Use MPI parallelization
        mpi_cores: Number of MPI cores
    """
    complex_dir = utils.get_complex_dir(complex_name)
    if not os.path.isdir(complex_dir):
        print(f"ERROR: Complex directory not found: {complex_dir}")
        return False

    if replicas is None:
        replicas = list(range(1, config.NUM_REPLICAS + 1))

    print(f"\n{'#'*60}")
    print(f"# Processing: {complex_name}")
    print(f"# Replicas: {replicas}")
    print(f"# Step: {step}")
    print(f"# Analysis windows: {analysis_window}")
    print(f"{'#'*60}")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    # Phases 1-3: Per-replica processing
    if step in ("all", "topology", "trajectory", "topology+trajectory", "mmpbsa"):
        for rep in replicas:
            print(f"\n--- Replica {rep}/{len(replicas)} ---")
            if process_single_replica(complex_name, rep, step,
                                      analysis_window, use_mpi,
                                      mpi_cores):
                success_count += 1
            else:
                fail_count += 1

        print(f"\nReplica processing: {success_count} success, {fail_count} failed")

    # Phase 4: Analysis (operates across all replicas)
    analysis_ok = False
    if step in ("all", "analyze"):
        try:
            result = analyze_results(complex_name, analysis_window)
            analysis_ok = result is not None
            if not analysis_ok:
                print(f"  WARNING: Analysis returned no results for {complex_name}. "
                      f"Report generation will be skipped.")
        except Exception as e:
            print(f"  ERROR in analysis: {e}")
            import traceback
            traceback.print_exc()

    # Phase 5: Report generation (skip if analysis produced no results)
    if step in ("all", "report", "analyze"):
        if step in ("report",) or analysis_ok:
            try:
                generate_reports(complex_name)
            except Exception as e:
                print(f"  ERROR in report generation: {e}")
                import traceback
                traceback.print_exc()
        elif step in ("all", "analyze") and not analysis_ok:
            print(f"  Skipping report generation: no analysis results available.")

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal time for {complex_name}: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

    return fail_count == 0


def generate_oar_scripts(complex_names, analysis_window="all",
                         use_mpi=False, mpi_cores=10):
    """
    Generate OAR job submission scripts for cluster execution.

    Creates:
    - One script per complex that processes all replicas sequentially
    - One script per replica for parallel execution
    - A master submission script
    """
    oar_dir = os.path.join(config.PACKAGE_DIR, "oar_scripts")
    os.makedirs(oar_dir, exist_ok=True)

    script_path = os.path.abspath(__file__)

    # Per-replica scripts
    submission_cmds = []

    for complex_name in complex_names:
        for rep in range(1, config.NUM_REPLICAS + 1):
            # Worker script for a single replica
            worker_name = f"run_{complex_name}_rep{rep}.sh"
            worker_path = os.path.join(oar_dir, worker_name)

            with open(worker_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"# OAR worker: {complex_name} replica {rep}\n")
                f.write(f"# Generated by run_amber_mmpbsa.py\n\n")
                f.write(f"set -euo pipefail\n\n")
                f.write(f"# Activate conda environment (set +u: .bashrc/conda may reference unset vars)\n")
                f.write(f"set +u\n")
                f.write(f"source ~/.bashrc\n")
                f.write("\n".join(config.CONDA_ACTIVATE_CMDS) + "\n")
                f.write(f"set -u\n\n")
                f.write(f"cd {config.PACKAGE_DIR}\n\n")
                if use_mpi:
                    # Auto-detect cores on the assigned node, reserve 4 for system
                    f.write("# Auto-detect MPI cores on this node (nproc - 4)\n")
                    f.write("MPI_CORES=$(( $(nproc) - 4 ))\n")
                    f.write("if [ $MPI_CORES -lt 1 ]; then MPI_CORES=1; fi\n")
                    f.write('echo "MPI cores: $MPI_CORES ($(nproc) total - 4 reserved)"\n\n')
                    mpi_flags = '--use-mpi --mpi-cores $MPI_CORES'
                else:
                    mpi_flags = ""
                f.write(f"echo 'Starting: {complex_name} rep_{rep}'\n")
                f.write(f"echo 'Hostname: '$(hostname)\n")
                f.write(f"echo 'Date: '$(date)\n\n")
                f.write(f"# Run phases 1-3 for this replica\n")
                f.write(f"python {script_path} --complex {complex_name} "
                        f"--replicas {rep} --step topology+trajectory "
                        f"--window {analysis_window} "
                        f"{mpi_flags}\n\n")
                f.write(f"python {script_path} --complex {complex_name} "
                        f"--replicas {rep} --step mmpbsa "
                        f"--window {analysis_window} "
                        f"{mpi_flags}\n\n")
                f.write(f"echo 'Finished: {complex_name} rep_{rep}'\n")
                f.write(f"echo 'Date: '$(date)\n")

            os.chmod(worker_path, 0o755)

            # OAR submission command
            job_name = f"mmpbsa_{complex_name}_rep{rep}"
            notify = f'--notify "mail:{config.OAR_EMAIL}" ' if config.OAR_EMAIL else ""
            cluster = f'-p "{config.OAR_CLUSTERS}" ' if config.OAR_CLUSTERS else ""
            submission_cmds.append(
                f'oarsub {notify}'
                f'-n "{job_name}" '
                f'-q {config.OAR_QUEUE} '
                f'{cluster}'
                f'-l "host=1,walltime={config.OAR_WALLTIME_HOURS}:00:00" '
                f'"{worker_path}"'
            )

    # Analysis script (run after all replicas are done)
    for complex_name in complex_names:
        analysis_worker = os.path.join(oar_dir, f"run_{complex_name}_analysis.sh")
        with open(analysis_worker, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# OAR worker: analysis for {complex_name}\n\n")
            f.write(f"set -euo pipefail\n\n")
            f.write(f"# Activate conda environment (set +u: .bashrc/conda may reference unset vars)\n")
            f.write(f"set +u\n")
            f.write(f"source ~/.bashrc\n")
            f.write("\n".join(config.CONDA_ACTIVATE_CMDS) + "\n")
            f.write(f"set -u\n\n")
            f.write(f"cd {config.PACKAGE_DIR}\n\n")
            f.write(f"python {script_path} --complex {complex_name} "
                    f"--step analyze --window {analysis_window}\n")
        os.chmod(analysis_worker, 0o755)

    # Master submission script
    submit_script = os.path.join(oar_dir, "submit_all.sh")
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all MMPBSA jobs to OAR\n")
        f.write(f"# Generated by run_amber_mmpbsa.py\n\n")
        f.write(f'echo "Submitting MMPBSA jobs to OAR"\n')
        f.write('echo "Email: %s"\n' % (config.OAR_EMAIL or "(not set)"))
        f.write(f'echo "Queue: {config.OAR_QUEUE}"\n')
        f.write(f'echo "Walltime: {config.OAR_WALLTIME_HOURS}h"\n')
        f.write(f'echo "========================================"\n\n')

        for cmd in submission_cmds:
            f.write(f"{cmd}\n\n")

        f.write(f'echo "========================================"\n')
        f.write(f'echo "Submitted {len(submission_cmds)} jobs"\n')
        f.write(f'echo "Use oarstat -u to check job status"\n')
        f.write(f'echo ""\n')
        f.write(f'echo "After all replica jobs complete, run analysis:"\n')
        for complex_name in complex_names:
            f.write(f'echo "  oarsub ... {oar_dir}/run_{complex_name}_analysis.sh"\n')

    os.chmod(submit_script, 0o755)

    print(f"\nOAR scripts generated in {oar_dir}/")
    print(f"  Worker scripts: {len(submission_cmds)} (one per replica)")
    print(f"  Analysis scripts: {len(complex_names)} (one per complex)")
    print(f"  Master submission: {submit_script}")
    print(f"\nTo submit all jobs:")
    print(f"  bash {submit_script}")


def main():
    parser = argparse.ArgumentParser(
        description="MM-PBSA binding free energy pipeline for peptide-HLA complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline
    python run_amber_mmpbsa.py list_dirs.txt

    # Only topology conversion + trajectory preparation
    python run_amber_mmpbsa.py list_dirs.txt --step topology+trajectory

    # Only MMPBSA computation
    python run_amber_mmpbsa.py list_dirs.txt --step mmpbsa

    # Only analysis + reports
    python run_amber_mmpbsa.py list_dirs.txt --step analyze

    # Generate OAR scripts for cluster
    python run_amber_mmpbsa.py list_dirs.txt --generate-oar-scripts

    # Specific complex without list file
    python run_amber_mmpbsa.py --complex pik3ca_e545k_neo_a_1101_strdplseitk

    # Specific replicas
    python run_amber_mmpbsa.py list_dirs.txt --replicas 1,2,3

    # With MPI
    python run_amber_mmpbsa.py list_dirs.txt --use-mpi --mpi-cores 10
        """
    )
    parser.add_argument("list_file", nargs="?", default=None,
                        help="File containing list of complex directories")
    parser.add_argument("--complex", type=str, default=None,
                        help="Process a single complex (alternative to list file)")
    parser.add_argument("--step", choices=[
        "all", "topology", "trajectory", "topology+trajectory",
        "mmpbsa", "analyze", "report"
    ], default="all", help="Pipeline step to run (default: all)")
    parser.add_argument("--replicas", type=str, default=None,
                        help="Comma-separated replica numbers (default: all)")
    parser.add_argument("--window", choices=config.ALL_WINDOW_NAMES + ["all"], default="all",
                        help="Analysis window (default: all)")
    parser.add_argument("--use-mpi", action="store_true",
                        help="Use MPI parallelization for MMPBSA.py")
    parser.add_argument("--mpi-cores", type=int, default=0,
                        help="Number of MPI cores (default: auto-detect, reserving 4 for system)")
    parser.add_argument("--generate-oar-scripts", action="store_true",
                        help="Generate OAR job submission scripts instead of running")

    args = parser.parse_args()

    # Determine complexes to process
    if args.complex:
        complex_names = [args.complex]
    elif args.list_file:
        if not os.path.exists(args.list_file):
            print(f"ERROR: List file not found: {args.list_file}")
            sys.exit(1)
        complex_names = utils.read_complexes_list(args.list_file)
    else:
        parser.print_help()
        sys.exit(1)

    if not complex_names:
        print("ERROR: No complexes to process")
        sys.exit(1)

    # Parse replicas
    replicas = None
    if args.replicas:
        replicas = [int(r) for r in args.replicas.split(",")]

    # Auto-detect MPI cores if not specified (reserve 4 for system)
    if args.use_mpi and args.mpi_cores <= 0:
        total_cores = os.cpu_count() or 1
        args.mpi_cores = max(1, total_cores - 4)

    # Validate AMBERHOME and binaries early
    config.validate_amberhome()

    print("=" * 60)
    print("MM-PBSA Binding Free Energy Pipeline")
    print("=" * 60)
    print(f"Complexes: {len(complex_names)}")
    print(f"Step: {args.step}")
    print(f"Analysis windows: {args.window}")
    print(f"Replicas: {replicas if replicas else 'all'}")
    print(f"MPI: {'Yes (' + str(args.mpi_cores) + ' cores)' if args.use_mpi else 'No'}")
    print(f"Entropy: IE + QHA (always computed)")
    print("=" * 60)

    # Generate OAR scripts (MPI enabled by default for cluster jobs)
    if args.generate_oar_scripts:
        generate_oar_scripts(complex_names, args.window,
                             use_mpi=True, mpi_cores=args.mpi_cores)
        return

    # Process each complex
    total_start = time.time()
    success = 0
    failed = 0

    for complex_name in complex_names:
        if process_complex(complex_name, replicas, args.step,
                           args.window, args.use_mpi,
                           args.mpi_cores):
            success += 1
        else:
            failed += 1

    total_elapsed = time.time() - total_start
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Complexes: {success} success, {failed} failed")
    print(f"  Total time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print(f"{'='*60}")

    # Exit non-zero if anything failed. Per-replica errors are caught and reported so that
    # one bad replica does not abort the batch, but the process must still signal failure:
    # a driver script or scheduler that only inspects the exit status would otherwise treat
    # a run that produced nothing as a complete success.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
