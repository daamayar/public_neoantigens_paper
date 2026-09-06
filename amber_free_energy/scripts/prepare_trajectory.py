#!/usr/bin/env python3
"""
Phase 2: Trajectory Preparation
Strip water/ions and select frames from DCD trajectories using cpptraj.

Creates stripped NetCDF trajectories for all analysis windows (100ns-500ns).

Usage:
    python -m scripts.prepare_trajectory <complex_name> <rep_num> [--window 500ns|100ns|200ns|300ns|400ns|all]
"""
import os
import argparse

from . import config
from . import utils


def prepare_trajectory(complex_name, rep_num, analysis_window="all"):
    """
    Prepare trajectory for MMPBSA analysis.

    Args:
        complex_name: Name of the complex directory
        rep_num: Replica number (1-based)
        analysis_window: A window name (e.g. "500ns"), or "all" for all windows

    Returns:
        dict: {window_name: output_trajectory_path}
    """
    rep_dir = utils.get_replica_dir(complex_name, rep_num)
    print(f"\n{'='*60}")
    print(f"Phase 2: Trajectory preparation for {complex_name} rep_{rep_num}")
    print(f"{'='*60}")

    # Find DCD file
    dcd_path, traj_form = utils.find_trajectory(complex_name, rep_num)
    print(f"  trajectory: {os.path.basename(dcd_path)} ({traj_form})")

    # Determine which windows to process
    if analysis_window == "all":
        windows = list(config.ANALYSIS_WINDOWS.keys())
    else:
        windows = [analysis_window]

    results = {}

    for window in windows:
        window_config = config.ANALYSIS_WINDOWS[window]
        workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, window)
        os.makedirs(workdir, exist_ok=True)

        output_traj = os.path.join(workdir, f"rep{rep_num}_dry_{window}.nc")

        # The solvated prmtop lives in the 500ns workdir; other windows symlink to it.
        solvated_prmtop = os.path.join(workdir, "solvated.prmtop")
        complex_prmtop = os.path.join(workdir, "complex.prmtop")

        expected_frames = (window_config['last_frame'] - window_config['first_frame']) \
            // window_config['stride'] + 1

        # Reuse an existing trajectory only when it is COMPLETE, in count and in content.
        # Testing existence alone accepts a half-written .nc left by an interrupted cpptraj:
        # the file is non-empty, so it passes, and every later phase then silently uses a
        # shorter ensemble than the window configuration implies.
        if utils.check_file_exists(output_traj):
            ok, reason = validate_trajectory(output_traj, expected_frames)
            if ok:
                print(f"  [{window}] Trajectory already exists: "
                      f"{os.path.basename(output_traj)} ({expected_frames} frames, verified)")
                results[window] = output_traj
                continue
            print(f"  [{window}] Existing trajectory rejected - {reason}; regenerating.")
            os.remove(output_traj)

        print(f"  [{window}] {window_config['description']}")
        print(f"    Frames: {window_config['first_frame']} to {window_config['last_frame']}, stride {window_config['stride']}")

        # Check the topology this path will actually use. The dry trajectory is read
        # with complex.prmtop and never touches solvated.prmtop, so demanding the
        # latter would reject a perfectly usable deposit.
        needed = solvated_prmtop if traj_form == "solvated" else complex_prmtop
        if not os.path.exists(needed):
            raise FileNotFoundError(
                f"{os.path.basename(needed)} not found in {workdir}. Run topology "
                f"conversion first: python -m scripts.convert_topology "
                f"{complex_name} {rep_num}"
            )

        # Build cpptraj input
        cpptraj_input = os.path.join(workdir, f"strip_traj_{window}.cpptraj")
        with open(cpptraj_input, "w") as f:
            if traj_form == "solvated":
                # Reimage before stripping so molecules broken across the periodic
                # boundary are made whole; the dry trajectory is already whole and
                # carries no box, so both steps are skipped for it.
                f.write(f"parm {solvated_prmtop}\n")
                f.write(f"trajin {dcd_path} {window_config['first_frame']} {window_config['last_frame']} {window_config['stride']}\n")
                f.write("autoimage\n")
                f.write(f"strip {config.STRIP_MASK_CPPTRAJ} nobox\n")
            else:
                f.write(f"parm {utils.dry_topology(complex_name, rep_num, workdir)}\n")
                f.write(f"trajin {dcd_path} {window_config['first_frame']} {window_config['last_frame']} {window_config['stride']}\n")
            f.write(f"trajout {output_traj} netcdf\n")
            f.write("run\n")

        # Run cpptraj
        cpptraj_log = os.path.join(workdir, f"cpptraj_{window}.log")
        cpptraj_bin = os.path.join(config.AMBERHOME, "bin", "cpptraj")
        result = utils.run_command(
            [cpptraj_bin, "-i", cpptraj_input],
            log_file=cpptraj_log,
            description=f"Running cpptraj for {window}..."
        )

        if result.returncode != 0:
            print(f"    ERROR: cpptraj failed. Check log: {cpptraj_log}")
            print(f"    STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"cpptraj failed for {complex_name} rep_{rep_num} {window}")

        # Validate what cpptraj actually wrote, and retry once on a bad write. A zero exit
        # status is not evidence that the file is complete: concurrent cpptraj processes
        # writing NetCDF to a network filesystem have been seen to lose individual records
        # while still reporting success.
        ok, reason = validate_trajectory(output_traj, expected_frames)
        if not ok:
            print(f"    WARNING: cpptraj reported success but the output is bad - {reason}. "
                  f"Retrying once.")
            os.remove(output_traj)
            result = utils.run_command(
                [cpptraj_bin, "-i", cpptraj_input],
                log_file=cpptraj_log,
                description=f"Re-running cpptraj for {window}..."
            )
            ok, reason = validate_trajectory(output_traj, expected_frames)
            if not ok:
                raise RuntimeError(
                    f"Trajectory {output_traj} is still invalid after a retry - {reason}. "
                    f"Check {cpptraj_log}."
                )
            print("    Retry produced a valid trajectory.")

        # Count frames in output trajectory and validate
        frame_count = count_frames(complex_prmtop, output_traj)
        print(f"    Output frames: {frame_count} (expected ~{expected_frames})")
        if frame_count > 0 and frame_count < expected_frames * 0.5:
            raise RuntimeError(
                f"Trajectory severely truncated: got {frame_count} frames, "
                f"expected ~{expected_frames} (< 50% threshold). "
                f"The DCD may have far fewer frames than "
                f"config.TOTAL_FRAMES={config.TOTAL_FRAMES}. "
                f"Check trajectory integrity before proceeding."
            )
        elif frame_count > 0 and frame_count < expected_frames * 0.9:
            print(f"    WARNING: Frame count ({frame_count}) is less than "
                  f"expected ({expected_frames}). The DCD may have fewer frames than "
                  f"config.TOTAL_FRAMES={config.TOTAL_FRAMES}. "
                  f"Results may have reduced statistical power.")
        print(f"    Output: {os.path.basename(output_traj)}")

        results[window] = output_traj

    print("  Trajectory preparation complete!")
    return results


#: NetCDF's default fill value for NC_FLOAT. A record slot that was allocated but never
#: written reads back as this, so the frame COUNT is correct while the content is missing.
NETCDF_FLOAT_FILL = 9.9692099683868690e+36


def validate_trajectory(traj_path, expected_frames=None):
    """Check that a NetCDF trajectory is complete in count *and* in content.

    Frame count alone is not sufficient. NetCDF pre-allocates record slots, so a write that
    loses records leaves holes filled with NETCDF_FLOAT_FILL while the record dimension
    still reports the full length. Such a frame places atoms ~1e37 A apart, which makes
    sander abort with "PB Bomb in pb_atmlist(): maxnba too short"; a trajectory that is
    merely short fails more quietly, by silently reducing the ensemble MM-PBSA averages
    over. Both have been observed after interrupted or heavily concurrent cpptraj runs.

    Args:
        traj_path (str): Path to the NetCDF trajectory to check.
        expected_frames (int | None): Frame count implied by the window configuration.
            When None, only the content is checked.

    Returns:
        tuple[bool, str]: (ok, reason). reason is "" when ok is True, otherwise a short
        human-readable description of the defect.
    """
    if not os.path.exists(traj_path) or os.path.getsize(traj_path) == 0:
        return False, "missing or empty"
    try:
        import numpy as np
        from scipy.io import netcdf_file
        with netcdf_file(traj_path, "r", mmap=False) as nc:
            coords = np.array(nc.variables["coordinates"][:], dtype=np.float64)
    except Exception as exc:
        return False, f"unreadable ({type(exc).__name__})"

    n = coords.shape[0]
    if expected_frames is not None and n != expected_frames:
        return False, f"{n} frames, expected {expected_frames}"

    nonfinite = ~np.isfinite(coords).all(axis=(1, 2))
    fill = np.isclose(np.abs(coords), NETCDF_FLOAT_FILL, rtol=1e-6).any(axis=(1, 2))
    absurd = np.abs(np.nan_to_num(coords)).max(axis=(1, 2)) > 1e4
    bad = np.where(nonfinite | fill | absurd)[0]
    if len(bad):
        return False, (f"{len(bad)}/{n} frames unwritten or non-physical "
                       f"(indices {bad[:5].tolist()}{'...' if len(bad) > 5 else ''})")
    return True, ""


def count_frames(prmtop, traj_path):
    """Count the number of frames in a trajectory file.

    The fast path reads the length of the unlimited 'frame' dimension straight from the
    NetCDF header, which is exact and costs no subprocess. Every trajectory this module
    writes is NetCDF, so that path is normally taken. The cpptraj fallback exists for
    non-NetCDF inputs and for environments without scipy, and parses the two strings
    cpptraj actually emits ("Coordinate processing will occur on N frames." and
    "Read N frames and processed N frames.").

    Args:
        prmtop (str): Path to the AMBER topology matching traj_path. Used only by the
            cpptraj fallback; the NetCDF fast path ignores it.
        traj_path (str): Path to the trajectory whose frames should be counted.

    Returns:
        int: Number of frames, or -1 if it could not be determined. Callers treat any
        value <= 0 as "unknown" and skip the size check.
    """
    try:
        from scipy.io import netcdf_file
        with netcdf_file(traj_path, "r", mmap=False) as nc:
            n = int(nc.variables["coordinates"].shape[0])
        if n > 0:
            return n
    except Exception:
        pass

    cpptraj_bin = os.path.join(config.AMBERHOME, "bin", "cpptraj")

    import re
    import tempfile
    count_script = f"parm {prmtop}\ntrajin {traj_path}\nrun\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpptraj", delete=False) as f:
        f.write(count_script)
        tmpfile = f.name

    try:
        result = utils.run_command([cpptraj_bin, "-i", tmpfile])
        output = (result.stdout or "") + (result.stderr or "")
        for pattern in (r"Read\s+(\d+)\s+frames",
                        r"processing will occur on\s+(\d+)\s+frames"):
            match = re.search(pattern, output)
            if match:
                n = int(match.group(1))
                if n > 0:
                    return n
    finally:
        os.unlink(tmpfile)

    return -1  # Unknown


def main():
    parser = argparse.ArgumentParser(description="Prepare trajectories for MMPBSA")
    parser.add_argument("complex_name", help="Complex directory name")
    parser.add_argument("rep_num", type=int, help="Replica number (1-based)")
    parser.add_argument("--window", choices=config.ALL_WINDOW_NAMES + ["all"], default="all",
                        help="Analysis window (default: all)")
    args = parser.parse_args()
    prepare_trajectory(args.complex_name, args.rep_num, args.window)


if __name__ == "__main__":
    main()
