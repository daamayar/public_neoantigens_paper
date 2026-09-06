"""
Shared utility functions for the MM-PBSA binding free energy pipeline.
"""
import os
import glob
import json
import subprocess

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types gracefully.

    Converts numpy scalars (float64, int64, etc.) to Python native types
    and numpy arrays to lists. Also handles np.nan/np.inf by converting
    to None (JSON-safe).
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

from . import config


def find_file(directory, pattern):
    """Find a single file matching a glob pattern in a directory."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files matching '{pattern}' in {directory}: {matches}")
    return matches[0]


def find_trajectory(complex_name, rep_num):
    """Resolve the production trajectory for one replica.

    Honours config.TRAJECTORY_SOURCE. Returns (path, form) where form is
    "solvated" or "dry"; callers use the form to decide whether the trajectory
    still has to be reimaged and stripped.
    """
    rep_dir = get_replica_dir(complex_name, rep_num)
    source = config.TRAJECTORY_SOURCE

    def _match(pattern):
        try:
            return find_file(rep_dir, pattern)
        except (FileNotFoundError, RuntimeError):
            return None

    solvated = None
    for pat in config.DCD_PATTERNS:
        solvated = _match(pat)
        if solvated:
            break
    dry = _match(config.DRY_DCD_PATTERN)

    if source == "solvated":
        if solvated:
            return solvated, "solvated"
        if dry:
            raise FileNotFoundError(
                f"PHLA_TRAJECTORY_SOURCE=solvated, but only the dry trajectory is "
                f"present for {complex_name} rep_{rep_num}.\n"
                f"  The solvated DCDs are not deposited (~840 GB). Use "
                f"PHLA_TRAJECTORY_SOURCE=dry (or auto): every analysis strips solvent "
                f"immediately, and the two forms give identical results.\n"
                f"  md_setup/resolvate.py rebuilds solvent around a single frame, for "
                f"restarts and visualisation. It cannot supply a solvated trajectory "
                f"here: a trajectory needs a fixed atom count, and solvating frames "
                f"independently gives a different water count each time.")
        raise FileNotFoundError(
            f"No solvated trajectory for {complex_name} rep_{rep_num} in {rep_dir}; "
            f"looked for {' or '.join(config.DCD_PATTERNS)}")

    if source == "dry":
        if dry:
            return dry, "dry"
        raise FileNotFoundError(
            f"PHLA_TRAJECTORY_SOURCE=dry, but no {config.DRY_DCD_PATTERN} for "
            f"{complex_name} rep_{rep_num} in {rep_dir}")

    if solvated:
        return solvated, "solvated"
    if dry:
        return dry, "dry"
    raise FileNotFoundError(
        f"No production trajectory for {complex_name} rep_{rep_num} in {rep_dir}: "
        f"looked for {' or '.join(config.DCD_PATTERNS)} (solvated) and "
        f"{config.DRY_DCD_PATTERN} (dry). "
        f"See docs/DATA.md for the expected layout.")


def dry_topology(complex_name, rep_num, workdir=None):
    """Topology matching the dry trajectory: the pipeline's own complex.prmtop.

    The deposited dry trajectory holds exactly the protein atoms of the solvated
    PSF in the same order, so complex.prmtop describes it without remapping
    (verified atom-by-atom on names and residues).
    """
    if workdir is None:
        workdir = get_mmpbsa_workdir(complex_name, rep_num, config.REFERENCE_WINDOW)
    return os.path.join(workdir, "complex.prmtop")


def get_complex_dir(complex_name):
    """Get the full path to a complex directory."""
    return os.path.join(config.BASE_DIR, complex_name)


def get_replica_dir(complex_name, rep_num):
    """Get the full path to a replica directory."""
    return os.path.join(get_complex_dir(complex_name), f"rep_{rep_num}")


def get_mmpbsa_workdir(complex_name, rep_num, analysis_window):
    """Get the MMPBSA working directory for a specific replica and analysis window."""
    workdir_name = f"mmpbsa_{analysis_window}"
    return os.path.join(get_replica_dir(complex_name, rep_num), workdir_name)


def get_results_dir(complex_name):
    """Get the results directory for a complex."""
    return os.path.join(config.RESULTS_DIR, complex_name)


def parse_xsc_file(xsc_path):
    """
    Parse a NAMD XSC file and extract box dimensions.

    Returns:
        list: [a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z] cell vectors
              For orthogonal boxes: [ax, 0, 0, 0, by, 0, 0, 0, cz]
    """
    with open(xsc_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                # step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z ...
                a_x = float(parts[1])
                a_y = float(parts[2])
                a_z = float(parts[3])
                b_x = float(parts[4])
                b_y = float(parts[5])
                b_z = float(parts[6])
                c_x = float(parts[7])
                c_y = float(parts[8])
                c_z = float(parts[9])

                # Validate orthogonal box assumption: off-diagonal elements
                # should be ~0 for the rectangular water boxes used in NAMD
                off_diag = [a_y, a_z, b_x, b_z, c_x, c_y]
                max_off_diag = max(abs(v) for v in off_diag)
                if max_off_diag > 0.01:
                    raise ValueError(
                        f"XSC file {xsc_path} has non-orthogonal cell vectors "
                        f"(max off-diagonal = {max_off_diag:.4f}). "
                        f"This pipeline assumes an orthogonal box. "
                        f"Off-diagonal elements: a_y={a_y}, a_z={a_z}, "
                        f"b_x={b_x}, b_z={b_z}, c_x={c_x}, c_y={c_y}"
                    )

                # Return as ParmEd box format: [a, b, c, alpha, beta, gamma]
                return [a_x, b_y, c_z, 90.0, 90.0, 90.0]
    raise ValueError(f"Could not parse XSC file: {xsc_path}")


def read_complexes_list(list_file):
    """Read complex names from a list file (one per line, skip comments/empty)."""
    complexes = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                complexes.append(line.rstrip("/"))
    return complexes


def save_metadata(filepath, metadata):
    """Save metadata dictionary as JSON (atomic write via temp file + rename).

    Uses NumpyEncoder to safely handle any numpy types that may have
    leaked into the metadata dict (np.float64, np.ndarray, etc.).
    Cleans up temp file on failure to prevent orphaned .tmp files.

    Args:
        filepath (str): Destination JSON file path.
        metadata (dict): Data to serialize.
    """
    import tempfile
    dir_name = os.path.dirname(filepath)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=dir_name, suffix=".tmp",
                                         delete=False) as f:
            tmp_path = f.name
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_metadata(filepath):
    """Load metadata dictionary from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def run_command(cmd, cwd=None, log_file=None, description=""):
    """
    Run a shell command and return the result.

    Args:
        cmd: Command as list of strings or single string.
        cwd: Working directory.
        log_file: Optional file path to write stdout/stderr.
        description: Description for logging.

    Returns:
        subprocess.CompletedProcess
    """
    if description:
        print(f"  {description}")

    if isinstance(cmd, str):
        import shlex
        result = subprocess.run(
            shlex.split(cmd), cwd=cwd, capture_output=True, text=True
        )
    else:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True
        )

    if log_file:
        with open(log_file, "w") as f:
            f.write(f"=== COMMAND ===\n{cmd}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
            f.write(f"=== RETURN CODE ===\n{result.returncode}\n")

    return result


def check_file_exists(filepath, description=""):
    """Check if a file exists and is non-empty."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False
    return True


def get_segment_residue_mapping(psf_structure):
    """
    Analyze a ParmEd structure to get residue ranges per segment.

    Args:
        psf_structure: ParmEd CharmmPsfFile or Structure object

    Returns:
        dict: {segment_id: {"start": first_resid, "end": last_resid, "count": n_residues}}
              Residue indices are 1-based (matching AMBER/cpptraj convention).
    """
    segment_info = {}
    current_resid = 1
    current_segment = None

    for res in psf_structure.residues:
        segid = res.segid
        if segid not in segment_info:
            segment_info[segid] = {
                "start": current_resid,
                "count": 0,
                "residue_names": [],
            }
        segment_info[segid]["count"] += 1
        segment_info[segid]["end"] = current_resid
        segment_info[segid]["residue_names"].append(res.name)
        current_resid += 1

    return segment_info
