#!/usr/bin/env python3
"""
Post-processing entropy corrections using Quasi-Harmonic Analysis.

Computes configurational entropy from mass-weighted covariance matrices
of atomic coordinates. Compatible with ChamberParm (CHARMM) topologies
since it only uses coordinate data, not force field terms.

Quasi-harmonic (QH) entropy from the quantum harmonic oscillator partition
function, evaluated on the eigenvalues of the mass-weighted covariance matrix:
  ΔS = S(complex) - S(receptor) - S(ligand)
  -TΔS in kcal/mol

Usage:
    Called from analyze_results.py during Phase 4, or standalone:
    python -m scripts.compute_entropy <complex_name> <rep_num> [--window 500ns]
"""
import os
import re
import json
import argparse
import subprocess
import numpy as np

from . import config
from . import utils


# =============================================================================
# PHYSICAL CONSTANTS (SI)
# =============================================================================
KB_SI = 1.380649e-23          # Boltzmann constant, J/K
HBAR_SI = 1.054571817e-34     # Reduced Planck constant, J·s
NA = 6.02214076e23            # Avogadro's number, /mol
AMU_TO_KG = 1.66053906660e-27 # kg per amu
ANG2_TO_M2 = 1e-20            # m² per Å²
CAL_TO_J = 4.184              # J per cal


# =============================================================================
# ENTROPY FORMULAS
# =============================================================================

def quasi_harmonic_entropy(eigenvalues, temperature):
    """
    Quasi-harmonic entropy using quantum harmonic oscillator formula.

    ω_i = sqrt(kBT / λ_i)
    α_i = ℏω_i / (kBT)
    S = kB * Σ [α_i/(exp(α_i) - 1) - ln(1 - exp(-α_i))]

    Reference: Andricioaei & Karplus, J. Chem. Phys. 2001, 115, 6289-6292.

    Args:
        eigenvalues: numpy array in amu·Å²
        temperature: in Kelvin

    Returns:
        S in cal/(mol·K)
    """
    kBT = KB_SI * temperature
    lambda_si = eigenvalues * AMU_TO_KG * ANG2_TO_M2

    # Filter: need λ > 0 (skip zero eigenvalues from translation/rotation/rank deficiency)
    valid = lambda_si > 1e-60
    if not np.any(valid):
        return 0.0

    lam = lambda_si[valid]

    # Frequencies: ω = sqrt(kBT / λ) [rad/s]
    omega = np.sqrt(kBT / lam)

    # Dimensionless: α = ℏω/(kBT)
    alpha = HBAR_SI * omega / kBT

    # QH entropy per mode
    # For very large α (stiff modes), use asymptotic to avoid overflow
    S_modes = np.where(
        alpha < 500,
        alpha / np.expm1(alpha) - np.log(-np.expm1(-alpha)),
        alpha * np.exp(-alpha)  # asymptotic for large α
    )

    S = KB_SI * NA * np.sum(S_modes) / CAL_TO_J  # cal/(mol·K)
    return float(S)


# =============================================================================
# CPPTRAJ INTERFACE
# =============================================================================

def _run_cpptraj_eigenvalues(prmtop, traj_file, atom_mask, rms_mask,
                              workdir, prefix, strip_mask=None):
    """
    Run cpptraj to compute mass-weighted covariance matrix eigenvalues.

    Args:
        prmtop: topology file path
        traj_file: trajectory file path
        atom_mask: atom selection for covariance (e.g., "@CA")
        rms_mask: atom selection for superposition (e.g., "@CA")
        workdir: working directory
        prefix: output file prefix
        strip_mask: optional mask to strip atoms before analysis

    Returns:
        numpy array of eigenvalues (amu·Å²), sorted descending
    """
    evecs_file = os.path.join(workdir, f"{prefix}_evecs.dat")
    script_file = os.path.join(workdir, f"{prefix}_covar.cpptraj")
    log_file = os.path.join(workdir, f"{prefix}_covar.log")

    # Build cpptraj script
    lines = [
        f"parm {prmtop}",
        f"trajin {traj_file}",
    ]
    if strip_mask:
        lines.append(f"strip {strip_mask}")
    lines.extend([
        f"rms first {rms_mask}",
        f"matrix covar name M {atom_mask} mass",
        f"diagmatrix M vecs 0 out {evecs_file}",
        "run",
    ])

    script_content = "\n".join(lines) + "\n"
    with open(script_file, "w") as f:
        f.write(script_content)

    # Run cpptraj
    cpptraj_bin = os.path.join(config.AMBERHOME, "bin", "cpptraj")
    cmd = [cpptraj_bin, "-i", script_file]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=workdir
    )

    with open(log_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"cpptraj failed for {prefix}: exit code {result.returncode}\n"
            f"Check log: {log_file}"
        )

    # Parse eigenvalues from evecs file
    eigenvalues = _parse_evecs_file(evecs_file)
    return eigenvalues


def _parse_evecs_file(filepath):
    """
    Parse cpptraj diagmatrix output file for eigenvalues.

    Format:
        Line 1: " Eigenvector file: COVAR nmodes N width W"
        Line 2: "N N"
        Lines 3-M: average structure coordinates
        Then repeating blocks of: "****" / "  mode_num  eigenvalue" / eigenvector values

    Returns:
        numpy array of eigenvalues, sorted descending
    """
    eigenvalues = []

    with open(filepath, "r") as f:
        prev_line_is_separator = False
        for line in f:
            stripped = line.strip()
            if stripped == "****":
                prev_line_is_separator = True
                continue
            if prev_line_is_separator:
                parts = stripped.split()
                if len(parts) == 2:
                    try:
                        eigenvalues.append(float(parts[1]))
                    except ValueError:
                        pass
                prev_line_is_separator = False

    if not eigenvalues:
        raise ValueError(f"No eigenvalues found in {filepath}")

    ev = np.array(eigenvalues)
    return ev  # Already sorted descending from cpptraj


# =============================================================================
# QHA TRAJECTORY PREPARATION
# =============================================================================

def _prepare_qha_trajectory(complex_name, rep_num, analysis_window, workdir):
    """
    Prepare a stride-1 dry trajectory for QHA covariance computation.

    QHA covariance matrices benefit from the maximum number of frames to
    improve the frame-to-DOF ratio and reduce rank deficiency.  MMPBSA uses
    strided trajectories (stride 3–10) because each frame requires an expensive
    PB solve, but QHA only needs coordinates.  This function creates a dry
    (water/ions stripped) trajectory with stride 1 from the original DCD,
    giving up to 10× more frames for the same analysis window.

    For windows that already use stride 1 (e.g. 100ns), the existing MMPBSA
    trajectory is returned directly — no new file is created.

    Args:
        complex_name: complex directory name
        rep_num: replica number (1-based)
        analysis_window: window name (e.g. "500ns")
        workdir: MMPBSA working directory (where output is written)

    Returns:
        tuple: (trajectory_path, n_frames_approx) on success,
               (None, None) if the DCD is unavailable or cpptraj fails.
               The caller should fall back to the strided MMPBSA trajectory.
    """
    window_cfg = config.ANALYSIS_WINDOWS[analysis_window]
    original_stride = window_cfg["stride"]
    first_frame = window_cfg["first_frame"]
    last_frame = window_cfg["last_frame"]
    n_frames_approx = last_frame - first_frame + 1

    from .prepare_trajectory import count_frames

    # If original stride is already 1, the MMPBSA trajectory has all frames
    if original_stride == 1:
        existing_traj = os.path.join(workdir, f"rep{rep_num}_dry_{analysis_window}.nc")
        if os.path.exists(existing_traj):
            n_actual = count_frames(None, existing_traj)
            return existing_traj, (n_actual if n_actual > 0 else n_frames_approx)
        return None, None

    # QHA-specific trajectory path
    qha_traj = os.path.join(workdir, f"rep{rep_num}_dry_{analysis_window}_qha.nc")

    # Reuse only a COMPLETE trajectory. A size test (exists and > 1 KB) accepts a .nc left
    # half-written by an interrupted cpptraj; the covariance matrix would then be built from
    # fewer frames than the reported n_frames, with nothing downstream able to detect it.
    if os.path.exists(qha_traj) and os.path.getsize(qha_traj) > 1024:
        n_actual = count_frames(None, qha_traj)
        if n_actual == n_frames_approx:
            return qha_traj, n_actual
        print(f"      QHA: existing stride-1 trajectory is incomplete "
              f"({n_actual} frames, expected {n_frames_approx}) - regenerating.")
        os.remove(qha_traj)

    # Find source files
    try:
        dcd_path, traj_form = utils.find_trajectory(complex_name, rep_num)
    except FileNotFoundError:
        return None, None

    solvated_prmtop = os.path.join(workdir, "solvated.prmtop")
    if traj_form == "solvated" and not os.path.exists(solvated_prmtop):
        return None, None
    dry_prmtop = utils.dry_topology(complex_name, rep_num, workdir)
    if traj_form == "dry" and not os.path.exists(dry_prmtop):
        return None, None

    # Build cpptraj script: read DCD with stride 1, strip water/ions, write dry NetCDF
    script_file = os.path.join(workdir, f"qha_prep_{analysis_window}.cpptraj")
    log_file = os.path.join(workdir, f"qha_prep_{analysis_window}.log")

    if traj_form == "solvated":
        script_lines = [
            f"parm {solvated_prmtop}",
            f"trajin {dcd_path} {first_frame} {last_frame} 1",
            "autoimage",
            f"strip {config.STRIP_MASK_CPPTRAJ} nobox",
            f"trajout {qha_traj} netcdf",
            "run",
        ]
    else:
        script_lines = [
            f"parm {dry_prmtop}",
            f"trajin {dcd_path} {first_frame} {last_frame} 1",
            f"trajout {qha_traj} netcdf",
            "run",
        ]
    with open(script_file, "w") as f:
        f.write("\n".join(script_lines) + "\n")

    cpptraj_bin = os.path.join(config.AMBERHOME, "bin", "cpptraj")
    result = subprocess.run(
        [cpptraj_bin, "-i", script_file],
        capture_output=True, text=True, cwd=workdir
    )

    with open(log_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0 or not os.path.exists(qha_traj):
        print(f"      WARNING: QHA trajectory preparation failed. "
              f"Falling back to strided trajectory. Check {log_file}")
        return None, None

    # Report the frames actually written, not the frames the window implies, so that
    # n_frames in the ensemble JSON and the frame/DOF ratio warning describe the real input.
    n_written = count_frames(None, qha_traj)
    if n_written <= 0:
        n_written = n_frames_approx
    elif n_written != n_frames_approx:
        print(f"      WARNING: QHA trajectory has {n_written} frames, "
              f"expected {n_frames_approx}.")
    return qha_traj, n_written


# =============================================================================
# MAIN ENTROPY COMPUTATION
# =============================================================================

def compute_qha_entropy(complex_name, rep_num, analysis_window,
                         atom_selection="ca", temperature=None):
    """
    Compute the quasi-harmonic entropy for a single replica.

    Runs cpptraj to compute covariance matrices for complex, receptor and
    ligand, then applies the quasi-harmonic formula to their eigenvalues.

    Uses a stride-1 trajectory when available (prepared from the original DCD)
    to maximize the frame-to-DOF ratio for the covariance matrix.  Falls back
    to the strided MMPBSA trajectory if the DCD is not accessible.

    Args:
        complex_name: complex directory name
        rep_num: replica number (1-based)
        analysis_window: window name (e.g. "500ns", "100ns", "300ns")
        atom_selection: "ca" for @CA, "backbone" for @CA,C,N,O
        temperature: K (default: from config)

    Returns:
        dict with entropy results, or None on failure
    """
    if temperature is None:
        temperature = config.QHA_PARAMS["temperature"]

    workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, analysis_window)

    # Required files (topology and metadata — trajectory resolved below)
    prmtop = os.path.join(workdir, "complex.prmtop")
    metadata_file = os.path.join(workdir, "topology_metadata.json")

    for f_path in [prmtop, metadata_file]:
        if not os.path.exists(f_path):
            print(f"    WARNING: Missing file for QHA: {f_path}")
            return None

    # Trajectory: prefer stride-1 QHA trajectory (more frames → better covariance).
    # Falls back to the strided MMPBSA trajectory if DCD is unavailable.
    qha_traj, n_qha_frames = _prepare_qha_trajectory(
        complex_name, rep_num, analysis_window, workdir)
    if qha_traj:
        traj_file = qha_traj
        window_cfg = config.ANALYSIS_WINDOWS[analysis_window]
        if window_cfg["stride"] > 1:
            print(f"      QHA: Using stride-1 trajectory (~{n_qha_frames} frames "
                  f"vs ~{n_qha_frames // window_cfg['stride']} with MMPBSA stride)")
        else:
            print(f"      QHA: Using trajectory (~{n_qha_frames} frames)")
    else:
        traj_file = os.path.join(workdir, f"rep{rep_num}_dry_{analysis_window}.nc")
        if not os.path.exists(traj_file):
            print(f"    WARNING: No trajectory available for QHA: {traj_file}")
            return None
        print(f"      QHA: Using MMPBSA trajectory (strided, DCD unavailable)")

    # Read metadata for receptor/ligand masks
    with open(metadata_file) as f:
        metadata = json.load(f)

    rec_range = metadata["receptor_residue_range"]  # e.g., "1-375"
    lig_range = metadata["ligand_residue_range"]    # e.g., "376-386"

    # Atom selection mask
    if atom_selection == "ca":
        atom_mask = "@CA"
    elif atom_selection == "backbone":
        atom_mask = "@CA,C,N,O"
    else:
        atom_mask = atom_selection

    # Strip masks for isolating receptor/ligand
    ligand_strip = f":{lig_range}"     # strip ligand → keep receptor
    receptor_strip = f":{rec_range}"   # strip receptor → keep ligand

    try:
        # Complex eigenvalues
        print(f"      QHA: Computing complex covariance ({atom_mask})...")
        ev_complex = _run_cpptraj_eigenvalues(
            prmtop, traj_file, atom_mask, atom_mask,
            workdir, f"qha_complex_{analysis_window}"
        )

        # Receptor eigenvalues (strip ligand)
        print(f"      QHA: Computing receptor covariance...")
        ev_receptor = _run_cpptraj_eigenvalues(
            prmtop, traj_file, atom_mask, atom_mask,
            workdir, f"qha_receptor_{analysis_window}",
            strip_mask=ligand_strip
        )

        # Ligand eigenvalues (strip receptor)
        print(f"      QHA: Computing ligand covariance...")
        ev_ligand = _run_cpptraj_eigenvalues(
            prmtop, traj_file, atom_mask, atom_mask,
            workdir, f"qha_ligand_{analysis_window}",
            strip_mask=receptor_strip
        )

    except Exception as e:
        print(f"      WARNING: QHA cpptraj failed: {e}")
        return None

    # Check frame/DOF ratio for covariance matrix quality
    n_frames_qha = n_qha_frames if qha_traj else None
    if n_frames_qha is not None:
        for label, n_ev in [("complex", len(ev_complex)),
                            ("receptor", len(ev_receptor)),
                            ("ligand", len(ev_ligand))]:
            dof = n_ev  # eigenvalues = 3 * n_atoms = DOF
            ratio = n_frames_qha / dof if dof > 0 else float('inf')
            if ratio < 5:
                print(f"      WARNING: Low frame/DOF ratio for {label}: "
                      f"{n_frames_qha}/{dof} = {ratio:.1f} "
                      f"(recommended >= 10). QHA entropy may be unreliable.")

    # Count positive eigenvalues (non-zero modes)
    pos_threshold = 1e-10 * max(np.max(np.abs(ev_complex)), 1e-30)
    n_pos_com = int(np.sum(ev_complex > pos_threshold))
    n_pos_rec = int(np.sum(ev_receptor > pos_threshold))
    n_pos_lig = int(np.sum(ev_ligand > pos_threshold))

    # Filter to positive eigenvalues only for entropy calculation
    ev_complex_pos = ev_complex[ev_complex > pos_threshold]
    ev_receptor_pos = ev_receptor[ev_receptor > pos_threshold]
    ev_ligand_pos = ev_ligand[ev_ligand > pos_threshold]

    S_qh_com = quasi_harmonic_entropy(ev_complex_pos, temperature)
    S_qh_rec = quasi_harmonic_entropy(ev_receptor_pos, temperature)
    S_qh_lig = quasi_harmonic_entropy(ev_ligand_pos, temperature)

    # Delta entropy: ΔS = S(complex) - S(receptor) - S(ligand)
    dS_qh = S_qh_com - S_qh_rec - S_qh_lig

    # -TΔS in kcal/mol (positive value = entropy penalty, unfavorable)
    TdS_qh = -temperature * dS_qh / 1000.0

    results = {
        "replica": rep_num,
        "temperature": temperature,
        "atom_selection": atom_selection,
        "n_frames": n_qha_frames if qha_traj else None,
        "n_eigenvalues": {
            "complex": len(ev_complex),
            "receptor": len(ev_receptor),
            "ligand": len(ev_ligand),
        },
        "n_positive_eigenvalues": {
            "complex": n_pos_com,
            "receptor": n_pos_rec,
            "ligand": n_pos_lig,
        },
        "quasi_harmonic": {
            "S_complex": S_qh_com,
            "S_receptor": S_qh_rec,
            "S_ligand": S_qh_lig,
            "dS": dS_qh,
            "TdS": TdS_qh,
        },
    }

    print(f"      Quasi-Harm: -TΔS = {TdS_qh:+.2f} kcal/mol "
          f"(ΔS = {dS_qh:.2f} cal/mol/K)")

    return results


def aggregate_qha_results(qha_results_list, temperature=None):
    """
    Aggregate QHA entropy results across replicas.

    Args:
        qha_results_list: list of dicts from compute_qha_entropy()
        temperature: K (default: from config)

    Returns:
        dict with ensemble QHA statistics
    """
    if temperature is None:
        temperature = config.QHA_PARAMS["temperature"]

    if not qha_results_list:
        return None

    n = len(qha_results_list)
    if n == 1:
        print("  WARNING: Only 1 replica has valid QHA results. "
              "Entropy SEM will be reported as 0.0 (no inter-replica "
              "variance estimate possible). At least 2 replicas are "
              "needed for meaningful uncertainty estimates.")

    # Extract per-replica values
    TdS_qh = np.array([r["quasi_harmonic"]["TdS"] for r in qha_results_list])

    ensemble = {
        "n_replicas": n,
        "atom_selection": qha_results_list[0]["atom_selection"],
        "temperature": temperature,
        "quasi_harmonic": {
            "TdS_mean": float(np.mean(TdS_qh)),
            "TdS_std": float(np.std(TdS_qh, ddof=1)) if n > 1 else 0.0,
            "TdS_sem": float(np.std(TdS_qh, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "per_replica_TdS": TdS_qh.tolist(),
        },
    }

    return ensemble


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute quasi-harmonic entropy for a single replica"
    )
    parser.add_argument("complex_name", help="Complex directory name")
    parser.add_argument("rep_num", type=int, help="Replica number (1-based)")
    parser.add_argument("--window", choices=config.ALL_WINDOW_NAMES, default=config.REFERENCE_WINDOW,
                        help=f"Analysis window (default: {config.REFERENCE_WINDOW})")
    parser.add_argument("--atoms", choices=["ca", "backbone"], default="ca",
                        help="Atom selection for covariance (default: ca)")
    args = parser.parse_args()

    result = compute_qha_entropy(args.complex_name, args.rep_num,
                                  args.window, args.atoms)
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
