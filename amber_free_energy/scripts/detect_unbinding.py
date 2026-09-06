#!/usr/bin/env python3
"""
Peptide-terminal unbinding detection pipeline (standalone).

Detects the simulation time at which at least one peptide terminal dissociates
from the HLA binding groove in 500 ns MD trajectories. Produces per-replica
event times and an aggregated per-complex summary over 10 replicas.

A terminal is scored as unbound when
      min_heavy_atom_distance(terminal, HLA) > D_CUT  [5.0 Å]
   OR n_contacts(terminal, HLA; < 4.5 Å) < F_CONTACT * n_contacts_reference
      (F_CONTACT 0.3, reference = median contacts over the first 5 ns)
   sustained for >= sustain_frames (17 frames ~= 2 ns at 0.12 ns/frame).

Each terminal is represented by its two outermost residues, i.e. the terminal
residue together with its adjacent neighbour, so that partial unzipping is
detected rather than strict terminal detachment.

The script runs as a pure post-processing step on completed MD trajectories.
It does NOT modify the MMPBSA pipeline (`run_amber_mmpbsa.py`). It only requires
that Phase 1 (topology conversion) has been run so the dry AMBER prmtop exists.

Usage
-----
    # All complexes in a list file
    python scripts/detect_unbinding.py list_dirs_all.txt

    # Single complex
    python scripts/detect_unbinding.py --complex flt3_d835y_neo_a_0201_yimsdsnyv

    # Single replica
    python scripts/detect_unbinding.py --complex flt3_d835y_neo_a_0201_yimsdsnyv --rep 3

    # Threshold tuning
    python scripts/detect_unbinding.py --complex X \
        --d-cut 5.0 --f-contact 0.3 --sustain-ns 2.0

    # Parallelise across replicas
    python scripts/detect_unbinding.py list_dirs_all.txt --n-workers 8

    # Force recomputation (overwrite existing outputs)
    python scripts/detect_unbinding.py --complex X --force

    # Typically used
    python scripts/detect_unbinding.py list_dirs_all.txt --n-workers 8 --force

Outputs (under results/<complex>/), suffixed with `dist-contact_residue-neighbor`:
  - unbinding_rep{N}_timeseries_{suffix}.npz    # per-frame arrays
  - unbinding_summary_{suffix}.json             # per-replica + aggregate
  - unbinding_report_{suffix}.md                # Markdown report
  - unbinding_plot_{complex}_{suffix}.png       # 10-panel + survival curve
"""

# Thread-limiting env vars must be set before importing pytraj / numpy linear-
# algebra backends when --n-workers > 1, otherwise each worker over-subscribes.
# Safe no-op when the user runs serially.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import os
import sys
import math
import argparse
import warnings
import traceback
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Make scripts.config / scripts.utils importable whether run as a module or a
# script. When invoked as `python scripts/detect_unbinding.py`, __file__ is in
# scripts/ and the parent directory (project root) must be on sys.path so that
# `from scripts import config, utils` resolves.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts import config, utils  # noqa: E402

# pytraj is imported lazily inside workers so process-pool spawns inherit the
# thread-limiting env vars.


# ---------------------------------------------------------------------------
# Defaults and parameter objects
# ---------------------------------------------------------------------------

DEFAULTS = {
    "d_cut": 5.0,               # Å, min heavy-atom distance threshold
    "contact_cutoff": 4.5,      # Å, heavy-atom contact radius
    "f_contact": 0.3,           # fraction of reference contacts threshold
    "sustain_ns": 2.0,          # ns — sustain-window duration
    "start_ns": float(config.EQUILIBRATION_NS),  # ns skipped from event detection
    "reference_window_ns": 5.0, # ns used to derive n_contacts_reference
}


@dataclass
class UnbindingParams:
    """User-selected parameters for a single run.

    The values are stored inside each output JSON so downstream scripts can
    verify reproducibility and warn on parameter mismatches.
    """
    d_cut: float = DEFAULTS["d_cut"]
    contact_cutoff: float = DEFAULTS["contact_cutoff"]
    f_contact: float = DEFAULTS["f_contact"]
    sustain_ns: float = DEFAULTS["sustain_ns"]
    start_ns: float = DEFAULTS["start_ns"]
    reference_window_ns: float = DEFAULTS["reference_window_ns"]

    # Terminal scope is the terminal residue plus its adjacent neighbour, and the
    # criterion is minimum anchor-groove distance combined with a normalized contact
    # count. Both are fixed; the suffix names every output file and is kept verbatim
    # so that filenames remain comparable with previously published results.
    scope = "residue+neighbor"
    suffix = "dist-contact_residue-neighbor"


def _sustain_frames(params):
    """Convert the sustain-window duration (ns) to an integer frame count."""
    return max(1, int(round(params.sustain_ns / config.TIME_PER_FRAME_NS)))


def _start_frame(params):
    """Convert --start-ns (event-detection skip) to a 0-based frame index."""
    return max(0, int(round(params.start_ns / config.TIME_PER_FRAME_NS)))


def _reference_frames(params):
    """Number of frames (from the start) to average for the reference state."""
    return max(1, int(round(params.reference_window_ns / config.TIME_PER_FRAME_NS)))


# ---------------------------------------------------------------------------
# Topology / residue-range helpers
# ---------------------------------------------------------------------------

def _load_topology_metadata(complex_name, rep_num):
    """Load the dry-topology metadata JSON written by Phase 1.

    Args:
        complex_name (str): Complex directory name.
        rep_num (int): Replica number (1-based).

    Returns:
        dict: Parsed metadata. Requires keys ``ligand_residue_range`` and
            ``receptor_residue_range``.

    Raises:
        FileNotFoundError: When topology conversion has not been run for this
            replica (Phase 1 output missing).
    """
    meta_path = os.path.join(
        utils.get_mmpbsa_workdir(complex_name, rep_num, "500ns"),
        "topology_metadata.json",
    )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"topology_metadata.json not found for {complex_name} rep_{rep_num}.\n"
            f"  Expected at: {meta_path}\n"
            f"  Fix: run Phase 1 first:\n"
            f"    python run_amber_mmpbsa.py --complex {complex_name} "
            f"--replicas {rep_num} --step topology"
        )
    return utils.load_metadata(meta_path)


def _parse_residue_range(s):
    """Parse a 'start-end' string into (start, end) ints (1-based, inclusive)."""
    a, b = s.split("-")
    return int(a), int(b)


def _peptide_terminal_residues(meta, scope):
    """Return residue numbers (1-based) that define each peptide terminal.

    Args:
        meta (dict): topology_metadata.json contents.
        scope (str): Either "residue" or "residue+neighbor".

    Returns:
        dict with keys ``nterm_residues``, ``cterm_residues`` (lists of ints),
        plus ``peptide_start``, ``peptide_end`` for reference.
    """
    start, end = _parse_residue_range(meta["ligand_residue_range"])
    if end < start:
        raise ValueError(
            f"Invalid ligand residue range '{meta['ligand_residue_range']}'"
        )
    if scope == "residue":
        nterm = [start]
        cterm = [end]
    elif scope == "residue+neighbor":
        # Include the two outermost residues on each side. For a 9-mer (start..end)
        # this gives Nterm = {1, 2} and Cterm = {8, 9}. For peptides shorter than
        # 4 residues this would overlap — guard against it.
        if end - start < 3:
            raise ValueError(
                f"Peptide too short ({end - start + 1} residues) for scope "
                f"'residue+neighbor'; use 'residue' instead."
            )
        nterm = [start, start + 1]
        cterm = [end - 1, end]
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return {
        "nterm_residues": nterm,
        "cterm_residues": cterm,
        "peptide_start": start,
        "peptide_end": end,
    }


# ---------------------------------------------------------------------------
# Trajectory loading (pytraj, streaming)
# ---------------------------------------------------------------------------

def _find_clean_topology(complex_name, rep_num):
    """Locate the dry AMBER prmtop produced by Phase 1 topology conversion."""
    workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, "500ns")
    prmtop = os.path.join(workdir, "complex.prmtop")
    if not os.path.exists(prmtop):
        raise FileNotFoundError(
            f"complex.prmtop not found for {complex_name} rep_{rep_num}.\n"
            f"  Expected at: {prmtop}\n"
            f"  Fix: run Phase 1 topology conversion first."
        )
    return prmtop


def _find_solvated_topology(complex_name, rep_num):
    """Locate the solvated AMBER prmtop matching the original wrapped DCD."""
    workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, "500ns")
    prmtop = os.path.join(workdir, "solvated.prmtop")
    if not os.path.exists(prmtop):
        raise FileNotFoundError(
            f"solvated.prmtop not found for {complex_name} rep_{rep_num}.\n"
            f"  Expected at: {prmtop}"
        )
    return prmtop




def _prepare_clean_trajectory(complex_name, rep_num, force=False):
    """Create a water-stripped, HLA-anchor-autoimaged NetCDF trajectory.

    The resulting NetCDF preserves the full 4166-frame resolution of the
    original wrapped DCD, is imaged so that the HLA receptor is always in the
    central cell (crucial for correct min_dist under PBC when the peptide
    partially exits the pocket), and contains only protein atoms matching
    the dry `complex.prmtop` topology.

    Args:
        complex_name (str): Complex directory name.
        rep_num (int): Replica number (1-based).
        force (bool): If True, overwrite an existing clean trajectory.

    Returns:
        str: Path to the clean NetCDF trajectory.
    """
    rep_dir = utils.get_replica_dir(complex_name, rep_num)
    out_dir = os.path.join(rep_dir, "unbinding")
    os.makedirs(out_dir, exist_ok=True)

    clean_nc = os.path.join(out_dir, f"rep{rep_num}_clean_500ns.nc")
    if utils.check_file_exists(clean_nc) and not force:
        return clean_nc

    dcd_path, traj_form = utils.find_trajectory(complex_name, rep_num)
    solvated_prmtop = (_find_solvated_topology(complex_name, rep_num)
                       if traj_form == "solvated" else None)

    # HLA anchor for autoimage: APRO + BPRO residues = 1..375 in the dry topology,
    # but the solvated topology has the same protein ordering (protein first,
    # water/ions after). Use the receptor_residue_range from metadata as the
    # authoritative anchor mask.
    meta = _load_topology_metadata(complex_name, rep_num)
    r_start, r_end = _parse_residue_range(meta["receptor_residue_range"])

    cpptraj_input = os.path.join(out_dir, f"prep_clean_rep{rep_num}.cpptraj")
    cpptraj_log = os.path.join(out_dir, f"prep_clean_rep{rep_num}.log")
    with open(cpptraj_input, "w") as f:
        if traj_form == "solvated":
            # Anchor the reimaging on the receptor so a peptide leaving the groove is
            # not folded back across the boundary.
            f.write(f"parm {solvated_prmtop}\n")
            f.write(f"trajin {dcd_path}\n")
            f.write(f"autoimage anchor :{r_start}-{r_end}\n")
            f.write(f"strip {config.STRIP_MASK_CPPTRAJ} nobox\n")
        else:
            # The dry trajectory is protein-only and already whole. It differs from
            # the receptor-anchored strip by a rigid-body superposition, which leaves
            # every inter-atomic distance measured here unchanged.
            f.write(f"parm {utils.dry_topology(complex_name, rep_num)}\n")
            f.write(f"trajin {dcd_path}\n")
        f.write(f"trajout {clean_nc} netcdf\n")
        f.write("run\n")

    cpptraj_bin = os.path.join(config.AMBERHOME, "bin", "cpptraj")
    res = utils.run_command(
        [cpptraj_bin, "-i", cpptraj_input],
        log_file=cpptraj_log,
        description=f"  cpptraj clean trajectory: rep_{rep_num}",
    )
    if res.returncode != 0 or not utils.check_file_exists(clean_nc):
        raise RuntimeError(
            f"cpptraj failed producing clean trajectory. See log: {cpptraj_log}"
        )
    return clean_nc


def _open_trajectory(clean_nc, complex_prmtop):
    """Open a clean NetCDF trajectory with pytraj as a streaming iterator."""
    import pytraj as pt  # local: avoid import cost / threading side-effects globally
    traj = pt.iterload(clean_nc, top=complex_prmtop)
    return pt, traj


# ---------------------------------------------------------------------------
# Atom selections
# ---------------------------------------------------------------------------

def _residues_mask(residues):
    """Build a cpptraj-style residue mask from a list of residue numbers.

    E.g. [376, 377] -> ":376,377".
    """
    return ":" + ",".join(str(r) for r in residues)


def _heavy_atoms_for_residues(topology, residues):
    """Return the 0-based atom indices of heavy (non-H) atoms for a residue set.

    Args:
        topology: pytraj Topology.
        residues (list[int]): 1-based residue numbers (matching cpptraj
            convention).

    Returns:
        np.ndarray[int]: Atom indices in the topology.
    """
    mask = _residues_mask(residues) + "&!@H="
    idx = topology.select(mask)
    if len(idx) == 0:
        raise RuntimeError(f"No heavy atoms found for mask '{mask}'.")
    return np.asarray(idx, dtype=np.int64)


def _receptor_heavy_atoms(topology, r_start, r_end):
    """Return the 0-based atom indices of HLA heavy atoms."""
    mask = f":{r_start}-{r_end}&!@H="
    idx = topology.select(mask)
    if len(idx) == 0:
        raise RuntimeError(f"No HLA heavy atoms found for mask '{mask}'.")
    return np.asarray(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def _min_dist_and_contacts(term_coords, recv_coords, contact_cutoff):
    """Return (min heavy-atom distance, number of contacts < cutoff) for a single frame.

    Uses a vectorised (N_term × N_receptor × 3) distance tensor. For the sizes
    involved (~10–20 × ~4000 atoms) this is ~0.5 ms/frame in numpy and much
    faster than a KDTree rebuild per frame.
    """
    diff = term_coords[:, None, :] - recv_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    return float(dists.min()), int((dists < contact_cutoff).sum())


# ---------------------------------------------------------------------------
# Time-series computation (one pass over the trajectory)
# ---------------------------------------------------------------------------

def _compute_series(traj, nterm_idx, cterm_idx, recv_idx, params):
    """Compute per-frame metric arrays by streaming the trajectory once.

    Returns:
        dict of numpy arrays: time_ns, min_dist_nterm, min_dist_cterm,
        n_contacts_nterm, n_contacts_cterm.
    """
    # First pass to count frames (iterload objects expose n_frames cheaply).
    try:
        n_frames = int(traj.n_frames)
    except Exception:
        n_frames = sum(1 for _ in traj)  # fallback; rarely hit

    ts = {
        "time_ns": np.arange(n_frames) * config.TIME_PER_FRAME_NS,
        "min_dist_nterm": np.empty(n_frames, dtype=np.float32),
        "min_dist_cterm": np.empty(n_frames, dtype=np.float32),
        "n_contacts_nterm": np.empty(n_frames, dtype=np.int32),
        "n_contacts_cterm": np.empty(n_frames, dtype=np.int32),
    }
    cutoff = params.contact_cutoff
    for i, frame in enumerate(traj):
        xyz = np.asarray(frame.xyz, dtype=np.float64)
        nterm_c = xyz[nterm_idx]
        cterm_c = xyz[cterm_idx]
        recv_c = xyz[recv_idx]

        d_n, k_n = _min_dist_and_contacts(nterm_c, recv_c, cutoff)
        d_c, k_c = _min_dist_and_contacts(cterm_c, recv_c, cutoff)
        ts["min_dist_nterm"][i] = d_n
        ts["min_dist_cterm"][i] = d_c
        ts["n_contacts_nterm"][i] = k_n
        ts["n_contacts_cterm"][i] = k_c

    return ts


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def _find_sustained_episodes(mask, sustain_frames):
    """Return [(start, end_exclusive), ...] runs of True with length >= sustain_frames.

    Both ends of the returned intervals are 0-based indices into the mask;
    ``end_exclusive`` is one past the last True frame of the run.
    """
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)
            if (e - s) >= sustain_frames]


def _event_metrics(unbound_mask, sustain_frames, start_frame, time_ns):
    """Derive the four unbinding metrics from a per-frame boolean mask.

    Args:
        unbound_mask (np.ndarray[bool]): True when the terminal meets the
            unbinding condition at that frame.
        sustain_frames (int): Min length of an unbinding episode to count.
        start_frame (int): Frames before this index are not used for event
            detection (equilibration skip).
        time_ns (np.ndarray[float]): Simulation time per frame.

    Returns:
        dict with keys:
            t_first_unbind_ns: first frame time of the earliest qualifying
                episode starting at or after start_frame, or None if none.
            final_state: "unbound" if the last sustain_frames frames are all
                unbound, else "bound".
            fraction_time_unbound: over [start_frame, end], fraction of frames
                with unbound_mask True.
            n_unbinding_episodes: number of distinct qualifying episodes
                separated by at least sustain_frames of bound frames.
    """
    n = unbound_mask.size
    if n == 0:
        return {
            "t_first_unbind_ns": None,
            "final_state": "bound",
            "fraction_time_unbound": 0.0,
            "n_unbinding_episodes": 0,
        }

    # Episodes over the whole trajectory (for episode count and final state).
    all_episodes = _find_sustained_episodes(unbound_mask, sustain_frames)

    # First-event detection restricted to frames >= start_frame.
    first_t = None
    for (s, e) in all_episodes:
        if s >= start_frame:
            first_t = float(time_ns[s])
            break

    # Final state: are the trailing sustain_frames frames all unbound?
    tail = unbound_mask[-sustain_frames:] if sustain_frames <= n else unbound_mask
    final_state = "unbound" if bool(tail.all()) else "bound"

    # Fraction of time unbound over the analysed range.
    analysed = unbound_mask[start_frame:]
    frac = float(analysed.mean()) if analysed.size else 0.0

    return {
        "t_first_unbind_ns": first_t,
        "final_state": final_state,
        "fraction_time_unbound": frac,
        "n_unbinding_episodes": len(all_episodes),
    }


def _detect_events(timeseries, params):
    """Apply the unbinding criterion and return per-terminal events.

    Returns:
        dict with keys "nterm", "cterm", "combined" (combined = earliest of the
        two terminals), plus sanity diagnostics.
    """
    t = timeseries["time_ns"]
    sustain = _sustain_frames(params)
    start = _start_frame(params)
    n_ref = _reference_frames(params)

    # Reference contact counts (median over first reference_window_ns).
    ref_slice = slice(0, n_ref)
    n_ref_nterm = float(np.median(timeseries["n_contacts_nterm"][ref_slice]))
    n_ref_cterm = float(np.median(timeseries["n_contacts_cterm"][ref_slice]))

    thresh_n = max(1.0, params.f_contact * n_ref_nterm)
    thresh_c = max(1.0, params.f_contact * n_ref_cterm)
    unbound_n = (
        (timeseries["min_dist_nterm"] > params.d_cut)
        | (timeseries["n_contacts_nterm"] < thresh_n)
    )
    unbound_c = (
        (timeseries["min_dist_cterm"] > params.d_cut)
        | (timeseries["n_contacts_cterm"] < thresh_c)
    )

    ev_n = _event_metrics(unbound_n, sustain, start, t)
    ev_c = _event_metrics(unbound_c, sustain, start, t)

    # Combined event = earlier of the two terminals, or None if both stable.
    times = [ev_n["t_first_unbind_ns"], ev_c["t_first_unbind_ns"]]
    non_none = [x for x in times if x is not None]
    combined_t = min(non_none) if non_none else None
    if combined_t is None:
        terminal_source = None
    elif ev_n["t_first_unbind_ns"] == combined_t and ev_c["t_first_unbind_ns"] == combined_t:
        terminal_source = "both"
    elif ev_n["t_first_unbind_ns"] == combined_t:
        terminal_source = "nterm"
    else:
        terminal_source = "cterm"

    return {
        "nterm": ev_n,
        "cterm": ev_c,
        "combined": {
            "t_first_unbind_ns": combined_t,
            "terminal_source": terminal_source,
        },
        "reference": {
            "n_contacts_ref_nterm": n_ref_nterm,
            "n_contacts_ref_cterm": n_ref_cterm,
        },
    }


# ---------------------------------------------------------------------------
# Per-replica driver
# ---------------------------------------------------------------------------

def _timeseries_path(complex_name, rep_num, params):
    return os.path.join(
        utils.get_results_dir(complex_name),
        f"unbinding_rep{rep_num}_timeseries_{params.suffix}.npz",
    )


def _sanity_check(timeseries):
    """Return a dict flagging suspect initial bound states (for user awareness)."""
    warns = []
    init_n = float(timeseries["min_dist_nterm"][0])
    init_c = float(timeseries["min_dist_cterm"][0])
    init_k_n = int(timeseries["n_contacts_nterm"][0])
    init_k_c = int(timeseries["n_contacts_cterm"][0])
    if init_n > 3.0:
        warns.append(f"initial N-term min_dist = {init_n:.2f} Å (> 3.0 Å)")
    if init_c > 3.0:
        warns.append(f"initial C-term min_dist = {init_c:.2f} Å (> 3.0 Å)")
    if init_k_n < 5:
        warns.append(f"initial N-term contacts = {init_k_n} (< 5)")
    if init_k_c < 5:
        warns.append(f"initial C-term contacts = {init_k_c} (< 5)")
    return {
        "initial_min_dist_nterm": init_n,
        "initial_min_dist_cterm": init_c,
        "initial_n_contacts_nterm": init_k_n,
        "initial_n_contacts_cterm": init_k_c,
        "warnings": warns,
    }


def analyze_replica(complex_name, rep_num, params, force=False):
    """Run the unbinding analysis for a single replica.

    Writes the per-frame time series to
    ``results/<complex>/unbinding_rep{rep_num}_timeseries_{suffix}.npz`` and
    returns the per-replica metrics dict used by the per-complex aggregator.

    Args:
        complex_name (str): Complex directory name.
        rep_num (int): Replica number (1-based).
        params (UnbindingParams): User-selected run parameters.
        force (bool): If True, overwrite existing time-series files.

    Returns:
        dict: Per-replica metrics (events, sanity checks, n_frames, replica).
    """
    results_dir = utils.get_results_dir(complex_name)
    os.makedirs(results_dir, exist_ok=True)

    # Skip if output exists and --force not set; still return re-loaded metrics
    # so the aggregator can proceed.
    ts_path = _timeseries_path(complex_name, rep_num, params)
    if utils.check_file_exists(ts_path) and not force:
        print(f"  [rep_{rep_num}] cached time series found, reusing.")
        data = np.load(ts_path, allow_pickle=False)
        timeseries = {k: data[k] for k in data.files}
        events = _detect_events(timeseries, params)
        sanity = _sanity_check(timeseries)
        return {
            "replica": rep_num,
            "n_frames": int(timeseries["time_ns"].size),
            "events": events,
            "sanity": sanity,
            "cached": True,
        }

    # Fresh computation
    meta = _load_topology_metadata(complex_name, rep_num)
    r_start, r_end = _parse_residue_range(meta["receptor_residue_range"])
    p_start, p_end = _parse_residue_range(meta["ligand_residue_range"])
    terminals = _peptide_terminal_residues(meta, params.scope)

    clean_nc = _prepare_clean_trajectory(complex_name, rep_num, force=force)
    complex_prmtop = _find_clean_topology(complex_name, rep_num)

    pt, traj = _open_trajectory(clean_nc, complex_prmtop)
    topology = traj.top

    nterm_idx = _heavy_atoms_for_residues(topology, terminals["nterm_residues"])
    cterm_idx = _heavy_atoms_for_residues(topology, terminals["cterm_residues"])
    recv_idx = _receptor_heavy_atoms(topology, r_start, r_end)

    print(f"  [rep_{rep_num}] computing time series "
          f"(scope={params.scope}, n_frames={traj.n_frames})...")
    timeseries = _compute_series(traj, nterm_idx, cterm_idx, recv_idx, params)

    np.savez_compressed(ts_path, **timeseries)
    events = _detect_events(timeseries, params)
    sanity = _sanity_check(timeseries)

    if sanity["warnings"]:
        print(f"  [rep_{rep_num}] sanity warnings: " + "; ".join(sanity["warnings"]))

    return {
        "replica": rep_num,
        "n_frames": int(timeseries["time_ns"].size),
        "events": events,
        "sanity": sanity,
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Per-complex aggregation
# ---------------------------------------------------------------------------

def _aggregate(per_replica, params):
    """Compute cross-replica summary statistics from the per-replica metrics."""
    n = len(per_replica)

    def _combined_times():
        return [r["events"]["combined"]["t_first_unbind_ns"]
                for r in per_replica]

    times = _combined_times()
    non_none = [t for t in times if t is not None]
    n_event = len(non_none)
    frac_unbound = n_event / n if n else 0.0

    t_mean = float(np.mean(non_none)) if non_none else None
    t_sem = (float(np.std(non_none, ddof=1) / math.sqrt(len(non_none)))
             if len(non_none) > 1 else (0.0 if non_none else None))
    t_med = float(np.median(non_none)) if non_none else None

    # Terminal breakdown: N-only / C-only / both
    n_only = c_only = both = 0
    for r in per_replica:
        tn = r["events"]["nterm"]["t_first_unbind_ns"]
        tc = r["events"]["cterm"]["t_first_unbind_ns"]
        if tn is not None and tc is not None:
            both += 1
        elif tn is not None:
            n_only += 1
        elif tc is not None:
            c_only += 1

    frac_time_unbound = [
        max(r["events"]["nterm"]["fraction_time_unbound"],
            r["events"]["cterm"]["fraction_time_unbound"])
        for r in per_replica
    ]
    mean_frac_unbound = float(np.mean(frac_time_unbound)) if n else 0.0

    # Kaplan-Meier-style empirical survival curve (time grid = union of event times)
    if non_none:
        event_times = np.sort(np.array(non_none))
        n_at_risk = n
        surv = []
        censored = n - n_event
        # Greenwood variance accumulator
        var_sum = 0.0
        prev_t = 0.0
        surv.append({"t_ns": prev_t, "survival": 1.0, "lower": 1.0, "upper": 1.0})
        s = 1.0
        for t_ev in event_times:
            d = 1  # one event at this time (times are unique in practice)
            p = (n_at_risk - d) / n_at_risk
            s *= p
            var_sum += d / (n_at_risk * (n_at_risk - d)) if n_at_risk > d else 0.0
            se = s * math.sqrt(var_sum)
            surv.append({"t_ns": float(t_ev), "survival": s,
                         "lower": max(0.0, s - 1.96 * se),
                         "upper": min(1.0, s + 1.96 * se)})
            n_at_risk -= 1
        # Final point at 500 ns if any replicas remain bound
        if censored:
            surv.append({"t_ns": float(config.TOTAL_SIM_TIME_NS),
                         "survival": s,
                         "lower": surv[-1]["lower"],
                         "upper": surv[-1]["upper"]})
    else:
        surv = [
            {"t_ns": 0.0, "survival": 1.0, "lower": 1.0, "upper": 1.0},
            {"t_ns": float(config.TOTAL_SIM_TIME_NS),
             "survival": 1.0, "lower": 1.0, "upper": 1.0},
        ]

    return {
        "n_replicas_total": n,
        "n_replicas_with_event": n_event,
        "fraction_unbound": frac_unbound,
        "t_first_unbind_mean_ns": t_mean,
        "t_first_unbind_sem_ns": t_sem,
        "t_first_unbind_median_ns": t_med,
        "mean_fraction_time_unbound": mean_frac_unbound,
        "terminal_breakdown": {
            "nterm_only": n_only,
            "cterm_only": c_only,
            "both": both,
            "neither": n - (n_only + c_only + both),
        },
        "survival_curve": surv,
    }


def _plot_complex(complex_name, per_replica, aggregate, params, out_path):
    """Write a diagnostic PNG with per-replica traces + survival curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plot.")
        return None

    n_reps = len(per_replica)
    fig, axes = plt.subplots(n_reps + 1, 1, figsize=(11, 1.6 * (n_reps + 1) + 1),
                             sharex=False)
    if n_reps == 0:
        plt.close(fig)
        return None

    # Per-replica traces.
    # Pre-pass: load every replica's time series once and find the global max of
    # the plotted metric so that all per-replica panels can share a common
    # y-scale (the x-axis is already uniform via set_xlim below). Plotting each
    # panel against its own max would make traces visually incomparable across
    # replicas.
    metric_keys = ("min_dist_nterm", "min_dist_cterm")
    threshold = params.d_cut
    ylabel_metric = "min dist (Å)"

    rep_data = {}
    global_max = 0.0
    for r in per_replica:
        rep = r["replica"]
        ts_file = _timeseries_path(complex_name, rep, params)
        if not os.path.exists(ts_file):
            continue
        data = np.load(ts_file, allow_pickle=False)
        rep_data[rep] = data
        m = float(np.nanmax([data[k] for k in metric_keys]))
        if np.isfinite(m):
            global_max = max(global_max, m)

    # Common y-limit: leave headroom above the data and keep the threshold line
    # comfortably inside the axes. Floor keeps tightly-bound replicas readable.
    y_top = max(global_max + 1.0, threshold + 2.0)

    for ax, r in zip(axes[:-1], per_replica):
        rep = r["replica"]
        data = rep_data.get(rep)
        if data is None:
            ax.set_visible(False)
            continue
        t = data["time_ns"]
        nterm_key, cterm_key = metric_keys
        ax.plot(t, data[nterm_key], label="N-term", lw=0.8, color="C0")
        ax.plot(t, data[cterm_key], label="C-term", lw=0.8, color="C3")
        ax.axhline(threshold, color="k", ls="--", lw=0.6, alpha=0.5)
        ax.set_ylabel(f"rep {rep}\n{ylabel_metric}")
        ax.set_ylim(0, y_top)
        # Event markers (vertical lines at each terminal's first-unbind time)
        # intentionally omitted per user request; the first-unbind times remain
        # available in the per-replica table of the Markdown report and JSON.
        ax.set_xlim(0, config.TOTAL_SIM_TIME_NS)
        ax.grid(True, alpha=0.3)
        if rep == per_replica[0]["replica"]:
            ax.legend(loc="upper right", fontsize=8)

    # Bottom panel: survival curve
    ax_surv = axes[-1]
    surv = aggregate["survival_curve"]
    ts_arr = np.array([p["t_ns"] for p in surv])
    s_arr = np.array([p["survival"] for p in surv])
    lo_arr = np.array([p["lower"] for p in surv])
    hi_arr = np.array([p["upper"] for p in surv])
    ax_surv.step(ts_arr, s_arr, where="post", color="k", lw=1.2,
                 label="Kaplan–Meier")
    ax_surv.fill_between(ts_arr, lo_arr, hi_arr, step="post", alpha=0.2,
                         color="k", label="95% CI (Greenwood)")
    ax_surv.set_ylim(0, 1.05)
    ax_surv.set_xlim(0, config.TOTAL_SIM_TIME_NS)
    ax_surv.set_xlabel("Time (ns)")
    ax_surv.set_ylabel("Survival\n(both terminals bound)")
    ax_surv.legend(loc="lower left", fontsize=8)
    ax_surv.grid(True, alpha=0.3)

    fig.suptitle(
        f"{complex_name} — unbinding detection "
        f"(dist-contact, scope={params.scope})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _write_markdown_report(complex_name, per_replica, aggregate, params,
                           plot_path, report_path):
    """Write a Markdown report summarising the per-replica and aggregate results."""
    def _fmt_t(t):
        return "stable" if t is None else f"{t:.1f}"

    lines = []
    lines.append(f"# Peptide-terminal unbinding — {complex_name}")
    lines.append("")
    lines.append("- Criterion: `dist-contact` (minimum anchor-groove distance "
                 "or normalized contact count)")
    lines.append(f"- Terminal scope: `{params.scope}`")
    lines.append(f"- Sustain window: {params.sustain_ns} ns "
                 f"({_sustain_frames(params)} frames)")
    lines.append(f"- Event detection skips first: {params.start_ns} ns")
    lines.append(f"- Thresholds: min_dist > {params.d_cut} Å OR "
                 f"contacts < {params.f_contact:.2f} × reference "
                 f"(contact cutoff {params.contact_cutoff} Å)")
    lines.append("")

    lines.append("## Per-replica events")
    lines.append("")
    lines.append(
        "| Rep | t_first (ns) | N-term (ns) | C-term (ns) | Final state | "
        "Frac unbound | Episodes | Warnings |"
    )
    lines.append(
        "|-----|-------------:|-----------:|-----------:|-------------|"
        "------------:|---------:|----------|"
    )
    for r in per_replica:
        ev = r["events"]
        t_first = ev["combined"]["t_first_unbind_ns"]
        tn = ev["nterm"]["t_first_unbind_ns"]
        tc = ev["cterm"]["t_first_unbind_ns"]
        final = ("unbound" if ev["nterm"]["final_state"] == "unbound"
                 or ev["cterm"]["final_state"] == "unbound" else "bound")
        frac = max(ev["nterm"]["fraction_time_unbound"],
                   ev["cterm"]["fraction_time_unbound"])
        eps = ev["nterm"]["n_unbinding_episodes"] + ev["cterm"]["n_unbinding_episodes"]
        warns = "; ".join(r["sanity"]["warnings"]) or "—"
        lines.append(
            f"| {r['replica']} | {_fmt_t(t_first)} | "
            f"{_fmt_t(tn)} | {_fmt_t(tc)} | {final} | "
            f"{frac:.2f} | {eps} | {warns} |"
        )
    lines.append("")

    lines.append("## Aggregate (10 replicas)")
    lines.append("")
    agg = aggregate
    frac = agg["fraction_unbound"]
    mean = agg["t_first_unbind_mean_ns"]
    sem = agg["t_first_unbind_sem_ns"]
    med = agg["t_first_unbind_median_ns"]
    tb = agg["terminal_breakdown"]
    lines.append(f"- Replicas with an unbinding event: "
                 f"{agg['n_replicas_with_event']} / {agg['n_replicas_total']} "
                 f"({frac * 100:.0f}%)")
    if mean is not None:
        lines.append(f"- First-unbind time (events only): mean = {mean:.1f} ± "
                     f"{(sem or 0):.1f} ns, median = {med:.1f} ns")
    else:
        lines.append("- All replicas remained stable over 500 ns.")
    lines.append(f"- Mean fraction of analysed time unbound: "
                 f"{agg['mean_fraction_time_unbound']:.3f}")
    lines.append(f"- Terminal breakdown: N-only = {tb['nterm_only']}, "
                 f"C-only = {tb['cterm_only']}, both = {tb['both']}, "
                 f"neither = {tb['neither']}")
    lines.append("")

    if plot_path and os.path.exists(plot_path):
        rel = os.path.basename(plot_path)
        lines.append("## Time traces and survival curve")
        lines.append("")
        lines.append("<figure>")
        lines.append(f'  <img src="{rel}" alt="Per-replica unbinding traces" '
                     f'width="900"/>')
        lines.append(f"  <figcaption>Per-replica min-distance (or RMSD) time "
                     f"traces for both peptide terminals, with the unbinding "
                     f"threshold line. Bottom panel: "
                     f"Kaplan–Meier empirical survival curve across 10 "
                     f"replicas with 95% Greenwood CI.</figcaption>")
        lines.append("</figure>")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/detect_unbinding.py`*")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def analyze_complex(complex_name, params, n_workers=1, force=False):
    """Run the unbinding analysis for every replica of a complex and aggregate.

    Writes ``unbinding_summary_{suffix}.json``, ``unbinding_report_{suffix}.md``,
    and ``unbinding_plot_{complex}_{suffix}.png`` under ``results/<complex>/``.

    Args:
        complex_name (str): Complex directory name.
        params (UnbindingParams): User-selected parameters.
        n_workers (int): ProcessPoolExecutor size for per-replica analysis.
        force (bool): If True, overwrite existing time-series files.

    Returns:
        dict: The full summary written to JSON.
    """
    print(f"\n=== {complex_name} — "
          f"scope={params.scope} ===")

    complex_dir = utils.get_complex_dir(complex_name)
    if not os.path.isdir(complex_dir):
        print(f"  WARNING: complex directory not found: {complex_dir}")
        return None

    results_dir = utils.get_results_dir(complex_name)
    os.makedirs(results_dir, exist_ok=True)

    # Discover available replicas.
    replicas = []
    for rep in range(1, config.NUM_REPLICAS + 1):
        if os.path.isdir(utils.get_replica_dir(complex_name, rep)):
            replicas.append(rep)
    if not replicas:
        print(f"  No replicas found.")
        return None

    per_replica = []
    if n_workers > 1 and len(replicas) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(analyze_replica, complex_name, rep, params, force): rep
                for rep in replicas
            }
            for fut in as_completed(futures):
                rep = futures[fut]
                try:
                    per_replica.append(fut.result())
                except Exception as e:
                    print(f"  rep_{rep} failed: {e}")
                    traceback.print_exc()
    else:
        for rep in replicas:
            try:
                per_replica.append(analyze_replica(complex_name, rep, params,
                                                   force))
            except Exception as e:
                print(f"  rep_{rep} failed: {e}")
                traceback.print_exc()

    per_replica.sort(key=lambda r: r["replica"])
    if not per_replica:
        print(f"  No replicas completed.")
        return None

    aggregate = _aggregate(per_replica, params)

    # Plot
    plot_path = os.path.join(
        results_dir,
        f"unbinding_plot_{complex_name}_{params.suffix}.png",
    )
    _plot_complex(complex_name, per_replica, aggregate, params, plot_path)

    # Summary JSON
    summary = {
        "complex": complex_name,
        "params": asdict(params),
        "per_replica": per_replica,
        "aggregate": aggregate,
    }
    summary_path = os.path.join(
        results_dir, f"unbinding_summary_{params.suffix}.json"
    )
    utils.save_metadata(summary_path, summary)

    # Markdown report
    report_path = os.path.join(
        results_dir, f"unbinding_report_{params.suffix}.md"
    )
    _write_markdown_report(complex_name, per_replica, aggregate, params,
                           plot_path, report_path)

    # Terminal summary
    agg = aggregate
    tb = agg["terminal_breakdown"]
    print(f"  events: {agg['n_replicas_with_event']}/{agg['n_replicas_total']} "
          f"({agg['fraction_unbound']*100:.0f}%); "
          f"mean t = "
          f"{(agg['t_first_unbind_mean_ns'] or float('nan')):.1f} ns; "
          f"N-only={tb['nterm_only']}, C-only={tb['cterm_only']}, "
          f"both={tb['both']}, stable={tb['neither']}")
    print(f"  wrote: {os.path.relpath(summary_path, config.BASE_DIR)}")
    print(f"         {os.path.relpath(report_path, config.BASE_DIR)}")
    if os.path.exists(plot_path):
        print(f"         {os.path.relpath(plot_path, config.BASE_DIR)}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _params_from_args(args):
    return UnbindingParams(
        d_cut=args.d_cut,
        contact_cutoff=args.contact_cutoff,
        f_contact=args.f_contact,
        sustain_ns=args.sustain_ns,
        start_ns=args.start_ns,
        reference_window_ns=args.reference_window_ns,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect peptide-terminal unbinding in peptide-HLA MD trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("list_file", nargs="?", default=None,
                        help="File listing complex names (one per line)")
    parser.add_argument("--complex", type=str, default=None,
                        help="Process a single complex (alternative to list file)")
    parser.add_argument("--rep", type=int, default=None,
                        help="Process only this replica (default: all 10)")

    # Detection thresholds
    parser.add_argument("--d-cut", type=float, default=DEFAULTS["d_cut"],
                        help="Min heavy-atom distance threshold (Å)")
    parser.add_argument("--contact-cutoff", type=float,
                        default=DEFAULTS["contact_cutoff"],
                        help="Heavy-atom contact radius (Å)")
    parser.add_argument("--f-contact", type=float,
                        default=DEFAULTS["f_contact"],
                        help="Fraction-of-reference contacts threshold")

    # rmsd thresholds
    parser.add_argument("--sustain-ns", type=float,
                        default=DEFAULTS["sustain_ns"],
                        help="Sustain-window duration (ns)")
    parser.add_argument("--start-ns", type=float, default=DEFAULTS["start_ns"],
                        help="Skip the first N ns for event detection")
    parser.add_argument("--reference-window-ns", type=float,
                        default=DEFAULTS["reference_window_ns"],
                        help="Window (ns) used for the reference contact count")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel replicas per complex (default: 1)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")

    args = parser.parse_args()

    if args.complex:
        complexes = [args.complex]
    elif args.list_file:
        if not os.path.exists(args.list_file):
            print(f"ERROR: list file not found: {args.list_file}")
            sys.exit(1)
        complexes = utils.read_complexes_list(args.list_file)
    else:
        parser.print_help()
        sys.exit(1)

    params = _params_from_args(args)

    print("=" * 60)
    print("Peptide-terminal unbinding detection")
    print("=" * 60)
    print(f"  Complexes: {len(complexes)}")
    print(f"  Criterion: dist-contact  |  scope: {params.scope}")
    print(f"  Thresholds: d_cut={params.d_cut} Å, "
          f"contact_cutoff={params.contact_cutoff} Å, "
          f"f_contact={params.f_contact}")
    print(f"  Sustain: {params.sustain_ns} ns ({_sustain_frames(params)} frames)")
    print(f"  Start:   {params.start_ns} ns ({_start_frame(params)} frames)")
    print(f"  Workers: {args.n_workers}   Force: {args.force}")
    print("=" * 60)

    # Early validation so the user gets a clean error when AMBERHOME is missing.
    config.validate_amberhome()

    for complex_name in complexes:
        if args.rep is not None:
            # Single-replica mode: run only, do not aggregate, but echo
            # the per-replica event metrics so the user sees results.
            result = analyze_replica(complex_name, args.rep, params,
                                     force=args.force)
            if result is not None:
                ev = result["events"]
                def _fmt(t):
                    return "stable" if t is None else f"{t:.1f} ns"
                print(f"  [rep_{args.rep}] events:")
                for term in ("nterm", "cterm"):
                    e = ev[term]
                    print(f"    {term:9s}: t_first={_fmt(e['t_first_unbind_ns'])}"
                          f", final={e['final_state']}, "
                          f"frac_unbound={e['fraction_time_unbound']:.2f}, "
                          f"episodes={e['n_unbinding_episodes']}")
                comb = ev["combined"]
                print(f"    combined : t_first={_fmt(comb['t_first_unbind_ns'])}"
                      f" (source={comb['terminal_source']})")
                san = result["sanity"]
                print(f"    sanity: init_min_dist "
                      f"(N={san['initial_min_dist_nterm']:.2f} Å, "
                      f"C={san['initial_min_dist_cterm']:.2f} Å), "
                      f"init_contacts "
                      f"(N={san['initial_n_contacts_nterm']}, "
                      f"C={san['initial_n_contacts_cterm']})")
                if san["warnings"]:
                    print(f"    warnings: {'; '.join(san['warnings'])}")
        else:
            analyze_complex(complex_name, params,
                            n_workers=args.n_workers, force=args.force)


if __name__ == "__main__":
    main()
