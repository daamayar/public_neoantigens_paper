#!/usr/bin/env python3
"""
Shared utilities for the interaction-network analyses.

Provides:
  * path helpers for the per-replica trajectories and outputs,
  * a topology "index map" that resolves, once per complex, every atom and
    residue selection the analyses need (peptide atoms, groove atoms, charged
    groups, hydrogen-bond donors/acceptors, per-residue atom slices),
  * extraction of the non-bonded force-field parameters (partial charges and
    the exact Lennard-Jones A/B coefficient matrix, including any CHARMM NBFIX
    corrections) needed for the MM interaction-energy decomposition,
  * fast trajectory loading via scipy's NetCDF reader,
  * the per-frame bound/unbound mask used for conditioning.

"""
import os
import json

import numpy as np
from scipy.io import netcdf_file

from . import inconfig as cfg


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def replica_dir(dirname, rep):
    """Absolute path of a replica directory.

    Args:
        dirname (str): complex directory name, e.g. "pik3ca_e545k_wt_a_1101_strdplseite".
        rep (int): replica number (1-based).

    Returns:
        str: absolute path to ``BASE_DIR/<dirname>/rep_<rep>``.
    """
    return os.path.join(cfg.BASE_DIR, dirname, f"rep_{rep}")


def clean_nc_path(dirname, rep):
    """Path of the water-stripped, receptor-autoimaged 4166-frame NetCDF.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        str: absolute path to the clean trajectory produced by
            ``scripts/detect_unbinding.py``.
    """
    return os.path.join(replica_dir(dirname, rep), cfg.CLEAN_NC.format(rep=rep))


def prmtop_path(dirname, rep):
    """Path of the dry AMBER complex topology for a replica.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        str: absolute path to ``mmpbsa_500ns/complex.prmtop``.
    """
    return os.path.join(replica_dir(dirname, rep), cfg.COMPLEX_PRMTOP)


def unbinding_npz_path(dirname, rep):
    """Path of the per-frame unbinding time series for a replica.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        str: absolute path to the .npz written by ``detect_unbinding.py``.
    """
    return os.path.join(cfg.RESULTS_DIR, dirname,
                        cfg.UNBINDING_NPZ.format(rep=rep))


def contacts_npz_path(dirname, rep):
    """Path of this package's Phase-1 per-frame output for a replica.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        str: absolute path under ``interaction_network/data/<dirname>/``.
    """
    return os.path.join(cfg.DATA_DIR, dirname, f"rep{rep}_network.npz")


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist.

    Args:
        path (str): directory path.

    Returns:
        str: the same path, for chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_system(dirname):
    """Return the SYSTEMS entry whose ``dirname`` matches.

    Args:
        dirname (str): complex directory name.

    Returns:
        dict: the matching entry of ``inconfig.SYSTEMS``.

    Raises:
        KeyError: if no system matches.
    """
    for s in cfg.SYSTEMS:
        if s["dirname"] == dirname:
            return s
    raise KeyError(f"unknown system: {dirname}")


# ---------------------------------------------------------------------------
# Topology index map
# ---------------------------------------------------------------------------

_THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _residue_atom_slices(residues, atom_subset=None):
    """Build ``reduceat`` start indices for a contiguous block of residues.

    AMBER topologies store the atoms of a residue contiguously, so a per-residue
    reduction over a per-atom array is exactly ``np.minimum.reduceat`` /
    ``np.add.reduceat`` at the residue start offsets. This helper returns those
    offsets relative to the *concatenated selection*, not to the topology.

    Args:
        residues (list): parmed Residue objects, in topology order.
        atom_subset (set[int] | None): if given, only atom indices in this set
            are kept (used to build heavy-atom-only selections).

    Returns:
        tuple:
            - np.ndarray[int]: the selected atom indices, residue-major.
            - np.ndarray[int]: start offset of each residue within that array.
            - np.ndarray[int]: number of selected atoms per residue.
    """
    idx, starts, counts = [], [], []
    for r in residues:
        starts.append(len(idx))
        n = 0
        for a in r.atoms:
            if atom_subset is None or a.idx in atom_subset:
                idx.append(a.idx)
                n += 1
        counts.append(n)
    return (np.asarray(idx, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(counts, dtype=np.int64))


def build_index_map(dirname, rep=1):
    """Resolve every atom/residue selection needed by the analyses, once.

    Loads the dry complex topology with ParmEd and derives:
      * peptide identity (sequence, residue range, length),
      * all-atom and heavy-atom selections for the peptide and the groove
        (HLA residues 1-180), with per-residue ``reduceat`` offsets,
      * charged-group atom indices per residue (for salt bridges),
      * hydrogen-bond donor(-H) and acceptor atom indices on both sides,
      * per-atom partial charges and Lennard-Jones type indices, plus the
        exact LJ A/B coefficient matrices from the topology.

    Args:
        dirname (str): complex directory name.
        rep (int): replica whose topology is read. PSF/prmtop files differ per
            replica only in their water/ion content, which is already stripped
            from ``complex.prmtop``, so any replica gives the same map. Default 1.

    Returns:
        dict: the index map. Keys of interest to callers:
            ``seq``, ``pep_len``, ``pep_resids`` (1-based topology residue
            numbers), ``groove_resids``, ``pep_heavy``/``groove_heavy`` (atom
            index arrays), ``pep_heavy_starts``/``groove_heavy_starts``
            (reduceat offsets), ``pep_all``/``groove_all`` and their offsets,
            ``charge`` (per-atom, elementary units), ``lj_a``/``lj_b``
            (n_types x n_types matrices), ``lj_type`` (per-atom, 0-based),
            ``pep_charged``/``groove_charged`` (dict resid -> atom indices),
            ``don_h``/``acc`` selections for both partners.

    Raises:
        FileNotFoundError: if the replica's ``complex.prmtop`` is missing.
    """
    import parmed as pmd

    top_path = prmtop_path(dirname, rep)
    if not os.path.exists(top_path):
        raise FileNotFoundError(f"missing topology: {top_path}")
    parm = pmd.load_file(top_path)

    residues = parm.residues
    n_res = len(residues)
    pep_res = residues[cfg.PEPTIDE_START - 1:]
    g_lo, g_hi = cfg.GROOVE_RANGE
    groove_res = residues[g_lo - 1:g_hi]

    seq = "".join(_THREE2ONE.get(r.name, "X") for r in pep_res)

    heavy = {a.idx for a in parm.atoms if a.atomic_number != 1}

    pep_all, pep_all_starts, pep_all_counts = _residue_atom_slices(pep_res)
    pep_hv, pep_hv_starts, pep_hv_counts = _residue_atom_slices(pep_res, heavy)
    grv_all, grv_all_starts, grv_all_counts = _residue_atom_slices(groove_res)
    grv_hv, grv_hv_starts, grv_hv_counts = _residue_atom_slices(groove_res, heavy)

    # --- non-bonded parameters -------------------------------------------
    charge = np.array([a.charge for a in parm.atoms], dtype=np.float64)
    lj_type = np.array([a.nb_idx - 1 for a in parm.atoms], dtype=np.int64)
    n_types = int(lj_type.max()) + 1

    # Reconstruct the exact A/B coefficient matrices from the topology arrays.
    # This reproduces sander's pair energies including any CHARMM NBFIX entries,
    # which a Lorentz-Berthelot recombination of per-atom rmin/epsilon would not.
    nbparm = np.asarray(parm.parm_data["NONBONDED_PARM_INDEX"], dtype=np.int64)
    acoef = np.asarray(parm.parm_data["LENNARD_JONES_ACOEF"], dtype=np.float64)
    bcoef = np.asarray(parm.parm_data["LENNARD_JONES_BCOEF"], dtype=np.float64)
    lj_a = np.zeros((n_types, n_types), dtype=np.float64)
    lj_b = np.zeros((n_types, n_types), dtype=np.float64)
    for i in range(n_types):
        for j in range(n_types):
            k = nbparm[n_types * i + j]
            if k > 0:
                lj_a[i, j] = acoef[k - 1]
                lj_b[i, j] = bcoef[k - 1]
            # k < 0 flags a 10-12 hydrogen-bond pair, absent from modern force
            # fields; k == 0 means the pair is excluded. Both stay at zero.

    # --- charged groups ---------------------------------------------------
    def _charged(res_list):
        out = {}
        for r in res_list:
            names, sign = cfg.CHARGED_GROUPS.get(r.name, (None, 0))
            sel = []
            if names:
                sel += [a.idx for a in r.atoms if a.name in names]
            # C-terminal carboxylate: an extra anionic group on the last residue
            ct = [a.idx for a in r.atoms if a.name in cfg.CTERM_CARBOXYLATE]
            if sel:
                out[(r.number + 1, "sc")] = (np.asarray(sel, dtype=np.int64), sign)
            if ct:
                out[(r.number + 1, "ct")] = (np.asarray(ct, dtype=np.int64), -1)
        return out

    # --- hydrogen-bond donors and acceptors -------------------------------
    # Donor-H pairs: any hydrogen covalently bonded to N, O or S.
    # Acceptors: any N or O. This is the standard geometric definition; the
    # angle criterion in compute_contacts.py does the discrimination.
    def _don_acc(atom_idx_set):
        dh, acc = [], []
        for a in parm.atoms:
            if a.idx not in atom_idx_set:
                continue
            if a.atomic_number in (7, 8):
                acc.append(a.idx)
            elif a.atomic_number == 1:
                for p in a.bond_partners:
                    if p.atomic_number in (7, 8, 16):
                        dh.append((p.idx, a.idx))
                        break
        return (np.asarray(dh, dtype=np.int64).reshape(-1, 2),
                np.asarray(acc, dtype=np.int64))

    pep_set = set(pep_all.tolist())
    grv_set = set(grv_all.tolist())
    pep_dh, pep_acc = _don_acc(pep_set)
    grv_dh, grv_acc = _don_acc(grv_set)

    # --- apolar atoms (hydrophobic contacts) ------------------------------
    # An atom counts as apolar if it is carbon NOT covalently bonded to any
    # nitrogen or oxygen, or if it is sulfur. This is the standard definition:
    # it keeps side-chain and backbone CB/CG/CD... carbons but excludes
    # carbonyl carbons, C-alpha (bonded to the backbone N) and any carbon
    # adjacent to a hydroxyl or amide, which are not hydrophobic.
    def _apolar(atom_idx_set):
        out = []
        for a in parm.atoms:
            if a.idx not in atom_idx_set:
                continue
            if a.atomic_number == 16:
                out.append(a.idx)
            elif a.atomic_number == 6:
                if not any(p.atomic_number in (7, 8) for p in a.bond_partners):
                    out.append(a.idx)
        return np.asarray(out, dtype=np.int64)

    pep_apolar = _apolar(pep_set)
    grv_apolar = _apolar(grv_set)

    # --- aromatic rings (cation-pi) ---------------------------------------
    # Ring atom names per residue type. Tryptophan contributes both rings of
    # its indole; each ring is tested independently.
    ring_defs = {
        "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
        "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
        "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
        "HSD": [["CG", "ND1", "CD2", "CE1", "NE2"]],
        "HSE": [["CG", "ND1", "CD2", "CE1", "NE2"]],
        "HSP": [["CG", "ND1", "CD2", "CE1", "NE2"]],
        "TRP": [["CG", "CD1", "NE1", "CE2", "CD2"],
                ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]],
    }

    def _rings(res_list, offset):
        """Return (list of atom-index arrays, array of residue slots)."""
        idx, slots = [], []
        for r in res_list:
            for names in ring_defs.get(r.name, []):
                sel = [a.idx for a in r.atoms if a.name in names]
                if len(sel) >= 5:
                    idx.append(np.asarray(sel, dtype=np.int64))
                    slots.append(r.idx - offset)
        return idx, np.asarray(slots, dtype=np.int64)

    pep_rings, pep_ring_res = _rings(pep_res, cfg.PEPTIDE_START - 1)
    grv_rings, grv_ring_res = _rings(groove_res, cfg.GROOVE_RANGE[0] - 1)

    # --- cationic groups (cation-pi donors) --------------------------------
    # Only the positively charged groups; the anionic ones cannot make a
    # cation-pi interaction. The "cation centre" is the guanidinium CZ for
    # arginine, NZ for lysine, and the ring centroid for protonated histidine.
    cation_names = {"ARG": ["CZ"], "LYS": ["NZ"], "HSP": ["ND1", "NE2"]}

    def _cations(res_list, offset):
        idx, slots = [], []
        for r in res_list:
            names = cation_names.get(r.name)
            if not names:
                continue
            sel = [a.idx for a in r.atoms if a.name in names]
            if sel:
                idx.append(np.asarray(sel, dtype=np.int64))
                slots.append(r.idx - offset)
        return idx, np.asarray(slots, dtype=np.int64)

    pep_cations, pep_cation_res = _cations(pep_res, cfg.PEPTIDE_START - 1)
    grv_cations, grv_cation_res = _cations(groove_res, cfg.GROOVE_RANGE[0] - 1)

    # Per-atom residue index (0-based, over the whole topology). Lets any
    # atom-level result be folded onto residues without re-consulting ParmEd.
    atom_res = np.empty(len(parm.atoms), dtype=np.int64)
    for r in residues:
        for a in r.atoms:
            atom_res[a.idx] = r.idx

    return dict(
        dirname=dirname,
        n_res=n_res,
        n_atoms=len(parm.atoms),
        atom_res=atom_res,
        seq=seq,
        pep_len=len(pep_res),
        pep_resids=np.arange(cfg.PEPTIDE_START, n_res + 1, dtype=np.int64),
        pep_resnames=[r.name for r in pep_res],
        groove_resids=np.arange(g_lo, g_hi + 1, dtype=np.int64),
        groove_resnames=[r.name for r in groove_res],
        # heavy-atom selections (contacts)
        pep_heavy=pep_hv, pep_heavy_starts=pep_hv_starts,
        pep_heavy_counts=pep_hv_counts,
        groove_heavy=grv_hv, groove_heavy_starts=grv_hv_starts,
        groove_heavy_counts=grv_hv_counts,
        # all-atom selections (energies)
        pep_all=pep_all, pep_all_starts=pep_all_starts,
        pep_all_counts=pep_all_counts,
        groove_all=grv_all, groove_all_starts=grv_all_starts,
        groove_all_counts=grv_all_counts,
        # force field
        charge=charge, lj_type=lj_type, lj_a=lj_a, lj_b=lj_b,
        # interaction-specific selections
        pep_charged=_charged(pep_res), groove_charged=_charged(groove_res),
        pep_don_h=pep_dh, pep_acc=pep_acc,
        groove_don_h=grv_dh, groove_acc=grv_acc,
        pep_apolar=pep_apolar, groove_apolar=grv_apolar,
        pep_rings=pep_rings, pep_ring_res=pep_ring_res,
        groove_rings=grv_rings, groove_ring_res=grv_ring_res,
        pep_cations=pep_cations, pep_cation_res=pep_cation_res,
        groove_cations=grv_cations, groove_cation_res=grv_cation_res,
    )


def residue_label(resid, resname):
    """Format an HLA residue as e.g. "Asp116".

    Args:
        resid (int): 1-based residue number in the topology (= HLA numbering).
        resname (str): three-letter residue name from the topology.

    Returns:
        str: capitalised three-letter code followed by the number.
    """
    name = resname.capitalize()
    if name in ("Hsd", "Hse", "Hsp"):
        name = "His"
    return f"{name}{resid}"
def load_coordinates(dirname, rep, atom_indices=None):
    """Read a clean trajectory into a numpy array.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).
        atom_indices (np.ndarray | None): if given, only these atom columns are
            returned (still in the given order). If None, all atoms are read.

    Returns:
        np.ndarray: float32 array of shape (n_frames, n_selected_atoms, 3).

    Raises:
        FileNotFoundError: if the clean trajectory does not exist.
        ValueError: if the frame count differs from ``inconfig.TOTAL_FRAMES``.
    """
    path = clean_nc_path(dirname, rep)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing clean trajectory: {path}\n"
            f"  Fix: python scripts/detect_unbinding.py --complex {dirname} "
            f"--rep {rep}")
    with netcdf_file(path, "r", mmap=False) as fh:
        xyz = np.asarray(fh.variables["coordinates"][:], dtype=np.float32)
    if xyz.shape[0] != cfg.TOTAL_FRAMES:
        raise ValueError(
            f"{path}: expected {cfg.TOTAL_FRAMES} frames, found {xyz.shape[0]}")
    if atom_indices is not None:
        xyz = xyz[:, atom_indices, :]
    return xyz


def time_axis(n_frames=None):
    """Simulation time of each frame, in ns.

    Args:
        n_frames (int | None): number of frames; defaults to TOTAL_FRAMES.

    Returns:
        np.ndarray: float64 array of frame times in ns, starting at 0.
    """
    n = cfg.TOTAL_FRAMES if n_frames is None else n_frames
    return np.arange(n) * cfg.TIME_PER_FRAME_NS


# ---------------------------------------------------------------------------
# Conditioning masks
# ---------------------------------------------------------------------------

def cterm_bound_mask(dirname, rep):
    """Per-frame boolean mask: is the peptide C-terminal region engaged?

    Reuses the per-frame arrays already written by ``scripts/detect_unbinding.py``
    (criterion ``dist-contact``, scope ``residue+neighbor``) and applies exactly
    that script's bound condition, so the two analyses cannot disagree about
    which frames count as bound:

        bound  <=>  min_dist_cterm <= BOUND_D_CUT
                    AND n_contacts_cterm >= BOUND_F_CONTACT * n_contacts_reference

    where the reference is the median contact count over the first
    ``BOUND_REF_WINDOW_NS`` ns.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        np.ndarray: boolean array of length ``TOTAL_FRAMES``; True = bound.

    Raises:
        FileNotFoundError: if the unbinding time series has not been computed.
    """
    path = unbinding_npz_path(dirname, rep)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing unbinding time series: {path}\n"
            f"  Fix: python -m scripts.detect_unbinding --complex {dirname}")
    with np.load(path) as data:
        min_d = np.asarray(data["min_dist_cterm"], dtype=np.float64)
        n_ct = np.asarray(data["n_contacts_cterm"], dtype=np.float64)
    n_ref = max(1, int(round(cfg.BOUND_REF_WINDOW_NS / cfg.TIME_PER_FRAME_NS)))
    ref = float(np.median(n_ct[:n_ref]))
    thresh = max(1.0, cfg.BOUND_F_CONTACT * ref)
    return (min_d <= cfg.BOUND_D_CUT) & (n_ct >= thresh)


def first_event_frame(dirname, rep):
    """Frame index of the first scored C-terminal unbinding event, if any.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        int | None: 0-based frame index of the event, or None if the replica
            was scored as stable.
    """
    path = os.path.join(cfg.RESULTS_DIR, dirname, cfg.UNBINDING_JSON)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        summary = json.load(fh)
    for r in summary["per_replica"]:
        if r["replica"] == rep:
            t = r["events"]["cterm"]["t_first_unbind_ns"]
            if t is None:
                return None
            return int(round(t / cfg.TIME_PER_FRAME_NS))
    return None


def window_mask(dirname, rep, window):
    """Build the per-frame selection mask for a named conditioning window.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).
        window (str): a key of ``inconfig.WINDOWS``.

    Returns:
        np.ndarray: boolean array of length ``TOTAL_FRAMES``; True = include.

    Raises:
        KeyError: if ``window`` is not a configured window name.
    """
    spec = cfg.WINDOWS[window]
    lo, hi = spec["frame_slice"]
    mask = np.zeros(cfg.TOTAL_FRAMES, dtype=bool)
    mask[lo:hi] = True
    if spec["bound_only"]:
        mask &= cterm_bound_mask(dirname, rep)
    return mask


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def mean_sem(values):
    """Mean and standard error of the mean of a 1-D sample.

    Args:
        values (array-like): sample values (e.g. one number per replica).

    Returns:
        tuple[float, float]: (mean, SEM). SEM is 0.0 for a single value and
            NaN for an empty input.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U test between two independent samples.

    Used for replica-level comparisons (n = 10 per group), where the
    distribution of per-replica occupancies is not necessarily normal and the
    sample is too small to check. Falls back to scipy, which handles ties and
    uses the exact distribution for small n.

    Args:
        a (array-like): sample 1 (e.g. 10 per-replica values for the neoantigen).
        b (array-like): sample 2 (e.g. 10 per-replica values for the wild-type).

    Returns:
        tuple[float, float]: (U statistic, two-sided p-value). Returns
            (nan, nan) if either sample is empty or entirely constant.
    """
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return float("nan"), 1.0
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return float("nan"), float("nan")


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values.

    The per-edge and per-salt-bridge comparisons run hundreds of tests over the
    same 10 vs 10 replicas, so raw p-values would be badly optimistic. BH
    controls the expected proportion of false positives among the calls
    declared significant, which is the appropriate criterion for a screen whose
    output is a ranked list of candidate interactions.

    NaN entries (tests that could not be run) are passed through unchanged and
    excluded from the ranking.

    Args:
        pvalues (array-like): raw two-sided p-values.

    Returns:
        np.ndarray: adjusted p-values (q-values), same length and order as the
            input, each clipped to at most 1.0 and enforced monotone.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p downwards
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q
