#!/usr/bin/env python3
"""
Shared utilities for the peptide-MD structural analyses (RMSD / RMSF).

Provides:
  * build_index_map()   -- maps the reduced trajectory cache back to named atoms
                           (receptor CA, peptide CA / backbone / heavy).
  * load_cache()        -- fast NetCDF read of a cached replica (~0.04 s).
  * kabsch_rotations()  -- batched optimal-rotation solver over all frames.
  * superpose()         -- fit every frame on a reference and return the rotated
                           coordinates of an arbitrary measured atom subset.
  * rmsd_to_reference() -- per-frame RMSD after superposition.
  * rmsf_about_mean()   -- per-atom RMSF about the ensemble mean structure.

THE REDUCED CACHE
-----------------
build_cache.py stores, for every replica, a NetCDF holding only:
    all C-alpha atoms (one per residue, receptor + peptide)  +  peptide heavy atoms
in the original topology's atom order. That is ~430-460 atoms instead of ~6100,
which is why an entire replica loads in 0.04 s. Frame 0 of the cache is the
t~0 RMSD reference; frames [EQUIL_FRAMES:] are the 50-500 ns analysis ensemble.

SIGN / CONVENTION NOTES
-----------------------
* Kabsch is solved for ROW-vector coordinates: given P (N,3) and Q (N,3) already
  centred, we return R such that ``P @ R.T`` best matches Q.
* When the fit group differs from the measured group (e.g. fit on peptide CA but
  measure backbone), the measured atoms are translated by the FIT group's
  centroid -- never by their own -- and then rotated by the fit's R. Centring the
  measured atoms on their own centroid would silently remove a real translation.
* The CHARMM36 C-terminal residue has OT1/OT2 and no "O". OT1 and OT2 are
  equivalent by resonance and swap when the carboxylate flips, so the backbone
  "O" of the C-terminal residue is the CENTROID of OT1 and OT2 (flip-invariant).
  This is why backbone coordinates are assembled by build_peptide_backbone()
  rather than by simple integer indexing.
"""
import os
import glob
import numpy as np
from scipy.io import netcdf_file

from . import mdconfig as cfg


# =============================================================================
# TOPOLOGY / INDEX MAPPING
# =============================================================================
def load_complex_topology(dirname, rep=1):
    """Load the dry complex topology (parmed) for a given complex.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number whose topology is read (topologies are
            equivalent across replicas for the dry complex; only the solvation
            differs, and that is stripped).

    Returns:
        parmed.Structure: the dry complex topology (~6100 atoms, 384-386 residues).
    """
    import parmed as pmd
    path = os.path.join(cfg.BASE_DIR, dirname, f"rep_{rep}", cfg.COMPLEX_PRMTOP)
    return pmd.load_file(path)


def build_index_map(dirname, rep=1):
    """Describe the reduced cache: which cache column is which atom.

    The reduced cache keeps, in original topology order,
        {every atom named CA}  UNION  {every non-hydrogen atom of the peptide}.
    This function reproduces that selection from the full topology and returns
    the column indices *into the cache array* for each group we analyse.

    Args:
        dirname (str): complex directory name.
        rep (int): replica whose topology is read (default 1).

    Returns:
        dict with keys:
            n_res (int)          : total residues in the complex (384/385/386)
            n_rec (int)          : receptor residues (always 375)
            pep_len (int)        : peptide length (9, 10 or 11)
            seq (str)            : peptide one-letter sequence
            n_cache_atoms (int)  : number of atoms stored per frame in the cache
            rec_ca (np.ndarray)  : cache columns of the 375 receptor CA, in residue order
            pep_ca (np.ndarray)  : cache columns of the peptide CA, in residue order
            pep_heavy (np.ndarray): cache columns of all peptide heavy atoms
            bb_groups (list)     : per peptide residue, a list of 4 entries
                                   (N, CA, C, O). Each entry is either an int
                                   (a single cache column) or a tuple of ints
                                   (columns to average -- used for the
                                   C-terminal OT1/OT2 carboxylate centroid).
            chain_a_ca, chain_b_ca (np.ndarray): cache columns of the CA of
                                   HLA alpha chain (res 1-276) and beta-2-m
                                   (res 277-375), for the wrapping QC.
    """
    cx = load_complex_topology(dirname, rep)
    n_res = len(cx.residues)
    n_rec = cfg.N_RECEPTOR_RES
    pep_len = n_res - n_rec

    # Reproduce exactly the cpptraj selection used by build_cache.py:
    #   strip !(@CA | (:376-<n_res> & !@H*))
    keep = [i for i, a in enumerate(cx.atoms)
            if a.name == "CA" or (a.residue.idx >= n_rec and a.atomic_number != 1)]
    col_of = {orig: k for k, orig in enumerate(keep)}   # original atom idx -> cache column

    rec_ca, chain_a_ca, chain_b_ca = [], [], []
    for i, a in enumerate(cx.atoms):
        if a.name == "CA" and a.residue.idx < n_rec:
            rec_ca.append(col_of[i])
            (chain_a_ca if a.residue.idx < cfg.CHAIN_A[1] else chain_b_ca).append(col_of[i])

    pep_ca, pep_heavy, bb_groups = [], [], []
    for r in range(n_rec, n_res):
        by_name = {}
        for a in cx.residues[r].atoms:
            if a.atomic_number == 1:
                continue
            pep_heavy.append(col_of[a.idx])
            by_name[a.name] = col_of[a.idx]
        pep_ca.append(by_name["CA"])

        # backbone N, CA, C, O -- with the flip-invariant C-terminal carboxylate
        if "O" in by_name:
            o_entry = by_name["O"]
        elif "OT1" in by_name and "OT2" in by_name:
            o_entry = (by_name["OT1"], by_name["OT2"])     # centroid of the carboxylate
        else:
            raise ValueError(
                f"{dirname}: peptide residue {r+1} has neither 'O' nor 'OT1'+'OT2' "
                f"(atoms: {sorted(by_name)})")
        bb_groups.append([by_name["N"], by_name["CA"], by_name["C"], o_entry])

    seqmap = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
              'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
              'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
              'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    seq = "".join(seqmap.get(r.name, "?") for r in cx.residues[n_rec:])

    return dict(
        n_res=n_res, n_rec=n_rec, pep_len=pep_len, seq=seq,
        n_cache_atoms=len(keep),
        rec_ca=np.array(rec_ca), pep_ca=np.array(pep_ca),
        pep_heavy=np.array(pep_heavy), bb_groups=bb_groups,
        chain_a_ca=np.array(chain_a_ca), chain_b_ca=np.array(chain_b_ca),
    )


def build_peptide_backbone(coords, imap):
    """Assemble peptide backbone (N, CA, C, O) coordinates from cache coordinates.

    Handles the CHARMM C-terminal carboxylate: for the last residue the "O"
    position is the centroid of OT1 and OT2, which is invariant to the 180-degree
    carboxylate flip that would otherwise cause spurious RMSD spikes.

    Args:
        coords (np.ndarray): (F, A, 3) cache coordinates for one replica.
        imap (dict): output of build_index_map().

    Returns:
        np.ndarray: (F, 4*pep_len, 3) backbone coordinates, ordered
            [res1 N, res1 CA, res1 C, res1 O, res2 N, ...].
    """
    cols = []
    for grp in imap["bb_groups"]:
        for entry in grp:
            if isinstance(entry, tuple):
                cols.append(coords[:, list(entry), :].mean(axis=1))    # OT1/OT2 centroid
            else:
                cols.append(coords[:, entry, :])
    return np.stack(cols, axis=1)


def get_selection(coords, imap, selection):
    """Return the coordinates of a named peptide atom selection.

    Args:
        coords (np.ndarray): (F, A, 3) cache coordinates.
        imap (dict): output of build_index_map().
        selection (str): "backbone" (N,CA,C,O), "ca" (peptide C-alpha),
            "heavy" (all peptide non-hydrogen atoms), or "receptor_ca".

    Returns:
        np.ndarray: (F, n_sel, 3) coordinates of the requested selection.
    """
    if selection == "backbone":
        return build_peptide_backbone(coords, imap)
    if selection == "ca":
        return coords[:, imap["pep_ca"], :]
    if selection == "heavy":
        return coords[:, imap["pep_heavy"], :]
    if selection == "receptor_ca":
        return coords[:, imap["rec_ca"], :]
    raise ValueError(f"unknown selection: {selection!r}")


def fit_coords(coords, imap, fit_mode=None):
    """Return the coordinates that define the superposition frame.

    Args:
        coords (np.ndarray): (F, A, 3) cache coordinates.
        imap (dict): output of build_index_map().
        fit_mode (str | None): "peptide" (peptide C-alpha) or "receptor"
            (receptor C-alpha). Defaults to mdconfig.FIT_MODE.

    Returns:
        np.ndarray: (F, n_fit, 3) coordinates of the fit group.
    """
    mode = fit_mode or cfg.FIT_MODE
    if mode == "peptide":
        return coords[:, imap["pep_ca"], :]
    if mode == "receptor":
        return coords[:, imap["rec_ca"], :]
    raise ValueError(f"unknown fit_mode: {mode!r}")


# =============================================================================
# CACHE I/O
# =============================================================================
def cache_path(dirname, rep):
    """Path of the reduced NetCDF cache for one replica.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        str: absolute path to md_analyses/cache/<dirname>/rep_<rep>.nc
    """
    return os.path.join(cfg.CACHE_DIR, dirname, f"rep_{rep}.nc")


def load_cache(dirname, rep):
    """Load one replica's reduced trajectory as a float64 array.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        np.ndarray: (F, A, 3) coordinates in Angstrom, native-endian float64.
            F should equal mdconfig.TOTAL_FRAMES (4166).

    Raises:
        FileNotFoundError: if the cache has not been built for this replica.
    """
    path = cache_path(dirname, rep)
    if not os.path.exists(path):
        raise FileNotFoundError(f"cache missing: {path} -- run build_cache.py first")
    with netcdf_file(path, "r", mmap=False) as f:
        xyz = np.array(f.variables["coordinates"][:], dtype=np.float64)
    return xyz


def find_dcd(dirname, rep):
    """Locate the production trajectory of one replica.

    Honours cfg.TRAJECTORY_SOURCE ("auto", "solvated" or "dry"). The dry form is
    the protein-only trajectory as deposited; it is read with complex.prmtop and
    needs neither reimaging nor stripping.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        tuple[str, str]: (path, form) with form "solvated" or "dry".

    Raises:
        FileNotFoundError: if nothing usable matches.
    """
    rep_dir = os.path.join(cfg.BASE_DIR, dirname, f"rep_{rep}")

    def _one(pattern):
        hits = sorted(glob.glob(os.path.join(rep_dir, pattern)))
        return hits[0] if len(hits) == 1 else None

    solvated = next((h for h in (_one(pat) for pat in cfg.DCD_PATTERNS) if h), None)
    dry = _one(cfg.DRY_DCD_PATTERN)
    source = cfg.TRAJECTORY_SOURCE

    if source == "solvated":
        if solvated:
            return solvated, "solvated"
        raise FileNotFoundError(
            f"PHLA_TRAJECTORY_SOURCE=solvated but no {' or '.join(cfg.DCD_PATTERNS)} for "
            f"{dirname} rep_{rep}")
    if source == "dry":
        if dry:
            return dry, "dry"
        raise FileNotFoundError(
            f"PHLA_TRAJECTORY_SOURCE=dry but no {cfg.DRY_DCD_PATTERN} for "
            f"{dirname} rep_{rep}")
    if solvated:
        return solvated, "solvated"
    if dry:
        return dry, "dry"
    raise FileNotFoundError(
        f"no production trajectory for {dirname} rep_{rep} in {rep_dir}")


# =============================================================================
# SUPERPOSITION (batched Kabsch)
# =============================================================================
def kabsch_rotations(P, Q):
    """Optimal rotations aligning each frame of P onto the reference Q.

    Solves, per frame f, the orthogonal Procrustes problem
        min_R  || P[f] @ R.T  -  Q ||_F      subject to det(R) = +1
    (proper rotations only -- reflections are excluded, which is essential for
    chiral molecules).

    Args:
        P (np.ndarray): (F, N, 3) mobile coordinates, ALREADY centred per frame.
        Q (np.ndarray): (N, 3) reference coordinates, ALREADY centred.

    Returns:
        np.ndarray: (F, 3, 3) rotation matrices R, to be applied as ``P @ R.T``.
    """
    H = np.einsum("fni,nj->fij", P, Q)                    # (F,3,3) = P^T Q per frame
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))                    # (F,) +-1, fixes reflections
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    return V @ D @ Ut


def superpose(fit_xyz, ref_fit, measured):
    """Superpose every frame on a reference and rotate a measured atom subset.

    The measured atoms are translated by the FIT group's centroid (not their own)
    and rotated by the fit's rotation, so any real displacement of the measured
    atoms relative to the fit group is preserved.

    Args:
        fit_xyz (np.ndarray): (F, Nfit, 3) coordinates of the fit group.
        ref_fit (np.ndarray): (Nfit, 3) reference coordinates of the fit group.
        measured (list[np.ndarray] | np.ndarray): one or more (F, M, 3) arrays to
            rotate into the reference frame.

    Returns:
        (np.ndarray | list[np.ndarray], np.ndarray): the superposed measured
            array(s) (same structure as the input), and the (Nfit, 3) centred
            reference used, i.e. ``ref_fit - ref_fit.mean(0)``.
    """
    single = isinstance(measured, np.ndarray)
    mlist = [measured] if single else list(measured)

    fit_cen = fit_xyz.mean(axis=1, keepdims=True)          # (F,1,3)
    ref_cen = ref_fit.mean(axis=0)                          # (3,)
    P = fit_xyz - fit_cen
    Q = ref_fit - ref_cen
    R = kabsch_rotations(P, Q)                              # (F,3,3)

    out = []
    for M in mlist:
        Mc = M - fit_cen                                    # translate by the FIT centroid
        out.append(np.einsum("fai,fji->faj", Mc, R))        # M @ R.T per frame
    return (out[0] if single else out), Q


def rmsd_to_reference(fit_xyz, ref_fit, measured, ref_measured):
    """Per-frame RMSD of a measured selection after superposition on a fit group.

    Args:
        fit_xyz (np.ndarray): (F, Nfit, 3) fit-group coordinates.
        ref_fit (np.ndarray): (Nfit, 3) reference fit-group coordinates.
        measured (np.ndarray): (F, M, 3) coordinates whose RMSD is reported.
        ref_measured (np.ndarray): (M, 3) reference coordinates of the same atoms.

    Returns:
        np.ndarray: (F,) RMSD in Angstrom.
    """
    moved, _ = superpose(fit_xyz, ref_fit, measured)
    ref0 = ref_measured - ref_fit.mean(axis=0)              # same translation as `moved`
    diff = moved - ref0[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2).mean(axis=1))


def rmsf_about_mean(fit_xyz, measured, n_iter=2):
    """Per-atom RMSF about the ensemble mean structure.

    Follows the standard cpptraj recipe: fit every frame to the first frame,
    compute the average structure, re-fit to that average, and only then measure
    the fluctuation. Iterating makes the result independent of the (arbitrary)
    choice of first frame as the initial reference.

    Args:
        fit_xyz (np.ndarray): (F, Nfit, 3) fit-group coordinates.
        measured (np.ndarray): (F, M, 3) coordinates whose RMSF is reported.
            Note: if the fit group is also the measured group, pass it twice.
        n_iter (int): number of fit-to-mean refinement passes (default 2).

    Returns:
        (np.ndarray, np.ndarray): (M,) RMSF in Angstrom, and the (M, 3) mean
            structure the fluctuation was measured about.
    """
    ref = fit_xyz[0]
    for _ in range(n_iter):
        moved_fit, _ = superpose(fit_xyz, ref, fit_xyz)
        ref = moved_fit.mean(axis=0)                       # new reference = mean structure

    moved_fit, _ = superpose(fit_xyz, ref, fit_xyz)
    R_ref = ref                                            # converged fit-group reference
    moved, _ = superpose(fit_xyz, R_ref, measured)
    mean_struct = moved.mean(axis=0)
    dev = moved - mean_struct[None, :, :]
    rmsf = np.sqrt((dev ** 2).sum(axis=2).mean(axis=0))
    return rmsf, mean_struct


# =============================================================================
# MISC
# =============================================================================
def frame_times_ns(n_frames=None):
    """Simulation time of each DCD frame, in ns.

    Frame i (0-based) is written at t = (i+1) * TIME_PER_FRAME_NS, because NAMD
    writes the first DCD frame after one DCDfreq interval, not at step 0.

    Args:
        n_frames (int | None): number of frames (default mdconfig.TOTAL_FRAMES).

    Returns:
        np.ndarray: (n_frames,) times in ns.
    """
    n = n_frames or cfg.TOTAL_FRAMES
    return (np.arange(n) + 1) * cfg.TIME_PER_FRAME_NS


def ensure_dir(path):
    """Create a directory (and parents) if it does not exist.

    Args:
        path (str): directory path.

    Returns:
        str: the same path, for chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path
