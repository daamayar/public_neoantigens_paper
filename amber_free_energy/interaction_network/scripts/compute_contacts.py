#!/usr/bin/env python3
"""
Phase 1 -- per-frame peptide<->HLA interaction network.

For every complex x replica this makes ONE pass over the 4166-frame clean
trajectory and writes, per frame, four peptide-residue x HLA-residue matrices
(11 x 180 for the A*11:01 11-mers) plus a set of tracked atom-pair distances:

    dmin      float16 (F, P, R)  minimum heavy-atom distance, Angstrom
    eel       float16 (F, P, R)  Coulomb interaction energy, eps = 1, kcal/mol
    eel_scr   float16 (F, P, R)  Debye-screened Coulomb, eps(r) = 4r, kcal/mol
    vdw       float16 (F, P, R)  Lennard-Jones 6-12 interaction energy, kcal/mol
    nhb       int8    (F, P, R)  hydrogen bonds
    nsb       int8    (F, P, R)  salt bridges (opposite-charge groups < 4.0 A)
    nphob     int8    (F, P, R)  hydrophobic contacts (apolar C/S pairs < 4.5 A)
    ncatpi    int8    (F, P, R)  cation-pi interactions (< 6.0 A, < 60 deg off
                                 the aromatic normal)
    salt      float32 (F, S)     minimum distance between tracked charged groups
    ncontact  int16   (F, P)     heavy-atom contacts of each peptide residue

`eel`, `eel_scr` and `vdw` include EVERY atom pair between the two partners
(hydrogens included, no cutoff, no exclusions -- the partners are separate
molecules so no 1-4 terms apply). Summed over all residue pairs they are
therefore exactly the pair decomposition of MM-PBSA's DELTA EEL and
DELTA VDWAALS, restricted to HLA residues 1-180. `--validate` checks this
against the pipeline's own FINAL_RESULTS_MMPBSA.dat using the full 1-375
receptor and the exact frames MMPBSA used.

These are gas-phase MM terms, NOT free energies: no solvent screening is
applied to `eel`, so charged pairs are hugely overestimated. `eel_scr` is the
qualitative "what survives screening" companion, using a distance-dependent
dielectric eps(r) = 4r and a Debye factor exp(-r/lambda_D) at the pipeline's
0.15 M ionic strength.

Usage
-----
    # both A*11:01 complexes, all 10 replicas, 8 parallel workers (default)
    python -m scripts.compute_contacts

    # one complex / a few replicas, force recomputation
    python -m scripts.compute_contacts --complex pik3ca_e545k_wt_a_1101_strdplseite \\
        --replicas 1,2,3 --force

    # validate the energy code against MM-PBSA before trusting any of it
    python -m scripts.compute_contacts --validate

Idempotent: a replica whose .npz already exists with the expected shapes is
skipped unless --force is given.
"""
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import inconfig as cfg
from . import common


# Frames processed per vectorised chunk. 32 frames x ~550k atom pairs x 4 B
# is ~70 MB per temporary array; with ~5 live temporaries that is ~350 MB per
# worker, so 8 workers stay comfortably inside the machine's memory budget.
CHUNK_FRAMES = 32


# ---------------------------------------------------------------------------
# Pairwise kernels
# ---------------------------------------------------------------------------

def _pair_distances(a, b):
    """Euclidean distances between two coordinate sets, for a chunk of frames.

    Args:
        a (np.ndarray): (n_frames, n_a, 3) float32 coordinates.
        b (np.ndarray): (n_frames, n_b, 3) float32 coordinates.

    Returns:
        np.ndarray: (n_frames, n_a, n_b) float32 distances in Angstrom.
    """
    diff = a[:, :, None, :] - b[:, None, :, :]
    return np.sqrt(np.einsum("fabx,fabx->fab", diff, diff, optimize=True),
                   dtype=np.float32)


def _reduce_pairs(mat, row_starts, col_starts, how):
    """Fold an atom-pair matrix onto residue pairs.

    AMBER stores each residue's atoms contiguously, so a per-residue reduction
    is exactly ``reduceat`` at the residue start offsets -- applied once per
    axis.

    Args:
        mat (np.ndarray): (n_frames, n_atoms_a, n_atoms_b) array.
        row_starts (np.ndarray): start offset of each residue along axis 1.
        col_starts (np.ndarray): start offset of each residue along axis 2.
        how (str): "min" (for distances) or "sum" (for energies).

    Returns:
        np.ndarray: (n_frames, n_res_a, n_res_b) reduced array.
    """
    op = np.minimum if how == "min" else np.add
    return op.reduceat(op.reduceat(mat, row_starts, axis=1), col_starts, axis=2)


def _energies(dist, q_outer, lj_a, lj_b):
    """Coulomb (raw and screened) and Lennard-Jones energies for an atom-pair chunk.

    Args:
        dist (np.ndarray): (n_frames, n_a, n_b) interatomic distances, Angstrom.
        q_outer (np.ndarray): (n_a, n_b) product of partial charges, e^2.
        lj_a (np.ndarray): (n_a, n_b) Lennard-Jones A coefficients.
        lj_b (np.ndarray): (n_a, n_b) Lennard-Jones B coefficients.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (eel, eel_screened, vdw),
            each (n_frames, n_a, n_b) float32, kcal/mol.

    Notes:
        The screened Coulomb uses eps(r) = SCREEN_DIELECTRIC_SLOPE * r together
        with a Debye-Huckel factor exp(-r / DEBYE_LENGTH_A). It is a qualitative
        descriptor of which electrostatic pairs survive aqueous screening, not a
        term of any thermodynamic cycle.
    """
    inv_r = np.reciprocal(dist, dtype=np.float32)
    eel = (cfg.COULOMB_K * q_outer)[None, :, :] * inv_r
    eel_scr = (eel * inv_r / np.float32(cfg.SCREEN_DIELECTRIC_SLOPE)
               * np.exp(-dist / np.float32(cfg.DEBYE_LENGTH_A)))
    inv_r6 = inv_r ** 6
    vdw = lj_a[None, :, :] * inv_r6 * inv_r6 - lj_b[None, :, :] * inv_r6
    return eel, eel_scr, vdw


def _hbonds(xyz_chunk, don_h, acc, don_res, acc_res, n_don_res, n_acc_res):
    """Count hydrogen bonds between one partner's donors and the other's acceptors.

    A hydrogen bond is scored when the hydrogen-to-acceptor distance is below
    ``HBOND_HA_CUTOFF`` and the donor-H...acceptor angle exceeds
    ``HBOND_ANGLE_CUTOFF``.

    Args:
        xyz_chunk (np.ndarray): (n_frames, n_atoms_total, 3) coordinates for the
            chunk, indexed by absolute topology atom index.
        don_h (np.ndarray): (n_donors, 2) array of [donor_idx, hydrogen_idx].
        acc (np.ndarray): (n_acceptors,) acceptor atom indices.
        don_res (np.ndarray): (n_donors,) residue slot of each donor, 0-based.
        acc_res (np.ndarray): (n_acceptors,) residue slot of each acceptor.
        n_don_res (int): number of residue slots on the donor side.
        n_acc_res (int): number of residue slots on the acceptor side.

    Returns:
        np.ndarray: (n_frames, n_don_res, n_acc_res) int16 hydrogen-bond counts.
    """
    nf = xyz_chunk.shape[0]
    out = np.zeros((nf, n_don_res, n_acc_res), dtype=np.int16)
    if don_h.size == 0 or acc.size == 0:
        return out

    d_xyz = xyz_chunk[:, don_h[:, 0], :]
    h_xyz = xyz_chunk[:, don_h[:, 1], :]
    a_xyz = xyz_chunk[:, acc, :]

    ha = a_xyz[:, None, :, :] - h_xyz[:, :, None, :]
    d_ha = np.sqrt(np.einsum("fhax,fhax->fha", ha, ha, optimize=True))
    close = d_ha < cfg.HBOND_HA_CUTOFF
    if not close.any():
        return out

    hd = (d_xyz - h_xyz)[:, :, None, :]              # H -> D vector
    d_hd = np.linalg.norm(hd, axis=-1)
    cos = np.einsum("fhax,fhax->fha", hd, ha, optimize=True) / np.maximum(
        d_hd * d_ha, 1e-6)
    # angle(D-H...A) = 180 deg - angle(H->D, H->A); the criterion
    # angle > 120 deg is therefore cos(H->D, H->A) < cos(60 deg) = 0.5.
    good = close & (cos < np.cos(np.radians(180.0 - cfg.HBOND_ANGLE_CUTOFF)))
    if not good.any():
        return out

    fi, hi, ai = np.nonzero(good)
    np.add.at(out, (fi, don_res[hi], acc_res[ai]), 1)
    return out


def _hydrophobic(xyz_chunk, p_idx, g_idx, p_res, g_res, n_pep, n_grv):
    """Count apolar-atom contacts between the two partners.

    A hydrophobic contact is an apolar carbon or sulfur of the peptide within
    ``HYDROPHOBIC_CUTOFF`` of an apolar carbon or sulfur of the HLA. Apolar
    carbons are those not covalently bonded to nitrogen or oxygen (see
    ``common.build_index_map``), which excludes carbonyl and C-alpha carbons.

    Args:
        xyz_chunk (np.ndarray): (n_frames, n_atoms_total, 3) coordinates.
        p_idx (np.ndarray): peptide apolar atom indices.
        g_idx (np.ndarray): HLA apolar atom indices.
        p_res (np.ndarray): residue slot of each peptide apolar atom.
        g_res (np.ndarray): residue slot of each HLA apolar atom.
        n_pep (int): number of peptide residues.
        n_grv (int): number of groove residues.

    Returns:
        np.ndarray: (n_frames, n_pep, n_grv) int16 contact counts.
    """
    nf = xyz_chunk.shape[0]
    out = np.zeros((nf, n_pep, n_grv), dtype=np.int16)
    if p_idx.size == 0 or g_idx.size == 0:
        return out
    d = _pair_distances(xyz_chunk[:, p_idx, :], xyz_chunk[:, g_idx, :])
    fi, ai, bi = np.nonzero(d < cfg.HYDROPHOBIC_CUTOFF)
    if fi.size:
        np.add.at(out, (fi, p_res[ai], g_res[bi]), 1)
    return out


def _salt_bridges(xyz_chunk, p_groups, g_groups, n_pep, n_grv):
    """Count salt bridges between oppositely charged groups.

    A salt bridge is scored when any atom of a cationic group comes within
    ``SALT_BRIDGE_CUTOFF`` of any atom of an anionic group. Like-charged pairs
    are never counted.

    Args:
        xyz_chunk (np.ndarray): (n_frames, n_atoms_total, 3) coordinates.
        p_groups (list): peptide groups as (residue_slot, atom_indices, sign).
        g_groups (list): HLA groups in the same form.
        n_pep (int): number of peptide residues.
        n_grv (int): number of groove residues.

    Returns:
        np.ndarray: (n_frames, n_pep, n_grv) int16 salt-bridge counts.
    """
    nf = xyz_chunk.shape[0]
    out = np.zeros((nf, n_pep, n_grv), dtype=np.int16)
    for p_slot, p_idx, p_sign in p_groups:
        for g_slot, g_idx, g_sign in g_groups:
            if p_sign * g_sign >= 0:
                continue                       # like charges: never a bridge
            d = _pair_distances(xyz_chunk[:, p_idx, :], xyz_chunk[:, g_idx, :])
            hit = d.reshape(nf, -1).min(axis=1) < cfg.SALT_BRIDGE_CUTOFF
            out[hit, p_slot, g_slot] += 1
    return out


def _ring_frames(xyz_chunk, rings):
    """Centroids and unit normals of a set of aromatic rings, per frame.

    The normal is taken from the singular vector of smallest variance of the
    centred ring coordinates, which is the least-squares plane normal and is
    robust to the small out-of-plane puckering seen in MD.

    Args:
        xyz_chunk (np.ndarray): (n_frames, n_atoms_total, 3) coordinates.
        rings (list[np.ndarray]): one atom-index array per ring.

    Returns:
        tuple[np.ndarray, np.ndarray]: centroids (n_frames, n_rings, 3) and
            unit normals (n_frames, n_rings, 3).
    """
    nf = xyz_chunk.shape[0]
    cent = np.empty((nf, len(rings), 3), dtype=np.float32)
    norm = np.empty((nf, len(rings), 3), dtype=np.float32)
    for k, idx in enumerate(rings):
        pts = xyz_chunk[:, idx, :].astype(np.float64)
        c = pts.mean(axis=1)
        cent[:, k, :] = c
        _, _, vt = np.linalg.svd(pts - c[:, None, :], full_matrices=False)
        norm[:, k, :] = vt[:, 2, :]            # smallest-variance direction
    return cent, norm


def _cation_pi(xyz_chunk, cations, cation_res, rings, ring_res,
               n_cat_res, n_ring_res):
    """Count cation-pi interactions between one partner's cations and the other's rings.

    Scored when the cation centre lies within ``CATION_PI_CUTOFF`` of the ring
    centroid AND within ``CATION_PI_ANGLE`` of the ring normal, i.e. above the
    aromatic face rather than in its plane.

    Args:
        xyz_chunk (np.ndarray): (n_frames, n_atoms_total, 3) coordinates.
        cations (list[np.ndarray]): atom indices defining each cation centre
            (averaged if more than one atom).
        cation_res (np.ndarray): residue slot of each cation.
        rings (list[np.ndarray]): atom indices of each aromatic ring.
        ring_res (np.ndarray): residue slot of each ring.
        n_cat_res (int): number of residue slots on the cation side.
        n_ring_res (int): number of residue slots on the ring side.

    Returns:
        np.ndarray: (n_frames, n_cat_res, n_ring_res) int16 counts.
    """
    nf = xyz_chunk.shape[0]
    out = np.zeros((nf, n_cat_res, n_ring_res), dtype=np.int16)
    if not cations or not rings:
        return out

    cat = np.stack([xyz_chunk[:, idx, :].mean(axis=1) for idx in cations],
                   axis=1).astype(np.float64)            # (nf, n_cat, 3)
    cent, norm = _ring_frames(xyz_chunk, rings)

    v = cat[:, :, None, :] - cent[:, None, :, :]         # (nf, n_cat, n_ring, 3)
    dist = np.linalg.norm(v, axis=-1)
    close = dist < cfg.CATION_PI_CUTOFF
    if not close.any():
        return out
    cos = np.abs(np.einsum("fcrx,frx->fcr", v, norm.astype(np.float64))
                 / np.maximum(dist, 1e-6))
    good = close & (cos > np.cos(np.radians(cfg.CATION_PI_ANGLE)))
    fi, ci, ri = np.nonzero(good)
    if fi.size:
        np.add.at(out, (fi, cation_res[ci], ring_res[ri]), 1)
    return out


# ---------------------------------------------------------------------------
# Tracked charged-group pairs
# ---------------------------------------------------------------------------

def _salt_pairs(imap):
    """Enumerate every peptide charged group x HLA charged group pair to track.

    Args:
        imap (dict): index map from ``common.build_index_map``.

    Returns:
        tuple:
            - list[str]: human-readable pair names, e.g.
              "P11K(NZ)--Asp116" or "P11-COO---Lys146".
            - list[tuple[np.ndarray, np.ndarray, int]]: for each pair, the
              peptide atom indices, the HLA atom indices, and the product of
              the two formal charges (-1 attractive, +1 repulsive).
    """
    names, sels = [], []
    pep_first = int(imap["pep_resids"][0])
    for (p_res, p_kind), (p_idx, p_sign) in sorted(imap["pep_charged"].items()):
        pos = p_res - pep_first + 1
        p_name = (f"P{pos}-COO-" if p_kind == "ct"
                  else f"P{pos}{imap['seq'][pos - 1]}")
        for (g_res, g_kind), (g_idx, g_sign) in sorted(imap["groove_charged"].items()):
            if g_kind == "ct":
                continue                       # the HLA C-terminus is not in 1-180
            g_name = common.residue_label(
                g_res, imap["groove_resnames"][g_res - cfg.GROOVE_RANGE[0]])
            names.append(f"{p_name}--{g_name}")
            sels.append((p_idx, g_idx, p_sign * g_sign))
    return names, sels


# ---------------------------------------------------------------------------
# Per-replica driver
# ---------------------------------------------------------------------------

def _expected_shapes(imap):
    """Array shapes a complete Phase-1 output must have.

    Args:
        imap (dict): index map from ``common.build_index_map``.

    Returns:
        dict: mapping of npz key -> expected shape tuple.
    """
    p, r = imap["pep_len"], cfg.N_GROOVE_RES
    f = cfg.TOTAL_FRAMES
    return {"dmin": (f, p, r), "eel": (f, p, r), "eel_scr": (f, p, r),
            "vdw": (f, p, r), "nhb": (f, p, r), "nsb": (f, p, r),
            "nphob": (f, p, r), "ncatpi": (f, p, r), "ncontact": (f, p)}


def _output_is_valid(path, imap):
    """Check an existing Phase-1 npz has every expected array at the right shape.

    Args:
        path (str): path to the .npz file.
        imap (dict): index map for the complex.

    Returns:
        bool: True if the file is present and complete.
    """
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as d:
            for k, shape in _expected_shapes(imap).items():
                if k not in d or d[k].shape != shape:
                    return False
    except Exception:
        return False
    return True


def compute_replica(dirname, rep, force=False, groove_only=True):
    """Compute and store the per-frame interaction network of one replica.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).
        force (bool): recompute even if a valid output already exists.
        groove_only (bool): restrict the HLA side to residues 1-180. Set False
            only for validation against MM-PBSA, which uses the full receptor.

    Returns:
        dict: {"dirname", "rep", "status", "seconds", "message"} with status
            one of "computed", "skipped", "failed".
    """
    t0 = time.time()
    out_path = common.contacts_npz_path(dirname, rep)
    common.ensure_dir(os.path.dirname(out_path))

    try:
        imap = common.build_index_map(dirname, rep)
        if not force and _output_is_valid(out_path, imap):
            return dict(dirname=dirname, rep=rep, status="skipped",
                        seconds=0.0, message="output already complete")

        xyz = common.load_coordinates(dirname, rep)
        n_frames = xyz.shape[0]
        n_pep = imap["pep_len"]
        n_grv = cfg.N_GROOVE_RES

        ph, gh = imap["pep_heavy"], imap["groove_heavy"]
        pa, ga = imap["pep_all"], imap["groove_all"]
        ph_s, gh_s = imap["pep_heavy_starts"], imap["groove_heavy_starts"]
        pa_s, ga_s = imap["pep_all_starts"], imap["groove_all_starts"]

        # Pre-broadcast the pair-constant force-field terms once.
        q = imap["charge"]
        q_outer = np.outer(q[pa], q[ga]).astype(np.float32)
        ti, tj = imap["lj_type"][pa], imap["lj_type"][ga]
        lj_a = imap["lj_a"][np.ix_(ti, tj)].astype(np.float32)
        lj_b = imap["lj_b"][np.ix_(ti, tj)].astype(np.float32)

        # Hydrogen-bond bookkeeping: residue slot of every donor/acceptor.
        ares = imap["atom_res"]
        pep_off = cfg.PEPTIDE_START - 1
        grv_off = cfg.GROOVE_RANGE[0] - 1
        p_dh, p_acc = imap["pep_don_h"], imap["pep_acc"]
        g_dh, g_acc = imap["groove_don_h"], imap["groove_acc"]
        p_dh_res = ares[p_dh[:, 0]] - pep_off if p_dh.size else np.zeros(0, int)
        p_acc_res = ares[p_acc] - pep_off
        g_dh_res = ares[g_dh[:, 0]] - grv_off if g_dh.size else np.zeros(0, int)
        g_acc_res = ares[g_acc] - grv_off

        salt_names, salt_sels = _salt_pairs(imap)

        # Apolar-atom bookkeeping (hydrophobic contacts).
        p_apol, g_apol = imap["pep_apolar"], imap["groove_apolar"]
        p_apol_res = ares[p_apol] - pep_off if p_apol.size else np.zeros(0, int)
        g_apol_res = ares[g_apol] - grv_off if g_apol.size else np.zeros(0, int)

        # Charged groups as (residue_slot, atom_indices, sign) triples.
        pep_groups = [(res - cfg.PEPTIDE_START, idx, sign)
                      for (res, _kind), (idx, sign) in imap["pep_charged"].items()]
        grv_groups = [(res - cfg.GROOVE_RANGE[0], idx, sign)
                      for (res, _kind), (idx, sign)
                      in imap["groove_charged"].items()]

        out = {
            "dmin": np.empty((n_frames, n_pep, n_grv), np.float16),
            "eel": np.empty((n_frames, n_pep, n_grv), np.float16),
            "eel_scr": np.empty((n_frames, n_pep, n_grv), np.float16),
            "vdw": np.empty((n_frames, n_pep, n_grv), np.float16),
            "nhb": np.empty((n_frames, n_pep, n_grv), np.int8),
            "nsb": np.empty((n_frames, n_pep, n_grv), np.int8),
            "nphob": np.empty((n_frames, n_pep, n_grv), np.int8),
            "ncatpi": np.empty((n_frames, n_pep, n_grv), np.int8),
            "ncontact": np.empty((n_frames, n_pep), np.int16),
            "salt": np.empty((n_frames, len(salt_sels)), np.float32),
        }

        for lo in range(0, n_frames, CHUNK_FRAMES):
            hi = min(lo + CHUNK_FRAMES, n_frames)
            chunk = xyz[lo:hi]

            # --- contacts (heavy atoms only) ---------------------------------
            d_hv = _pair_distances(chunk[:, ph, :], chunk[:, gh, :])
            out["dmin"][lo:hi] = _reduce_pairs(d_hv, ph_s, gh_s, "min")
            out["ncontact"][lo:hi] = np.add.reduceat(
                (d_hv < cfg.CONTACT_CUTOFF).sum(axis=2), ph_s, axis=1)
            del d_hv

            # --- energies (all atoms) ----------------------------------------
            d_all = _pair_distances(chunk[:, pa, :], chunk[:, ga, :])
            eel, eel_scr, vdw = _energies(d_all, q_outer, lj_a, lj_b)
            del d_all
            out["eel"][lo:hi] = _reduce_pairs(eel, pa_s, ga_s, "sum")
            out["eel_scr"][lo:hi] = _reduce_pairs(eel_scr, pa_s, ga_s, "sum")
            out["vdw"][lo:hi] = _reduce_pairs(vdw, pa_s, ga_s, "sum")
            del eel, eel_scr, vdw

            # --- hydrogen bonds (both directions) -----------------------------
            nhb = _hbonds(chunk, p_dh, g_acc, p_dh_res, g_acc_res, n_pep, n_grv)
            nhb += _hbonds(chunk, g_dh, p_acc, g_dh_res, p_acc_res,
                           n_grv, n_pep).transpose(0, 2, 1)
            out["nhb"][lo:hi] = np.clip(nhb, 0, 127).astype(np.int8)

            # --- salt bridges, hydrophobic contacts, cation-pi ----------------
            out["nsb"][lo:hi] = np.clip(
                _salt_bridges(chunk, pep_groups, grv_groups, n_pep, n_grv),
                0, 127).astype(np.int8)
            out["nphob"][lo:hi] = np.clip(
                _hydrophobic(chunk, p_apol, g_apol, p_apol_res, g_apol_res,
                             n_pep, n_grv), 0, 127).astype(np.int8)
            catpi = _cation_pi(chunk, imap["pep_cations"], imap["pep_cation_res"],
                               imap["groove_rings"], imap["groove_ring_res"],
                               n_pep, n_grv)
            catpi += _cation_pi(chunk, imap["groove_cations"],
                                imap["groove_cation_res"], imap["pep_rings"],
                                imap["pep_ring_res"], n_grv,
                                n_pep).transpose(0, 2, 1)
            out["ncatpi"][lo:hi] = np.clip(catpi, 0, 127).astype(np.int8)

            # --- tracked charged-group distances ------------------------------
            for k, (pi, gi, _sign) in enumerate(salt_sels):
                dd = _pair_distances(chunk[:, pi, :], chunk[:, gi, :])
                out["salt"][lo:hi, k] = dd.reshape(hi - lo, -1).min(axis=1)

        np.savez_compressed(
            out_path,
            salt_names=np.array(salt_names, dtype=object),
            salt_sign=np.array([s for _, _, s in salt_sels], dtype=np.int8),
            pep_resids=imap["pep_resids"],
            groove_resids=imap["groove_resids"],
            groove_resnames=np.array(imap["groove_resnames"], dtype=object),
            pep_resnames=np.array(imap["pep_resnames"], dtype=object),
            seq=np.array(imap["seq"]),
            **out)
        return dict(dirname=dirname, rep=rep, status="computed",
                    seconds=time.time() - t0, message="")

    except Exception as exc:                                       # noqa: BLE001
        if os.path.exists(out_path):
            os.unlink(out_path)             # never leave a partial output behind
        import traceback
        return dict(dirname=dirname, rep=rep, status="failed",
                    seconds=time.time() - t0,
                    message=f"{exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Validation against MM-PBSA
# ---------------------------------------------------------------------------

def _parse_mmpbsa_delta(path):
    """Read the DELTA VDWAALS/EEL averages and `indi` from FINAL_RESULTS_MMPBSA.dat.

    Args:
        path (str): path to the MM-PBSA results file.

    Returns:
        dict: {"VDWAALS": float, "EEL": float, "indi": float}. The energies come
            from the "Differences (Complex - Receptor - Ligand)" block, in
            kcal/mol; ``indi`` is the solute dielectric echoed in the input-file
            header of the same file.

    Notes:
        The reported EEL is the Coulomb energy computed with the SOLUTE
        DIELECTRIC: when `indi` != 1, sander's PB path divides the Coulomb sum
        by `indi`, so ``EEL_reported = EEL_vacuum / indi``. This pipeline runs
        with indi = 4.0, so its EEL values are one quarter of the vacuum
        Coulomb energy. The convention is self-consistent (EPB is computed with
        the same interior dielectric), but any comparison against an
        independently computed Coulomb sum must apply the same factor.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if the DELTA block cannot be located.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as fh:
        lines = fh.readlines()

    indi = 1.0
    for ln in lines[:60]:
        if "indi" in ln and "=" in ln:
            try:
                indi = float(ln.split("indi")[1].split("=")[1].split(",")[0])
            except (IndexError, ValueError):
                pass

    start = next((i for i, ln in enumerate(lines)
                  if "Differences" in ln and "Complex" in ln), None)
    if start is None:
        raise ValueError(f"no DELTA block in {path}")
    out = {}
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) >= 2 and parts[0] in ("VDWAALS", "EEL"):
            out[parts[0]] = float(parts[1])
        if len(out) == 2:
            break
    if len(out) != 2:
        raise ValueError(f"incomplete DELTA block in {path}")
    out["indi"] = indi
    return out


def validate(dirname, rep=1):
    """Check the MM energy code against the pipeline's own MM-PBSA numbers.

    Recomputes DELTA EEL and DELTA VDWAALS from scratch on the *exact* frames
    MM-PBSA used (``mmpbsa_500ns/rep{rep}_dry_500ns.nc``, 375 frames) with the
    FULL receptor (residues 1-375, matching MM-PBSA's receptor mask) and
    compares the frame averages with ``FINAL_RESULTS_MMPBSA.dat``.

    Agreement to well under 1 kcal/mol on a ~1000 kcal/mol quantity confirms
    that the charges, the Lennard-Jones A/B coefficients (including any CHARMM
    NBFIX entries) and the pair bookkeeping are all correct.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based). Default 1.

    Returns:
        dict: {"EEL_mine", "EEL_mmpbsa", "EEL_diff", "VDW_mine", "VDW_mmpbsa",
            "VDW_diff", "n_frames"} -- all energies in kcal/mol.
    """
    from scipy.io import netcdf_file
    import parmed as pmd

    rdir = common.replica_dir(dirname, rep)
    traj = os.path.join(rdir, "mmpbsa_500ns", f"rep{rep}_dry_500ns.nc")
    parm = pmd.load_file(common.prmtop_path(dirname, rep))

    rec = parm.residues[:cfg.RECEPTOR_RANGE[1]]
    pep = parm.residues[cfg.PEPTIDE_START - 1:]
    rec_idx = np.array([a.idx for r in rec for a in r.atoms], dtype=np.int64)
    pep_idx = np.array([a.idx for r in pep for a in r.atoms], dtype=np.int64)

    imap = common.build_index_map(dirname, rep)
    q = imap["charge"]
    q_outer = np.outer(q[pep_idx], q[rec_idx]).astype(np.float64)
    ti, tj = imap["lj_type"][pep_idx], imap["lj_type"][rec_idx]
    lj_a = imap["lj_a"][np.ix_(ti, tj)]
    lj_b = imap["lj_b"][np.ix_(ti, tj)]

    with netcdf_file(traj, "r", mmap=False) as fh:
        xyz = np.asarray(fh.variables["coordinates"][:], dtype=np.float64)

    eel_sum = vdw_sum = 0.0
    for i in range(xyz.shape[0]):
        diff = xyz[i, pep_idx][:, None, :] - xyz[i, rec_idx][None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        inv = 1.0 / d
        eel_sum += float((cfg.COULOMB_K * q_outer * inv).sum())
        inv6 = inv ** 6
        vdw_sum += float((lj_a * inv6 * inv6 - lj_b * inv6).sum())
    n = xyz.shape[0]
    ref = _parse_mmpbsa_delta(os.path.join(rdir, cfg.MMPBSA_RESULTS))
    # Compare like with like: rescale the vacuum Coulomb sum by the solute
    # dielectric MM-PBSA ran with (see _parse_mmpbsa_delta).
    mine = {"EEL": eel_sum / n / ref["indi"], "VDWAALS": vdw_sum / n}

    return {
        "n_frames": n, "indi": ref["indi"],
        "EEL_vacuum": eel_sum / n,
        "EEL_mine": mine["EEL"], "EEL_mmpbsa": ref["EEL"],
        "EEL_diff": mine["EEL"] - ref["EEL"],
        "VDW_mine": mine["VDWAALS"], "VDW_mmpbsa": ref["VDWAALS"],
        "VDW_diff": mine["VDWAALS"] - ref["VDWAALS"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Phase 1: per-frame peptide<->HLA interaction network.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--complex", action="append", default=None,
                    help="complex directory name (repeatable; default: both "
                         "A*11:01 systems)")
    ap.add_argument("--replicas", default=None,
                    help="comma-separated replica numbers (default 1-10)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel replicas (default 8)")
    ap.add_argument("--force", action="store_true",
                    help="recompute replicas that already have output")
    ap.add_argument("--validate", action="store_true",
                    help="check the MM energy code against MM-PBSA and exit")
    args = ap.parse_args()

    names = args.complex or [s["dirname"] for s in cfg.SYSTEMS]
    reps = ([int(x) for x in args.replicas.split(",")] if args.replicas
            else list(range(1, cfg.NUM_REPLICAS + 1)))

    if args.validate:
        print("Validating MM interaction energies against MM-PBSA")
        print("(full receptor 1-375, exact MM-PBSA frames)\n")
        ok = True
        for name in names:
            v = validate(name, reps[0])
            print(f"  {name}  rep_{reps[0]}  ({v['n_frames']} frames, "
                  f"indi={v['indi']})")
            print(f"    Coulomb (vacuum, eps=1)  = {v['EEL_vacuum']:12.3f}")
            print(f"    DELTA EEL     mine = {v['EEL_mine']:12.3f}   "
                  f"MMPBSA = {v['EEL_mmpbsa']:12.3f}   "
                  f"diff = {v['EEL_diff']:+.4f} kcal/mol")
            print(f"    DELTA VDWAALS mine = {v['VDW_mine']:12.3f}   "
                  f"MMPBSA = {v['VDW_mmpbsa']:12.3f}   "
                  f"diff = {v['VDW_diff']:+.4f} kcal/mol")
            if abs(v["EEL_diff"]) > 0.5 or abs(v["VDW_diff"]) > 0.5:
                ok = False
                print("    *** MISMATCH ***")
        print("\nvalidation:", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)

    jobs = [(n, r) for n in names for r in reps]
    print(f"Phase 1: {len(jobs)} replica jobs, {args.workers} workers")
    t0 = time.time()
    done = computed = skipped = failed = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(compute_replica, n, r, args.force): (n, r)
                for n, r in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res["status"] == "computed":
                computed += 1
            elif res["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(res)
                print(f"  [FAIL] {res['dirname']} rep_{res['rep']}: "
                      f"{res['message'][:400]}")
            el = time.time() - t0
            eta = (len(jobs) - done) / (done / el) if done else 0
            print(f"  {done}/{len(jobs)}  computed={computed} "
                  f"skipped={skipped} failed={failed}  "
                  f"elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min: computed={computed} "
          f"skipped={skipped} failed={failed}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
