#!/usr/bin/env python3
"""
Phase 2 -- aggregate the per-frame networks into comparable statistics.

Reads the Phase-1 .npz files and produces, for each conditioning window
(see ``inconfig.WINDOWS``), a JSON summary containing:

  * ``occupancy``   -- peptide-residue x HLA-residue contact occupancy
                       (fraction of window frames with min heavy-atom distance
                       below CONTACT_CUTOFF), per replica and ensemble-averaged.
  * ``hbond``       -- mean number of hydrogen bonds per residue pair.
  * ``saltbridge``  -- mean number of salt bridges per residue pair.
  * ``hydrophobic`` -- mean number of apolar-atom contacts per residue pair.
  * ``cation_pi``   -- mean number of cation-pi interactions per residue pair.
  * ``energy``      -- mean Coulomb / screened-Coulomb / Lennard-Jones
                       interaction energy per residue pair.
  * ``per_position``-- everything above collapsed onto the 11 peptide positions,
                       with per-replica values so a Mann-Whitney test can be run
                       on the replica-level distributions.
  * ``salt``        -- occupancy and mean distance of every tracked charged-group
                       pair, per replica.
  * ``network``     -- graph descriptors of the weighted bipartite residue
                       interaction network (edge weight = contact occupancy):
                       total edge weight, edge count, per-position weighted
                       degree, and betweenness centrality of every node.

and a cross-complex comparison JSON (neo vs wt) holding the per-position and
per-edge differences together with replica-level Mann-Whitney p-values.

Two standalone analyses are also provided:


  ``--cross-system`` Relate the P-Omega (C-terminal anchor) residue identity of
                    all 17 simulated complexes to their observed C-terminal
                    unbinding frequency, using only the existing unbinding
                    summaries. This is the dataset-wide control for the
                    mechanism proposed for the A*11:01 pair.

Usage
-----
    python -m scripts.analyze_network                  # all windows, both systems
    python -m scripts.analyze_network --window bound   # one window
    python -m scripts.analyze_network --cross-system
"""
import os
import json
import argparse

import numpy as np

from . import inconfig as cfg
from . import common


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_replica(dirname, rep):
    """Load one Phase-1 output into memory, with energies cast back to float32.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).

    Returns:
        dict: arrays keyed as in the .npz, with ``dmin``/``eel``/``eel_scr``/
            ``vdw`` promoted from float16 to float32 and metadata kept as
            Python lists/strings.

    Raises:
        FileNotFoundError: if Phase 1 has not been run for this replica.
    """
    path = common.contacts_npz_path(dirname, rep)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing Phase-1 output: {path}\n"
            f"  Fix: python -m scripts.compute_contacts --complex {dirname} "
            f"--replicas {rep}")
    with np.load(path, allow_pickle=True) as d:
        out = {k: (d[k].astype(np.float32) if d[k].dtype == np.float16 else d[k])
               for k in ("dmin", "eel", "eel_scr", "vdw", "nhb", "nsb",
                         "nphob", "ncatpi", "ncontact", "salt")}
        out["salt_names"] = [str(x) for x in d["salt_names"]]
        out["salt_sign"] = d["salt_sign"]
        out["groove_resids"] = d["groove_resids"]
        out["groove_resnames"] = [str(x) for x in d["groove_resnames"]]
        out["pep_resnames"] = [str(x) for x in d["pep_resnames"]]
        out["seq"] = str(d["seq"])
    return out


# ---------------------------------------------------------------------------
# Per-replica window statistics
# ---------------------------------------------------------------------------

def replica_window_stats(dirname, rep, window):
    """Reduce one replica's per-frame arrays over one conditioning window.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).
        window (str): key of ``inconfig.WINDOWS``.

    Returns:
        dict | None: per-replica statistics, or None if the window selects no
            frames at all (possible for "bound" in a fully-detached replica).
            Keys: ``n_frames``, ``occupancy`` (P x R), ``hbond`` (P x R),
            ``eel``/``eel_scr``/``vdw`` (P x R means), ``salt_occ``/``salt_dist``
            (S,), ``n_contacts`` (P,), and the scalar totals ``tot_occupancy``,
            ``tot_hbond``, ``tot_eel``, ``tot_eel_scr``, ``tot_vdw``.
    """
    d = load_replica(dirname, rep)
    mask = common.window_mask(dirname, rep, window)
    n = int(mask.sum())
    if n == 0:
        return None

    dmin = d["dmin"][mask]
    contact = dmin < cfg.CONTACT_CUTOFF
    occ = contact.mean(axis=0)
    hb = d["nhb"][mask].mean(axis=0)
    sb = d["nsb"][mask].mean(axis=0)
    phob = d["nphob"][mask].mean(axis=0)
    catpi = d["ncatpi"][mask].mean(axis=0)
    eel = d["eel"][mask].mean(axis=0)
    eel_scr = d["eel_scr"][mask].mean(axis=0)
    vdw = d["vdw"][mask].mean(axis=0)

    salt = d["salt"][mask]
    salt_occ = (salt < cfg.SALT_BRIDGE_CUTOFF).mean(axis=0)
    salt_dist = salt.mean(axis=0)

    return dict(
        n_frames=n,
        occupancy=occ, hbond=hb, saltbridge=sb, hydrophobic=phob,
        cation_pi=catpi, eel=eel, eel_scr=eel_scr, vdw=vdw,
        salt_occ=salt_occ, salt_dist=salt_dist,
        n_contacts=d["ncontact"][mask].mean(axis=0),
        tot_occupancy=float(occ.sum()), tot_hbond=float(hb.sum()),
        tot_saltbridge=float(sb.sum()), tot_hydrophobic=float(phob.sum()),
        tot_cation_pi=float(catpi.sum()),
        tot_eel=float(eel.sum()), tot_eel_scr=float(eel_scr.sum()),
        tot_vdw=float(vdw.sum()),
        salt_names=d["salt_names"], groove_resids=d["groove_resids"],
        groove_resnames=d["groove_resnames"], seq=d["seq"],
    )


def complex_window_stats(dirname, window, replicas=None):
    """Aggregate every replica of one complex over one conditioning window.

    Args:
        dirname (str): complex directory name.
        window (str): key of ``inconfig.WINDOWS``.
        replicas (list[int] | None): replicas to include (default 1..NUM_REPLICAS).

    Returns:
        dict: with ``per_replica`` (list of the per-replica dicts, None entries
            dropped), ``replicas`` (the replica numbers that contributed),
            ``mean``/``sem`` dicts holding the ensemble mean and SEM of every
            matrix and scalar, and the metadata needed for labelling.
    """
    reps = replicas or list(range(1, cfg.NUM_REPLICAS + 1))
    stats, used = [], []
    for r in reps:
        s = replica_window_stats(dirname, r, window)
        if s is not None:
            stats.append(s)
            used.append(r)
    if not stats:
        raise RuntimeError(f"{dirname}/{window}: no replica had any frame")

    def _stack(key):
        return np.stack([s[key] for s in stats], axis=0)

    mean, sem = {}, {}
    for key in ("occupancy", "hbond", "saltbridge", "hydrophobic",
                "cation_pi", "eel", "eel_scr", "vdw",
                "salt_occ", "salt_dist", "n_contacts"):
        arr = _stack(key)
        mean[key] = arr.mean(axis=0)
        sem[key] = (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                    if arr.shape[0] > 1 else np.zeros_like(arr[0]))
    for key in ("tot_occupancy", "tot_hbond", "tot_saltbridge",
                "tot_hydrophobic", "tot_cation_pi", "tot_eel", "tot_eel_scr",
                "tot_vdw", "n_frames"):
        vals = [s[key] for s in stats]
        mean[key], sem[key] = common.mean_sem(vals)

    return dict(dirname=dirname, window=window, replicas=used,
                per_replica=stats, mean=mean, sem=sem,
                salt_names=stats[0]["salt_names"],
                groove_resids=stats[0]["groove_resids"],
                groove_resnames=stats[0]["groove_resnames"],
                seq=stats[0]["seq"])


# ---------------------------------------------------------------------------
# Network descriptors
# ---------------------------------------------------------------------------

def network_descriptors(occ, seq, groove_resids, groove_resnames,
                        min_weight=0.05):
    """Graph descriptors of the weighted bipartite peptide<->HLA contact network.

    Nodes are peptide positions ("P1".."P11") and HLA residues ("Asp116", ...);
    an edge carries the contact occupancy as its weight. Betweenness centrality
    uses 1/weight as the edge length, so a high-occupancy contact is a *short*
    path and a hub is a residue that many short interface paths run through.

    Args:
        occ (np.ndarray): (P, R) contact-occupancy matrix.
        seq (str): peptide one-letter sequence, length P.
        groove_resids (np.ndarray): (R,) HLA residue numbers.
        groove_resnames (list[str]): (R,) HLA residue names.
        min_weight (float): edges below this occupancy are dropped, to keep the
            graph from being dominated by transient brushing contacts.

    Returns:
        dict: ``total_weight`` (sum of kept edge occupancies), ``n_edges``,
            ``n_hla_partners`` (distinct HLA residues engaged),
            ``degree`` (per peptide position, weighted),
            ``hla_degree`` (dict HLA label -> weighted degree),
            ``betweenness`` (dict node label -> centrality),
            ``edges`` (list of dicts with the retained edges, occupancy-sorted).
    """
    import networkx as nx

    g = nx.Graph()
    p_labels = [f"P{i+1}{seq[i]}" for i in range(occ.shape[0])]
    h_labels = [common.residue_label(int(groove_resids[j]), groove_resnames[j])
                for j in range(occ.shape[1])]

    edges = []
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            w = float(occ[i, j])
            if w >= min_weight:
                g.add_edge(p_labels[i], h_labels[j], weight=w, length=1.0 / w)
                edges.append(dict(peptide=p_labels[i], hla=h_labels[j],
                                  occupancy=w))
    if g.number_of_edges() == 0:
        return dict(total_weight=0.0, n_edges=0, n_hla_partners=0,
                    degree={lab: 0.0 for lab in p_labels},
                    hla_degree={}, betweenness={}, edges=[])

    btw = nx.betweenness_centrality(g, weight="length", normalized=True)
    deg = dict(g.degree(weight="weight"))
    edges.sort(key=lambda e: -e["occupancy"])

    return dict(
        total_weight=float(sum(e["occupancy"] for e in edges)),
        n_edges=len(edges),
        n_hla_partners=len({e["hla"] for e in edges}),
        degree={lab: float(deg.get(lab, 0.0)) for lab in p_labels},
        hla_degree={lab: float(deg[lab]) for lab in h_labels if lab in deg},
        betweenness={k: float(v) for k, v in btw.items()},
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Neo vs wt comparison
# ---------------------------------------------------------------------------

def compare_pair(window, replicas=None):
    """Compare the neoantigen and wild-type networks over one window.

    Args:
        window (str): key of ``inconfig.WINDOWS``.
        replicas (list[int] | None): replicas to include (default all).

    Returns:
        dict: JSON-serialisable comparison containing, for each system, the
            ensemble means and network descriptors; the per-position deltas
            with replica-level Mann-Whitney p-values; the per-edge occupancy
            deltas ranked by magnitude; and the salt-bridge table.
    """
    neo = next(s for s in cfg.SYSTEMS if s["kind"] == "neo")
    wt = next(s for s in cfg.SYSTEMS if s["kind"] == "wt")
    a = complex_window_stats(neo["dirname"], window, replicas)
    b = complex_window_stats(wt["dirname"], window, replicas)

    seq_a, seq_b = a["seq"], b["seq"]
    n_pos = len(seq_a)
    gids, gnames = a["groove_resids"], a["groove_resnames"]

    # ---- per-position summary with replica-level statistics ---------------
    positions = []
    for i in range(n_pos):
        row = dict(position=i + 1, neo_res=seq_a[i], wt_res=seq_b[i],
                   mutated=(seq_a[i] != seq_b[i]))
        for key, agg in (("occupancy", "sum"), ("hbond", "sum"),
                         ("saltbridge", "sum"), ("hydrophobic", "sum"),
                         ("cation_pi", "sum"),
                         ("eel", "sum"), ("eel_scr", "sum"), ("vdw", "sum")):
            va = [float(getattr(np, agg)(s[key][i])) for s in a["per_replica"]]
            vb = [float(getattr(np, agg)(s[key][i])) for s in b["per_replica"]]
            ma, sa = common.mean_sem(va)
            mb, sb = common.mean_sem(vb)
            _, p = common.mann_whitney_u(va, vb)
            row[key] = dict(neo=ma, neo_sem=sa, wt=mb, wt_sem=sb,
                            delta=ma - mb, p=p,
                            neo_per_replica=va, wt_per_replica=vb)
        positions.append(row)

    # ---- per-edge occupancy differences -----------------------------------
    d_occ = a["mean"]["occupancy"] - b["mean"]["occupancy"]
    edges = []
    for i in range(n_pos):
        for j in range(len(gids)):
            if max(abs(d_occ[i, j]), a["mean"]["occupancy"][i, j],
                   b["mean"]["occupancy"][i, j]) < 0.05:
                continue
            va = [float(s["occupancy"][i, j]) for s in a["per_replica"]]
            vb = [float(s["occupancy"][i, j]) for s in b["per_replica"]]
            _, p = common.mann_whitney_u(va, vb)
            edges.append(dict(
                position=i + 1, neo_res=seq_a[i], wt_res=seq_b[i],
                hla=common.residue_label(int(gids[j]), gnames[j]),
                neo_occ=float(a["mean"]["occupancy"][i, j]),
                wt_occ=float(b["mean"]["occupancy"][i, j]),
                delta=float(d_occ[i, j]), p=p,
                neo_eel=float(a["mean"]["eel"][i, j]),
                wt_eel=float(b["mean"]["eel"][i, j]),
                neo_hbond=float(a["mean"]["hbond"][i, j]),
                wt_hbond=float(b["mean"]["hbond"][i, j])))
    # Hundreds of edges are tested against the same 10 vs 10 replicas, so the
    # raw p-values need a false-discovery-rate correction before any edge is
    # called significant.
    q_edges = common.benjamini_hochberg([e["p"] for e in edges])
    for e, q in zip(edges, q_edges):
        e["q"] = float(q)
    edges.sort(key=lambda e: -abs(e["delta"]))

    # ---- salt bridges ------------------------------------------------------
    # Map each neoantigen pair onto its wild-type counterpart. Exact names match
    # everywhere except at the substituted position, where the residue letter is
    # baked into the name ("P11K--Asp116" vs "P11E--Asp116"); those are matched
    # on position + HLA partner instead, so the mutated anchor -- the whole
    # point of the comparison -- is not silently dropped.
    def _counterpart(name):
        if name in b["salt_names"]:
            return b["salt_names"].index(name)
        head, _, tail = name.partition("--")
        if head.endswith("-COO-"):
            return None
        pos = head[1:-1]                      # strip leading "P" and the residue
        for j, other in enumerate(b["salt_names"]):
            oh, _, ot = other.partition("--")
            if ot == tail and not oh.endswith("-COO-") and oh[1:-1] == pos:
                return j
        return None

    salts = []
    for k, name in enumerate(a["salt_names"]):
        kb = _counterpart(name)
        va = [float(s["salt_occ"][k]) for s in a["per_replica"]]
        vb = ([float(s["salt_occ"][kb]) for s in b["per_replica"]]
              if kb is not None else [])
        ma, sa = common.mean_sem(va)
        mb, sb = (common.mean_sem(vb) if vb else (float("nan"), float("nan")))
        if (np.nanmax([ma, mb if vb else 0.0]) < 0.02):
            continue
        _, p = common.mann_whitney_u(va, vb) if vb else (None, float("nan"))
        salts.append(dict(
            pair=name,
            wt_pair=(b["salt_names"][kb] if kb is not None else None),
            neo_occ=ma, neo_sem=sa, wt_occ=mb, wt_sem=sb,
            delta=(ma - mb) if vb else None, p=p,
            neo_dist=float(np.mean([s["salt_dist"][k] for s in a["per_replica"]])),
            wt_dist=(float(np.mean([s["salt_dist"][kb] for s in b["per_replica"]]))
                     if kb is not None else None),
            neo_per_replica=va, wt_per_replica=vb))
    q_salt = common.benjamini_hochberg([s["p"] for s in salts])
    for s, q in zip(salts, q_salt):
        s["q"] = float(q)
    salts.sort(key=lambda s: -(abs(s["delta"]) if s["delta"] is not None else 0))

    # ---- global network descriptors ---------------------------------------
    net_a = network_descriptors(a["mean"]["occupancy"], seq_a, gids, gnames)
    net_b = network_descriptors(b["mean"]["occupancy"], seq_b, gids, gnames)

    # replica-level totals, for testing the headline numbers
    totals = {}
    for key in ("tot_occupancy", "tot_hbond", "tot_saltbridge",
                "tot_hydrophobic", "tot_cation_pi", "tot_eel", "tot_eel_scr",
                "tot_vdw", "n_frames"):
        va = [s[key] for s in a["per_replica"]]
        vb = [s[key] for s in b["per_replica"]]
        ma, sa = common.mean_sem(va)
        mb, sb = common.mean_sem(vb)
        _, p = common.mann_whitney_u(va, vb)
        totals[key] = dict(neo=ma, neo_sem=sa, wt=mb, wt_sem=sb,
                           delta=ma - mb, p=p,
                           neo_per_replica=va, wt_per_replica=vb)

    return dict(
        window=window,
        window_label=cfg.WINDOWS[window]["label"],
        window_description=cfg.WINDOWS[window]["description"],
        neo=dict(dirname=neo["dirname"], label=neo["label"], seq=seq_a,
                 replicas=a["replicas"], network=net_a),
        wt=dict(dirname=wt["dirname"], label=wt["label"], seq=seq_b,
                replicas=b["replicas"], network=net_b),
        totals=totals, positions=positions, edges=edges, salt_bridges=salts,
    )
def cross_system_pomega():
    """Relate P-Omega anchor identity to C-terminal unbinding across all 17 systems.

    Reads only the existing ``unbinding_summary_*.json`` files, so it needs no
    trajectory access. For each complex it records the C-terminal anchor residue
    of the peptide and the fraction of the 10 replicas that were scored with a
    C-terminal unbinding event, plus the mean fraction of time unbound.

    This is the dataset-wide control for the mechanism proposed for the A*11:01
    pair: if a basic P-Omega anchor is what holds the C-terminus in the
    A*11:01 F pocket, then every A*11:01 peptide ending in Lys should be
    C-terminally stable and only the Glu-terminated one should fail.

    Returns:
        dict: ``rows`` (one record per complex: dirname, label, allele,
            sequence, P-Omega residue, C-terminal event count and fraction,
            fraction of time unbound) and ``by_allele`` grouping the A*11:01
            subset, which is the internally controlled comparison.
    """
    rows = []
    for dirname, seq, allele, label in cfg.ALL_COMPLEXES:
        path = os.path.join(cfg.RESULTS_DIR, dirname, cfg.UNBINDING_JSON)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            s = json.load(fh)
        n_c = sum(1 for r in s["per_replica"]
                  if r["events"]["cterm"]["t_first_unbind_ns"] is not None)
        n_n = sum(1 for r in s["per_replica"]
                  if r["events"]["nterm"]["t_first_unbind_ns"] is not None)
        frac_c = float(np.mean([r["events"]["cterm"]["fraction_time_unbound"]
                                for r in s["per_replica"]]))
        frac_n = float(np.mean([r["events"]["nterm"]["fraction_time_unbound"]
                                for r in s["per_replica"]]))
        rows.append(dict(dirname=dirname, label=label, allele=allele, seq=seq,
                         pomega=seq[-1], p2=seq[1], length=len(seq),
                         n_replicas=len(s["per_replica"]),
                         cterm_events=n_c, nterm_events=n_n,
                         cterm_event_fraction=n_c / len(s["per_replica"]),
                         cterm_time_unbound=frac_c,
                         nterm_time_unbound=frac_n))
    a1101 = [r for r in rows if r["allele"] == "A*11:01"]
    return dict(rows=rows,
                by_allele=dict(A1101=sorted(a1101,
                                            key=lambda r: -r["cterm_time_unbound"])))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Recursively convert numpy types/arrays into plain JSON-serialisable data.

    Args:
        obj: any nested combination of dicts, lists, numpy arrays and scalars.

    Returns:
        The same structure with numpy arrays as lists and numpy scalars as
        Python floats/ints. NaN is preserved (``json`` writes it as ``NaN``).
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write(name, payload):
    """Write a JSON payload into the package's data directory.

    Args:
        name (str): file name (without directory).
        payload (dict): data to serialise.

    Returns:
        str: the path written.
    """
    common.ensure_dir(cfg.DATA_DIR)
    path = os.path.join(cfg.DATA_DIR, name)
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=1)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Phase 2: aggregate per-frame networks into comparisons.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--window", action="append", default=None,
                    help="conditioning window (repeatable; default: all)")
    ap.add_argument("--replicas", default=None,
                    help="comma-separated replica numbers (default 1-10)")
    ap.add_argument("--cross-system", action="store_true",
                    help="run the cross-system P-Omega control")
    args = ap.parse_args()

    reps = ([int(x) for x in args.replicas.split(",")] if args.replicas
            else None)

    if args.cross_system:
        print(_write("cross_system_pomega.json", cross_system_pomega()))
        return

    windows = args.window or list(cfg.WINDOWS)
    for w in windows:
        print(f"window '{w}' ({cfg.WINDOWS[w]['label']}) ...", flush=True)
        cmp_ = compare_pair(w, reps)
        path = _write(f"compare_{w}.json", cmp_)
        t = cmp_["totals"]
        print(f"  frames/replica: neo {t['n_frames']['neo']:.0f}, "
              f"wt {t['n_frames']['wt']:.0f}")
        print(f"  total contact occupancy: neo {t['tot_occupancy']['neo']:.2f} "
              f"+/- {t['tot_occupancy']['neo_sem']:.2f}, "
              f"wt {t['tot_occupancy']['wt']:.2f} "
              f"+/- {t['tot_occupancy']['wt_sem']:.2f}  "
              f"(p={t['tot_occupancy']['p']:.4f})")
        print(f"  peptide<->HLA H-bonds:   neo {t['tot_hbond']['neo']:.2f}, "
              f"wt {t['tot_hbond']['wt']:.2f}  (p={t['tot_hbond']['p']:.4f})")
        print(f"  wrote {path}")

    # The windows are also the input to the cross-system view;
    # produce them by default so a single invocation yields a complete dataset.
    print(_write("cross_system_pomega.json", cross_system_pomega()))


if __name__ == "__main__":
    main()
