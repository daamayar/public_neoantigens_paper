#!/usr/bin/env python3
"""
Task 2 -- per-residue peptide RMSF, restricted to two residue windows.

Two plots, both over ALL 17 peptides (7 SNV mutants + 7 wild-types + 3 frameshift
neoantigens), peptide atoms only -- the HLA is excluded from every value.

PLOT A -- termini (exactly 8 x-positions)
    The first 4 and the last 4 residues of each peptide:
        9-mer  -> residues 1,2,3,4 + 6,7,8,9      (residue 5 dropped)
        10-mer -> residues 1,2,3,4 + 7,8,9,10
        11-mer -> residues 1,2,3,4 + 8,9,10,11
    Peptides of different length therefore align on their termini, which is what
    makes the comparison meaningful: P1/P2 and P-Omega are the HLA anchor residues.

PLOT B -- core (exactly 5 x-slots)
    The middle residues, mapped onto 5 shared slots:
        9-mer  -> positions 3,4,5,6,7   -> slots 1,2,3,4,5
        11-mer -> positions 4,5,6,7,8   -> slots 1,2,3,4,5
        10-mer -> positions 4,5,6,7     -> slots   2,3,4,5   (slot 1 left EMPTY)
    A 10-mer has an even length, so it has no single central residue; taking 4
    residues symmetric about the 5.5 midpoint and leaving the first slot empty. 

Protocol
--------
* Superposition: each frame is fitted on the PEPTIDE's own C-alpha atoms (internal
  conformation).
* RMSF is measured on C-alpha atoms, about the ENSEMBLE MEAN structure, using the
  standard cpptraj recipe: fit to the first frame, compute the average structure,
  re-fit to that average, then measure the fluctuation. Iterating makes the result
  independent of the arbitrary choice of first frame as initial reference.
* RMSF is computed INDEPENDENTLY PER REPLICA and then averaged over the 10 replicas,
  with the error bar being the standard error of the mean across replicas. Pooling
  all replicas' frames into one ensemble first would fold inter-replica drift
  (different basins) into what is supposed to be an intra-replica fluctuation, and
  would inflate the RMSF.
* Window: frames 418-4166 (50-500 ns); the first 50 ns is discarded as equilibration.

Outputs
-------
    figures/rmsf/rmsf_A_termini_8pos.png            17 curves, 8 x-positions
    figures/rmsf/rmsf_B_core_5pos.png               17 curves, 5 x-slots
    figures/rmsf/rmsf_A_termini_8pos_panels.png     small multiples (readable)
    figures/rmsf/rmsf_B_core_5pos_panels.png        small multiples (readable)
    data/rmsf_termini8.csv, data/rmsf_core5.csv     mean, SEM and per-replica values
    data/rmsf_full.csv                              every residue of every peptide

Usage
-----
    python -m scripts.analyze_rmsf
"""
import os
import csv
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import mdconfig as cfg
from . import common


TERMINI_LABELS = ["P1", "P2", "P3", "P4", "P$\\Omega$-3", "P$\\Omega$-2", "P$\\Omega$-1", "P$\\Omega$"]
CORE_LABELS = ["M1", "M2", "M3", "M4", "M5"]


def termini8_mapping(pep_len):
    """Map a peptide's first-4 and last-4 residues onto the 8 shared x-positions.

    Args:
        pep_len (int): peptide length (9, 10 or 11).

    Returns:
        (list[int], list[int]): (slot indices 0-7, 0-based residue indices).
    """
    res = [0, 1, 2, 3, pep_len - 4, pep_len - 3, pep_len - 2, pep_len - 1]
    if len(set(res)) != 8:
        raise ValueError(f"peptide too short for a first-4/last-4 split: {pep_len}")
    return list(range(8)), res


def core5_mapping(pep_len):
    """Map a peptide's middle residues onto the 5 shared x-slots.

    9-mer  -> positions 3-7 in slots 1-5
    11-mer -> positions 4-8 in slots 1-5
    10-mer -> positions 4-7 in slots 2-5 (slot 1 empty; a 10-mer has no central residue)

    Args:
        pep_len (int): peptide length (9, 10 or 11).

    Returns:
        (list[int], list[int]): (slot indices 0-4, 0-based residue indices).

    Raises:
        ValueError: for an unsupported peptide length.
    """
    if pep_len == 9:
        return [0, 1, 2, 3, 4], [2, 3, 4, 5, 6]          # positions 3,4,5,6,7
    if pep_len == 10:
        return [1, 2, 3, 4], [3, 4, 5, 6]                # positions 4,5,6,7 (slot 0 empty)
    if pep_len == 11:
        return [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]          # positions 4,5,6,7,8
    raise ValueError(f"no core-5 mapping defined for a {pep_len}-mer")


def compute_system_rmsf(dirname, fit_mode):
    """Per-residue C-alpha RMSF of one system, computed per replica then averaged.

    Args:
        dirname (str): complex directory name.
        fit_mode (str): "peptide" or "receptor" -- the superposition group.

    Returns:
        dict: {"per_rep": (10, pep_len), "mean": (pep_len,), "sem": (pep_len,),
               "seq": str, "pep_len": int}
    """
    imap = common.build_index_map(dirname)
    per_rep = []
    for rep in range(1, cfg.NUM_REPLICAS + 1):
        xyz = common.load_cache(dirname, rep)[cfg.ANALYSIS_SLICE]      # 50-500 ns
        fit = common.fit_coords(xyz, imap, fit_mode)
        ca = common.get_selection(xyz, imap, cfg.RMSF_SELECTION)       # peptide C-alpha
        rmsf, _ = common.rmsf_about_mean(fit, ca)
        per_rep.append(rmsf)
    per_rep = np.array(per_rep)                                        # (10, pep_len)
    return dict(per_rep=per_rep,
                mean=per_rep.mean(axis=0),
                sem=per_rep.std(axis=0, ddof=1) / np.sqrt(per_rep.shape[0]),
                seq=imap["seq"], pep_len=imap["pep_len"])


def _style(s):
    """Line colour / style / marker for a system, so 17 curves stay distinguishable.

    Args:
        s (dict): a mdconfig.SYSTEMS entry.

    Returns:
        dict: matplotlib kwargs (color, linestyle, marker).
    """
    key = s["pair"] if s["pair"] else s["dirname"]
    colour = cfg.SYSTEM_COLORS[key]
    if s["kind"] == "snv_neo":
        return dict(color=colour, linestyle="-", marker="o")
    if s["kind"] == "snv_wt":
        return dict(color=colour, linestyle="--", marker="s")
    return dict(color=colour, linestyle=":", marker="^")               # frameshift


def _overlay(ax, RM, mapping_fn, labels, title):
    """Draw all 17 peptides on one axis for a given residue-window mapping.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        RM (dict): {dirname: compute_system_rmsf() result}.
        mapping_fn (callable): termini8_mapping or core5_mapping.
        labels (list[str]): x tick labels.
        title (str): axis title.

    Returns:
        None.
    """
    for s in cfg.SYSTEMS:
        r = RM[s["dirname"]]
        slots, res = mapping_fn(r["pep_len"])
        ax.errorbar(slots, r["mean"][res], yerr=r["sem"][res],
                    capsize=2, lw=1.5, ms=4, elinewidth=0.9, alpha=0.9,
                    label=f"{s['label']}", **_style(s))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_ylabel("C$\\alpha$ RMSF ($\\AA$)")
    ax.set_title(title, fontsize=11.5)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    ap = argparse.ArgumentParser(description="Task 2: peptide RMSF (termini + core).")
    args = ap.parse_args()
    fit_mode = cfg.FIT_MODE
    suffix = ""

    figdir = common.ensure_dir(os.path.join(cfg.FIG_DIR, "rmsf"))
    common.ensure_dir(cfg.DATA_DIR)

    print(f"Computing per-residue C-alpha RMSF (fit = {fit_mode} C-alpha, "
          f"per replica then averaged, 50-500 ns) ...")
    RM = {}
    for s in cfg.SYSTEMS:
        RM[s["dirname"]] = compute_system_rmsf(s["dirname"], fit_mode)
        r = RM[s["dirname"]]
        print(f"  {s['label']:28s} {r['seq']:11s} "
              f"RMSF {r['mean'].min():.2f}-{r['mean'].max():.2f} A  "
              f"(mean {r['mean'].mean():.2f})")

    # ---------------- CSVs ----------------
    with open(os.path.join(cfg.DATA_DIR, f"rmsf_full{suffix}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["complex", "label", "allele", "kind", "peptide", "length",
                    "position", "residue", "rmsf_mean_A", "rmsf_sem_A"] +
                   [f"rep{i}_A" for i in range(1, cfg.NUM_REPLICAS + 1)])
        for s in cfg.SYSTEMS:
            r = RM[s["dirname"]]
            for i in range(r["pep_len"]):
                w.writerow([s["dirname"], s["label"], s["allele"], s["kind"],
                            r["seq"], r["pep_len"], i + 1, r["seq"][i],
                            f"{r['mean'][i]:.4f}", f"{r['sem'][i]:.4f}"] +
                           [f"{x:.4f}" for x in r["per_rep"][:, i]])

    for name, fn, labels in [("termini8", termini8_mapping, TERMINI_LABELS),
                             ("core5", core5_mapping, CORE_LABELS)]:
        with open(os.path.join(cfg.DATA_DIR, f"rmsf_{name}{suffix}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["complex", "label", "kind", "peptide", "length",
                        "slot", "slot_label", "position", "residue",
                        "rmsf_mean_A", "rmsf_sem_A"] +
                       [f"rep{i}_A" for i in range(1, cfg.NUM_REPLICAS + 1)])
            for s in cfg.SYSTEMS:
                r = RM[s["dirname"]]
                slots, res = fn(r["pep_len"])
                for sl, ri in zip(slots, res):
                    plain = labels[sl].replace("$\\Omega$", "Omega")
                    w.writerow([s["dirname"], s["label"], s["kind"], r["seq"], r["pep_len"],
                                sl + 1, plain, ri + 1, r["seq"][ri],
                                f"{r['mean'][ri]:.4f}", f"{r['sem'][ri]:.4f}"] +
                               [f"{x:.4f}" for x in r["per_rep"][:, ri]])

    # ---------------- PLOT A: termini, 8 positions ----------------
    # The legend is placed OUTSIDE the axes: 17 curves inside one panel leave no
    # empty region big enough for it, and an inset legend covered the data.
    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    _overlay(ax, RM, termini8_mapping, TERMINI_LABELS,
             "Peptide C$\\alpha$ RMSF - first 4 and last 4 residues (all 17 peptides)")
    ax.set_xlabel("Peptide position (P1-P4 = N-terminal; "
                  "P$\\Omega$ = C-terminal residue)")
    ax.axvline(3.5, color="0.4", ls="-", lw=1.0, alpha=0.6)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.10 * (y1 - y0))
    ax.text(1.75, ax.get_ylim()[1] * 0.985, "N-terminal 4", ha="center", va="top",
            fontsize=9.5, color="0.3")
    ax.text(5.75, ax.get_ylim()[1] * 0.985, "C-terminal 4", ha="center", va="top",
            fontsize=9.5, color="0.3")
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="solid = mutant\ndashed = wild-type\ndotted = frameshift",
              title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"rmsf_A_termini_8pos{suffix}.png"), dpi=cfg.DPI)
    plt.close(fig)

    # ---------------- PLOT B: core, 5 slots ----------------
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    _overlay(ax, RM, core5_mapping, CORE_LABELS,
             "Peptide C$\\alpha$ RMSF - middle residues (all 17 peptides)")
    ax.set_xlabel("Core slot")
    note = ("slot mapping:\n"
            "  9-mer   : positions 3-7  -> M1-M5\n"
            "  11-mer  : positions 4-8  -> M1-M5\n"
            "  10-mer  : positions 4-7  -> M2-M5  (M1 empty)")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.16 * (y1 - y0), y1)
    ax.text(0.015, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.95))
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="solid = mutant\ndashed = wild-type\ndotted = frameshift",
              title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"rmsf_B_core_5pos{suffix}.png"), dpi=cfg.DPI)
    plt.close(fig)

    # ---------------- small multiples (far more readable than 17 overlaid curves) ----
    # Every text element of the small-multiples grids (tick labels, panel titles,
    # y labels and suptitle) is drawn at PANELS_FS: both grids are embedded in
    # REPORT.md at ~1/3 of their rendered pixel width, where smaller sizes are
    # unreadable. The legend is one step smaller (PANELS_LEGEND_FS) so the full
    # peptide sequences stay clear of the curves. The overlay figures above keep
    # their own (smaller) sizes.
    PANELS_FS = 14
    PANELS_LEGEND_FS = 12
    # xrot: at 14 pt the eight termini labels (P1..P4, POmega-3..POmega) no longer fit
    # side by side in a 1/5-width panel and run into each other, so they are rotated;
    # the five core labels (M1..M5) still fit horizontally.
    for name, fn, labels, ttl, xrot in [
            ("A_termini_8pos", termini8_mapping, TERMINI_LABELS,
             "first 4 + last 4 residues", 45),
            ("B_core_5pos", core5_mapping, CORE_LABELS, "middle residues", 0)]:
        xtick_kw = (dict(rotation=xrot, ha="right", rotation_mode="anchor")
                    if xrot else {})
        fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), sharey=True)
        axes = axes.ravel()
        for i, pk in enumerate(cfg.SNV_PAIRS):
            neo, wt = cfg.pair_members(pk)
            ax = axes[i]
            for s, col, lab in [(neo, cfg.NEO_COLOR, "mutant"), (wt, cfg.WT_COLOR, "wild-type")]:
                r = RM[s["dirname"]]
                slots, res = fn(r["pep_len"])
                ax.errorbar(slots, r["mean"][res], yerr=r["sem"][res], capsize=2,
                            color=col, marker="o", ms=4, lw=1.6,
                            label=f"{s['seq']} ({lab})")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=PANELS_FS, **xtick_kw)
            ax.tick_params(axis="y", labelsize=PANELS_FS)
            # title already has gene + paper mutation
            ax.set_title(neo["pair_label"], fontsize=PANELS_FS)
            ax.legend(fontsize=PANELS_LEGEND_FS, frameon=False)
            ax.grid(alpha=0.25, lw=0.5)
        for j, s in enumerate(cfg.systems_by_kind("fs_neo")):
            ax = axes[7 + j]
            r = RM[s["dirname"]]
            slots, res = fn(r["pep_len"])
            ax.errorbar(slots, r["mean"][res], yerr=r["sem"][res], capsize=2,
                        color=cfg.FS_COLOR, marker="^", ms=4, lw=1.6,
                        label=f"{s['seq']} (fs)")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=PANELS_FS, **xtick_kw)
            ax.tick_params(axis="y", labelsize=PANELS_FS)
            ax.set_title(s["pair_label"], fontsize=PANELS_FS)
            ax.legend(fontsize=PANELS_LEGEND_FS, frameon=False)
            ax.grid(alpha=0.25, lw=0.5)
        # the 2x5 grid is exactly filled: 7 SNV pairs (axes 0-6) + 3 frameshift (axes 7-9)
        axes[0].set_ylabel("C$\\alpha$ RMSF ($\\AA$)", fontsize=PANELS_FS)
        axes[5].set_ylabel("C$\\alpha$ RMSF ($\\AA$)", fontsize=PANELS_FS)
        fig.suptitle(f"Peptide C$\\alpha$ RMSF - {ttl}  |  fit on {fit_mode} C$\\alpha$  |  "
                     f"mean $\\pm$ SEM over 10 replicas, 50-500 ns", fontsize=PANELS_FS)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(figdir, f"rmsf_{name}_panels{suffix}.png"), dpi=140)
        plt.close(fig)

    print(f"\nwrote 4 figures to {figdir}")
    print(f"wrote data/rmsf_termini8{suffix}.csv, data/rmsf_core5{suffix}.csv, "
          f"data/rmsf_full{suffix}.csv")


if __name__ == "__main__":
    main()
