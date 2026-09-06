#!/usr/bin/env python3
"""
Task 1 -- peptide RMSD distribution (mutant vs wild-type).

"RMSD distribution" is the frequency / probability distribution of the RMSD 
values sampled over the trajectory -- NOT an RMSD-vs-time trace. Its peaks 
correspond to distinct conformational states. We reproduce that, one panel 
per system-group, with the mutant and its wild-type overlaid so the pair can 
be compared directly.

Protocol
--------
* Reference   : frame 1 of each replica's own production DCD (t ~ 0) = cache[0].
* Ensemble    : frames 418-4166 (50-500 ns), i.e. the first 50 ns is discarded as
                equilibration; 3749 frames x 10 replicas = 37,490 values per system.
* Superposition: each frame is fitted on the PEPTIDE's own C-alpha atoms, so the
                RMSD reports the peptide's INTERNAL conformational change. Rigid-body
                motion of the peptide inside the groove is removed by this fit.
* RMSD atoms  : peptide backbone N, CA, C, O (primary, plotted). The C-terminal
                "O" is the OT1/OT2 carboxylate centroid -- see common.py. C-alpha-only
                and all-heavy-atom RMSD are also computed and written to CSV.
* The HLA is never included in a reported RMSD value.

Statistics
----------
Frames within a replica are strongly autocorrelated, so a frame-level test would
be pseudo-replication and would report absurdly small p-values. Inference is
therefore done at the REPLICA level: a two-sided Mann-Whitney U on the 10 neo vs
10 wt per-replica mean RMSDs (n=10 vs n=10). The frame-level Kolmogorov-Smirnov
distance D is also reported, but purely as a DESCRIPTIVE measure of how different
the two distributions are -- its p-value is meaningless here and is not reported.

Outputs
-------
    figures/rmsd/rms_distribution_<pair>.png        x7  (SNV mutant vs wild-type)
    figures/rmsd/rms_distribution_fs_<name>.png     x3  (frameshift neoantigens)
    figures/rmsd/rms_distribution_10_neoantigens.png    (all 10 neoantigens)
    figures/rmsd/rms_distribution_overview.png          (all 11 panels in one grid)
    data/rmsd_values.npz            pooled per-system RMSD arrays (all 3 selections)
    data/rmsd_summary.csv           mean/sd/median/min/max per system and selection
    data/rmsd_pair_stats.csv        neo-vs-wt comparison per SNV pair

Usage
-----
    python -m scripts.analyze_rmsd
"""
import os
import csv
import argparse
import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu, ks_2samp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import mdconfig as cfg
from . import common


def compute_system_rmsd(dirname, fit_mode):
    """Compute per-replica RMSD time series for one system, for all atom selections.

    Args:
        dirname (str): complex directory name.
        fit_mode (str): "peptide" or "receptor" -- the superposition group.

    Returns:
        dict: {selection: np.ndarray (NUM_REPLICAS, n_window_frames)} for each of
            mdconfig.RMSD_SELECTIONS, plus key "time_ns" -> (n_window_frames,) and
            "full" -> {selection: (NUM_REPLICAS, TOTAL_FRAMES)} for the QC time plot.
    """
    imap = common.build_index_map(dirname)
    out = {sel: [] for sel in cfg.RMSD_SELECTIONS}
    full = {sel: [] for sel in cfg.RMSD_SELECTIONS}

    for rep in range(1, cfg.NUM_REPLICAS + 1):
        xyz = common.load_cache(dirname, rep)                 # (4166, A, 3)
        fit_all = common.fit_coords(xyz, imap, fit_mode)      # (4166, Nfit, 3)
        ref_fit = fit_all[cfg.REFERENCE_FRAME_INDEX]          # t ~ 0

        for sel in cfg.RMSD_SELECTIONS:
            meas_all = common.get_selection(xyz, imap, sel)   # (4166, M, 3)
            ref_meas = meas_all[cfg.REFERENCE_FRAME_INDEX]
            r = common.rmsd_to_reference(fit_all, ref_fit, meas_all, ref_meas)
            full[sel].append(r)
            out[sel].append(r[cfg.ANALYSIS_SLICE])

    res = {sel: np.array(v) for sel, v in out.items()}
    res["full"] = {sel: np.array(v) for sel, v in full.items()}
    res["time_ns"] = common.frame_times_ns()[cfg.ANALYSIS_SLICE]
    return res


def _panel(ax, series, title, xlim=None, ylim=None,
           show_xlabel=True, show_ylabel=True, title_fontsize=10.5,
           legend_fontsize=7.5, label_fontsize=None, tick_labelsize=None):
    """Draw one RMS-distribution panel: filled histogram + KDE per series.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        series (list[tuple]): (label, colour, values 1-D np.ndarray) per curve.
        title (str): panel title.
        xlim (tuple | None): x-range; computed from the data if None. When given,
            the histogram bins and KDE grid also span exactly this range, so panels
            that share an `xlim` are directly comparable bin-for-bin.
        ylim (tuple | None): y-range; matplotlib autoscales if None.
        show_xlabel (bool): draw the per-axes x-axis label (turn off when a single
            figure-level label is used instead).
        show_ylabel (bool): draw the per-axes y-axis label.
        title_fontsize (float): panel title font size.
        legend_fontsize (float): legend font size (sequence + mean +/- sd entries).
        label_fontsize (float | None): per-axes x/y label font size; matplotlib
            default when None. Ignored when the corresponding show_* flag is False.
        tick_labelsize (float | None): x/y tick-label font size; matplotlib default
            (10 pt) when None.

    Returns:
        None. Draws on `ax`.
    """
    allv = np.concatenate([v for _, _, v in series])
    lo, hi = (xlim if xlim else (0.0, float(np.percentile(allv, 99.9)) * 1.10))
    grid = np.linspace(lo, hi, 400)
    bins = np.linspace(lo, hi, 70)

    for label, colour, v in series:
        ax.hist(v, bins=bins, density=True, alpha=0.35, color=colour,
                edgecolor="none", zorder=1)
        # KDE on a subsample: 37,490 autocorrelated frames add no shape information
        # beyond ~8k, and gaussian_kde is O(n*grid).
        sub = v if len(v) <= 8000 else v[np.linspace(0, len(v) - 1, 8000).astype(int)]
        kde = gaussian_kde(sub)
        ax.plot(grid, kde(grid), color=colour, lw=2.0, zorder=3,
                label=f"{label}\n  {v.mean():.2f} $\\pm$ {v.std():.2f} $\\AA$")
        ax.axvline(v.mean(), color=colour, ls=":", lw=1.2, alpha=0.9, zorder=2)

    ax.set_xlim(lo, hi)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_xlabel:
        ax.set_xlabel("Peptide backbone RMSD ($\\AA$)", fontsize=label_fontsize)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=label_fontsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", labelsize=tick_labelsize)
    ax.set_title(title, fontsize=title_fontsize, pad=6)
    ax.legend(fontsize=legend_fontsize, frameon=False, loc="upper right",
              handlelength=1.2)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    ap = argparse.ArgumentParser(description="Task 1: peptide RMS distributions.")
    args = ap.parse_args()
    fit_mode = cfg.FIT_MODE
    suffix = ""

    figdir = common.ensure_dir(os.path.join(cfg.FIG_DIR, "rmsd"))
    common.ensure_dir(cfg.DATA_DIR)

    print(f"Computing peptide RMSD (fit = {fit_mode} C-alpha, "
          f"reference = frame 1 / t~0, window = 50-500 ns) ...")
    R = {}
    for s in cfg.SYSTEMS:
        R[s["dirname"]] = compute_system_rmsd(s["dirname"], fit_mode)
        v = R[s["dirname"]][cfg.RMSD_PRIMARY].ravel()
        print(f"  {s['label']:28s} n={v.size:6d}  "
              f"mean={v.mean():5.2f}  sd={v.std():5.2f}  "
              f"range={v.min():4.2f}-{v.max():5.2f} A")

    # ---------------- save raw values + summary ----------------
    np.savez_compressed(
        os.path.join(cfg.DATA_DIR, f"rmsd_values{suffix}.npz"),
        **{f"{d}__{sel}": R[d][sel] for d in R for sel in cfg.RMSD_SELECTIONS},
        time_ns=R[cfg.SYSTEMS[0]["dirname"]]["time_ns"])

    with open(os.path.join(cfg.DATA_DIR, f"rmsd_summary{suffix}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["complex", "label", "allele", "peptide", "length", "kind", "pair",
                    "selection", "n_frames", "mean_A", "sd_A", "median_A",
                    "min_A", "max_A", "sem_across_replicas_A"] +
                   [f"rep{i}_mean_A" for i in range(1, cfg.NUM_REPLICAS + 1)])
        for s in cfg.SYSTEMS:
            for sel in cfg.RMSD_SELECTIONS:
                a = R[s["dirname"]][sel]                      # (10, F)
                v = a.ravel()
                rm = a.mean(axis=1)                            # per-replica means
                w.writerow([s["dirname"], s["label"], s["allele"], s["seq"], len(s["seq"]),
                            s["kind"], s["pair"] or "", sel, v.size,
                            f"{v.mean():.4f}", f"{v.std():.4f}", f"{np.median(v):.4f}",
                            f"{v.min():.4f}", f"{v.max():.4f}",
                            f"{rm.std(ddof=1)/np.sqrt(len(rm)):.4f}"] +
                           [f"{x:.4f}" for x in rm])

    # ---------------- neo vs wt statistics (replica-level) ----------------
    pair_rows = []
    for pk in cfg.SNV_PAIRS:
        neo, wt = cfg.pair_members(pk)
        pos, mres, wres = cfg.mutation_position(pk)
        a = R[neo["dirname"]][cfg.RMSD_PRIMARY]
        b = R[wt["dirname"]][cfg.RMSD_PRIMARY]
        am, bm = a.mean(axis=1), b.mean(axis=1)                # 10 vs 10 replica means
        U, p = mannwhitneyu(am, bm, alternative="two-sided")
        ks = ks_2samp(a.ravel(), b.ravel()).statistic          # descriptive only
        sp = np.sqrt((am.var(ddof=1) + bm.var(ddof=1)) / 2)
        pair_rows.append(dict(
            pair=pk, pair_label=neo["pair_label"], allele=neo["allele"],
            mutation=f"{wres}{pos}{mres}", position=pos,
            neo_seq=neo["seq"], wt_seq=wt["seq"],
            neo_mean_A=am.mean(), neo_sd_A=a.ravel().std(),
            wt_mean_A=bm.mean(), wt_sd_A=b.ravel().std(),
            delta_neo_minus_wt_A=am.mean() - bm.mean(),
            cohens_d=(am.mean() - bm.mean()) / sp if sp > 0 else np.nan,
            mannwhitney_U=U, mannwhitney_p=p,
            ks_distance_D=ks))
    with open(os.path.join(cfg.DATA_DIR, f"rmsd_pair_stats{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        for r in pair_rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    print("\nneo vs wt (replica-level Mann-Whitney, n=10 vs 10):")
    for r in pair_rows:
        star = "*" if r["mannwhitney_p"] < 0.05 else " "
        print(f"  {r['pair_label']:34s} {r['mutation']:>6s}  "
              f"neo={r['neo_mean_A']:.2f}  wt={r['wt_mean_A']:.2f}  "
              f"d={r['delta_neo_minus_wt_A']:+.2f} A  p={r['mannwhitney_p']:.3f}{star}")

    # ---------------- 7 SNV pair panels ----------------
    # pair_label already carries the gene + paper-notation mutation
    # (e.g. "A*11:01 - PIK3CA E11K"), so the title only adds the peptide length.
    sel = cfg.RMSD_PRIMARY
    for pk in cfg.SNV_PAIRS:
        neo, wt = cfg.pair_members(pk)
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        _panel(ax,
               [(f"{neo['seq']} (mutant)", cfg.NEO_COLOR, R[neo["dirname"]][sel].ravel()),
                (f"{wt['seq']} (wild-type)", cfg.WT_COLOR, R[wt["dirname"]][sel].ravel())],
               f"{neo['pair_label']}  ({len(neo['seq'])}-mer)")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"rms_distribution_{pk}{suffix}.png"),
                    dpi=cfg.DPI)
        plt.close(fig)

    # ---------------- 3 frameshift panels ----------------
    for s in cfg.systems_by_kind("fs_neo"):
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        _panel(ax, [(f"{s['seq']} (frameshift neoantigen)", cfg.FS_COLOR,
                     R[s["dirname"]][sel].ravel())],
               f"{s['pair_label']}\n{s['seq']}  ({len(s['seq'])}-mer)")
        fig.tight_layout()
        tag = s["dirname"].split("_")[0]
        fig.savefig(os.path.join(figdir, f"rms_distribution_fs_{tag}{suffix}.png"),
                    dpi=cfg.DPI)
        plt.close(fig)

    # ---------------- the 10 neoantigens together ----------------
    neos = cfg.systems_by_kind("snv_neo", "fs_neo")
    assert len(neos) == 10, f"expected 10 neoantigens, got {len(neos)}"
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    allv = np.concatenate([R[s["dirname"]][sel].ravel() for s in neos])
    lo, hi = 0.0, float(np.percentile(allv, 99.9)) * 1.10
    grid = np.linspace(lo, hi, 500)
    for s in neos:
        key = s["pair"] if s["pair"] else s["dirname"]
        v = R[s["dirname"]][sel].ravel()
        sub = v[np.linspace(0, len(v) - 1, 8000).astype(int)]
        ls = "-" if s["kind"] == "snv_neo" else "--"
        ax.plot(grid, gaussian_kde(sub)(grid), lw=2.0, ls=ls,
                color=cfg.SYSTEM_COLORS[key],
                label=f"{s['label']}  ({v.mean():.2f} $\\AA$)")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Peptide backbone RMSD ($\\AA$)")
    ax.set_ylabel("Density")
    ax.set_title("RMS distribution of the 10 neoantigens\n"
                 "(solid = SNV neoantigen, dashed = frameshift neoantigen)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"rms_distribution_10_neoantigens{suffix}.png"),
                dpi=cfg.DPI)
    plt.close(fig)

    # ---------------- overview grid (10 panels: 7 SNV pairs + 3 frameshift) -------
    # One shared X scale and one shared Y scale across ALL panels, so the
    # distributions are directly comparable; a single figure-level X and Y label
    # replaces the per-panel labels.
    overview_systems = list(cfg.SYSTEMS)                       # all 17, over 10 panels
    pooled = [R[s["dirname"]][sel].ravel() for s in overview_systems]
    # Shared x: cover the bulk of every system (99.7th percentile is robust to the
    # rare long fraying tails) with a little headroom.
    x_hi = max(float(np.percentile(v, 99.7)) for v in pooled) * 1.05
    shared_xlim = (0.0, x_hi)
    # Shared y: tallest KDE peak among all systems, on the shared grid.
    _grid = np.linspace(0.0, x_hi, 400)
    y_hi = 0.0
    for v in pooled:
        sv = v if len(v) <= 8000 else v[np.linspace(0, len(v) - 1, 8000).astype(int)]
        y_hi = max(y_hi, float(gaussian_kde(sv)(_grid).max()))
    shared_ylim = (0.0, y_hi * 1.12)

    nrow, ncol = 2, 5
    # Every text element of this overview is drawn at OVERVIEW_FS (panel titles, tick
    # labels, figure-level labels and suptitle) - the figure is embedded in REPORT.md
    # at ~1/3 of its rendered pixel width, so smaller sizes are unreadable there. The
    # legend is one step smaller (OVERVIEW_LEGEND_FS): its entries are two lines each
    # and would otherwise crowd the KDE curves. The per-pair figures below/above keep
    # _panel's smaller defaults.
    OVERVIEW_FS = 14
    OVERVIEW_LEGEND_FS = 12
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, pk in enumerate(cfg.SNV_PAIRS):                     # axes 0-6
        neo, wt = cfg.pair_members(pk)
        _panel(axes[i],
               [(f"{neo['seq']} (mut)", cfg.NEO_COLOR, R[neo["dirname"]][sel].ravel()),
                (f"{wt['seq']} (wt)", cfg.WT_COLOR, R[wt["dirname"]][sel].ravel())],
               neo["pair_label"], xlim=shared_xlim, ylim=shared_ylim,
               show_xlabel=False, show_ylabel=False, title_fontsize=OVERVIEW_FS,
               legend_fontsize=OVERVIEW_LEGEND_FS, tick_labelsize=OVERVIEW_FS)
    for j, s in enumerate(cfg.systems_by_kind("fs_neo")):     # axes 7-9
        _panel(axes[7 + j], [(f"{s['seq']} (fs)", cfg.FS_COLOR,
                              R[s["dirname"]][sel].ravel())],
               s["pair_label"], xlim=shared_xlim, ylim=shared_ylim,
               show_xlabel=False, show_ylabel=False, title_fontsize=OVERVIEW_FS,
               legend_fontsize=OVERVIEW_LEGEND_FS, tick_labelsize=OVERVIEW_FS)

    fig.supxlabel("Peptide backbone RMSD ($\\AA$)", fontsize=OVERVIEW_FS)
    fig.supylabel("Density", fontsize=OVERVIEW_FS)
    fig.suptitle(
        f"Peptide RMS distribution  |  backbone N,CA,C,O  |  fit on {fit_mode} C$\\alpha$  |  "
        f"reference = first frame (t$\\approx$0)  |  50-500 ns, 10 replicas  |  "
        f"shared axes",
        fontsize=OVERVIEW_FS, y=0.995)
    fig.tight_layout(rect=[0.012, 0.02, 1, 0.965])
    fig.savefig(os.path.join(figdir, f"rms_distribution_overview{suffix}.png"), dpi=150)
    plt.close(fig)

    print(f"\nwrote 11 panels + overview to {figdir}")
    print(f"wrote data/rmsd_summary{suffix}.csv, data/rmsd_pair_stats{suffix}.csv, "
          f"data/rmsd_values{suffix}.npz")


if __name__ == "__main__":
    main()
