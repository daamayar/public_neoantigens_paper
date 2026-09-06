#!/usr/bin/env python3
"""
Phase 3 -- figures for the interaction-network comparison.

Reads the Phase-2 JSON files (and, for the time-series figures, the Phase-1
per-frame arrays) and writes the figure set into ``interaction_network/figures/``:

  fig1_contact_maps_<window>.png   Contact-occupancy heat maps: neoantigen,
                                   wild-type, and their difference, over the
                                   HLA residues either peptide ever engages.
  fig2_per_position_<window>.png   Per-peptide-position contact occupancy,
                                   hydrogen bonds and screened electrostatics,
                                   neo vs wt, with replica SEM and FDR calls.
  fig4_network_graph_<window>.png  The bipartite residue interaction network
                                   drawn for both peptides.

The figN_ prefixes match the order the figures are referenced in REPORT.md, and
make_report.py asserts that correspondence when it builds the document.

Usage
-----
    python -m scripts.make_figures                 # every figure
    python -m scripts.make_figures --figure 1      # one figure
    python -m scripts.make_figures --window bound  # heat maps for one window
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

from . import inconfig as cfg                                      # noqa: E402
from . import common                                               # noqa: E402
from . import analyze_network as an                                # noqa: E402


def _load(name):
    """Load a Phase-2 JSON payload from the data directory.

    Args:
        name (str): file name, e.g. "compare_bound.json".

    Returns:
        dict: the parsed payload.

    Raises:
        FileNotFoundError: if Phase 2 has not been run.
    """
    path = os.path.join(cfg.DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {path}\n  Fix: python -m scripts.analyze_network")
    with open(path) as fh:
        return json.load(fh)


def _save(fig, name):
    """Write a figure to the figures directory and close it.

    Args:
        fig (matplotlib.figure.Figure): the figure.
        name (str): output file name.

    Returns:
        str: the path written.
    """
    common.ensure_dir(cfg.FIG_DIR)
    path = os.path.join(cfg.FIG_DIR, name)
    fig.savefig(path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, cfg.OUT_DIR)}")
    return path


# Uniform text size for the figures that are read as stand-alone panels
# (fig1 and fig4). Every text element in those two figures — tick labels,
# titles, colorbar labels, node annotations — is drawn at this size.
FIG_FONT_SIZE = 14


def _font_ctx(size=FIG_FONT_SIZE):
    """Context manager forcing every matplotlib text element to one size.

    Explicit ``fontsize=`` arguments still win locally, so this only fixes the
    text that is created implicitly (colorbar tick labels and colorbar axis
    labels, in particular), which is why the figures that use it also pass
    ``size`` explicitly wherever they set a font size themselves.

    Args:
        size (float): point size applied to ticks, labels, titles, legends and
            the figure suptitle.

    Returns:
        matplotlib.rc_context: context manager to wrap figure construction,
        layout and saving (text properties are frozen when the artists are
        created, so the whole build must happen inside it).
    """
    return plt.rc_context({
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "figure.titlesize": size,
    })


def _grid(ax, axis="both", alpha=0.3):
    """Draw a light grid that always sits BEHIND every other element.

    ``Axes.set_axisbelow(True)`` drops the axis layer (ticks and grid lines) to
    zorder 0.5, below the default zorder of patches (1) and lines (2), so bars,
    traces, markers and annotations are never crossed by grid lines. Passing
    ``zorder`` to ``grid()`` would not work: gridline zorder is governed by the
    axis artist, not by the individual lines.

    Args:
        ax (matplotlib.axes.Axes): the axes to draw on.
        axis (str): "both", "x" or "y" — which grid lines to show.
        alpha (float): grid line opacity.

    Returns:
        matplotlib.axes.Axes: the same axes, for chaining.
    """
    ax.set_axisbelow(True)
    ax.grid(axis=axis, alpha=alpha)
    return ax


def _stars(q):
    """Significance marker for an FDR-adjusted p-value.

    Args:
        q (float): adjusted p-value.

    Returns:
        str: "***" (q<0.001), "**" (q<0.01), "*" (q<0.05) or "" otherwise.
    """
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


# ---------------------------------------------------------------------------
# Figure 1 -- contact-occupancy heat maps
# ---------------------------------------------------------------------------

def fig_contact_maps(window):
    """Heat maps of contact occupancy for both peptides and their difference.

    Args:
        window (str): conditioning window name.

    Returns:
        str: path of the written figure.
    """
    cmp_ = _load(f"compare_{window}.json")
    neo = an.complex_window_stats(cmp_["neo"]["dirname"], window)
    wt = an.complex_window_stats(cmp_["wt"]["dirname"], window)

    A, B = neo["mean"]["occupancy"], wt["mean"]["occupancy"]
    keep = np.where((A.max(axis=0) >= 0.10) | (B.max(axis=0) >= 0.10))[0]
    gids, gnames = neo["groove_resids"], neo["groove_resnames"]
    xlab = [common.residue_label(int(gids[j]), gnames[j]) for j in keep]
    ylab_n = [f"P{i+1} {cmp_['neo']['seq'][i]}" for i in range(A.shape[0])]
    ylab_w = [f"P{i+1} {cmp_['wt']['seq'][i]}" for i in range(B.shape[0])]
    ylab_d = ylab_n
    ylab_d[-1] = "P11 K/E"
    
    # At FIG_FONT_SIZE the tick labels need more room than the original 8/6 pt
    # ones did: 0.42 in per column keeps the 90°-rotated HLA labels apart, and
    # the taller canvas keeps the 11 peptide rows above one line height each.
    with _font_ctx():
        fig, axes = plt.subplots(3, 1,
                                 figsize=(max(12, 0.42 * len(keep)), 12.5))
        for ax, M, lab, title, cmapname, vmin, vmax in (
                (axes[0], A[:, keep], ylab_n,
                 f"Neoantigen  {cmp_['neo']['seq']}  (PIK3CA E11K)",
                 "viridis", 0, 1),
                (axes[1], B[:, keep], ylab_w,
                 f"Wild-type   {cmp_['wt']['seq']}", "viridis", 0, 1),
                (axes[2], (A - B)[:, keep], ylab_d,
                 "Difference  (neoantigen - wild-type)", cfg.DIFF_CMAP, -1, 1)):
            im = ax.imshow(M, aspect="auto", cmap=cmapname, vmin=vmin,
                           vmax=vmax)
            ax.set_yticks(range(len(lab)))
            ax.set_yticklabels(lab, fontsize=FIG_FONT_SIZE)
            ax.set_xticks(range(len(keep)))
            ax.set_xticklabels(xlab, rotation=90, fontsize=FIG_FONT_SIZE)
            ax.set_title(title, fontsize=FIG_FONT_SIZE, loc="left")
            cb = plt.colorbar(
                im, ax=ax, pad=0.01, fraction=0.02,
                label="contact occupancy" if vmin == 0 else "Δ occupancy")
            cb.ax.tick_params(labelsize=FIG_FONT_SIZE)
            cb.set_label(cb.ax.get_ylabel(), size=FIG_FONT_SIZE)
            # mark the mutated position
            for i, (a, b) in enumerate(zip(cmp_["neo"]["seq"],
                                           cmp_["wt"]["seq"])):
                if a != b:
                    ax.axhline(i - 0.5, color="k", lw=0.8)
                    ax.axhline(i + 0.5, color="k", lw=0.8)

        fig.suptitle(
            f"Peptide-HLA contact occupancy — {cfg.WINDOWS[window]['label']}\n"
            f"(heavy-atom contact < {cfg.CONTACT_CUTOFF} Å; mean over 10 "
            f"replicas; black lines mark the mutated position P11)",
            fontsize=FIG_FONT_SIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save(fig, f"fig1_contact_maps_{window}.png")


# ---------------------------------------------------------------------------
# Figure 2 -- per-position summary
# ---------------------------------------------------------------------------

def fig_per_position(window):
    """Per-peptide-position contacts, hydrogen bonds and screened electrostatics.

    Args:
        window (str): conditioning window name.

    Returns:
        str: path of the written figure.
    """
    cmp_ = _load(f"compare_{window}.json")
    pos = cmp_["positions"]
    x = np.arange(len(pos))
    labels = [f"P{p['position']}\n{p['neo_res']}/{p['wt_res']}"
              if p["mutated"] else f"P{p['position']}\n{p['neo_res']}"
              for p in pos]

    panels = [("occupancy", "Summed contact occupancy\n(residue-pairs engaged)"),
              ("hbond", "Hydrogen bonds to HLA\n(mean per frame)"),
              ("eel_scr", "Screened electrostatics\n(kcal/mol)")]

    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 9), sharex=True)
    qs = {k: common.benjamini_hochberg([p[k]["p"] for p in pos])
          for k, _ in panels}

    for ax, (key, ylabel) in zip(axes, panels):
        n = [p[key]["neo"] for p in pos]
        w = [p[key]["wt"] for p in pos]
        ne = [p[key]["neo_sem"] for p in pos]
        we = [p[key]["wt_sem"] for p in pos]
        ax.bar(x - 0.2, n, 0.38, yerr=ne, capsize=2, color=cfg.NEO_COLOR,
               label=f"neoantigen {cmp_['neo']['seq']}", alpha=0.9)
        ax.bar(x + 0.2, w, 0.38, yerr=we, capsize=2, color=cfg.WT_COLOR,
               label=f"wild-type {cmp_['wt']['seq']}", alpha=0.9)
        ax.set_ylabel(ylabel, fontsize=9)
        _grid(ax, "y")
        ax.axvline(len(pos) - 1.5, color="k", ls=":", lw=1, alpha=0.6)
        for i, q in enumerate(qs[key]):
            s = _stars(q)
            if s:
                top = max(n[i] + ne[i], w[i] + we[i])
                bot = min(n[i] - ne[i], w[i] - we[i])
                y = top + 0.06 * (top - bot + 1e-9) if top > 0 else bot * 1.10
                ax.text(i, y, s, ha="center", fontsize=9)
        if key == "occupancy":
            ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=8)
    axes[-1].set_xlabel("peptide position")
    fig.suptitle(
        f"Interaction network per peptide position — "
        f"{cfg.WINDOWS[window]['label']}\n"
        f"mean ± SEM over 10 replicas; * / ** / *** = "
        f"Benjamini-Hochberg q < 0.05 / 0.01 / 0.001 (Mann-Whitney, n=10 vs 10)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, f"fig2_per_position_{window}.png")


# ---------------------------------------------------------------------------
# Figure 3 -- the P-Omega anchor
# ---------------------------------------------------------------------------

def _anchor_series(dirname, hla="Asp116", group="sidechain"):
    """Per-frame distance from the P-Omega charged group to an HLA residue.

    The tracked-pair names are built as ``P<pos><aa>--<HLA>`` for side chains
    and ``P<pos>-COO---<HLA>`` for the C-terminal carboxylate. Matching on the
    HLA residue alone is NOT safe: several peptide positions are charged, and
    e.g. "P3R--Asp116" would match a plain "--Asp116" substring test before
    "P11K--Asp116" does. The P-Omega pair is therefore resolved explicitly from
    the peptide length and sequence.

    Args:
        dirname (str): complex directory name.
        hla (str): HLA residue label, e.g. "Asp116" or "Lys146".
        group (str): "sidechain" for the P-Omega side chain, "carboxylate" for
            its C-terminal COO-.

    Returns:
        tuple[np.ndarray, str]: (n_replicas, n_frames) distance array and the
            resolved pair name.

    Raises:
        KeyError: if the resolved pair name is not among the tracked pairs.
    """
    rows, name = [], None
    for rep in range(1, cfg.NUM_REPLICAS + 1):
        d = an.load_replica(dirname, rep)
        if name is None:
            seq = d["seq"]
            pos = len(seq)
            prefix = (f"P{pos}-COO-" if group == "carboxylate"
                      else f"P{pos}{seq[-1]}")
            name = f"{prefix}--{hla}"
            if name not in d["salt_names"]:
                raise KeyError(
                    f"{dirname}: tracked pair '{name}' not found; available "
                    f"P{pos} pairs: "
                    f"{[n for n in d['salt_names'] if n.startswith(f'P{pos}')]}")
        rows.append(d["salt"][:, d["salt_names"].index(name)])
    return np.stack(rows), name
def fig_network_graph(window):
    """Bipartite peptide<->HLA interaction network for both peptides.

    Peptide positions are laid out along the bottom, HLA partners along the top
    ordered by residue number; edge width and opacity encode contact occupancy.

    Args:
        window (str): conditioning window name.

    Returns:
        str: path of the written figure.
    """
    cmp_ = _load(f"compare_{window}.json")

    hla_all = sorted({e["hla"] for side in ("neo", "wt")
                      for e in cmp_[side]["network"]["edges"]},
                     key=lambda s: int("".join(c for c in s if c.isdigit())))
    hx = {h: i for i, h in enumerate(hla_all)}

    # At FIG_FONT_SIZE the 90°-rotated HLA labels are ~0.19 in wide and ~0.9 in
    # long, so the canvas is widened (label width x number of partners) and
    # heightened (label length above each row of nodes) relative to the 7.5 pt
    # version, otherwise neighbouring labels touch and the top row runs into
    # the suptitle.
    with _font_ctx():
        fig, axes = plt.subplots(
            2, 1, figsize=(max(13, 0.34 * len(hla_all)), 11.0))

        for ax, side, color in ((axes[0], "neo", cfg.NEO_COLOR),
                                (axes[1], "wt", cfg.WT_COLOR)):
            net = cmp_[side]["network"]
            seq = cmp_[side]["seq"]
            p_labels = [f"P{i+1}{seq[i]}" for i in range(len(seq))]
            px = {p: i * (len(hla_all) - 1) / max(len(seq) - 1, 1)
                  for i, p in enumerate(p_labels)}

            for e in net["edges"]:
                w = e["occupancy"]
                ax.plot([px[e["peptide"]], hx[e["hla"]]], [0, 1],
                        color=color, lw=0.4 + 3.2 * w, alpha=0.15 + 0.65 * w,
                        solid_capstyle="round", zorder=1)

            # Node labels are offset in POINTS, not data units, by that node's
            # own marker radius. A scatter size `s` is an area in points^2, so
            # the radius is sqrt(s/pi) points; offsetting by radius + padding
            # therefore clears the circle exactly, however large it is. Using a
            # fixed offset in data units (as before) made the biggest hubs —
            # precisely the ones worth reading — collide with their own labels.
            def _radius_pt(s):
                return float(np.sqrt(s / np.pi))

            deg = net["degree"]
            for p, xx in px.items():
                s = 40 + 130 * deg.get(p, 0)
                ax.scatter(xx, 0, s=s, color=color, edgecolor="k", lw=0.6,
                           zorder=3)
                ax.annotate(p, (xx, 0), textcoords="offset points",
                            xytext=(0, -(_radius_pt(s) + 5)), ha="center",
                            va="top", fontsize=FIG_FONT_SIZE,
                            fontweight="bold",
                            annotation_clip=False, zorder=4)
            hdeg = net["hla_degree"]
            for h, xx in hx.items():
                d = hdeg.get(h, 0.0)
                s = 18 + 90 * d
                ax.scatter(xx, 1, s=s, color="0.55", edgecolor="k", lw=0.4,
                           zorder=3)
                # rotation_mode="anchor" applies ha/va AFTER the rotation, so
                # ha="left" reliably means "the text starts here and grows
                # upwards". With the default rotation_mode the text is aligned
                # first and rotated about the anchor afterwards, which shifts it
                # back over the marker.
                ax.annotate(h, (xx, 1), textcoords="offset points",
                            xytext=(0, _radius_pt(s) + 4), ha="left",
                            va="center", rotation=90, rotation_mode="anchor",
                            fontsize=FIG_FONT_SIZE,
                            annotation_clip=False, zorder=4)

            ax.set_ylim(-0.75, 1.75)
            ax.set_xlim(-1, len(hla_all))
            ax.axis("off")
            ax.set_title(
                f"{cmp_[side]['label']}   —   {net['n_edges']} edges, "
                f"{net['n_hla_partners']} HLA partners, "
                f"total occupancy weight {net['total_weight']:.1f}",
                fontsize=FIG_FONT_SIZE, loc="left", color=color)

        fig.suptitle(
            f"Peptide-HLA residue interaction network — "
            f"{cfg.WINDOWS[window]['label']}\n"
            f"edge width ∝ contact occupancy (edges with occupancy ≥ 0.05 "
            f"shown); node size ∝ weighted degree", fontsize=FIG_FONT_SIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _save(fig, f"fig4_network_graph_{window}.png")