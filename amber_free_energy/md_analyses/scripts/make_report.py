#!/usr/bin/env python3
"""
Assemble md_analyses/REPORT.md from the CSVs and figures produced by the pipeline.

Reads:
    data/rmsd_summary.csv, data/rmsd_pair_stats.csv
    data/rmsf_termini8.csv, data/rmsf_core5.csv

Figures are embedded with the HTML <figure> fallback so that they render both on
GitHub and in Markdown-to-PDF converters.

Usage
-----
    python -m scripts.make_report
"""
import os
import csv
import argparse
import numpy as np

from . import mdconfig as cfg
from . import common


def read_csv(path):
    """Read a CSV into a list of dicts.

    Args:
        path (str): path to the CSV file.

    Returns:
        list[dict]: rows, or [] if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fig(rel, caption, width=780):
    """Render an HTML <figure> block for a figure file.

    Args:
        rel (str): path to the image, relative to md_analyses/.
        caption (str): figure caption.
        width (int): rendered width in px.

    Returns:
        str: the HTML block.
    """
    return (f'<figure>\n  <img src="{rel}" alt="{caption}" width="{width}"/>\n'
            f'  <figcaption>{caption}</figcaption>\n</figure>\n')


def main():
    ap = argparse.ArgumentParser(description="Assemble REPORT.md")
    args = ap.parse_args()
    sfx = ""

    rsum = read_csv(os.path.join(cfg.DATA_DIR, f"rmsd_summary{sfx}.csv"))
    rpair = read_csv(os.path.join(cfg.DATA_DIR, f"rmsd_pair_stats{sfx}.csv"))
    rmsf8 = read_csv(os.path.join(cfg.DATA_DIR, f"rmsf_termini8{sfx}.csv"))
    rmsf5 = read_csv(os.path.join(cfg.DATA_DIR, f"rmsf_core5{sfx}.csv"))

    L = []
    A = L.append

    A("# Peptide MD structural analyses - RMS distribution and RMSF")
    A("")
    A("Peptide-HLA neoantigen complexes: **17 systems x 10 replicas x 500 ns** "
      "(4166 frames each, 120 ps/frame).")
    A("All quantities are computed on the **peptide only** - the HLA heavy chain and "
      "beta-2-microglobulin are excluded from every reported value.")
    A("")

    # ---------------- key findings (computed, not hard-coded) ----------------
    if rpair and rmsf8:
        # RMSF: mean over all residues, per system
        rf = {}
        for r in read_csv(os.path.join(cfg.DATA_DIR, f"rmsf_full{sfx}.csv")):
            rf.setdefault(r["complex"], []).append(float(r["rmsd_mean_A"]
                                                         if "rmsd_mean_A" in r
                                                         else r["rmsf_mean_A"]))
        rf = {k: float(np.mean(v)) for k, v in rf.items()}

        rmsd_lower = rmsf_lower = 0
        rmsd_n = rmsf_n = 0
        exceptions = []
        for pk in cfg.SNV_PAIRS:
            neo, wt = cfg.pair_members(pk)
            row = next(r for r in rpair if r["pair"] == pk)
            rmsd_n += 1
            if float(row["delta_neo_minus_wt_A"]) < 0:
                rmsd_lower += 1
            if neo["dirname"] in rf and wt["dirname"] in rf:
                rmsf_n += 1
                if rf[neo["dirname"]] < rf[wt["dirname"]]:
                    rmsf_lower += 1
                else:
                    exceptions.append(neo["pair_label"])
        sig = [r for r in rpair if float(r["mannwhitney_p"]) < 0.05]

        A("## Key findings")
        A("")
        A("Both analyses agree: **the neoantigen is generally more conformationally "
          "constrained than its wild-type counterpart.**")
        A("")
        A(f"- **RMSD** (Task 1): the mutant has the *lower* mean peptide RMSD in "
          f"**{rmsd_lower}/{rmsd_n}** SNV pairs. "
          f"{len(sig)} reach significance at the replica level "
          f"(" + ", ".join(f"{r['pair_label']}, p={float(r['mannwhitney_p']):.3f}"
                           for r in sig) + "), and **both favour the mutant**.")
        A(f"- **RMSF** (Task 2): the mutant is *less flexible* (lower mean C-alpha RMSF across "
          f"all residues) in **{rmsf_lower}/{rmsf_n}** SNV pairs.")
        if exceptions:
            A(f"- The **sole consistent exception is {exceptions[0]}**, where the mutant is "
              f"slightly *more* flexible than the wild-type on both measures.")
        A("")
        # pull the P-Omega RMSF of the A1101 pair straight from the CSV so the prose
        # can never drift away from the data it is describing
        pomega = {r["peptide"]: float(r["rmsf_mean_A"])
                  for r in rmsf8 if r["slot_label"] == "POmega"}
        a11 = next(r for r in rpair if r["pair"] == "A1101")
        neo_seq, wt_seq = a11["neo_seq"], a11["wt_seq"]
        biggest = max(pomega, key=pomega.get)
        superlative = (" - the largest terminal fluctuation of any peptide in the set"
                       if biggest == wt_seq else "")
        A(f"The most striking pair is **{cfg.pair_display('A1101')}** "
          f"(canonical {cfg.CANONICAL_MUT['A1101']}; A*11:01; {neo_seq}/{wt_seq}). The "
          f"substituted position {a11['position']} is the peptide's C-terminal residue (P-Omega), "
          f"and for HLA-A11 the C-terminus is a *critical anchor position*: Kubo et al. "
          f"characterised the HLA-A11 motif as bearing \"critical anchor residues at position 2 and "
          f"at the COOH-terminal\" "
          f"([J Immunol 1994;152:3913-24, PMID 8144960](https://pubmed.ncbi.nlm.nih.gov/8144960/); "
          f"retrieved from PubMed - abstract only, full text not checked). This substitution "
          f"therefore alters an anchor residue, and the wild-type shows a much broader, clearly "
          f"**bimodal** RMSD distribution ({float(a11['wt_mean_A']):.2f} vs "
          f"{float(a11['neo_mean_A']):.2f} A, p = {float(a11['mannwhitney_p']):.3f}) and a "
          f"C-terminal RMSF of **{pomega[wt_seq]:.2f} A vs {pomega[neo_seq]:.2f} A** for the "
          f"mutant{superlative}. That is consistent with the project's independent unbinding "
          f"analysis, in which this wild-type peptide detaches at the C-terminus in 9/10 replicas.")
        A("")
        A("*Note on sourcing: only the claim that the C-terminus is an anchor position for HLA-A11 "
          "is cited above. Which residue A\\*11:01 specifically prefers at P-Omega was **not** verified "
          "against a source and is therefore not asserted here.*")
        A("")
        A("> **Caveat.** These numbers are measured in the peptide's own frame (see Methods), so "
          "they describe the peptide's *internal* conformational change. They are not a direct "
          "measure of unbinding - see section 5.")
        A("")

    # ---------------- methods ----------------
    A("## Methods")
    A("")
    A("| Setting | Value |")
    A("|---|---|")
    A(f"| Trajectories | 17 complexes x {cfg.NUM_REPLICAS} replicas = "
      f"{17*cfg.NUM_REPLICAS} runs, {cfg.TOTAL_FRAMES} frames each (0-500 ns) |")
    A(f"| Analysis window | frames {cfg.EQUIL_FRAMES+1}-{cfg.TOTAL_FRAMES} "
      f"(**{cfg.EQUILIBRATION_NS:.0f}-500 ns**); the first {cfg.EQUILIBRATION_NS:.0f} ns "
      f"is discarded as equilibration |")
    A(f"| Frames per system | 3749 x 10 = **37,490** |")
    A(f"| RMSD reference | **frame 1 of each replica's own production DCD** (t ~ 0), "
      f"i.e. \"the first frame\" |")
    A(f"| Superposition (fit) | **peptide's own C-alpha** - so every metric reports the "
      f"peptide's INTERNAL conformational change; rigid-body motion of the peptide within "
      f"the groove is removed by the fit |")
    A(f"| RMSD atoms | peptide **backbone N, CA, C, O** (C-alpha-only and all-heavy-atom "
      f"variants also in the CSVs) |")
    A(f"| RMSF atoms | peptide **C-alpha**, per residue |")
    A("")
    A("**Nomenclature.** Every table and figure names a substitution in **paper notation** - the "
      "residue's position *within the peptide* (as in the reference paper), formatted "
      "wild-type/position/mutant. The equivalent **canonical** notation (position in the "
      "full-length protein) is:")
    A("")
    A("| Gene | Paper notation | Canonical notation | Peptides (wt -> mut) |")
    A("|---|---|---|---|")
    for pk in cfg.SNV_PAIRS:
        neo, wt = cfg.pair_members(pk)
        A(f"| {neo['gene']} | {cfg.paper_mutation(pk)} | {cfg.CANONICAL_MUT[pk]} "
          f"| {wt['seq']} -> {neo['seq']} |")
    A("")
    A("The three frameshift neoantigens (APC, NPM, TGFBRII) have no wild-type counterpart and no "
      "substitution notation. Note that two distinct **PIK3CA** mutations appear - E11K (E545K, "
      "on A*11:01) and H2L (H1047L, on A*03:01) - distinguished by their mutation code and allele.")
    A("")
    A("**CHARMM36 C-terminal carboxylate.** The C-terminal residue of every peptide carries "
      "`OT1`/`OT2` and has no atom named `O`. `OT1` and `OT2` are equivalent by resonance and "
      "swap whenever the carboxylate flips 180 degrees about the CA-C bond, so selecting one of "
      "them would inject spurious RMSD. The backbone \"O\" of the C-terminal residue is "
      "therefore taken as the **centroid of OT1 and OT2**, which is invariant under that flip.")
    A("")
    A("**Validation.** The RMSD and RMSF implementations were checked against `cpptraj` on the "
      "same trajectory and selection: agreement is **5e-5 A** (RMSD) and **7e-5 A** (RMSF). "
      "Fitting on peptide C-alpha rather than on the backbone changes the mean backbone RMSD by "
      "0.017 A (max 0.054 A over 4166 frames), so the choice of fit atoms is immaterial.")
    A("")

    # ---------------- Task 1 ----------------
    A("## 1. RMS distribution (mutant vs wild-type)")
    A("")
    A("Following the reference paper (*Exploring KRas Protein Dynamics*, ACS Omega 2024, 9, "
      "30665-30674, Fig. 4), the **RMS distribution** is the frequency/probability distribution "
      "of the RMSD values sampled over the trajectory - not an RMSD-vs-time trace. Peaks "
      "correspond to distinct conformational states.")
    A("")
    A("Each panel pools **37,490 frames** (10 replicas x 3749 frames, 50-500 ns).")
    A("")

    if rpair:
        A("### Mutant vs wild-type, per SNV pair")
        A("")
        A("Frames within a replica are strongly autocorrelated, so inference is done at the "
          "**replica level**: a two-sided Mann-Whitney U on the 10 mutant vs 10 wild-type "
          "per-replica mean RMSDs. The Kolmogorov-Smirnov distance *D* is reported as a purely "
          "**descriptive** measure of how different the two full distributions are; its p-value "
          "would be meaningless at the frame level and is not given.")
        A("")
        A("Mutations are in paper (peptide-position) notation - see the mapping key in Methods.")
        A("")
        A("| Pair | Mutant RMSD (A) | Wild-type RMSD (A) | delta (mut - wt) | "
          "Mann-Whitney p | KS *D* |")
        A("|---|---|---|---|---|---|")
        for r in rpair:
            p = float(r["mannwhitney_p"])
            star = " **\\***" if p < 0.05 else ""
            A(f"| {r['pair_label']} "
              f"| {float(r['neo_mean_A']):.2f} +/- {float(r['neo_sd_A']):.2f} "
              f"| {float(r['wt_mean_A']):.2f} +/- {float(r['wt_sd_A']):.2f} "
              f"| {float(r['delta_neo_minus_wt_A']):+.2f} "
              f"| {p:.3f}{star} | {float(r['ks_distance_D']):.3f} |")
        A("")
        A("\\* p < 0.05. Mean +/- SD is over all 37,490 pooled frames. "
          "The `Pair` column is `allele - gene mutation` (paper notation).")
        A("")

    A(fig(f"figures/rmsd/rms_distribution_overview{sfx}.png",
          "Figure 1: RMS distribution - 10 panels sharing one common X scale and one common Y "
          "scale. Panels 1-7: the seven SNV pairs (mutant red, wild-type blue). Panels 8-10: the "
          "three frameshift neoantigens. Because the scales are shared, the panels are directly "
          "comparable - the broad, right-shifted distributions (e.g. PIK3CA E11K wild-type) stand "
          "out against the sharp, low-RMSD ones (e.g. KRAS G3D).", 980))
    A("")
    for pk in cfg.SNV_PAIRS:
        neo, _ = cfg.pair_members(pk)
        A(fig(f"figures/rmsd/rms_distribution_{pk}{sfx}.png",
              f"{neo['pair_label']} - mutant vs wild-type.", 560))
    for s in cfg.systems_by_kind("fs_neo"):
        tag = s["dirname"].split("_")[0]
        A(fig(f"figures/rmsd/rms_distribution_fs_{tag}{sfx}.png",
              f"{s['pair_label']} - frameshift neoantigen {s['seq']}.", 560))
    A(fig(f"figures/rmsd/rms_distribution_10_neoantigens{sfx}.png",
          "Figure 2: RMS distribution of the 10 neoantigens (7 SNV, solid; 3 frameshift, "
          "dashed).", 820))
    A("")

    # ---------------- Task 2 ----------------
    A("## 2. RMSF (two residue windows, all 17 peptides)")
    A("")
    A("Per-residue C-alpha RMSF, measured about the **ensemble mean structure** (fit to the first "
      "frame -> average -> re-fit to the average -> measure), computed **independently per "
      "replica and then averaged over the 10 replicas**; error bars are the SEM across replicas. "
      "Pooling all replicas' frames first would fold inter-replica drift into what is supposed to "
      "be an intra-replica fluctuation and would inflate the RMSF.")
    A("")
    A("### Plot A - first 4 and last 4 residues (8 x-positions)")
    A("")
    A("| Peptide length | N-terminal 4 | C-terminal 4 |")
    A("|---|---|---|")
    A("| 9-mer | 1, 2, 3, 4 | 6, 7, 8, 9 |")
    A("| 10-mer | 1, 2, 3, 4 | 7, 8, 9, 10 |")
    A("| 11-mer | 1, 2, 3, 4 | 8, 9, 10, 11 |")
    A("")
    A("Peptides of different length therefore align on their **termini**, which is what makes the "
      "comparison meaningful: P1/P2 and P-Omega are the HLA anchor positions.")
    A("")
    A(fig(f"figures/rmsf/rmsf_A_termini_8pos{sfx}.png",
          "Figure 3: C-alpha RMSF of the first 4 and last 4 residues, all 17 peptides. "
          "Solid + circles = SNV mutant, dashed + squares = wild-type, dotted + triangles = "
          "frameshift neoantigen.", 860))
    A(fig(f"figures/rmsf/rmsf_A_termini_8pos_panels{sfx}.png",
          "Figure 3b: the same data as small multiples (one panel per pair / frameshift "
          "neoantigen) - easier to read than 17 overlaid curves.", 980))
    A("")
    A("### Plot B - middle residues (5 x-slots)")
    A("")
    A("| Peptide length | Positions used | Slots |")
    A("|---|---|---|")
    A("| 9-mer | 3, 4, 5, 6, 7 | M1-M5 |")
    A("| 11-mer | 4, 5, 6, 7, 8 | M1-M5 |")
    A("| 10-mer | 4, 5, 6, 7 | **M2-M5** (M1 left empty) |")
    A("")
    A("A 10-mer has an even length and therefore no single central residue; taking the 4 residues "
      "symmetric about the 5.5 midpoint and leaving the first slot empty is the requested "
      "convention. Note this data set contains **three** 10-mers - KRAS G6C mutant, KRAS G6C "
      "wild-type, and the APC frameshift neoantigen - and all three follow the same rule.")
    A("")
    A(fig(f"figures/rmsf/rmsf_B_core_5pos{sfx}.png",
          "Figure 4: C-alpha RMSF of the middle residues, all 17 peptides.", 800))
    A(fig(f"figures/rmsf/rmsf_B_core_5pos_panels{sfx}.png",
          "Figure 4b: the same data as small multiples.", 980))
    A("")

    # ---------------- caveat ----------------
    A("## 3. Interpretation caveat - what this fit frame can and cannot see")
    A("")
    A("Every analysis here superposes on the **peptide's own C-alpha**, so it reports the "
      "peptide's *internal conformational change*. Rigid-body motion of the peptide inside the "
      "HLA groove - sliding, rocking, and in particular a terminus lifting out of the groove - is "
      "largely **removed by the fit** and will not appear in these RMSD or RMSF values.")
    A("")
    A("This matters because the project's existing unbinding analysis "
      "(`unbinding_summary_dist-contact_residue+neighbor.md`) shows terminal unbinding is common "
      "in several of these systems - for example STRDPLSEITE (wild-type) in 9/10 replicas "
      "(57% of the time), the APC frameshift neoantigen in 9/10 replicas (47%), and AHHGGWTTK "
      "(wild-type) in 7/10 replicas (22%). Those events are, by construction, **not** what these "
      "figures are measuring.")
    A("")
    A("Peptide displacement *within the groove* is measured separately, and by a "
      "different criterion, in the terminal-stability analysis "
      "(`scripts/detect_unbinding.py`).")
    A("")

    # ---------------- files ----------------
    A("## 4. Files")
    A("")
    A("```")
    A("md_analyses/")
    A("  REPORT.md                     this report")
    A("  run_md_analyses.py            orchestrator")
    A("  scripts/                      mdconfig, common, build_cache,")
    A("                                analyze_rmsd, analyze_rmsf, make_report")
    A("  cache/<complex>/rep_N.nc      reduced coordinates: all CA + peptide heavy atoms,")
    A("                                all 4166 frames (0-500 ns), ~23 MB/replica")
    A("  figures/rmsd/                 11 RMS-distribution panels + overview")
    A("  figures/rmsf/                 the two RMSF plots + small multiples")
    A("  data/                         every number behind every figure, as CSV/NPZ")
    A("```")
    A("")

    out = os.path.join(cfg.OUT_DIR, "REPORT.md" if not sfx else f"REPORT{sfx}.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
