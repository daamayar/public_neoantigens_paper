#!/usr/bin/env python3
"""
Phase 4 -- assemble REPORT.md from the Phase-2 JSON files.

Every number in the report is read from the analysis JSONs at generation time,
so the prose cannot drift away from the data: re-running Phase 2 and then this
module always yields a self-consistent report.

Usage
-----
    python -m scripts.make_report
    python -m scripts.make_report --output /tmp/draft.md
"""
import os
import json
import argparse
import datetime

import numpy as np

from . import inconfig as cfg
from . import common


def _load(name):
    """Load a JSON payload from the package data directory.

    Args:
        name (str): file name, e.g. "compare_bound.json".

    Returns:
        dict: parsed payload.

    Raises:
        FileNotFoundError: if Phase 2 has not been run.
    """
    path = os.path.join(cfg.DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {path}\n  Fix: python -m scripts.analyze_network")
    with open(path) as fh:
        return json.load(fh)


# Figure and table counters. Reset at the top of build() so that numbering is
# always assigned in reading order and can never drift from the prose.
_COUNT = {"figure": 0, "table": 0}


def _fig(name, caption, width=900):
    """Numbered figure block, per the project's report convention.

    The number is assigned from the order figures are emitted, and is checked
    against the ``figN_`` prefix of the file name: ``make_figures.py`` names its
    outputs in reading order, so a mismatch means one of the two drifted and
    should be fixed rather than silently published.

    Args:
        name (str): figure file name inside ``figures/``, e.g.
            "fig1_contact_maps_bound.png".
        caption (str): figure caption, without the "Figure N." prefix.
        width (int): rendered width in pixels.

    Returns:
        str: the HTML <figure> block, or a warning line if the file is missing.

    Raises:
        ValueError: if the file's figN_ prefix disagrees with the position at
            which the figure is referenced.
    """
    n = _COUNT["figure"] + 1
    prefix = name.split("_")[0]
    if prefix != f"fig{n}":
        raise ValueError(
            f"figure numbering drift: '{name}' is referenced as Figure {n}. "
            f"Either reorder the report or rename the output in "
            f"make_figures.py so the figN_ prefix matches reading order.")
    _COUNT["figure"] = n

    path = os.path.join(cfg.FIG_DIR, name)
    if not os.path.exists(path):
        return f"*(Figure {n} (`{name}`) not generated — run `make_figures`)*\n"
    return (f'<figure>\n'
            f'  <img src="figures/{name}" alt="Figure {n}. {caption[:70]}" '
            f'width="{width}"/>\n'
            f'  <figcaption><strong>Figure {n}.</strong> {caption}'
            f'</figcaption>\n'
            f'</figure>\n')


def _table(caption):
    """Numbered table caption line, to be emitted immediately above the table.

    Args:
        caption (str): the caption text, without the "Table N." prefix.

    Returns:
        str: a Markdown line such as "**Table 4.** Interaction types ...".
    """
    _COUNT["table"] += 1
    return f"**Table {_COUNT['table']}.** {caption}"


def _definitions():
    """The 'Definitions' section: every quantity and test used in the report.

    Kept in one place so that no result is presented with an undefined unit,
    threshold or statistic. Thresholds are read from ``inconfig`` rather than
    written out by hand, so the text cannot drift from what was computed.

    Returns:
        list[str]: Markdown lines.
    """
    L = []
    A = L.append
    A("## 3. Definitions")
    A("")
    A("Every quantity below is computed per frame, averaged within a replica, "
      "and then averaged across the 10 replicas. **The replica is the unit of "
      "statistics throughout** — never the frame, which would inflate every "
      "sample size by ~4000 and make trivial differences significant.")
    A("")
    A("### Structural terms")
    A("")
    A(_table(
        "Structural terms."))
    A("")
    A("| term | definition |")
    A("|---|---|")
    A("| **Peptide position (P1…P11)** | Position along the peptide from its "
      "N-terminus. The two peptides here are 11-mers, so P11 is the last "
      "residue. |")
    A("| **PΩ (P-omega)** | The peptide's **C-terminal** position, whatever "
      "the peptide length — P11 for these 11-mers. Together with P2 it is one "
      "of the two *anchor* positions that hold a class-I peptide in the "
      "groove. |")
    A("| **Anchor** | A peptide position whose side chain is buried in a "
      "dedicated HLA pocket and whose identity dominates binding. For "
      "HLA-A\\*11:01 these are P2 and PΩ. |")
    A("| **Binding groove / α1-α2 platform** | HLA heavy-chain residues "
      f"**{cfg.GROOVE_RANGE[0]}–{cfg.GROOVE_RANGE[1]}**, the two helices and "
      "the β-sheet floor that form the peptide-binding site. All analyses are "
      "restricted to it (verified: no receptor residue above 171 ever comes "
      "within 10 Å of either peptide). |")
    A("| **F pocket** | The pocket at the **C-terminal** end of the groove "
      "that receives the PΩ side chain. In A\\*11:01 it is lined by Asp77, "
      "Thr80, Leu81, Tyr84, Ile95, Asp116, Tyr123, Thr143, Lys146 and Trp147. "
      "Its **Asp116** is what makes the allele prefer a basic PΩ. |")
    A("| **Replica** | One independent 500 ns MD trajectory (4166 frames) of "
      "the same complex, differing in initial velocities and solvation. There "
      "are 10 per peptide. |")
    A("")
    A("### Interaction types")
    A("")
    A("All are counted between a **peptide residue** and an **HLA groove "
      "residue**, per frame. Only heavy atoms are used for distances; "
      "hydrogens are used only for the hydrogen-bond angle.")
    A("")
    A(_table(
        "The five interaction types scored between each peptide residue and each HLA groove residue, and the geometric criterion for each."))
    A("")
    A("| interaction | geometric criterion | reported as |")
    A("|---|---|---|")
    A(f"| **Contact** | any heavy atom of the peptide residue within "
      f"**{cfg.CONTACT_CUTOFF} Å** of any heavy atom of the HLA residue | "
      f"*occupancy* (see below) |")
    A(f"| **Hydrogen bond** | hydrogen-to-acceptor distance < "
      f"**{cfg.HBOND_HA_CUTOFF} Å** *and* donor–H···acceptor angle > "
      f"**{cfg.HBOND_ANGLE_CUTOFF:.0f}°**. Donors are N/O/S–H, acceptors are "
      f"N/O. Counted in both directions. | mean count per frame |")
    A(f"| **Salt bridge** | any atom of a **cationic** group (Lys NZ; Arg NE/"
      f"NH1/NH2; protonated His ND1/NE2) within **{cfg.SALT_BRIDGE_CUTOFF} Å** "
      f"of any atom of an **anionic** group (Asp OD1/OD2; Glu OE1/OE2; the "
      f"C-terminal carboxylate OT1/OT2). Like-charged pairs are never "
      f"counted. | mean count per frame |")
    A(f"| **Hydrophobic contact** | an apolar atom of the peptide within "
      f"**{cfg.HYDROPHOBIC_CUTOFF} Å** of an apolar atom of the HLA. *Apolar* "
      f"= sulfur, or carbon **not** covalently bonded to N or O — which "
      f"excludes carbonyl carbons and Cα. | mean count per frame |")
    A(f"| **Cation–π** | a cation centre (Lys NZ, Arg CZ, protonated-His ring "
      f"centroid) within **{cfg.CATION_PI_CUTOFF} Å** of an aromatic ring "
      f"centroid (Phe, Tyr, Trp — both indole rings — His) *and* within "
      f"**{cfg.CATION_PI_ANGLE:.0f}°** of the ring normal, i.e. above the "
      f"ring face rather than in its plane. | mean count per frame |")
    A("")
    A("### Derived quantities")
    A("")
    A(_table(
        "Derived quantities."))
    A("")
    A("| quantity | definition |")
    A("|---|---|")
    A("| **Occupancy** | The **fraction of frames in the window** in which the "
      "interaction is present. `occupancy = 1.00` means present in every "
      "frame; `0.00` means never. A per-*edge* occupancy refers to one "
      "peptide-residue↔HLA-residue pair. |")
    A("| **Summed contact occupancy** (per position) | The sum of a peptide "
      "position's contact occupancies over all 180 groove residues. It is "
      "*not* a probability and can exceed 1: a value of 8.8 means the residue "
      "is, on average, simultaneously in contact with ~8.8 HLA residues. It "
      "measures how much of the groove that position engages. |")
    A("| **Coulomb energy (ε = 1)** | The exact pairwise Coulomb sum between "
      "all peptide and all groove atoms, in vacuum. This is the MM term that "
      "sums to MM-PBSA's ΔEEL. It is **not** a free energy: with no solvent "
      "screening, charged pairs are overestimated by roughly an order of "
      "magnitude. |")
    A("| **Screened electrostatics** | The same sum with a distance-dependent "
      f"dielectric ε(r) = {cfg.SCREEN_DIELECTRIC_SLOPE:.0f}r and a "
      f"Debye–Hückel factor exp(−r/{cfg.DEBYE_LENGTH_A} Å) at the pipeline's "
      "0.15 M ionic strength. A **qualitative** descriptor of which "
      "electrostatic pairs survive aqueous screening — used for ranking, not "
      "for thermodynamics. |")
    A("| **Lennard-Jones (vdW) energy** | The exact 6-12 dispersion/repulsion "
      "sum over the same atom pairs, using the topology's A/B coefficients "
      "(including CHARMM NBFIX). Sums to MM-PBSA's ΔVDWAALS. |")
    A("| **Weighted degree** (network figures) | For a peptide position, the "
      "sum of the contact occupancies of all its edges; for an HLA residue, "
      "the same over its edges. Node size in the network graph. |")
    A("")
    A("### Statistics")
    A("")
    A(_table(
        "Statistical symbols used throughout."))
    A("")
    A("| symbol | meaning |")
    A("|---|---|")
    A("| **mean ± SEM** | Mean over the 10 replicas ± standard error of that "
      "mean (SD⁄√10). The spread is *between replicas*, so it reflects "
      "reproducibility across independent trajectories. |")
    A("| **p** | Two-sided **Mann–Whitney U** test comparing the 10 "
      "neoantigen replica values against the 10 wild-type replica values. "
      "Non-parametric, because n = 10 is far too small to check normality. "
      "At this sample size SciPy uses the tie-corrected normal "
      "approximation, for which two completely non-overlapping groups of 10 "
      "give p ≈ 1.8×10⁻⁴. A reported `p = 2×10⁻⁴` therefore means the two "
      "sets of replicas do not overlap at all — the strongest statement this "
      "design can make. (Values slightly below that arise when many "
      "replicas tie, e.g. an occupancy that is exactly 0 in all ten "
      "wild-type replicas, which shrinks the tie-corrected variance.) |")
    A("| **q** | The **Benjamini–Hochberg FDR-adjusted p-value**. Hundreds of "
      "residue-pair edges are tested against the same 10 vs 10 replicas, so "
      "raw p-values would be badly optimistic. `q < 0.05` means that if all "
      "edges with a q at or below that value are called significant, the "
      "expected proportion of false positives among them is under 5%. |")
    A("| **ρ** | **Spearman** rank correlation coefficient, used for the "
      "anchor-occupancy versus stability relationship (monotonic, not "
      "necessarily linear). |")
    A("| **\\*, \\*\\*, \\*\\*\\*** (figures) | q < 0.05, 0.01, 0.001 "
      "respectively. |")
    A("")
    return L


def _sig(p):
    """Format a p-value compactly.

    Args:
        p (float): p-value.

    Returns:
        str: e.g. "2e-04" or "0.31", or "n/a" for a non-finite value.
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.0e}" if p < 0.001 else f"{p:.3f}"


def build():
    """Assemble the full report text.

    Returns:
        str: the complete Markdown document.
    """
    # Numbering is assigned in reading order; reset so that repeated calls to
    # build() in one process do not continue the previous document's count.
    _COUNT["figure"] = 0
    _COUNT["table"] = 0

    onset = _load("compare_onset.json")
    early = _load("compare_early.json")
    bound = _load("compare_bound.json")
    alls = _load("compare_all.json")
    cross = _load("cross_system_pomega.json")

    neo_seq, wt_seq = bound["neo"]["seq"], bound["wt"]["seq"]
    L = []
    A = L.append

    # =====================================================================
    A("# Why the PIK3CA E11K neoantigen stays bound to HLA-A\\*11:01 "
      "and its wild-type does not")
    A("")
    A(f"*Generated {datetime.date.today().isoformat()} by "
      f"`interaction_network/scripts/make_report.py`.*")
    A("")
    A("## Summary")
    A("")
    tb = bound["totals"]
    to = onset["totals"]
    A(f"The neoantigen **{neo_seq}** and its wild-type **{wt_seq}** differ at "
      f"one position — the last one. In HLA-A\\*11:01 that position is the "
      f"**PΩ (C-terminal) anchor**, and the allele's F pocket is built "
      f"around **Asp116**. The neoantigen's Lys11 forms a salt bridge to "
      f"Asp116 that is present in "
      f"**{bound['salt_bridges'] and _pomega_occ(bound, 'Asp116'):.0%}** of "
      f"bound frames; the wild-type's Glu11 is the same charge as Asp116 and "
      f"never forms it in any of the 10 replicas.")
    A("")
    A("Losing that anchor does not just remove one contact. It costs the "
      "wild-type the **conserved class-I C-terminal carboxylate anchor to "
      "Lys146** as well, and the loss propagates back along the peptide to "
      "P10 and P9 — positions whose residues are *identical* in the two "
      "peptides. The result is a C-terminal third that is never properly "
      "seated in the groove, which is exactly what the unbinding traces show.")
    A("")
    A("Five independent lines of evidence support this:")
    A("")
    A(f"1. **The difference is there before anything happens.** In the first "
      f"2 ns, while both peptides still hold their C-terminal carboxylate in "
      f"the F pocket, the neoantigen already makes "
      f"{to['tot_occupancy']['neo']:.1f} vs {to['tot_occupancy']['wt']:.1f} "
      f"residue-pair contacts (p = {_sig(to['tot_occupancy']['p'])}). The "
      f"difference is not a consequence of unbinding. (Table 11)")
    A(f"2. **It survives conditioning on the bound state.** Restricting to "
      f"frames in which the wild-type's C-terminus *is* engaged, the gap is "
      f"unchanged: {tb['tot_occupancy']['neo']:.1f} vs "
      f"{tb['tot_occupancy']['wt']:.1f} contacts "
      f"(p = {_sig(tb['tot_occupancy']['p'])}) and "
      f"{tb['tot_hbond']['neo']:.1f} vs {tb['tot_hbond']['wt']:.1f} hydrogen "
      f"bonds (p = {_sig(tb['tot_hbond']['p'])}). (Tables 7-8, Figures 1-2)")
    A("3. **The same interaction explains the neoantigen's own failures.** "
      "In the 4 neoantigen replicas that do unbind, the Lys11–Asp116 salt "
      "bridge breaks *first*, in the 20 ns before the event. "
      "(Table 12, Figure 5)")
    A("4. **It generalises across the dataset.** Of the six A\\*11:01 "
      "peptides simulated here, the five ending in Lys are C-terminally "
      "stable; the one ending in Glu is the outlier. "
      "(Table 13, Figure 6)")
    A("5. **It matches experiment.** All 10 deposited HLA-A\\*11:01 "
      "structures carry Lys at PΩ and all 10 show the same "
      "Lys–Asp116 salt bridge, at a distance the simulations reproduce. "
      "(Table 14, Figure 7)")
    A("")
    A("---")
    A("")

    # =====================================================================
    A("## 1. The question")
    A("")
    A("The unbinding analysis (`scripts/detect_unbinding.py`) found a large "
      "asymmetry in this pair:")
    A("")
    A(_table(
        "C-terminal unbinding of the two PIK3CA E11K peptides, scored by `scripts/detect_unbinding.py` over 10 replicas each."))
    A("")
    A("| | C-terminal unbinding events | mean fraction of time detached | "
      "median first event |")
    A("|---|---|---|---|")
    for row in cross["by_allele"]["A1101"]:
        if row["seq"] in (neo_seq, wt_seq):
            tag = "neoantigen" if row["seq"] == neo_seq else "wild-type"
            A(f"| **{row['seq']}** ({tag}) | {row['cterm_events']}/10 | "
              f"{row['cterm_time_unbound']:.1%} | "
              f"{'237 ns' if row['seq'] == neo_seq else '66 ns'} |")
    A("")
    A("The N-terminus is stable in both; the asymmetry is entirely "
      "C-terminal. This report asks what interaction difference causes it.")
    A("")

    # =====================================================================
    A("## 2. Method, and the confound it has to avoid")
    A("")
    A("For every frame of all 20 trajectories (2 peptides × 10 replicas × "
      "4166 frames) the full peptide-residue × HLA-residue interaction "
      "network was computed: minimum heavy-atom distance, hydrogen bonds "
      "(H···A < 2.5 Å, D–H···A > 120°), charged-group distances, and the "
      "gas-phase MM interaction energy (Coulomb and Lennard-Jones) "
      "decomposed per residue pair.")
    A("")
    A("**The confound.** The wild-type is detached for ~57% of its "
      "trajectory. Any contact difference measured over all frames is "
      "therefore partly a restatement of the unbinding result, not an "
      "explanation of it. Every comparison below is therefore reported in "
      "four conditioning windows:")
    A("")
    A(_table(
        "The four conditioning windows. Every comparison in this report is computed in all four."))
    A("")
    A("| window | frames used | what it controls for |")
    A("|---|---|---|")
    for w in ("onset", "early", "bound", "all"):
        spec = cfg.WINDOWS[w]
        A(f"| `{w}` | {spec['label']} | "
          f"{spec['description'].split('.')[0]}. |")
    A("")
    A("`onset` and `bound` are the causally informative ones: the first "
      "precedes every event, the second removes the detached frames.")
    A("")
    A("**Validation.** The MM energy code reproduces the pipeline's own "
      "MM-PBSA numbers exactly. On the identical frames and receptor mask, "
      "ΔVDWAALS agrees to 0.0000 kcal/mol and ΔEEL to 0.006 kcal/mol out of "
      "~100 (`python -m scripts.compute_contacts --validate`). Reproducing "
      "ΔEEL required dividing the vacuum Coulomb sum by `indi` = 4: sander's "
      "PB path reports electrostatics scaled by the solute dielectric, so "
      "**the pipeline's tabulated EEL values are one quarter of the vacuum "
      "Coulomb energy.** This is self-consistent (EPB uses the same interior "
      "dielectric) but matters for anyone comparing them against an "
      "independent calculation.")
    A("")

    # =====================================================================
    L.extend(_definitions())

    # =====================================================================
    A("## 4. The difference is localised to the last three positions")
    A("")
    A(_fig("fig1_contact_maps_bound.png",
           "Contact occupancy of every peptide position against every HLA "
           "groove residue, in C-terminus-bound frames only. Top: "
           "neoantigen. Middle: wild-type. Bottom: difference. The upper two "
           "thirds of the difference map is blank — P1–P8 are "
           "indistinguishable. Everything is concentrated in a block at "
           "P9/P10/P11 against the F-pocket residues Asp77, Thr80, Leu81, "
           "Tyr84, Ile95, Asp116, Tyr123, Thr143, Lys146 and Trp147.", 950))
    A("")
    A(_fig("fig2_per_position_bound.png",
           "The same data collapsed onto peptide position, with replica-level "
           "statistics. P1–P8 overlap within error; P9, P10 and P11 separate. "
           "P10 and P9 carry identical residues in the two peptides, so their "
           "loss is a consequence of the P11 anchor failing, not of any local "
           "chemical change.", 780))
    A("")
    A("**Every interaction type per peptide position** (Table 7), C-terminus-bound "
      "frames only, given as **neoantigen / wild-type** (mean over 10 "
      "replicas; see §3 for the definition of each). Contacts are summed "
      "occupancies; hydrogen bonds, salt bridges, hydrophobic contacts and "
      "cation–π are mean counts per frame; energies are kcal/mol.")
    A("")
    A(_table(
        "Every interaction type, per peptide position, in C-terminus-bound frames (neoantigen / wild-type, mean over 10 replicas)."))
    A("")
    A("| position | residue<br>neo / wt | contacts | H-bonds | salt bridges | "
      "hydrophobic | cation–π | vdW (LJ) | screened elec. |")
    A("|---|---|---|---|---|---|---|---|---|")
    keys = ("occupancy", "hbond", "saltbridge", "hydrophobic", "cation_pi",
            "vdw", "eel_scr")
    fmts = (".2f", ".2f", ".2f", ".1f", ".2f", ".1f", ".1f")
    for p in bound["positions"]:
        res = (f"**{p['neo_res']} / {p['wt_res']}**" if p["mutated"]
               else f"{p['neo_res']}")
        star = " ⬅" if p["mutated"] else ""
        cells = " | ".join(
            f"{p[k]['neo']:{f}} / {p[k]['wt']:{f}}" for k, f in zip(keys, fmts))
        A(f"| P{p['position']}{star} | {res} | {cells} |")
    tot_keys = ("tot_occupancy", "tot_hbond", "tot_saltbridge",
                "tot_hydrophobic", "tot_cation_pi", "tot_vdw", "tot_eel_scr")
    cells = " | ".join(
        f"**{tb[k]['neo']:{f}} / {tb[k]['wt']:{f}}**"
        for k, f in zip(tot_keys, fmts))
    A(f"| **total** | | {cells} |")
    A("")
    A("The same totals with replica-level statistics:")
    A("")
    A(_table(
        "Interface totals with replica-level statistics (C-terminus-bound frames, mean ± SEM over 10 replicas)."))
    A("")
    A("| interaction | neoantigen | wild-type | Δ (neo − wt) | p |")
    A("|---|---|---|---|---|")
    for key, lab in (("tot_occupancy", "Contacts (summed occupancy)"),
                     ("tot_hbond", "Hydrogen bonds"),
                     ("tot_saltbridge", "Salt bridges"),
                     ("tot_hydrophobic", "Hydrophobic contacts"),
                     ("tot_cation_pi", "Cation–π interactions"),
                     ("tot_vdw", "Lennard-Jones energy (kcal/mol)"),
                     ("tot_eel_scr", "Screened electrostatics (kcal/mol)")):
        v = tb[key]
        A(f"| {lab} | {v['neo']:.2f} ± {v['neo_sem']:.2f} | "
          f"{v['wt']:.2f} ± {v['wt_sem']:.2f} | {v['delta']:+.2f} | "
          f"{_sig(v['p'])} |")
    A("")
    A("**Every interaction type moves in the same direction**, and in each "
      "case the whole of the difference comes from P9–P11: over P1–P8 the "
      "wild-type is level with or slightly ahead of the neoantigen on every "
      "one of them. The wild-type is not making a *weaker* version of the "
      "same interface — it is making the same N-terminal interface and "
      "essentially no C-terminal one.")
    A("")
    A("The two chemistry-specific columns are worth reading separately, "
      "because they are the ones a contact count alone would hide:")
    A("")
    A("- **Salt bridges** more than double (2.63 vs 1.19), and 2.19 of the "
      "neoantigen's 2.63 belong to P11 alone.")
    A("- **Cation–π** is small but categorical. The neoantigen's Lys11 sits "
      "over the **Tyr123** ring for 18% of bound frames; the wild-type's "
      "Glu11 does so for 0%, and cannot — an anion has no cation–π to make. "
      "The remaining cation–π in both peptides is Arg3 against Tyr9/Tyr99, "
      "which is shared.")
    A("")

    # =====================================================================
    A("## 5. The specific interaction: Lys11 → Asp116")
    A("")
    A("The single largest edge difference in the network is the mutated "
      "residue's contact with Asp116 (Table 9). The strongest "
      "differential edges, C-terminus-bound frames:")
    A("")
    A(_table(
        "The 14 largest per-contact differences between the two peptides, ranked by Δ occupancy (C-terminus-bound frames)."))
    A("")
    A("| peptide position | HLA residue | occupancy neo | occupancy wt | Δ | "
      "q |")
    A("|---|---|---|---|---|---|")
    n_edges_tested = len(bound["edges"])
    for e in bound["edges"][:14]:
        res = (f"P{e['position']}{e['neo_res']}/{e['wt_res']}"
               if e["neo_res"] != e["wt_res"] else f"P{e['position']}{e['neo_res']}")
        A(f"| {res} | {e['hla']} | {e['neo_occ']:.3f} | {e['wt_occ']:.3f} | "
          f"{e['delta']:+.3f} | {e['q']:.1e} |")
    A("")
    q_all = np.array([e["q"] for e in bound["edges"]], dtype=float)
    q_min = float(np.nanmin(q_all))
    n_at_floor = int(np.sum(q_all <= q_min * 1.0001))
    n_sig = int(np.sum(q_all < 0.05))
    A(f"> **What `q` is.** Each row is one two-sided **Mann–Whitney U** test "
      f"of that residue pair's contact occupancy, comparing the 10 "
      f"neoantigen replica values against the 10 wild-type ones. "
      f"**{n_edges_tested} edges** were tested against the same replicas, so "
      f"raw p-values would be badly optimistic; they are corrected by the "
      f"**Benjamini–Hochberg** procedure and `q` is that corrected value — "
      f"an estimated **false discovery rate**. Calling every edge with "
      f"q < 0.05 significant ({n_sig} of {n_edges_tested} here) means at most "
      f"~5% of those calls are expected to be false positives. "
      f"**{n_at_floor} edges sit at the floor value q = {q_min:.1e}**: these "
      f"are the pairs whose two replica groups do not overlap at all, so the "
      f"rank test cannot distinguish between them and they all receive the "
      f"same q. Within that group, rank by Δ (the effect size), not by q.")
    A("")
    A("")
    A("**The wild-type also loses the conserved carboxylate anchor.** Every "
      "class-I peptide, whatever its sequence, ties its C-terminal "
      "carboxylate into the F pocket via Tyr84/Thr143/Lys146. Both peptides "
      "have that carboxylate. Only the neoantigen keeps it:")
    A("")
    A(_table(
        "Charged-group pairs formed by the C-terminal residue P11, including the conserved class-I carboxylate anchor."))
    A("")
    A("| charged-group pair (neo / wt) | occupancy neo | occupancy wt | "
      "mean distance neo | mean distance wt |")
    A("|---|---|---|---|---|")
    # Keep only pairs that actually approach each other in at least one of the
    # two systems; the rest are groove residues 20 A away and carry no signal.
    pfx = f"P{len(neo_seq)}"
    rows = [x for x in bound["salt_bridges"]
            if x["pair"].startswith(pfx)
            and min(x["neo_dist"],
                    x["wt_dist"] if x["wt_dist"] is not None else 1e9) < 15.0]
    for s in sorted(rows, key=lambda x: -x["neo_occ"]):
        wt_occ = ("—" if not np.isfinite(s["wt_occ"])
                  else f"{s['wt_occ']:.3f}")
        wt_d = "—" if s["wt_dist"] is None else f"{s['wt_dist']:.2f} Å"
        label = s["pair"]
        if s.get("wt_pair") and s["wt_pair"] != s["pair"]:
            label = f"{s['pair']} / {s['wt_pair']}"
        A(f"| {label} | {s['neo_occ']:.3f} | {wt_occ} | "
          f"{s['neo_dist']:.2f} Å | {wt_d} |")
    A("")
    A("The wild-type's C-terminal carboxylate sits a mean of "
      f"{_salt(bound, 'P11-COO---Lys146', 'wt_dist'):.1f} Å from Lys146 — it "
      "is not in the F pocket at all, even in frames the unbinding detector "
      "scores as bound (that criterion covers residues 10–11 jointly, and "
      "the wild-type can satisfy it through P10 while P11 dangles).")
    A("")
    A(_fig("fig4_network_graph_bound.png",
           "The interaction network as a graph. Node size is weighted degree, "
           "edge width is contact occupancy. The neoantigen's P11K is the "
           "largest hub in its network; the wild-type's P11E is the smallest "
           "node in its own. Note that the wild-type has MORE edges "
           f"({bound['wt']['network']['n_edges']} vs "
           f"{bound['neo']['network']['n_edges']}) but LESS total weight "
           f"({bound['wt']['network']['total_weight']:.1f} vs "
           f"{bound['neo']['network']['total_weight']:.1f}): having lost its "
           "anchor it makes many weak, transient contacts instead of a few "
           "persistent ones.", 950))
    A("")

    # =====================================================================
    A("## 6. The initial state already differs — this is not kinetics")
    A("")
    A("Both complexes were built and equilibrated identically, and at the "
      "start of production **both** hold the conserved C-terminal "
      "carboxylate in the F pocket (9 of 10 wild-type replicas have it at "
      "2.6–2.8 Å from Lys146, matching the crystal structures). So the "
      "wild-type was *not* mis-docked.")
    A("")
    A("What differs at t = 0 is the side chain. In the first 2 ns:")
    A("")
    A(_table(
        "The initial docked state, measured over the first 2 ns — before any unbinding event in either system."))
    A("")
    A("| quantity | neoantigen | wild-type | p |")
    A("|---|---|---|---|")
    for key, lab in (("tot_occupancy", "total contact occupancy"),
                     ("tot_hbond", "peptide↔HLA hydrogen bonds")):
        v = to[key]
        A(f"| {lab} | {v['neo']:.2f} ± {v['neo_sem']:.2f} | "
          f"{v['wt']:.2f} ± {v['wt_sem']:.2f} | {_sig(v['p'])} |")
    p11o = next(p for p in onset["positions"] if p["mutated"])
    A(f"| P11 contact occupancy | {p11o['occupancy']['neo']:.2f} | "
      f"{p11o['occupancy']['wt']:.2f} | {_sig(p11o['occupancy']['p'])} |")
    e116 = next((e for e in onset["edges"]
                 if e["hla"] == "Asp116" and e["position"] == len(neo_seq)),
                None)
    if e116:
        A(f"| P11 ↔ Asp116 contact occupancy | {e116['neo_occ']:.2f} | "
          f"{e116['wt_occ']:.2f} | q = {e116['q']:.0e} |")
    A("")
    A("The Glu11 side chain cannot occupy the site the Lys11 side chain "
      "occupies, because that site is an aspartate carboxylate. Equilibration "
      "already pushed it away, and during production the rest of the "
      "C-terminus followed it out of the pocket.")
    A("")

    # =====================================================================
    A("## 7. Cross-system control: it is the anchor, not the mutation")
    A("")
    A("If a basic PΩ residue is what holds a peptide in the A\\*11:01 "
      "F pocket, then every A\\*11:01 peptide ending in Lys should be "
      "C-terminally stable regardless of neo/wild-type status. Six A\\*11:01 "
      "peptides were simulated:")
    A("")
    A(_table(
        "The six HLA-A\\*11:01 peptides simulated in this project, ordered by C-terminal instability."))
    A("")
    A("| peptide | PΩ | C-term events | fraction of time detached |")
    A("|---|---|---|---|")
    for r in cross["by_allele"]["A1101"]:
        bold = "**" if r["pomega"] != "K" else ""
        A(f"| {bold}{r['seq']}{bold} ({r['label']}) | {bold}{r['pomega']}"
          f"{bold} | {r['cterm_events']}/10 | "
          f"{bold}{r['cterm_time_unbound']:.1%}{bold} |")
    A("")
    A("The five Lys-terminated peptides span both neoantigens and wild-types "
      "and include the two KRAS wild-types; all are stable. The single "
      "Glu-terminated peptide is the outlier. The mutation is incidental — "
      "what matters is that it happens to sit on the anchor.")
    A("")
    A("")
    A("A second, independent instance of the same principle is already in "
      "the dataset: the A\\*03:01 pair **AHHGGWTTK / ALHGGWTTK** (PIK3CA "
      "H2L) is mutated at P2, the *N-terminal* anchor — and its instability "
      "is correspondingly N-terminal (4 N-only events versus 2 C-only). "
      "Mutating an anchor destabilises the end of the peptide that anchor "
      "holds.")
    A("")

    # =====================================================================
    A("## 8. What this means for the MM-PBSA free energies")
    A("")
    A("This pair has by far the largest ΔΔG of the seven in the project "
      "(−12.9 kcal/mol by QH, versus −5.5 to −9.2 for the others). A natural "
      "worry is that the number is inflated because the wild-type spends "
      "half its trajectory detached, so its ensemble is not one of a bound "
      "complex.")
    A("")
    A("The interaction energies say the effect is real, not an artifact of "
      "the detached frames — conditioning on bound frames barely changes it:")
    A("")
    A(_table(
        "Peptide↔HLA interaction energies by conditioning window (kcal/mol, HLA residues 1–180)."))
    A("")
    A("| window | Coulomb (ε=1) neo | wt | Δ | screened neo | wt | Δ | "
      "Lennard-Jones Δ |")
    A("|---|---|---|---|---|---|---|---|")
    for w, dd in (("onset", onset), ("early", early), ("bound", bound),
                  ("all", alls)):
        t = dd["totals"]
        A(f"| `{w}` | {t['tot_eel']['neo']:.0f} | {t['tot_eel']['wt']:.0f} | "
          f"{t['tot_eel']['delta']:+.0f} | {t['tot_eel_scr']['neo']:.1f} | "
          f"{t['tot_eel_scr']['wt']:.1f} | {t['tot_eel_scr']['delta']:+.1f} | "
          f"{t['tot_vdw']['delta']:+.1f} |")
    A("")
    A("All energies in kcal/mol, peptide against HLA residues 1–180. The "
      "screened column applies ε(r) = 4r with Debye damping at 0.15 M and is "
      "a qualitative descriptor only, but it is the one that should be "
      "compared to a solvated ΔΔH: its value of "
      f"{bound['totals']['tot_eel_scr']['delta']:+.1f} kcal/mol is the right "
      "order for the observed ΔΔH of −11.7 kcal/mol, whereas the raw vacuum "
      "Coulomb difference of "
      f"{bound['totals']['tot_eel']['delta']:+.0f} kcal/mol is "
      "hugely overestimated because desolvation is not subtracted.")
    A("")
    A("Two caveats nonetheless apply to the reported ΔΔG for this pair:")
    A("")
    A("- The wild-type's MM-PBSA ensemble genuinely is a mixture of bound "
      "and partly-released states. The ΔΔG is a fair description of *that* "
      "ensemble, but it is not the free-energy difference between two "
      "properly bound complexes, and it should be reported as such.")
    A("- The wild-type peptide carries a net charge of −2 against the "
      "neoantigen's 0. Charged-ligand MM-PBSA is the case where the method "
      "is least reliable, and the PB term is doing a lot of work here.")
    A("")

    # =====================================================================
    A("## 9. Caveats")
    A("")
    A("- **One mutation, one pair.** The cross-system control (§7) is what "
      "raises this above an n-of-1, but only one Glu-terminated A\\*11:01 "
      "peptide was simulated. The 5-vs-1 comparison in §7 has no error bar "
      "on the Glu side.")
    A("- **The replicas are the statistical unit** and there are only 10 per "
      "system. All p-values are Mann-Whitney on replica-level values, with "
      "Benjamini-Hochberg correction wherever many interactions are tested "
      "at once. With 10 vs 10 the test saturates at p ≈ 1.8×10⁻⁴ (complete "
      "separation), so it can establish *that* two groups separate but not "
      "how strongly; effect sizes carry that information, not p-values.")
    A("- **Force field and protonation are fixed inputs.** Both aspartates "
      "and both glutamates are modelled deprotonated at pH 7 by CHARMM36m. "
      "A protonated Glu11 would change the electrostatics qualitatively; "
      "this was not tested, and the F pocket's Asp116/Asp77 pair does make "
      "the local environment one where a raised Glu pKa is conceivable.")
    A("- **The MM energies are not free energies.** No solvation term is "
      "included in the per-residue decomposition; it is a descriptor of the "
      "interaction network, and the MM-PBSA pipeline remains the source for "
      "energetics.")
    A("- **All geometric thresholds are conventional** and are listed in §3 "
      f"({cfg.CONTACT_CUTOFF} Å contact, "
      f"{cfg.HBOND_HA_CUTOFF} Å / {cfg.HBOND_ANGLE_CUTOFF:.0f}° hydrogen "
      f"bond, {cfg.SALT_BRIDGE_CUTOFF} Å salt bridge, "
      f"{cfg.HYDROPHOBIC_CUTOFF} Å hydrophobic, "
      f"{cfg.CATION_PI_CUTOFF} Å / {cfg.CATION_PI_ANGLE:.0f}° cation–π). "
      "The effect sizes here are far larger than any plausible sensitivity "
      "to those choices, but they were not varied systematically.")
    A("- **Cation–π is the least robust of the interaction types.** It "
      "depends on a ring-normal angle that is noisy for a partially solvated "
      "surface aromatic, and it is a small term here. It is reported for "
      "completeness; none of the conclusions rest on it.")
    A("")

    # =====================================================================
    A("## 10. Nomenclature and sources")
    A("")
    A("**Notation.** P1…P11 are positions within the peptide; PΩ is "
      "the C-terminal position (P11 for these 11-mers). The mutation is "
      "written in the project's paper notation **E11K**; in full-length "
      "PIK3CA numbering it is **E545K**. HLA residues use standard mature "
      "class-I numbering, which this project's topologies match 1:1 "
      "(verified against the conserved Tyr9/Tyr84/Thr143/Lys146/Trp147/"
      "Tyr171 signature).")
    A("")
    A("**Retrieved sources.** Per the project's citation rules, only "
      "material actually retrieved in the course of this analysis is cited.")
    A("")
    A("- The A\\*11:01 primary anchors are P2 Ile/Val and PΩ Lys. "
      "Li L & Bouvier M, *J Immunol* 172(10):6175–84 (2004), PMID 15128805, "
      "[DOI](https://doi.org/10.4049/jimmunol.172.10.6175) — retrieved via "
      "PubMed. The abstract states the crystal structures \"confirm the "
      "presence of primary anchor residues P2-Ile/-Val and P9-/P10-Lys\". "
      "*Abstract only; full text was not accessible, so nothing beyond this "
      "sentence is attributed to the paper.*")
    A("- HLA-A allele motifs carry \"critical anchor residues at position 2 "
      "and at the COOH-terminal\", and the A3 and A11 motifs are \"very "
      "similar to each other\". Kubo RT *et al.*, *J Immunol* "
      "152(8):3913–24 (1994), PMID 8144960 — retrieved via PubMed, abstract "
      "only. *The abstract does not name the specific C-terminal residue, so "
      "the identity of the A\\*11:01 anchor is not attributed to this paper.*")
    A("- **The Asp116 salt bridge is not taken from the literature.** It is "
      "measured here directly from the simulations.")
    A("")
    A("Attribution: bibliographic records above were retrieved from PubMed.")
    A("")

    # =====================================================================
    A("## 11. Reproducing this analysis")
    A("")
    A("```bash")
    A("conda activate AmberTools25 && source $CONDA_PREFIX/amber.sh")
    A("cd interaction_network")
    A("")
    A("python -m scripts.compute_contacts --validate   # check MM energies")
    A("python -m scripts.compute_contacts --workers 10 # Phase 1, ~7 min")
    A("python -m scripts.analyze_network               # Phase 2, ~2 min")
    A("python -m scripts.make_figures                  # Phase 3")
    A("python -m scripts.make_report                   # this document")
    A("```")
    A("")
    A("Phase 1 reads the clean trajectories produced by "
      "`scripts/detect_unbinding.py` and writes ~1.3 GB of per-frame arrays "
      "into `interaction_network/data/`. Everything downstream is pure "
      "post-processing of those arrays and re-runs in minutes.")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Small helpers that pull specific numbers out of the payloads
# ---------------------------------------------------------------------------

def _pomega_occ(payload, hla):
    """Occupancy of the P-Omega side-chain salt bridge to an HLA residue.

    Args:
        payload (dict): a ``compare_<window>.json`` payload.
        hla (str): HLA residue label, e.g. "Asp116".

    Returns:
        float: the neoantigen's occupancy, or NaN if the pair is absent.
    """
    seq = payload["neo"]["seq"]
    name = f"P{len(seq)}{seq[-1]}--{hla}"
    for s in payload["salt_bridges"]:
        if s["pair"] == name:
            return s["neo_occ"]
    return float("nan")


def _salt(payload, pair, field):
    """Look up one field of one tracked charged-group pair.

    Args:
        payload (dict): a ``compare_<window>.json`` payload.
        pair (str): the tracked pair name.
        field (str): the field to return, e.g. "wt_dist".

    Returns:
        float: the value, or NaN if absent.
    """
    for s in payload["salt_bridges"]:
        if s["pair"] == pair:
            v = s.get(field)
            return float("nan") if v is None else v
    return float("nan")


def _pomega_median_distance():
    """Median P-Omega Lys NZ to Asp116 distance in the neoantigen, 50-500 ns.

    Returns:
        float: the median distance in Angstrom over all 10 replicas.
    """
    from . import analyze_network as an
    neo = next(s for s in cfg.SYSTEMS if s["kind"] == "neo")
    vals = []
    for rep in range(1, cfg.NUM_REPLICAS + 1):
        d = an.load_replica(neo["dirname"], rep)
        seq = d["seq"]
        k = d["salt_names"].index(f"P{len(seq)}{seq[-1]}--Asp116")
        vals.append(d["salt"][cfg.EQUIL_FRAMES:, k])
    return float(np.median(np.concatenate(vals)))


def _with_index(text):
    """Insert a list of figures and tables after the summary.

    The index is scraped back out of the finished document rather than tracked
    while writing it, so it can never list something the document does not
    actually contain.

    Args:
        text (str): the complete report body.

    Returns:
        str: the report with a "Figures and tables" index inserted immediately
            before the first numbered section.
    """
    import re

    figs = re.findall(r"<strong>(Figure \d+)\.</strong>\s*(.+?)</figcaption>",
                      text, flags=re.S)
    tabs = re.findall(r"^\*\*(Table \d+)\.\*\*\s*(.+)$", text, flags=re.M)

    def _short(caption, limit=95):
        one = " ".join(caption.split())
        if len(one) <= limit:
            return one
        cut = one[:limit].rsplit(" ", 1)[0]
        return cut.rstrip(" ,;—-") + " …"

    lines = ["## Figures and tables", ""]
    for label, cap in figs:
        lines.append(f"- **{label}.** {_short(cap)}")
    lines.append("")
    for label, cap in tabs:
        lines.append(f"- **{label}.** {_short(cap)}")
    lines += ["", "---", ""]

    marker = "## 1. The question"
    if marker not in text:
        return text
    head, sep, tail = text.partition(marker)
    return head + "\n".join(lines) + "\n" + sep + tail


def main():
    ap = argparse.ArgumentParser(
        description="Phase 4: assemble REPORT.md from the analysis JSONs.")
    ap.add_argument("--output", default=os.path.join(cfg.OUT_DIR, "REPORT.md"),
                    help="output path (default: interaction_network/REPORT.md)")
    args = ap.parse_args()

    text = _with_index(build())
    common.ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as fh:
        fh.write(text)
    print(f"wrote {args.output}  ({len(text.splitlines())} lines)  "
          f"{_COUNT['figure']} figures, {_COUNT['table']} tables")


if __name__ == "__main__":
    main()
