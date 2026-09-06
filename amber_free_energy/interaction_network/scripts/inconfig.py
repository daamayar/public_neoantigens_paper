#!/usr/bin/env python3
"""
Central configuration for the peptide-HLA INTERACTION NETWORK analyses.

This package answers a mechanistic question raised by the unbinding analysis
(`scripts/detect_unbinding.py`): the PIK3CA E11K neoantigen STRDPLSEITK stays
bound to HLA-A*11:01 in 6/10 replicas, whereas its wild-type STRDPLSEITE
releases its C-terminus in 9/10 replicas within ~100 ns.

The package builds, per frame, the full peptide<->HLA residue interaction
network (contacts, hydrogen bonds, salt bridges and gas-phase MM interaction
energies), then compares the two peptides under three *conditioning regimes*
that control for the fact that the wild-type spends most of its trajectory
partially detached (see WINDOWS below).


Design decisions encoded here
-----------------------------
RECEPTOR SCOPE : HLA residues 1-180 (the alpha1+alpha2 peptide-binding
                 platform). 

CONDITIONING   : the wild-type is *unbound* for ~57% of its trajectory, so any
                 contact difference measured over all frames is partly a
                 restatement of the unbinding result rather than an
                 explanation of it. Three windows are therefore always
                 computed (see WINDOWS):
                   "all"    - 50-500 ns, every frame (descriptive ensemble)
                   "bound"  - 50-500 ns, ONLY frames in which the C-terminal
                              region satisfies the bound criterion. Asks the
                              causal question: *even while engaged*, is the
                              wild-type network weaker?
                   "early"  - 0-25 ns, before any unbinding event in either
                              system. Both peptides start from the same
                              modelled pose, so this is the fairest matched
                              initial-state comparison and is causally
                              upstream of every event.

ENERGIES       : gas-phase MM interaction energies (Coulomb with eps=1 and
                 Lennard-Jones) decomposed per peptide-residue x HLA-residue
                 pair. These are exactly the pair terms that sum to MM-PBSA's
                 DELTA EEL and DELTA VDWAALS, so the totals can be validated
                 against the existing FINAL_RESULTS_MMPBSA.dat (done by
                 `compute_contacts.py --validate`). They are NOT free
                 energies: no solvent screening is included, so charged terms
                 are strongly overestimated in magnitude. A Debye-Huckel
                 screened Coulomb variant (eps = 4r, kappa from 0.15 M ionic
                 strength, matching the pipeline's istrng) is computed
                 alongside as a qualitative "what survives screening" check.

SIGN/UNITS     : distances in Angstrom, energies in kcal/mol, time in ns.
"""
import os

# =============================================================================
# PATHS
# =============================================================================
# OUT_DIR   interaction_network/ inside this repository; outputs are written here.
# BASE_DIR  simulation data root, <BASE_DIR>/<complex>/rep_<n>/... Override with
#           PHLA_DATA_ROOT; identical semantics to scripts/config.py. See docs/DATA.md.
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # interaction_network/
PACKAGE_DIR = os.path.dirname(OUT_DIR)                                 # amber_free_energy/
REPO_DIR = os.path.dirname(PACKAGE_DIR)                                # repository root

BASE_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("PHLA_DATA_ROOT") or os.path.join(REPO_DIR, "data")))

DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR = os.path.join(OUT_DIR, "figures")

RESULTS_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("PHLA_RESULTS_DIR") or os.path.join(PACKAGE_DIR, "results")))

# Per-replica inputs (relative to BASE_DIR/<dirname>/rep_<n>/)
# The "clean" trajectory was produced by scripts/detect_unbinding.py: the full
# 4166-frame wrapped DCD, autoimaged on the receptor and stripped of water and
# ions, so its atom order matches complex.prmtop exactly.
CLEAN_NC = os.path.join("unbinding", "rep{rep}_clean_500ns.nc")
COMPLEX_PRMTOP = os.path.join("mmpbsa_500ns", "complex.prmtop")
MMPBSA_RESULTS = os.path.join("mmpbsa_500ns", "FINAL_RESULTS_MMPBSA.dat")

# Per-frame unbinding time series already produced by scripts/detect_unbinding.py
UNBINDING_NPZ = "unbinding_rep{rep}_timeseries_dist-contact_residue-neighbor.npz"
UNBINDING_JSON = "unbinding_summary_dist-contact_residue-neighbor.json"

# =============================================================================
# TRAJECTORY GEOMETRY  (identical to scripts/config.py and md_analyses)
# =============================================================================
NUM_REPLICAS = 10
TOTAL_FRAMES = 4166
TOTAL_SIM_TIME_NS = 500.0
TIME_PER_FRAME_NS = TOTAL_SIM_TIME_NS / TOTAL_FRAMES        # ~0.12002 ns

EQUILIBRATION_NS = 50.0
EQUIL_FRAMES = int(round(EQUILIBRATION_NS / TIME_PER_FRAME_NS))          # 417

# =============================================================================
# TOPOLOGY LAYOUT
# =============================================================================
RECEPTOR_RANGE = (1, 375)        # HLA heavy chain 1-276 + beta-2-microglobulin
GROOVE_RANGE = (1, 180)          # alpha1 + alpha2 platform -- the analysed scope
PEPTIDE_START = 376              # peptide = 376 .. n_residues

N_GROOVE_RES = GROOVE_RANGE[1] - GROOVE_RANGE[0] + 1        # 180

# =============================================================================
# GEOMETRIC CRITERIA
# =============================================================================
CONTACT_CUTOFF = 4.5      # A, heavy-atom-to-heavy-atom; matches detect_unbinding.py
SALT_BRIDGE_CUTOFF = 4.0  # A, charged-group N/O to charged-group N/O
HBOND_HA_CUTOFF = 2.5     # A, hydrogen-to-acceptor distance
HBOND_ANGLE_CUTOFF = 120.0  # degrees, donor-H...acceptor angle
HYDROPHOBIC_CUTOFF = 4.5  # A, apolar-carbon/sulfur to apolar-carbon/sulfur
CATION_PI_CUTOFF = 6.0    # A, cation centre to aromatic-ring centroid
CATION_PI_ANGLE = 60.0    # degrees, max angle between the ring normal and the
                          #   ring-centroid -> cation vector (Gallivan-Dougherty
                          #   style geometric criterion: the cation must sit
                          #   above the ring face, not in its plane)

# Bound-state criterion used to build the "bound" conditioning mask. Identical
# in form to detect_unbinding.py's dist-contact criterion so the two analyses
# cannot disagree about which frames are bound.
BOUND_D_CUT = 5.0         # A, min heavy-atom distance C-term region <-> HLA
BOUND_F_CONTACT = 0.3     # fraction of the reference contact count
BOUND_REF_WINDOW_NS = 5.0  # ns averaged to define the reference contact count

# =============================================================================
# ELECTROSTATICS
# =============================================================================
COULOMB_K = 332.0522173  # kcal*A/(mol*e^2), AMBER's conversion constant
IONIC_STRENGTH_M = 0.15  # M, matches the MM-PBSA istrng setting
DEBYE_LENGTH_A = 7.86    # A, Debye length of 0.15 M 1:1 electrolyte at 310 K
SCREEN_DIELECTRIC_SLOPE = 4.0  # eps(r) = 4r, the standard distance-dependent model

# =============================================================================
# CONDITIONING WINDOWS
# =============================================================================
# name -> dict(frame_slice=(lo, hi) or None, bound_only=bool, description)
# `frame_slice` is a 0-based [lo, hi) slice of the 4166-frame trajectory.
WINDOWS = {
    "onset": dict(
        frame_slice=(0, int(round(2.0 / TIME_PER_FRAME_NS))),
        bound_only=False,
        label="first 2 ns (initial docked state)",
        description=(
            "The initial state of the production run. Both complexes were "
            "built and equilibrated in the same way and both still hold the "
            "conserved C-terminal carboxylate in the F pocket here, so any "
            "difference in this window is a property of the docked complex "
            "itself rather than of anything that happens during production. "
            "This is the earliest point at which the two systems can be "
            "compared, and it precedes every unbinding event by design."),
    ),
    "all": dict(
        frame_slice=(EQUIL_FRAMES, TOTAL_FRAMES),
        bound_only=False,
        label="all frames, 50-500 ns",
        description=(
            "Every production frame after equilibration. This is the ensemble "
            "the MM-PBSA free energies were computed over. It mixes bound and "
            "detached configurations, so it DESCRIBES the difference rather "
            "than explaining it."),
    ),
    "bound": dict(
        frame_slice=(EQUIL_FRAMES, TOTAL_FRAMES),
        bound_only=True,
        label="C-terminus-bound frames only, 50-500 ns",
        description=(
            "Only frames in which the peptide C-terminal region satisfies the "
            "bound criterion. Controls for the different amounts of time each "
            "peptide spends detached, and asks whether the wild-type network "
            "is already weaker WHILE it is still engaged."),
    ),
    "early": dict(
        frame_slice=(0, int(round(25.0 / TIME_PER_FRAME_NS))),
        bound_only=False,
        label="first 25 ns",
        description=(
            "The matched initial-state window. Both peptides start from the "
            "same modelled bound pose and no unbinding event has been scored "
            "in either system yet, so differences here are causally upstream "
            "of the unbinding events."),
    ),
}
REFERENCE_WINDOW = "bound"   # the window whose conclusions the report leads with

# =============================================================================
# THE SYSTEMS
# =============================================================================
# The primary comparison. `pomega` is the C-terminal (P-Omega) anchor residue.
PRIMARY_PAIR = "A1101_PIK3CA_E11K"

SYSTEMS = [
    dict(dirname="pik3ca_e545k_neo_a_1101_strdplseitk", seq="STRDPLSEITK", allele="A*11:01",
         kind="neo", pair="A1101_PIK3CA_E11K", gene="PIK3CA", mut="E11K",
         label="PIK3CA E11K neo (STRDPLSEITK)", short="neo K11"),
    dict(dirname="pik3ca_e545k_wt_a_1101_strdplseite", seq="STRDPLSEITE", allele="A*11:01",
         kind="wt", pair="A1101_PIK3CA_E11K", gene="PIK3CA", mut="E11K",
         label="PIK3CA E11K wt (STRDPLSEITE)", short="wt E11"),
]

# The 15 other complexes, used ONLY by the cross-system P-Omega analysis
# (analyze_network.py --cross-system), which needs no trajectory reading: it
# reads the existing unbinding summaries.
ALL_COMPLEXES = [
    ("flt3_d835y_neo_a_0201_yimsdsnyv", "YIMSDSNYV", "A*02:01", "FLT3 D1Y neo"),
    ("flt3_d835y_wt_a_0201_dimsdsnyv", "DIMSDSNYV", "A*02:01", "FLT3 D1Y wt"),
    ("pik3ca_e545k_neo_a_1101_strdplseitk", "STRDPLSEITK", "A*11:01", "PIK3CA E11K neo"),
    ("pik3ca_e545k_wt_a_1101_strdplseite", "STRDPLSEITE", "A*11:01", "PIK3CA E11K wt"),
    ("kras_g12c_neo_a_1101_vvvgacgvgk", "VVVGACGVGK", "A*11:01", "KRAS G6C neo"),
    ("kras_g12c_wt_a_1101_vvvgaggvgk", "VVVGAGGVGK", "A*11:01", "KRAS G6C wt"),
    ("kras_g12d_neo_c_0802_gadgvgksa", "GADGVGKSA", "C*08:02", "KRAS G3D neo"),
    ("kras_g12d_wt_c_0802_gaggvgksa", "GAGGVGKSA", "C*08:02", "KRAS G3D wt"),
    ("kras_g12v_neo_a_1101_vvgavgvgk", "VVGAVGVGK", "A*11:01", "KRAS G5V neo"),
    ("kras_g12v_wt_a_1101_vvgaggvgk", "VVGAGGVGK", "A*11:01", "KRAS G5V wt"),
    ("p53_r175h_neo_a_0201_hmtevvrhc", "HMTEVVRHC", "A*02:01", "p53 R8H neo"),
    ("p53_r175h_wt_a_0201_hmtevvrrc", "HMTEVVRRC", "A*02:01", "p53 R8H wt"),
    ("pik3ca_h1047l_neo_a_0301_alhggwttk", "ALHGGWTTK", "A*03:01", "PIK3CA H2L neo"),
    ("pik3ca_h1047l_wt_a_0301_ahhggwttk", "AHHGGWTTK", "A*03:01", "PIK3CA H2L wt"),
    ("apc_neo_a_0201_lqmdflvhpa", "LQMDFLVHPA", "A*02:01", "APC fs neo"),
    ("npm_neo_a_0201_claveevsl", "CLAVEEVSL", "A*02:01", "NPM fs neo"),
    ("tgfbrii_neo_a_0201_rlsscvpva", "RLSSCVPVA", "A*02:01", "TGFBRII fs neo"),
]

# =============================================================================
# CHARGED-GROUP DEFINITIONS (for salt bridges)
# =============================================================================
# Residue name -> (list of atom names forming the charged group, formal sign).
# CHARMM36 histidine variants: only HSP (doubly protonated) is cationic.
CHARGED_GROUPS = {
    "ARG": (["NE", "NH1", "NH2"], +1),
    "LYS": (["NZ"], +1),
    "HSP": (["ND1", "NE2"], +1),
    "ASP": (["OD1", "OD2"], -1),
    "GLU": (["OE1", "OE2"], -1),
}
# The C-terminal carboxylate is named OT1/OT2 in CHARMM36 and belongs to
# whatever the last residue is, in addition to that residue's own side chain.
CTERM_CARBOXYLATE = ["OT1", "OT2"]

# =============================================================================
# HLA CLASS-I RESIDUE ANNOTATION
# =============================================================================
# Conventional pocket assignments for HLA class I, used for LABELLING ONLY.
# Every quantitative pocket statement in the report is derived from the
# trajectories themselves (which HLA residues actually contact which peptide
# position); these names are the standard nomenclature attached afterwards.
# Source: see REPORT.md "Nomenclature and sources".
POCKET_LABELS = {
    "A": [5, 7, 59, 63, 66, 99, 159, 163, 167, 171],
    "B": [7, 9, 24, 25, 34, 45, 63, 66, 67, 70, 99],
    "C": [9, 70, 73, 74, 97],
    "D": [99, 114, 155, 156, 159, 160],
    "E": [97, 114, 133, 147, 152, 156],
    "F": [77, 80, 81, 84, 95, 116, 123, 143, 146, 147],
}

# The conserved class-I residues that hydrogen-bond the peptide's C-terminal
# main chain / carboxylate. Verified present in this topology.
CTERM_ANCHOR_RESIDUES = [77, 80, 84, 116, 143, 146, 147]

# =============================================================================
# PLOTTING
# =============================================================================
NEO_COLOR = "#d62728"    # neoantigen / mutant (red)
WT_COLOR = "#1f77b4"     # wild-type (blue)
DIFF_CMAP = "RdBu_r"
DPI = 200
