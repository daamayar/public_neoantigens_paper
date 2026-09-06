#!/usr/bin/env python3
"""
Central configuration for the peptide-MD structural analyses (RMSD / RMSF).

Design decisions encoded here
-----------------------------
FIT FRAME      : "peptide" -- every frame is superposed on the *peptide's own*
                 C-alpha atoms, so all metrics report the peptide's INTERNAL
                 conformational change. Rigid-body motion of the peptide within
                 the HLA groove (sliding, rocking, terminal lift-out) is removed
                 by the fit. The metrics are computed on peptide atoms only --
                 the HLA is never included in a reported value. Displacement of
                 the peptide relative to the groove is measured separately by
                 scripts/detect_unbinding.py.

ANALYSIS WINDOW: frames 418-4166 (50-500 ns). The first 50 ns is discarded as
                 equilibration, matching the convention already established by
                 this project's MM-PBSA pipeline (scripts/config.py). Including
                 it would add a relaxation transient to the RMSD distributions and
                 inflate the RMSF.

RMSD REFERENCE : frame 1 of each replica's own production DCD (t ~ 0), i.e. the
                 literal "first frame". Because the cache stores all 4166 frames,
                 the reference is simply cache frame index 0, while the analysis
                 ensemble is cache frames [EQUIL_FRAMES:].

RMSD ATOMS     : peptide backbone N, CA, C, O (primary). C-alpha-only and
                 all-heavy-atom variants are also computed and written to CSV.

                 IMPORTANT CHARMM36 DETAIL: the C-terminal residue carries a
                 carboxylate named OT1/OT2 and has NO atom called "O". OT1 and
                 OT2 are equivalent by resonance and swap whenever the
                 carboxylate flips 180 degrees about the CA-C bond, so picking
                 one of them arbitrarily would inject spurious RMSD. The
                 backbone "O" of the C-terminal residue is therefore taken as the
                 CENTROID of OT1 and OT2, which is invariant under that flip.
                 See common.build_index_map().
"""
import os

# =============================================================================
# PATHS
# =============================================================================
# OUT_DIR   md_analyses/ inside this repository; every output is written here.
# BASE_DIR  simulation data root, <BASE_DIR>/<complex>/rep_<n>/... Override with
#           PHLA_DATA_ROOT; identical semantics to scripts/config.py so the two
#           packages always read the same trajectories. See docs/DATA.md.
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # md_analyses/
PACKAGE_DIR = os.path.dirname(OUT_DIR)                                   # amber_free_energy/
REPO_DIR = os.path.dirname(PACKAGE_DIR)                                  # repository root

BASE_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("PHLA_DATA_ROOT") or os.path.join(REPO_DIR, "data")))

CACHE_DIR = os.path.join(OUT_DIR, "cache")
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR = os.path.join(OUT_DIR, "figures")

AMBERHOME = os.environ.get("AMBERHOME", os.path.expanduser("~/anaconda3/envs/AmberTools25"))
CPPTRAJ = os.path.join(AMBERHOME, "bin", "cpptraj")

# Per-replica input files (relative to BASE_DIR/<dirname>/rep_<n>/)
DCD_PATTERNS = ["*_wrapped_500ns.dcd", "*_production_*_stride*.dcd"]
DCD_PATTERN = DCD_PATTERNS[0]
# Dry (protein-only) trajectory as deposited; same atom order as complex.prmtop.
DRY_DCD_PATTERN = os.path.join("cleanDCD", "*_wrapped_500ns.clean.dcd")
COMPLEX_PRMTOP_REL = os.path.join("mmpbsa_500ns", "complex.prmtop")
# "auto" | "solvated" | "dry"; see scripts/config.py for the rationale.
TRAJECTORY_SOURCE = os.environ.get("PHLA_TRAJECTORY_SOURCE", "auto").strip().lower()
SOLVATED_PRMTOP = os.path.join("mmpbsa_500ns", "solvated.prmtop")
COMPLEX_PRMTOP = os.path.join("mmpbsa_500ns", "complex.prmtop")

# =============================================================================
# TRAJECTORY GEOMETRY
# =============================================================================
NUM_REPLICAS = 10
TOTAL_FRAMES = 4166                      # frames in *_wrapped_500ns.dcd
TOTAL_SIM_TIME_NS = 500.0
TIME_PER_FRAME_NS = TOTAL_SIM_TIME_NS / TOTAL_FRAMES     # ~0.12002 ns

EQUILIBRATION_NS = 50.0
EQUIL_FRAMES = int(round(EQUILIBRATION_NS / TIME_PER_FRAME_NS))   # 417
# 0-based slice of the cache used for every distribution / RMSF:
ANALYSIS_SLICE = slice(EQUIL_FRAMES, TOTAL_FRAMES)                # 3749 frames, 50-500 ns
REFERENCE_FRAME_INDEX = 0                                          # t ~ 0, "the first frame"

# =============================================================================
# TOPOLOGY LAYOUT (identical for all 17 complexes -- verified)
# =============================================================================
N_RECEPTOR_RES = 375        # HLA heavy chain (1-276) + beta-2-microglobulin (277-375)
CHAIN_A = (1, 276)          # HLA alpha chain          (1-based, inclusive)
CHAIN_B = (277, 375)        # beta-2-microglobulin
GROOVE_RES = (1, 180)       # alpha1+alpha2 peptide-binding platform
# Peptide = residues 376 .. n_residues (length 9, 10 or 11 -- read from the topology)

# Water / ion residues stripped before anything else
STRIP_MASK_CPPTRAJ = ":TIP3,WAT,SOD,CLA,POT,Na+,Cl-,K+,MG,CAL"

# Genuine chain junctions: consecutive CA-CA distance is legitimately large here,
# so the PBC-break detector must skip these boundaries.
CHAIN_JUNCTIONS = (276, 375)     # between residue i and i+1 (1-based)
CA_CA_BREAK_CUTOFF = 4.5         # A; within a chain, consecutive CA-CA is ~3.8 A

# =============================================================================
# SUPERPOSITION ("FIT") CONVENTION
# =============================================================================
# "peptide"  -> fit on the peptide's own C-alpha  (internal conformation; DEFAULT)
# "receptor" -> fit on receptor C-alpha (res 1-375) (peptide motion in groove frame)
FIT_MODE = "peptide"

# =============================================================================
# ANALYSIS WINDOWS / ATOM SELECTIONS
# =============================================================================
RMSD_SELECTIONS = ["backbone", "ca", "heavy"]   # all computed; "backbone" is plotted
RMSD_PRIMARY = "backbone"
RMSF_SELECTION = "ca"                            # per-residue C-alpha RMSF

# =============================================================================
# THE 17 SYSTEMS
# =============================================================================
# kind: "snv_neo" | "snv_wt" | "fs_neo"
# pair: shared key linking an SNV mutant to its wild-type partner (None for frameshift)
# gene: HGNC gene symbol the peptide derives from. This drives all display names.
#
# NOTATION. Two schemes exist for an SNV:
#   * canonical  = the substitution's position in the FULL-LENGTH protein
#                  (e.g. FLT3 D835Y, PIK3CA E545K, KRAS G12C).
#   * paper      = the position WITHIN THE PEPTIDE, matching the reference paper's
#                  convention (e.g. FLT3 D1Y, PIK3CA E11K, KRAS G6C). This is what
#                  mutation_position() computes from the two sequences.
# Per the report spec, ALL tables and figures use the *paper* notation; the
# canonical form is kept (CANONICAL_MUT, below) only for a one-line mapping key in
# the report prose. The display fields `label` / `pair_label` are BUILT from
# `gene` + the paper mutation at the bottom of this section, so they can never
# drift from the sequences.
SYSTEMS = [
    dict(dirname="flt3_d835y_neo_a_0201_yimsdsnyv",
         seq="YIMSDSNYV", allele="A*02:01", kind="snv_neo", pair="A0201", gene="FLT3"),
    dict(dirname="flt3_d835y_wt_a_0201_dimsdsnyv",
         seq="DIMSDSNYV", allele="A*02:01", kind="snv_wt", pair="A0201", gene="FLT3"),

    dict(dirname="pik3ca_e545k_neo_a_1101_strdplseitk",
         seq="STRDPLSEITK", allele="A*11:01", kind="snv_neo", pair="A1101", gene="PIK3CA"),
    dict(dirname="pik3ca_e545k_wt_a_1101_strdplseite",
         seq="STRDPLSEITE", allele="A*11:01", kind="snv_wt", pair="A1101", gene="PIK3CA"),

    dict(dirname="kras_g12c_neo_a_1101_vvvgacgvgk",
         seq="VVVGACGVGK", allele="A*11:01", kind="snv_neo", pair="KRAS_G12C", gene="KRAS"),
    dict(dirname="kras_g12c_wt_a_1101_vvvgaggvgk",
         seq="VVVGAGGVGK", allele="A*11:01", kind="snv_wt", pair="KRAS_G12C", gene="KRAS"),

    dict(dirname="kras_g12d_neo_c_0802_gadgvgksa",
         seq="GADGVGKSA", allele="C*08:02", kind="snv_neo", pair="KRAS_G12D", gene="KRAS"),
    dict(dirname="kras_g12d_wt_c_0802_gaggvgksa",
         seq="GAGGVGKSA", allele="C*08:02", kind="snv_wt", pair="KRAS_G12D", gene="KRAS"),

    dict(dirname="kras_g12v_neo_a_1101_vvgavgvgk",
         seq="VVGAVGVGK", allele="A*11:01", kind="snv_neo", pair="KRAS_G12V", gene="KRAS"),
    dict(dirname="kras_g12v_wt_a_1101_vvgaggvgk",
         seq="VVGAGGVGK", allele="A*11:01", kind="snv_wt", pair="KRAS_G12V", gene="KRAS"),

    dict(dirname="p53_r175h_neo_a_0201_hmtevvrhc",
         seq="HMTEVVRHC", allele="A*02:01", kind="snv_neo", pair="P53_R175H", gene="p53"),
    dict(dirname="p53_r175h_wt_a_0201_hmtevvrrc",
         seq="HMTEVVRRC", allele="A*02:01", kind="snv_wt", pair="P53_R175H", gene="p53"),

    dict(dirname="pik3ca_h1047l_neo_a_0301_alhggwttk",
         seq="ALHGGWTTK", allele="A*03:01", kind="snv_neo", pair="PIK3CA_H1047L", gene="PIK3CA"),
    dict(dirname="pik3ca_h1047l_wt_a_0301_ahhggwttk",
         seq="AHHGGWTTK", allele="A*03:01", kind="snv_wt", pair="PIK3CA_H1047L", gene="PIK3CA"),

    # Frameshift neoantigens -- no wild-type counterpart, no substitution notation
    dict(dirname="apc_neo_a_0201_lqmdflvhpa",
         seq="LQMDFLVHPA", allele="A*02:01", kind="fs_neo", pair=None, gene="APC"),
    dict(dirname="npm_neo_a_0201_claveevsl",
         seq="CLAVEEVSL", allele="A*02:01", kind="fs_neo", pair=None, gene="NPM"),
    dict(dirname="tgfbrii_neo_a_0201_rlsscvpva",
         seq="RLSSCVPVA", allele="A*02:01", kind="fs_neo", pair=None, gene="TGFBRII"),
]

# Ordered list of the 7 SNV pair keys (one RMS-distribution panel each)
SNV_PAIRS = ["A0201", "A1101", "KRAS_G12C", "KRAS_G12D", "KRAS_G12V",
             "P53_R175H", "PIK3CA_H1047L"]

# Canonical (full-length protein) mutation per SNV pair. Used ONLY for the
# paper<->canonical mapping key in the report prose; never in tables or figures.
CANONICAL_MUT = {
    "A0201":         "D835Y",    # FLT3
    "A1101":         "E545K",    # PIK3CA
    "KRAS_G12C":     "G12C",
    "KRAS_G12D":     "G12D",
    "KRAS_G12V":     "G12V",
    "P53_R175H":     "R175H",
    "PIK3CA_H1047L": "H1047L",
}


def get_system(dirname):
    """Return the SYSTEMS entry whose 'dirname' matches, or raise KeyError.

    Args:
        dirname (str): complex directory name, e.g. "kras_g12v_neo_a_1101_vvgavgvgk".

    Returns:
        dict: the matching entry of SYSTEMS.
    """
    for s in SYSTEMS:
        if s["dirname"] == dirname:
            return s
    raise KeyError(f"unknown system: {dirname}")


def systems_by_kind(*kinds):
    """Return all SYSTEMS entries whose 'kind' is one of *kinds, in SYSTEMS order.

    Args:
        *kinds (str): any of "snv_neo", "snv_wt", "fs_neo".

    Returns:
        list[dict]: matching system entries.
    """
    return [s for s in SYSTEMS if s["kind"] in kinds]


def pair_members(pair_key):
    """Return (neo_system, wt_system) for an SNV pair key.

    Args:
        pair_key (str): one of SNV_PAIRS, e.g. "KRAS_G12V".

    Returns:
        tuple[dict, dict]: the mutant and the wild-type system entries.
    """
    neo = next(s for s in SYSTEMS if s["pair"] == pair_key and s["kind"] == "snv_neo")
    wt = next(s for s in SYSTEMS if s["pair"] == pair_key and s["kind"] == "snv_wt")
    return neo, wt


def mutation_position(pair_key):
    """Locate the substituted position of an SNV pair by diffing the two sequences.

    Args:
        pair_key (str): one of SNV_PAIRS.

    Returns:
        tuple[int, str, str]: (1-based position, mutant residue, wild-type residue).

    Raises:
        ValueError: if the two sequences differ in length or at != 1 position.
    """
    neo, wt = pair_members(pair_key)
    a, b = neo["seq"], wt["seq"]
    if len(a) != len(b):
        raise ValueError(f"{pair_key}: neo/wt lengths differ ({len(a)} vs {len(b)})")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) != 1:
        raise ValueError(f"{pair_key}: expected exactly 1 substitution, found {len(diff)}")
    i = diff[0]
    return i + 1, a[i], b[i]


def paper_mutation(pair_key):
    """Paper-notation mutation string of an SNV pair, e.g. "E11K".

    The number is the substituted residue's position *within the peptide*
    (1-based), formatted as <wild-type><position><mutant>. This is the notation
    used in every table and figure.

    Args:
        pair_key (str): one of SNV_PAIRS.

    Returns:
        str: e.g. "D1Y", "E11K", "G6C".
    """
    pos, mres, wres = mutation_position(pair_key)
    return f"{wres}{pos}{mres}"


def pair_display(pair_key):
    """Gene + paper-notation mutation for an SNV pair, e.g. "PIK3CA E11K".

    Args:
        pair_key (str): one of SNV_PAIRS.

    Returns:
        str: "<gene> <paper mutation>", e.g. "FLT3 D1Y", "KRAS G6C".
    """
    neo, _ = pair_members(pair_key)
    return f"{neo['gene']} {paper_mutation(pair_key)}"


# ---------------------------------------------------------------------------
# Build the display fields (`label`, `pair_label`) from gene + paper notation.
# Doing this programmatically guarantees the on-figure names never drift from the
# sequences, and that ONLY paper notation ever appears (never the canonical form).
#   label       : per-curve / legend / per-system-row name
#                   snv_neo -> "KRAS G6C VVVGACGVGK"
#                   snv_wt  -> "KRAS G6C wt VVVGAGGVGK"
#                   fs_neo  -> "APC LQMDFLVHPA"
#   pair_label  : panel-title identifier
#                   snv     -> "A*11:01 - KRAS G6C"
#                   fs      -> "A*02:01 - APC (fs)"
# ---------------------------------------------------------------------------
def _build_display_names():
    """Populate `label` and `pair_label` on every SYSTEMS entry (paper notation)."""
    for s in SYSTEMS:
        if s["kind"] == "fs_neo":
            s["label"] = f"{s['gene']} {s['seq']}"
            s["pair_label"] = f"{s['allele']} - {s['gene']} (fs)"
        else:
            mut = paper_mutation(s["pair"])
            wt = " wt" if s["kind"] == "snv_wt" else ""
            s["label"] = f"{s['gene']} {mut}{wt} {s['seq']}"
            s["pair_label"] = f"{s['allele']} - {s['gene']} {mut}"


_build_display_names()


# =============================================================================
# PLOTTING
# =============================================================================
NEO_COLOR = "#d62728"    # mutant / neoantigen  (red)
WT_COLOR = "#1f77b4"     # wild-type            (blue)
FS_COLOR = "#2ca02c"     # frameshift neoantigen(green)

# One colour per pair / frameshift system, used when many curves share an axis
SYSTEM_COLORS = {
    "A0201":          "#1f77b4",
    "A1101":          "#ff7f0e",
    "KRAS_G12C":      "#2ca02c",
    "KRAS_G12D":      "#d62728",
    "KRAS_G12V":      "#9467bd",
    "P53_R175H":      "#8c564b",
    "PIK3CA_H1047L":  "#e377c2",
    "apc_neo_a_0201_lqmdflvhpa":  "#7f7f7f",
    "npm_neo_a_0201_claveevsl":   "#bcbd22",
    "tgfbrii_neo_a_0201_rlsscvpva": "#17becf",
}

DPI = 200
