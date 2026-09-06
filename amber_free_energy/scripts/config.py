"""
Shared configuration for the MM-PBSA binding free energy pipeline.
All paths, constants, and parameters are defined here.
"""
import os

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================
# Three roots are kept distinct so that code, bulk trajectory data and analysis
# outputs may sit on different filesystems:
#
#   REPO_DIR     this repository (code and documentation)
#   BASE_DIR     simulation data, laid out as <BASE_DIR>/<complex>/rep_<n>/...
#                Override with PHLA_DATA_ROOT.
#   RESULTS_DIR  analysis outputs written by phases 4-5.
#                Override with PHLA_RESULTS_DIR.
#
# The production trajectories are ~50 GB per complex and are distributed
# separately (see docs/DATA.md), so BASE_DIR usually points outside the
# repository. RESULTS_DIR defaults to amber_free_energy/results/ and is the
# single location both the pipeline and summarize_mmpbsa.py operate on.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPTS_DIR)          # amber_free_energy/
REPO_DIR = os.path.dirname(PACKAGE_DIR)             # repository root

BASE_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("PHLA_DATA_ROOT") or os.path.join(REPO_DIR, "data")))
RESULTS_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("PHLA_RESULTS_DIR") or os.path.join(PACKAGE_DIR, "results")))

# =============================================================================
# FORCE FIELD PARAMETER FILES
# =============================================================================
# CHARMM36m parameter files. Only par_water_ions.prm is redistributed here; the
# MacKerell-lab files and VMD's wat.top carry their own licences and must be
# added locally. See forcefield/README.md. Override with CHARMM_FF_DIR.
NAMD_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("CHARMM_FF_DIR") or os.path.join(REPO_DIR, "forcefield")))

# VMD water topology, required for the TIP3P residue and atom types (OT, HT)
# that the CHARMM protein RTF does not define. Ships with VMD under
# lib/vmd/plugins/noarch/tcl/solvate<version>/wat.top. Override with VMD_WAT_TOP.
VMD_WAT_TOP = os.path.abspath(os.path.expanduser(
    os.environ.get("VMD_WAT_TOP") or os.path.join(NAMD_DIR, "wat.top")))

# ParmEd requires parameter files in a specific order:
# 1. Protein RTF (topology definitions)
# 2. VMD water topology (TIP3P atom types OT, HT)
# 3. Protein parameters (CMAP, bonds, angles, etc.)
# 4. Additional parameter files (lipid, carb, cgenff)
# 5. Water/ion parameter file (par_water_ions.prm - extracted from NAMD str)
# 6. Water/ion stream file (toppar_water_ions_namd.str - for RTF section)
#
# Note: The NAMD-formatted toppar_water_ions_namd.str has commented-out 'read para'
# commands, so ParmEd cannot parse its parameter section. The water/ion bond, angle,
# nonbonded, and NBFIX parameters were extracted to par_water_ions.prm.
PARAM_FILES = [
    os.path.join(NAMD_DIR, "top_all36_prot.rtf"),
    VMD_WAT_TOP,
    os.path.join(NAMD_DIR, "par_all36m_prot.prm"),
    os.path.join(NAMD_DIR, "par_all36_lipid.prm"),
    os.path.join(NAMD_DIR, "par_all36_carb.prm"),
    os.path.join(NAMD_DIR, "par_all36_cgenff_namd.prm"),
    os.path.join(NAMD_DIR, "par_water_ions.prm"),
    os.path.join(NAMD_DIR, "toppar_water_ions_namd.str"),
]

# =============================================================================
# FILE NAMING PATTERNS
# =============================================================================
PSF_PATTERN = "*_ionized.psf"
PDB_PATTERN = "*_ionized.pdb"
XSC_PATTERN = "*_ionized_bb.0.xsc"
CONF_PATTERN = "*_production_HMR.conf"

# Solvated production trajectory, in preference order. Both forms are read the same
# way: phase 2 runs cpptraj `autoimage` before stripping, which reassembles molecules
# across the periodic boundary, so an unwrapped concatenation needs no separate
# wrapping pass. `scripts/wrap_trajectory.py` produces the wrapped form explicitly for
# anyone who wants that intermediate on disk.
#   *_wrapped_500ns.dcd            wrapped and RMS-fitted (wrapping.tcl output)
#   *_production_*_stride*.dcd     raw catdcd concatenation, not yet imaged
DCD_PATTERNS = ["*_wrapped_500ns.dcd", "*_production_*_stride*.dcd"]
# Kept for callers that want the canonical name.
DCD_PATTERN = DCD_PATTERNS[0]

# Dry (protein-only) trajectory, as distributed on Zenodo. Its atom ordering is
# identical to complex.prmtop, so it is read with that topology directly.
DRY_DCD_PATTERN = os.path.join("cleanDCD", "*_wrapped_500ns.clean.dcd")
DRY_PSF_PATTERN = os.path.join("cleanDCD", "*_ionized.clean.psf")

# =============================================================================
# TRAJECTORY SOURCE
# =============================================================================
# Which form of the production trajectory the analyses read.
#
#   "auto"     (default) solvated if present, otherwise dry
#   "solvated" the ~5 GB/replica solvated DCD, stripped on the fly as before
#   "dry"      the deposited protein-only trajectory, used directly
#
# The two forms give identical results. Every analysis measures inter-atomic
# quantities after discarding solvent, and the dry trajectory differs from the
# stripped solvated one only by a rigid-body superposition, which leaves all
# internal distances unchanged (verified: max deviation 8e-6 A on the C-terminal
# anchor distance across a 500 ns replica). The solvent is therefore never read
# for any reported quantity; it only lets cpptraj reimage before stripping, and
# the dry trajectory is already whole.
TRAJECTORY_SOURCE = os.environ.get("PHLA_TRAJECTORY_SOURCE", "auto").strip().lower()
VALID_TRAJECTORY_SOURCES = ("auto", "solvated", "dry")
if TRAJECTORY_SOURCE not in VALID_TRAJECTORY_SOURCES:
    raise ValueError(
        f"PHLA_TRAJECTORY_SOURCE={TRAJECTORY_SOURCE!r} is not one of "
        f"{VALID_TRAJECTORY_SOURCES}")

# md_setup/resolvate.py rebuilds solvent around a SINGLE frame, for restarting a
# simulation or visualising a solvated system. It cannot produce a solvated
# trajectory for this pipeline: a DCD or NetCDF trajectory requires a fixed atom
# count, and solvating each frame independently yields a different water count
# every time. "solvated" therefore means the original production DCDs, not
# reconstructed ones.

# =============================================================================
# SYSTEM STRUCTURE (CHARMM segment names)
# =============================================================================
RECEPTOR_SEGMENTS = ["APRO", "BPRO"]  # HLA alpha + beta-2-microglobulin
LIGAND_SEGMENTS = ["CPRO"]            # Peptide

# Residues/segments to strip (water and ions)
STRIP_RESIDUE_NAMES = ["TIP3", "WAT", "SOD", "CLA", "POT", "Na+", "Cl-", "K+", "MG", "CAL"]
STRIP_MASK_CPPTRAJ = ":TIP3,WAT,SOD,CLA,POT,Na+,Cl-,K+,MG,CAL"
# Note: MMPBSA.py uses the prmtop which is already stripped, so no separate mask needed

# =============================================================================
# TRAJECTORY PARAMETERS
# =============================================================================
TOTAL_SIM_TIME_NS = 500.0
TOTAL_FRAMES = 4166
TIME_PER_FRAME_NS = TOTAL_SIM_TIME_NS / TOTAL_FRAMES  # ~0.12002 ns/frame

# Number of replicas
NUM_REPLICAS = 10

# =============================================================================
# ANALYSIS WINDOWS
# =============================================================================
EQUILIBRATION_NS = 50.0  # Skip first 50 ns
EQUILIBRATION_FRAMES = int(round(EQUILIBRATION_NS / TIME_PER_FRAME_NS))  # ~417

# 500ns analysis: skip 50ns, stride 10, analyze remaining ~450ns
ANALYSIS_500NS = {
    "name": "500ns",
    "first_frame": EQUILIBRATION_FRAMES + 1,  # 1-based indexing for cpptraj
    "last_frame": TOTAL_FRAMES,
    "stride": 10,
    "description": "Full 500ns trajectory (skip first 50ns, stride 10)",
}

# 100ns analysis: skip first 50ns, analyze 50-100ns, no stride
ANALYSIS_100NS_END_NS = 100.0
ANALYSIS_100NS_END_FRAME = int(round(ANALYSIS_100NS_END_NS / TIME_PER_FRAME_NS))  # ~833
ANALYSIS_100NS = {
    "name": "100ns",
    "first_frame": EQUILIBRATION_FRAMES + 1,  # 1-based
    "last_frame": ANALYSIS_100NS_END_FRAME,
    "stride": 1,  # No stride → ~416 frames
    "description": "First 100ns trajectory (skip first 50ns, no stride)",
}

# 200ns analysis: skip first 50ns, analyze 50-200ns, stride 3
ANALYSIS_200NS_END_NS = 200.0
ANALYSIS_200NS_END_FRAME = int(round(ANALYSIS_200NS_END_NS / TIME_PER_FRAME_NS))  # ~1666
ANALYSIS_200NS = {
    "name": "200ns",
    "first_frame": EQUILIBRATION_FRAMES + 1,
    "last_frame": ANALYSIS_200NS_END_FRAME,
    "stride": 3,  # ~417 frames
    "description": "First 200ns trajectory (skip first 50ns, stride 3)",
}

# 300ns analysis: skip first 50ns, analyze 50-300ns, stride 5
ANALYSIS_300NS_END_NS = 300.0
ANALYSIS_300NS_END_FRAME = int(round(ANALYSIS_300NS_END_NS / TIME_PER_FRAME_NS))  # ~2500
ANALYSIS_300NS = {
    "name": "300ns",
    "first_frame": EQUILIBRATION_FRAMES + 1,
    "last_frame": ANALYSIS_300NS_END_FRAME,
    "stride": 5,  # ~417 frames
    "description": "First 300ns trajectory (skip first 50ns, stride 5)",
}

# 400ns analysis: skip first 50ns, analyze 50-400ns, stride 7
ANALYSIS_400NS_END_NS = 400.0
ANALYSIS_400NS_END_FRAME = int(round(ANALYSIS_400NS_END_NS / TIME_PER_FRAME_NS))  # ~3333
ANALYSIS_400NS = {
    "name": "400ns",
    "first_frame": EQUILIBRATION_FRAMES + 1,
    "last_frame": ANALYSIS_400NS_END_FRAME,
    "stride": 7,  # ~417 frames
    "description": "First 400ns trajectory (skip first 50ns, stride 7)",
}

# All analysis windows (ordered ascending by simulation time)
ANALYSIS_WINDOWS = {
    "100ns": ANALYSIS_100NS,
    "200ns": ANALYSIS_200NS,
    "300ns": ANALYSIS_300NS,
    "400ns": ANALYSIS_400NS,
    "500ns": ANALYSIS_500NS,
}

# Convenience constants
ALL_WINDOW_NAMES = list(ANALYSIS_WINDOWS.keys())  # ["100ns", "200ns", ..., "500ns"]
REFERENCE_WINDOW = "500ns"  # Window used as reference for convergence comparisons

# Binding free energy quantity plotted in convergence_across_windows.png:
#   "dh" -> MM-PBSA enthalpy only (DELTA TOTAL)
#   "qh" -> dH + TdS_QH, the quasi-harmonic entropy-corrected binding free energy
CONVERGENCE_DG_METHOD = "qh"

# =============================================================================
# MMPBSA PARAMETERS
# =============================================================================
MMPBSA_PARAMS = {
    "indi": 4.0,           # Interior dielectric
    "exdi": 80.0,          # Exterior dielectric
    "istrng": 0.150,       # Ionic strength (M)
    "radiopt": 0,          # Use radii from prmtop (mbondi_pb3 assigned during conversion)
    "inp": 1,              # Linear nonpolar solvation (inp=2 not suitable with mbondi_pb3 + CHARMM36m)
    "fillratio": 4.0,      # Grid fill ratio
    "scale": 2.0,          # Grid resolution (0.5 A spacing)
    "linit": 1000,         # Max PB iterations
    "prbrad": 1.4,         # Solvent probe radius (A)
    "use_sander": 1,       # Use sander for energy evaluation (mandatory for ChamberParm)
    "temperature": 310.0,  # Simulation temperature (K)
}

# =============================================================================
# ENTROPY CORRECTION
# =============================================================================
# The configurational entropy of binding is obtained by quasi-harmonic analysis
# of cpptraj covariance matrices, as post-processing in Phase 4:
#   -TΔS = -T [ S(complex) - S(receptor) - S(ligand) ]
QHA_PARAMS = {
    "atom_selection": "ca",  # Atom selection for covariance matrix
    "temperature": 310.0,    # Temperature (K)
}

# =============================================================================
# OAR JOB SUBMISSION
# =============================================================================
# Site-specific: these describe the Grid'5000 installation the study was run on.
# --generate-oar-scripts is a convenience for that scheduler and is not required
# to reproduce any result. Override through the environment for another site;
# an empty OAR_EMAIL omits the --notify flag from the generated commands.
OAR_EMAIL = os.environ.get("OAR_EMAIL", "")
OAR_QUEUE = os.environ.get("OAR_QUEUE", "production")
OAR_CLUSTERS = os.environ.get("OAR_CLUSTERS", "")
OAR_WALLTIME_HOURS = int(os.environ.get("OAR_WALLTIME_HOURS", "12"))

# =============================================================================
# CONDA ENVIRONMENT
# =============================================================================
CONDA_ENV = "AmberTools25"

# Conda activation commands (AmberTools25 requires sourcing amber.sh)
CONDA_ACTIVATE_CMDS = [
    "conda activate AmberTools25",
    "source $CONDA_PREFIX/amber.sh",
]

# =============================================================================
# AMBERHOME
# =============================================================================
AMBERHOME = os.environ.get("AMBERHOME", os.path.expanduser("~/anaconda3/envs/AmberTools25"))

def validate_amberhome():
    """Check that AMBERHOME is set and key binaries exist."""
    for binary in ["cpptraj", "MMPBSA.py"]:
        path = os.path.join(AMBERHOME, "bin", binary)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Binary not found: {path}\n"
                f"  AMBERHOME={AMBERHOME}\n"
                f"  Fix: run 'conda activate {CONDA_ENV} && source $CONDA_PREFIX/amber.sh' "
                f"before running the pipeline."
            )


def validate_data_root():
    """Check that BASE_DIR exists and holds at least one complex directory."""
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(
            f"Simulation data root not found: {BASE_DIR}\n"
            f"  PHLA_DATA_ROOT={os.environ.get('PHLA_DATA_ROOT', '(unset)')}\n"
            f"  Set PHLA_DATA_ROOT to the directory holding <complex>/rep_<n>/. "
            f"See docs/DATA.md for the expected layout and the Zenodo archives."
        )


def validate_force_field():
    """
    Check that every CHARMM36m parameter file needed by ParmEd is present.

    Only par_water_ions.prm is distributed with this repository; the remaining
    files must be obtained from the MacKerell lab (CHARMM36m) and from a VMD
    installation (wat.top). See forcefield/README.md.
    """
    missing = [p for p in PARAM_FILES if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing CHARMM36m force field files:\n"
            + "".join(f"  {p}\n" for p in missing)
            + f"  CHARMM_FF_DIR={NAMD_DIR}\n"
            f"  VMD_WAT_TOP={VMD_WAT_TOP}\n"
            "  See forcefield/README.md for how to obtain them."
        )
