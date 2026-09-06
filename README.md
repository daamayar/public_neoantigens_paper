# Peptide-HLA neoantigen binding free energies

Code accompanying the paper: *Accelerating Molecular Dynamics with Hydrogen Mass Repartitioning to Prioritize Immunogenic Neoantigens for Cancer Vaccines through Peptide–HLA Binding Free-Energy Calculations*.

Twenty-one peptide-HLA class I complexes — nine neoantigen/wild-type pairs arising
from single-nucleotide variants, plus three frameshift neoantigens — were each
simulated in ten independent 500 ns replicas with CHARMM36m in NAMD. This
repository contains everything used to set up those simulations and to derive
every number and figure in the paper from them:

- **Binding free energies** by MM-PBSA with Poisson-Boltzmann solvation, corrected
  for entropy, computed over five  cumulative time windows.
- **Peptide structural analyses**: RMSD distributions and RMSF, both computed
  on peptide atoms only.
- **Interaction networks**: per-frame peptide-HLA contacts, hydrogen bonds, salt
  bridges and gas-phase MM interaction energies, plus C-terminal anchor unbinding
  detection.

The trajectories themselves are deposited separately on Zenodo;
see [docs/DATA.md](docs/DATA.md).

## Quick start

```bash
git clone <this repository> && cd public_neoantigens_paper

conda env create -f environment.yml
conda activate phla-mmpbsa
source $CONDA_PREFIX/amber.sh

# Force field files that carry their own licence — see forcefield/README.md
export CHARMM_FF_DIR="$PWD/forcefield"
export VMD_WAT_TOP=/path/to/vmd/lib/vmd/plugins/noarch/tcl/solvate1.7/wat.top

# Trajectories from Zenodo, unpacked so that <root>/<complex>/rep_<n>/ exists
export PHLA_DATA_ROOT="$PWD/data"

cd amber_free_energy
python run_amber_mmpbsa.py --complex pik3ca_e545k_neo_a_1101_strdplseitk \
                           --replicas 1 --window 500ns
```

Check the environment before starting a long run:

```bash
cd amber_free_energy
python -c "from scripts import config; config.validate_amberhome(); \
           config.validate_force_field(); config.validate_data_root(); \
           print('environment OK')"
```

## Repository layout

| Path | Contents |
|---|---|
| `amber_free_energy/` | MM-PBSA pipeline: orchestrator, `scripts/` modules, cross-complex summaries |
| `amber_free_energy/md_analyses/` | RMSD distributions, RMSF. Self-contained; shares only the trajectories |
| `amber_free_energy/interaction_network/` | Contact, hydrogen-bond, salt-bridge and MM-energy networks |
| `md_setup/` | System construction, NAMD configurations, trajectory concatenation and wrapping; `resolvate.py` rebuilds a solvent box around a dry frame |
| `forcefield/` | `par_water_ions.prm` plus instructions for the files that cannot be redistributed |
| `docs/` | Data placement, reproduction recipes, pipeline architecture |

## Documentation

| Document | Purpose |
|---|---|
| [docs/DATA.md](docs/DATA.md) | Zenodo archives, download, where to unpack, disk budget |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Each paper figure and table, and the command that regenerates it |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Phase-by-phase architecture and configuration reference |


## The twenty-one complexes

Directory names use the substitution's position in the full-length protein
(FLT3 D835Y, PIK3CA E545K). Figures and tables in the paper use the position
within the peptide (FLT3 D1Y, PIK3CA E11K); the mapping is held in
`md_analyses/scripts/mdconfig.py` as `CANONICAL_MUT` and is applied automatically.

| Directory | Gene / variant | Peptide | HLA |
|---|---|---|---|
| `akap6_e6k_neo_a_0201_wlidmkslv` | AKAP6 E6K, neo | WLIDMKSLV | A\*02:01 |
| `akap6_e6k_wt_a_0201_wlidmeslv` | AKAP6 E6K, wild type | WLIDMESLV | A\*02:01 |
| `astn1_p2l_neo_a_0201_klygldwael` | ASTN1 P2L, neo | KLYGLDWAEL | A\*02:01 |
| `astn1_p2l_wt_a_0201_kpygldwael` | ASTN1 P2L, wild type | KPYGLDWAEL | A\*02:01 |
| `flt3_d835y_neo_a_0201_yimsdsnyv` | FLT3 D835Y, neo | YIMSDSNYV | A\*02:01 |
| `flt3_d835y_wt_a_0201_dimsdsnyv` | FLT3, wild type | DIMSDSNYV | A\*02:01 |
| `pik3ca_e545k_neo_a_1101_strdplseitk` | PIK3CA E545K, neo | STRDPLSEITK | A\*11:01 |
| `pik3ca_e545k_wt_a_1101_strdplseite` | PIK3CA, wild type | STRDPLSEITE | A\*11:01 |
| `kras_g12c_neo_a_1101_vvvgacgvgk` | KRAS G12C, neo | VVVGACGVGK | A\*11:01 |
| `kras_g12c_wt_a_1101_vvvgaggvgk` | KRAS, wild type | VVVGAGGVGK | A\*11:01 |
| `kras_g12d_neo_c_0802_gadgvgksa` | KRAS G12D, neo | GADGVGKSA | C\*08:02 |
| `kras_g12d_wt_c_0802_gaggvgksa` | KRAS, wild type | GAGGVGKSA | C\*08:02 |
| `kras_g12v_neo_a_1101_vvgavgvgk` | KRAS G12V, neo | VVGAVGVGK | A\*11:01 |
| `kras_g12v_wt_a_1101_vvgaggvgk` | KRAS, wild type | VVGAGGVGK | A\*11:01 |
| `p53_r175h_neo_a_0201_hmtevvrhc` | TP53 R175H, neo | HMTEVVRHC | A\*02:01 |
| `p53_r175h_wt_a_0201_hmtevvrrc` | TP53, wild type | HMTEVVRRC | A\*02:01 |
| `pik3ca_h1047l_neo_a_0301_alhggwttk` | PIK3CA H1047L, neo | ALHGGWTTK | A\*03:01 |
| `pik3ca_h1047l_wt_a_0301_ahhggwttk` | PIK3CA, wild type | AHHGGWTTK | A\*03:01 |
| `apc_neo_a_0201_lqmdflvhpa` | APC frameshift | LQMDFLVHPA | A\*02:01 |
| `npm_neo_a_0201_claveevsl` | NPM1 frameshift | CLAVEEVSL | A\*02:01 |
| `tgfbrii_neo_a_0201_rlsscvpva` | TGFBR2 frameshift | RLSSCVPVA | A\*02:01 |

Each complex has ten replicas of 500 ns, saved every ~0.12002 ns (4166 frames).
Chains follow the CHARMM segment names `APRO` (HLA heavy chain, 276 residues),
`BPRO` (beta-2-microglobulin, 99 residues) and `CPRO` (peptide, 8-11 residues).

## Configuration through the environment

Nothing needs editing to run this elsewhere; all machine-dependent paths are read
from the environment, each with a repository-relative default.

| Variable | Default | Meaning |
|---|---|---|
| `PHLA_DATA_ROOT` | `<repo>/data` | Trajectory root, `<root>/<complex>/rep_<n>/` |
| `PHLA_TRAJECTORY_SOURCE` | `auto` | `auto`, `solvated` or `dry`; which form of the trajectory the analyses read |
| `PHLA_RESULTS_DIR` | `<repo>/amber_free_energy/results` | Where analysis outputs are written and read |
| `CHARMM_FF_DIR` | `<repo>/forcefield` | CHARMM36m parameter files |
| `VMD_WAT_TOP` | `$CHARMM_FF_DIR/wat.top` | VMD TIP3P topology |
| `AMBERHOME` | `~/anaconda3/envs/AmberTools25` | AmberTools installation |
| `OAR_EMAIL`, `OAR_QUEUE`, `OAR_CLUSTERS`, `OAR_WALLTIME_HOURS` | unset / `production` / unset / `12` | OAR submission, site-specific |

## Requirements

AmberTools 24 or later (`cpptraj`, `MMPBSA.py`, `sander`, ParmEd), Python 3.10+
with NumPy, SciPy and Matplotlib, and CHARMM36m plus VMD's `wat.top` as described
in [forcefield/README.md](forcefield/README.md). NAMD 3 and VMD are additionally
needed to regenerate the trajectories from scratch, and VMD alone to rebuild a solvent
box with `md_setup/resolvate.py` (see [docs/DATA.md](docs/DATA.md)).

`MMPBSA.py` is run with `use_sander=1`, which is mandatory for CHARMM-derived
topologies. MPI execution (`--use-mpi`) additionally requires `MMPBSA.py.MPI` and
an MPI runtime.

## Licence

Code is released under the NC-MIT licence ([LICENSE](LICENSE)). The deposited
trajectories are released under CC-BY-4.0. CHARMM36m and VMD files are not
redistributed here and remain under their own terms.
