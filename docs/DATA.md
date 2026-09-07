# Simulation data

The production trajectories are distributed through Zenodo. They are not in this
repository.

**The deposited trajectories are dry**: water and ions were stripped, leaving protein
atoms only. This reduces the deposit from ~840 GB to **48.8 GB**, which fits 
Zenodo allocation. 

`md_setup/resolvate.py` rebuilds an explicit solvent box around any frame, for uses
that need one (restarting a simulation, visualising a solvated system). Read
[Rebuilding the solvent](#rebuilding-the-solvent) before relying on it: the solvent is
**rebuilt, not recovered**.

It rebuilds *single frames*, and cannot supply a solvated trajectory to the pipeline: a
DCD or NetCDF trajectory needs a fixed atom count, and solvating frames independently
gives a different water count each time. The pipeline reads the dry trajectory directly
instead — see [Choosing the trajectory form](#choosing-the-trajectory-form).

> DOI: [10.5281/zenodo.22226388](https://doi.org/10.5281/zenodo.22226388)

## What is deposited

| Record | Contents |
|---|---|
| Documentation | `README.md` |
| Topologies and inputs | CHARMM topologies, reference coordinates, box vectors, NAMD configurations |
| One per complex (17) | The ten dry production trajectories for that complex |


## Layout after unpacking

Both archives expand into the same tree, so unpack them into the same root:

```
<PHLA_DATA_ROOT>/
└── <complex>/                                  e.g. pik3ca_e545k_neo_a_1101_strdplseitk/
    ├── rep_1/
    │   ├── <complex>_ionized.psf               10 MB   CHARMM36m solvated topology
    │   ├── <complex>_ionized.pdb                8 MB   reference coordinates
    │   ├── <complex>_ionized_bb.0.xsc           1 KB   final box vectors
    │   ├── <complex>_production_HMR.conf        4 KB   NAMD production settings as run
    │   └── cleanDCD/
    │       ├── <complex>_ionized.clean.psf    1.5 MB   protein-only topology
    │       ├── <complex>_wrapped_500ns.clean.dcd
    │       │                                  292 MB   dry trajectory, 4166 frames
    │       └── clean.ind                       30 KB   indices kept from the solvated PSF
    └── rep_2/ … rep_10/
    
```

The dry DCD retains its **per-frame unit cell**, so the box volume at every frame is
preserved even though the solvent is not. `clean.ind` maps the kept atoms back onto the
original solvated PSF ordering.



## Retrieving and placing it

```bash
cd public_neoantigens_paper
export PHLA_DATA_ROOT="$PWD/data"          # or any location with enough space
mkdir -p "$PHLA_DATA_ROOT"

# From the record for the complex you want
unzip tgfbrii_neo_a_0201_rlsscvpva.zip -d "$PHLA_DATA_ROOT"


```

Then confirm the pipeline sees it:

```bash
cd amber_free_energy
python -c "from scripts import config; config.validate_data_root(); print(config.BASE_DIR)"
```

`PHLA_DATA_ROOT` is the only thing that has to change to run from a different
location; no source file needs editing.

## What each analysis consumes

| Analysis | Entry point | Reads |
|---|---|---|
| MM-PBSA and entropy | `run_amber_mmpbsa.py` | DCD, PSF, force field |
| RMSD distributions, RMSF | `md_analyses/run_md_analyses.py` | DCD, `solvated.prmtop` |
| Unbinding detection | `python -m scripts.detect_unbinding` | DCD, `solvated.prmtop` |
| Interaction networks | `interaction_network/scripts/compute_contacts.py` | `unbinding/rep{N}_clean_500ns.nc` |
| Re-running the simulations | `md_setup/pipeline_run_MD_HLAs.py` | PSF, PDB, `*_production_HMR.conf` |
| Rebuilding a solvent box | `md_setup/resolvate.py` | dry DCD + clean PSF (see below) |

The interaction-network package does not read the DCD directly. It reuses the
dry, receptor-autoimaged NetCDF that `detect_unbinding.py` writes to
`rep_<n>/unbinding/`, so that step must be run first. The dependency chain is
given in full in [REPRODUCE.md](REPRODUCE.md).

### Trajectory wrapping

`scripts/wrap_trajectory.py` materialises the wrapped trajectory on disk when it is
wanted as a file — to reproduce the preprocessing exactly, or to hand a wrapped
trajectory to another tool. It reproduces `wrapping.tcl` in cpptraj:

| `wrapping.tcl` (VMD) | cpptraj |
|---|---|
| `pbc join` / `pbc unwrap` / `pbc wrap` | `autoimage anchor :1-375` |
| `measure fit` on protein backbone | `rms first :1-375@N,CA,C,O` |

```bash
python -m scripts.wrap_trajectory <complex> <rep>            # writes *_wrapped_500ns.dcd
```

It writes a full solvated trajectory (~5 GB per replica), so it is not cheap, and it
needs topology conversion to have run first.

### Choosing the trajectory form

`PHLA_TRAJECTORY_SOURCE` selects which form the analyses read:

| Value | Meaning |
|---|---|
| `auto` (default) | the solvated DCD if present, otherwise the dry one |
| `solvated` | the original `*_wrapped_500ns.dcd`, reimaged and stripped on the fly |
| `dry` | the deposited `cleanDCD/*.clean.dcd`, used directly |

```bash
export PHLA_TRAJECTORY_SOURCE=dry     # working from the Zenodo deposit
```

The dry trajectory is read with `mmpbsa_500ns/complex.prmtop`: it holds exactly the
protein atoms of the solvated PSF, in the same order, so no remapping is needed. 
Being protein-only and already whole, it skips the `autoimage` and `strip` steps the 
solvated form requires.

**The two forms give the same results.** Every analysis measures inter-atomic
quantities after discarding solvent, and the dry trajectory differs from the stripped
solvated one only by a rigid-body superposition, which leaves internal geometry
untouched.

## Rebuilding the solvent

`md_setup/resolvate.py` puts a frame of a dry trajectory back into an explicit water
box with neutralising ions, following the protocol that built the original systems
(VMD `solvate` + `autoionize -neutralize`, CHARMM36m/TIP3P).

```bash
export VMD_BIN=/path/to/vmd          # only if vmd is not on PATH

# Inspect a frame: how many frames, and the cell it was recorded at
python md_setup/resolvate.py --data-root "$PHLA_DATA_ROOT" \
    --complex pik3ca_e545k_neo_a_1101_strdplseitk --rep 1 --frame 0 --info

# Rebuild that frame
python md_setup/resolvate.py --data-root "$PHLA_DATA_ROOT" \
    --complex pik3ca_e545k_neo_a_1101_strdplseitk --rep 1 --frame 0 \
    --out-prefix rebuilt/rep1_frame0
```

It writes `<prefix>.psf`, `.pdb`, `.xsc` (a NAMD-ready cell) and `.resolvate.json`
recording the provenance and the verification numbers.


## Working from a partial download

The deposit is one record per complex, so a partial download is the normal case. What
it supports:

| With | You can run |
|---|---|
| one replica of one complex | topology conversion, trajectory preparation, MM-PBSA, entropy, unbinding detection |
| all ten replicas of one complex | the above plus per-complex ensemble statistics, convergence, and the per-complex report |
| the two HLA-A\*11:01 systems (`pik3ca_e545k_neo/wt`) | the interaction-network analysis |
| all seventeen complexes | the cross-complex summary, the RMSD/RMSF figures, and every table in the paper |

The steps that pool across complexes need all seventeen and fail with the missing path
named if any is absent: `summarize_mmpbsa.py`, `summarize_unbinding.py` and the
`md_analyses` `rmsd`/`rmsf` steps build panels covering the whole panel of systems.
`md_analyses` `cache` accepts `--complex` and `--replicas`, so the cache can be built
for whatever subset is on disk.

**Topology conversion must run before anything else**, in either trajectory form: it
produces the `mmpbsa_500ns/*.prmtop` files that the deposit does not carry, and the dry
trajectory is read with `complex.prmtop`. It needs the CHARMM36m force field
(`forcefield/README.md`); `run_amber_mmpbsa.py` runs it  automatically, but a
bare `--step trajectory` or `--step mmpbsa` on a fresh deposit will stop and tell you to
run it first.

