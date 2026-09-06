# Pipeline architecture

Three packages share the trajectories.

| Package | Purpose | Entry point |
|---|---|---|
| `amber_free_energy/` | MM-PBSA binding free energies and entropy | `run_amber_mmpbsa.py` |
| `amber_free_energy/md_analyses/` | Peptide RMSD distributions, RMSF | `run_md_analyses.py` |
| `amber_free_energy/interaction_network/` | Contact, hydrogen-bond and MM-energy networks | `scripts/compute_contacts.py` |

## The five MM-PBSA phases

Each phase checks for its own output and skips work already done, so an
interrupted run is resumed by re-issuing the same command.

**0. Wrapping (optional)** — `scripts/wrap_trajectory.py`

Images a raw `catdcd` concatenation and superposes it on the protein backbone of
frame 0. Not required by the analyses: phase 2 runs `autoimage` on whatever it 
is given, so an unwrapped trajectory is imaged in passing. Run it only to put 
`*_wrapped_500ns.dcd` on disk.

**1. Topology conversion** — `scripts/convert_topology.py`

CHARMM PSF and PDB to AMBER prmtop and inpcrd through ParmEd's `chamber`.
Produces four topologies per replica: solvated, dry complex, receptor (HLA) and
ligand (peptide). Runs per replica, because each replica was solvated and
neutralised independently and its PSF therefore differs.

**2. Trajectory preparation** — `scripts/prepare_trajectory.py`

cpptraj strips water and ions and selects frames for each of the five analysis
windows, writing NetCDF. Always `autoimage` before `strip … nobox`.

**3. MM-PBSA** — `scripts/run_mmpbsa.py`

`MMPBSA.py` or `MMPBSA.py.MPI` for the Poisson-Boltzmann enthalpy, single
trajectory protocol. `use_sander=1` throughout.

**4. Analysis** — `scripts/analyze_results.py`

Parses the per-replica output, aggregates ensemble statistics, and computes the
 entropy corrections. Also runs the convergence diagnostics: autocorrelation, 
 block averages, cumulative means, and the across-window comparison.

**5. Reports** — `scripts/generate_reports.py`

A sixth, standalone step, `summarize_mmpbsa.py`, reads the per-complex JSON and
produces the cross-complex tables with ddG for each neoantigen/wild-type pair.

## Configuration

`scripts/config.py`: directory roots, force field
file list and load order, file-naming patterns, segment definitions
(`APRO`/`BPRO` receptor, `CPRO` ligand), trajectory geometry, the five analysis
windows, the PB parameters, the entropy parameters, and the OAR settings.
`md_analyses/scripts/mdconfig.py` and `interaction_network/scripts/inconfig.py`
play the same role for their packages.

All machine-dependent paths come from the environment with repository-relative
defaults, listed in the top-level README. The three packages resolve
`PHLA_DATA_ROOT` identically, so they always read the same trajectories.
