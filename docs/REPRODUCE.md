# Reproducing the results

Every number, table and figure in the paper is produced by one of the commands
below.

Before starting, set up the environment and the data as described in
[DATA.md](DATA.md), then confirm:

```bash
cd amber_free_energy
python -c "from scripts import config; config.validate_amberhome(); \
           config.validate_force_field(); config.validate_data_root(); \
           print('environment OK')"
```

## Dependency chain

The analyses are not independent. This is the order in which analyses are built:

```
<complex>_wrapped_500ns.dcd  +  <complex>_ionized.psf          [from Zenodo]
        │
        ├─ scripts.convert_topology     → solvated/complex/receptor/ligand .prmtop
        │        │
        │        ├─ scripts.prepare_trajectory  → rep{N}_dry_{window}.nc
        │        │        └─ scripts.run_mmpbsa → FINAL_RESULTS_MMPBSA.dat
        │        │                 └─ scripts.analyze_results
        │        │                          → ensemble_summary_{window}.json
        │        │                          → scripts.generate_reports  
        │        │                          → summarize_mmpbsa.py       (cross-complex)
        │        │
        │        └─ scripts.detect_unbinding    → unbinding/rep{N}_clean_500ns.nc
        │                 ├─ summarize_unbinding.py
        │                 └─ interaction_network/*   (reads the clean NetCDF only)
        │
        └─ md_analyses.build_cache      → md_analyses/cache/
                 └─ RMSD distributions, RMSF
```

`detect_unbinding` must run before the interaction-network package: the latter
never opens a DCD, it reads the dry receptor-autoimaged NetCDF that
`detect_unbinding` writes.

## Whole study, one command per stage

```bash
cd amber_free_energy

# 1. Binding free energies, all complexes, all five windows       
python run_amber_mmpbsa.py list_dirs.txt

# 2. Cross-complex tables for one window
python summarize_mmpbsa.py --window 500ns

# 3. C-terminal unbinding detection
for c in $(cat list_dirs.txt); do python -m scripts.detect_unbinding --complex "$c"; done
python summarize_unbinding.py

# 4. Peptide structural analyses
cd md_analyses && python run_md_analyses.py && cd ..

# 5. Interaction networks 
cd interaction_network
python -m scripts.compute_contacts --workers 10
python -m scripts.analyze_network
python -m scripts.make_figures
python -m scripts.make_report
```

`list_dirs.txt` is a plain list of complex directory names, one per line (blank
lines and `#` comments are ignored). A ready-to-use copy listing every complex
ships at [`amber_free_energy/list_dirs.txt`](../amber_free_energy/list_dirs.txt);
the commands above pick it up directly. Trim it to the subset you downloaded, or
pass a single name to run one complex.

## Figures and tables

Paths are relative to `amber_free_energy/`. `$W` is an analysis window
(`100ns`, `200ns`, `300ns`, `400ns`, `500ns`).

The binding free energies are reported at the **300 ns working window**, identified
as the shortest window on the 300/400/500 ns convergence plateau that still
preserves the experimental ranking. The window comparison itself needs all five.
The conformational, unbinding and interaction-network analyses are separate
readouts and use the full 50-500 ns trajectory rather than a free-energy window.

### Binding free energies

| Output | Command |
|---|---|
| `results/<complex>/ensemble_summary_$W.json` — dH, dG_QH | `python run_amber_mmpbsa.py --complex <c> --step analyze --window $W` |
| `results/<complex>/plots/energy_components_$W.png` | same |
| `results/<complex>/plots/convergence_{acf,blocks,cumulative}_$W.png` | same |
| `results/<complex>/plots/convergence_across_windows.png` — entropy-corrected dG against window length | `python run_amber_mmpbsa.py --complex <c> --step analyze` (needs all five windows) |
| `results/<complex>/reports/report.md` | `python run_amber_mmpbsa.py --complex <c> --step report` |
| `mmpbsa_summary_$W.{csv,md}` — all complexes, dH / dG_QH / ddG for each neo-wt pair | `python summarize_mmpbsa.py --window $W` |

The convergence figure plots the entropy-corrected binding free energy selected by
`config.CONVERGENCE_DG_METHOD`, quasi-harmonic by default, not the bare MM-PBSA
enthalpy. 

### Peptide structure

Run from `amber_free_energy/md_analyses/`. Outputs land in `figures/` and `data/`.

| Output | Command |
|---|---|
| `figures/rmsd/rms_distribution_overview.png` | `python run_md_analyses.py --step rmsd` |
| `figures/rmsd/rms_distribution_<pair>.png` — one per SNV pair, plus the three frameshift panels | same |
| `figures/rmsd/rms_distribution_10_neoantigens.png` | same |
| `figures/rmsf/rmsf_A_termini_8pos{,_panels}.png` | `python run_md_analyses.py --step rmsf` |
| `figures/rmsf/rmsf_B_core_5pos{,_panels}.png` | same |
| `data/rmsd_summary.csv`, `rmsd_pair_stats.csv`, `rmsf_*.csv` | the corresponding step |
| `REPORT.md` | `python run_md_analyses.py --step report` |

The first invocation builds a coordinate cache from the DCDs
(`--step cache`). Every later step reads the
cache, so re-running an analysis takes minutes. `--skip-cache` reuses it.

### Interaction networks

Run from `amber_free_energy/interaction_network/`, after `detect_unbinding`.

| Output | Command |
|---|---|
| `figures/fig1_contact_maps_bound.png` — contact occupancy, neo, wt, difference | `python -m scripts.make_figures --figure 1` |
| `figures/fig2_per_position_bound.png` — per-position occupancy, H-bonds, screened electrostatics | `--figure 2` |
| `figures/fig4_network_graph_bound.png` — bipartite residue network | `--figure 4` |
| `data/compare_{all,bound,early,onset}.json`, `cross_system_pomega.json` | `python -m scripts.analyze_network` |
| `REPORT.md` | `python -m scripts.make_report` |

`python -m scripts.make_figures` with no `--figure` builds all three.


### Unbinding

| Output | Command |
|---|---|
| `results/<complex>/unbinding_summary_*.json`, `unbinding_report_*.md` | `python -m scripts.detect_unbinding --complex <c>` |
| `results/<complex>/unbinding_rep{N}_timeseries_*.npz` | same |
| `results/<complex>/unbinding_plot_<complex>_*.png` | same |
| Cross-complex unbinding table | `python summarize_unbinding.py` |

## Verifying an installation before a long run

One replica of one complex exercises every phase and finishes in minutes:

```bash
cd amber_free_energy
python run_amber_mmpbsa.py --complex pik3ca_e545k_neo_a_1101_strdplseitk \
                           --replicas 1 --window 500ns
```

Compare `DELTA TOTAL` in
`$PHLA_DATA_ROOT/pik3ca_e545k_neo_a_1101_strdplseitk/rep_1/mmpbsa_500ns/FINAL_RESULTS_MMPBSA.dat`
against the per-replica value recorded in the deposited
`results/pik3ca_e545k_neo_a_1101_strdplseitk/ensemble_summary_500ns.json`. The two
must agree to the printed precision. A disagreement means the force field files or
the PB radii differ from those used in the study, and no later result will be
comparable.

`--generate-oar-scripts` writes per-replica job scripts for an OAR scheduler into
`amber_free_energy/oar_scripts/`. It is a convenience for the cluster this study
ran on and is not needed elsewhere; `--use-mpi --mpi-cores N` runs the same work
directly.
