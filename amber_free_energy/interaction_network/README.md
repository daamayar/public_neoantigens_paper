# `interaction_network/` — peptide↔HLA interaction-network analysis

A self-contained package that explore the particular case of the 
**PIK3CA E11K neoantigen**.

**Main deliverable: `REPORT.md`**, written by `scripts/make_report.py` from the
JSON produced by Phase 2. It is generated rather than stored: run the pipeline
below to build it.

It shares the simulation data with the MM-PBSA pipeline.
It reads the *clean* trajectories that `scripts/detect_unbinding.py` already
produced (`<complex>/rep_N/unbinding/repN_clean_500ns.nc`).

## Running it

```bash
conda activate AmberTools25 && source $CONDA_PREFIX/amber.sh
cd interaction_network

python -m scripts.compute_contacts --validate   # check MM energies vs MM-PBSA
python -m scripts.compute_contacts --workers 10 # Phase 1, ~7 min, writes ~1.3 GB
python -m scripts.analyze_network               # Phase 2, ~2 min
python -m scripts.make_figures                  # Phase 3
python -m scripts.make_report                   # regenerates REPORT.md
```

Phase 1 is the only expensive step and is idempotent. Everything downstream is
pure post-processing of `data/*.npz` and re-runs in minutes.

## Layout

```
interaction_network/
  REPORT.md                  # the deliverable
  scripts/
    inconfig.py              # paths, thresholds, conditioning windows, systems
    common.py                # topology index map, force-field extraction, masks
    compute_contacts.py      # Phase 1: per-frame networks (the expensive pass)
    analyze_network.py       # Phase 2: window statistics, graph metrics, tests
    make_figures.py          # Phase 3: three figures
    make_report.py           # Phase 4: REPORT.md, every number read from JSON
  data/                      # per-replica .npz + per-window .json + pdb/ cache
  figures/                   # fig1, fig2, fig4 .png
```

`REPORT.md` numbers its 7 figures and 15 tables in reading order, and opens
with a "Figures and tables" index. The `figN_` file-name prefixes are kept in
the same order, and `make_report.py` **raises** if a figure is referenced at a
position that disagrees with its file name.

## The one methodological point that matters

Every comparison is computed in four conditioning windows (`inconfig.WINDOWS`):

| window    | frames                                     | why                                                                       |
| --------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| `onset` | first 2 ns                                 | both peptides still hold the C-terminal carboxylate; precedes every event |
| `early` | first 25 ns                                | matched initial-state window                                              |
| `bound` | 50–500 ns, C-terminus-engaged frames only | removes the detached frames                                               |
| `all`   | 50–500 ns, everything                     | the ensemble MM-PBSA used                                                 |



