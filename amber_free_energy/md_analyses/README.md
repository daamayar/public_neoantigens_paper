# md_analyses - peptide structural analyses (RMSD distribution, RMSF)

Structural analysis of the peptide in the peptide-HLA neoantigen complexes
(10 replicas x 500 ns each). **Every reported value is computed on the peptide
only** - the HLA heavy chain and beta-2-microglobulin are excluded.

The main deliverable is **`REPORT.md`**, assembled by `scripts/make_report.py`
from the CSV and NPZ written by the analysis steps. It is generated rather than
stored: run the pipeline below to build it.

## Quick start

```bash
conda activate AmberTools25
source $CONDA_PREFIX/amber.sh

python run_md_analyses.py                 # everything (cache -> RMSD -> RMSF -> report)
python run_md_analyses.py --skip-cache    # cache already built
python run_md_analyses.py --step rmsd     # a single step
```

## What each step does

| Step | Module | Output |
|---|---|---|
| `cache` | `scripts/build_cache.py` | Reduces the 170 x 5 GB DCDs to a ~4 GB coordinate cache |
| `rmsd` | `scripts/analyze_rmsd.py` | RMSD distributions (11 panels) |
| `rmsf` | `scripts/analyze_rmsf.py` | RMSF, both variants (all 17 peptides) |
| `report` | `scripts/make_report.py` | Assembles `REPORT.md` |

## Key conventions

- **Analysis window**: frames 418-4166 = **50-500 ns**. The first 50 ns is discarded
  as equilibration, matching this project's MM-PBSA pipeline.
- **RMSD reference**: frame 1 of each replica's own production DCD (t ~ 0), i.e.
  "the first frame". The cache stores all 4166 frames, so `cache[0]` *is* that
  reference and `cache[417:]` is the analysis ensemble.
- **Superposition**: the peptide's own C-alpha (`FIT_MODE = "peptide"` in
  `scripts/mdconfig.py`). Metrics therefore report the peptide's **internal**
  conformational change; rigid-body motion in the groove is removed by the fit.
- **CHARMM36 C-terminus**: the last residue has `OT1`/`OT2` and no `O`. They are
  equivalent by resonance and swap on a carboxylate flip, so the backbone "O" of
  the C-terminal residue is their **centroid** (flip-invariant).
