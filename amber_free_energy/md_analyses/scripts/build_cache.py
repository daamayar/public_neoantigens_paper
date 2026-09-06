#!/usr/bin/env python3
"""
Phase 1 -- build the reduced trajectory cache.

For every complex x replica, read the wrapped production DCD (4166 frames,
0-500 ns), autoimage it, strip water and ions, and then strip again down to just
the atoms the structural analyses need:

    all C-alpha atoms (receptor + peptide)  +  every peptide heavy atom

That is ~430-460 atoms instead of ~6100, giving a ~23 MB NetCDF per replica
(~4 GB for all 170) that loads into numpy in 0.04 s. Every later analysis reads
only this cache, so the 5 GB DCDs are touched exactly once.

Why the FULL 0-500 ns and not the existing 50-500 ns pipeline trajectories?
Because the RMSD reference is "the first frame" (t ~ 0), which the existing
`rep{N}_dry_500ns_qha.nc` files do not contain -- they start at 50 ns. Caching
all 4166 frames means:
    cache[0]                  -> the t~0 RMSD reference
    cache[EQUIL_FRAMES:]      -> the 50-500 ns analysis ensemble
so one artifact serves both, and the equilibration window stays inspectable.

Receptor C-alpha atoms are kept even though the default fit is on the peptide:
they are needed for the wrapping / alignment QC (chain radius of gyration,
peptide-receptor centre-of-mass distance, PBC-break detection), and they make
the receptor-frame superposition available if it is ever needed.

Usage
-----
    # all 17 complexes x 10 replicas, 10 workers (default)
    python -m scripts.build_cache

    # a subset, more workers, force rebuild
    python -m scripts.build_cache --complex kras_g12v_neo_a_1101_vvgavgvgk --workers 4
    python -m scripts.build_cache --replicas 1,2,3 --force

Idempotent: a replica whose cache already exists with the right frame/atom count
is skipped.
"""
import os
import sys
import argparse
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.io import netcdf_file

from . import mdconfig as cfg
from . import common
def _cache_is_valid(path, n_atoms_expected):
    """Check an existing cache file has the expected shape.

    Args:
        path (str): path to a cached NetCDF.
        n_atoms_expected (int): atom count the file must have.

    Returns:
        bool: True if the file exists and has (TOTAL_FRAMES, n_atoms_expected, 3).
    """
    if not os.path.exists(path):
        return False
    try:
        with netcdf_file(path, "r", mmap=False) as f:
            shape = f.variables["coordinates"].shape
    except Exception:
        return False
    return shape == (cfg.TOTAL_FRAMES, n_atoms_expected, 3)


def build_one(dirname, rep, n_res, n_atoms_expected, force=False):
    """Build the reduced cache for a single complex/replica.

    Args:
        dirname (str): complex directory name.
        rep (int): replica number (1-based).
        n_res (int): total residues in the complex (peptide = 376..n_res).
        n_atoms_expected (int): atoms the output must contain.
        force (bool): rebuild even if a valid cache already exists.

    Returns:
        dict: {"dirname", "rep", "status", "seconds", "message"} where status is
            one of "built", "skipped", "failed".
    """
    t0 = time.time()
    out_nc = common.cache_path(dirname, rep)
    common.ensure_dir(os.path.dirname(out_nc))

    if not force and _cache_is_valid(out_nc, n_atoms_expected):
        return dict(dirname=dirname, rep=rep, status="skipped",
                    seconds=0.0, message="cache already valid")

    try:
        dcd, traj_form = common.find_dcd(dirname, rep)
        rep_dir = os.path.join(cfg.BASE_DIR, dirname, f"rep_{rep}")
        topo_rel = cfg.SOLVATED_PRMTOP if traj_form == "solvated" else cfg.COMPLEX_PRMTOP_REL
        topo = os.path.join(rep_dir, topo_rel)
        if not os.path.exists(topo):
            raise FileNotFoundError(f"missing {topo}")

        # keep: every CA, plus every non-hydrogen atom of the peptide (376..n_res)
        keep_mask = f"!(@CA|(:376-{n_res}&!@H*))"

        lines = [f"parm {topo}", f"trajin {dcd} 1 {cfg.TOTAL_FRAMES} 1"]
        if traj_form == "solvated":
            # Reimage while the box is still present, then discard solvent. The dry
            # trajectory is protein-only and already whole, so neither step applies.
            lines += ["autoimage", f"strip {cfg.STRIP_MASK_CPPTRAJ} nobox"]
        lines += [f"strip {keep_mask}", f"trajout {out_nc} netcdf", "run", "quit"]
        script = "\n".join(lines) + "\n"

        with tempfile.NamedTemporaryFile("w", suffix=".cpptraj", delete=False) as fh:
            fh.write(script)
            script_path = fh.name
        try:
            res = subprocess.run([cfg.CPPTRAJ, "-i", script_path],
                                 capture_output=True, text=True)
        finally:
            os.unlink(script_path)

        if res.returncode != 0:
            raise RuntimeError(f"cpptraj rc={res.returncode}: {res.stderr[-400:]}")

        if not _cache_is_valid(out_nc, n_atoms_expected):
            with netcdf_file(out_nc, "r", mmap=False) as f:
                shape = f.variables["coordinates"].shape
            raise RuntimeError(
                f"unexpected cache shape {shape}, "
                f"expected ({cfg.TOTAL_FRAMES}, {n_atoms_expected}, 3)")

        return dict(dirname=dirname, rep=rep, status="built",
                    seconds=time.time() - t0, message="")

    except Exception as exc:                                     # noqa: BLE001
        if os.path.exists(out_nc):
            os.unlink(out_nc)                                    # never leave a partial cache
        return dict(dirname=dirname, rep=rep, status="failed",
                    seconds=time.time() - t0, message=str(exc))


def main():
    ap = argparse.ArgumentParser(description="Build the reduced trajectory cache.")
    ap.add_argument("--complex", action="append", default=None,
                    help="complex directory name (repeatable; default: all 17)")
    ap.add_argument("--replicas", default=None,
                    help="comma-separated replica numbers (default: 1-10)")
    ap.add_argument("--workers", type=int, default=10,
                    help="parallel cpptraj jobs (default 10; the job is I/O-bound "
                         "on the shared filesystem, so more is not faster)")
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()

    systems = ([cfg.get_system(c) for c in args.complex] if args.complex else cfg.SYSTEMS)
    reps = ([int(x) for x in args.replicas.split(",")] if args.replicas
            else list(range(1, cfg.NUM_REPLICAS + 1)))

    # Resolve topology facts once per complex (parmed load is the slow part).
    print(f"Resolving topologies for {len(systems)} complexes ...")
    meta = {}
    for s in systems:
        imap = common.build_index_map(s["dirname"])
        meta[s["dirname"]] = imap
        print(f"  {s['dirname']:38s} peptide={imap['seq']:11s} "
              f"({imap['pep_len']}-mer)  cache atoms={imap['n_cache_atoms']}")

    jobs = [(s["dirname"], r, meta[s["dirname"]]["n_res"],
             meta[s["dirname"]]["n_cache_atoms"], args.force)
            for s in systems for r in reps]

    print(f"\nBuilding {len(jobs)} replica caches with {args.workers} workers ...")
    t0 = time.time()
    done = built = skipped = failed = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r["status"] == "built":
                built += 1
            elif r["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(r)
                print(f"  [FAIL] {r['dirname']} rep_{r['rep']}: {r['message']}")
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  {done}/{len(jobs)}  built={built} skipped={skipped} "
                      f"failed={failed}  elapsed={el/60:.1f}m  eta={eta/60:.1f}m")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min: "
          f"built={built} skipped={skipped} failed={failed}")
    if failures:
        print("\nFAILURES:")
        for r in failures:
            print(f"  {r['dirname']} rep_{r['rep']}: {r['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
