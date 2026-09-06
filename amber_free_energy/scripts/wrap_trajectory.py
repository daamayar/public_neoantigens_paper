#!/usr/bin/env python3
"""
Wrap a raw production trajectory: reassemble molecules across the periodic
boundary and superpose every frame on the protein backbone of frame 0.

This reproduces, with cpptraj, what `md_setup/wrapping.tcl` does under VMD, so the
step that turns a `catdcd` concatenation into the analysed trajectory is part of
this repository rather than an external prerequisite.

    wrapping.tcl                                  cpptraj equivalent
    pbc join res -ref "name CA"                   autoimage anchor <receptor>
    pbc unwrap -sel protein                         "
    pbc wrap -centersel protein -center com ...     "
    measure fit on "protein and backbone"         rms first @N,CA,C,O

`autoimage` performs join, unwrap and wrap in one pass: it images each molecule as a
unit so none is split across the boundary, and centres the cell on the anchor mask.

This script writes a full solvated trajectory, roughly the size of its input (~5 GB per
replica), so it is not a cheap step.

Usage
-----
    python -m scripts.wrap_trajectory <complex_name> <rep_num>
    python -m scripts.wrap_trajectory <complex_name> <rep_num> --output custom.dcd
    python -m scripts.wrap_trajectory <complex_name> <rep_num> --dry-run
"""
import argparse
import os
import subprocess
import sys

from . import config
from . import utils


def wrapped_output_path(complex_name, rep_num):
    """Canonical wrapped trajectory path for a replica."""
    rep_dir = utils.get_replica_dir(complex_name, rep_num)
    return os.path.join(rep_dir, f"{complex_name}_wrapped_500ns.dcd")


def build_cpptraj_input(solvated_prmtop, in_dcd, out_dcd, receptor_range):
    """Return the cpptraj script that images and superposes the trajectory."""
    r_start, r_end = receptor_range
    return "\n".join([
        f"parm {solvated_prmtop}",
        f"trajin {in_dcd}",
        # Anchor on the receptor so the complex is centred on the HLA rather than on
        # whichever molecule happens to be first; a peptide that leaves the groove is
        # then not folded back across the boundary.
        f"autoimage anchor :{r_start}-{r_end}",
        # Superpose on the protein backbone of frame 0, as wrapping.tcl does. This is
        # cosmetic for the analyses, which all re-superpose on their own fit group,
        # but it reproduces the deposited trajectory.
        "rms first :%d-%d@N,CA,C,O" % (r_start, r_end),
        f"trajout {out_dcd} dcd",
        "run",
        "quit",
    ]) + "\n"


def wrap_trajectory(complex_name, rep_num, output=None, dry_run=False, force=False):
    """Image and superpose one replica's production trajectory.

    Returns the output path, or None when nothing was done.
    """
    in_dcd, form = utils.find_trajectory(complex_name, rep_num)
    if form == "dry":
        raise ValueError(
            f"{complex_name} rep_{rep_num}: the available trajectory is the dry "
            f"(protein-only) one, which is already imaged and carries no box. There is "
            f"nothing to wrap.")

    out_dcd = output or wrapped_output_path(complex_name, rep_num)
    if os.path.abspath(out_dcd) == os.path.abspath(in_dcd):
        print(f"  {complex_name} rep_{rep_num}: input is already the wrapped "
              f"trajectory; nothing to do.")
        return in_dcd
    if os.path.exists(out_dcd) and not force:
        print(f"  {complex_name} rep_{rep_num}: {os.path.basename(out_dcd)} exists "
              f"(use --force to rebuild).")
        return out_dcd

    workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, config.REFERENCE_WINDOW)
    solvated_prmtop = os.path.join(workdir, "solvated.prmtop")
    if not os.path.exists(solvated_prmtop):
        raise FileNotFoundError(
            f"solvated.prmtop not found in {workdir}. Run topology conversion first: "
            f"python -m scripts.convert_topology {complex_name} {rep_num}")

    meta_path = os.path.join(workdir, "topology_metadata.json")
    receptor_range = (1, 375)
    if os.path.exists(meta_path):
        meta = utils.load_metadata(meta_path)
        rng = meta.get("receptor_residue_range")
        if rng:
            r = str(rng).replace(":", "").split("-")
            receptor_range = (int(r[0]), int(r[1]))

    script = build_cpptraj_input(solvated_prmtop, in_dcd, out_dcd, receptor_range)
    print(f"  input : {os.path.basename(in_dcd)} ({form})")
    print(f"  output: {os.path.basename(out_dcd)}")
    print(f"  anchor: :{receptor_range[0]}-{receptor_range[1]} (receptor)")
    if dry_run:
        print("  --dry-run, cpptraj script:")
        for line in script.strip().split("\n"):
            print(f"    {line}")
        return None

    script_path = os.path.join(workdir, f"wrap_rep{rep_num}.cpptraj")
    log_path = os.path.join(workdir, f"wrap_rep{rep_num}.log")
    with open(script_path, "w") as fh:
        fh.write(script)

    cpptraj = os.path.join(config.AMBERHOME, "bin", "cpptraj")
    with open(log_path, "w") as log:
        res = subprocess.run([cpptraj, "-i", script_path], stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not os.path.exists(out_dcd):
        raise RuntimeError(f"cpptraj failed; see {log_path}")
    print(f"  wrote {out_dcd} ({os.path.getsize(out_dcd) / 1e9:.2f} GB)")
    return out_dcd


def main():
    ap = argparse.ArgumentParser(
        description="Image and superpose a production trajectory (cpptraj).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("complex_name", help="Complex directory name")
    ap.add_argument("rep_num", type=int, help="Replica number (1-based)")
    ap.add_argument("--output", default=None, help="Output DCD (default: <complex>_wrapped_500ns.dcd)")
    ap.add_argument("--dry-run", action="store_true", help="Print the cpptraj script and stop")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the output exists")
    args = ap.parse_args()
    wrap_trajectory(args.complex_name, args.rep_num, args.output, args.dry_run, args.force)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)
