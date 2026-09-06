#!/usr/bin/env python3
"""
Rebuild a solvated peptide-HLA system from a dry (water-stripped) trajectory frame.

The deposited trajectories contain protein atoms only. This script puts a frame of
one back into an explicit water box and neutralising ions, reproducing the protocol
that built the original systems (VMD `solvate` + `autoionize -neutralize`,
CHARMM36m/TIP3P).

What is and is not reconstructed
--------------------------------
The solvent is REBUILT, not recovered. Stripping deleted ~94% of the atoms
(108,538 -> 6,143 for a 9-mer complex), and those coordinates are gone. The result
is a fresh, physically valid solvent configuration at the recorded box volume, not
the water configuration of the original run. It is therefore suitable for

  * re-running MM-PBSA (which strips solvent as its first step anyway),
  * any implicit-solvent or protein-only quantity,
  * restarting a simulation from that frame, after re-equilibrating the solvent,

and unsuitable for anything that depends on individual water positions: water-mediated
hydrogen bonds, residence times, hydration-site analysis, or reproducing the original
per-frame energies.

The dry trajectory was additionally superposed on the protein Calpha of its own first
frame, and the per-frame rigid-body transforms were not stored. The protein's
orientation relative to the periodic box is therefore not the original one. Internal
geometry is untouched, so the box is rebuilt around the protein as it sits in the
requested frame.

Box dimensions are read from the per-frame unit-cell record that the dry DCD still
carries, so each frame is rebuilt at the volume the NPT run actually had at that
point, rather than at the padded volume of the initial build.

Usage
-----
    # By explicit files
    python resolvate.py --clean-psf X_ionized.clean.psf \
                        --clean-dcd X_wrapped_500ns.clean.dcd \
                        --frame 0 --out-prefix rebuilt/X_frame0

    # By replica, against a deposit laid out as <root>/<complex>/rep_<n>/cleanDCD/
    python resolvate.py --data-root ./ --complex p53_r175h_wt_a_0201_hmtevvrrc \
                        --rep 9 --frame 0

    # Inspect the frame without building anything
    python resolvate.py --clean-dcd X.clean.dcd --frame 0 --info

Outputs `<prefix>.psf`, `<prefix>.pdb`, `<prefix>.xsc` (NAMD-ready cell) and
`<prefix>.resolvate.json` (provenance and verification numbers).

Requires VMD (>= 1.9.3) with the solvate, autoionize and psfgen plugins. Set VMD_BIN
if `vmd` is not on PATH.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

# CHARMM DCD unit-cell record order: a, gamma, b, beta, alpha, c
_CELL_A, _CELL_B, _CELL_C = 0, 2, 5

# TIP3P number density at 310 K / 1 atm, molecules per nm^3.
BULK_TIP3P_PER_NM3 = 33.42
# Protein partial specific volume, cm^3/g. Used only to estimate the volume the
# solute denies to water when computing the target water count.
PARTIAL_SPECIFIC_VOLUME = 0.73


def expected_waters(cell_a, cell_b, cell_c, solute_mass_da):
    """Waters that a box of this volume should hold at bulk density.

    VMD's solvate fills from a pre-equilibrated lattice that is a few per cent
    under-dense, so the count it returns for a given box is not the equilibrium
    one. Targeting this number instead of the box keeps the rebuilt system at the
    right density; the cell then relaxes back to the recorded volume during NPT,
    which doubles as a check that the rebuild is sound.
    """
    v_cell_nm3 = cell_a * cell_b * cell_c / 1000.0
    v_solute_nm3 = solute_mass_da * PARTIAL_SPECIFIC_VOLUME / 602.2
    return int(round(BULK_TIP3P_PER_NM3 * (v_cell_nm3 - v_solute_nm3)))


def psf_atom_mass(psf_path):
    """Total mass in Da of a PSF, read from the mass column of the atom records."""
    with open(psf_path, errors="replace") as fh:
        lines = fh.read().split("\n")
    i = next(k for k, l in enumerate(lines) if "!NATOM" in l)
    n = int(lines[i].split()[0])
    total = 0.0
    for line in lines[i + 1:i + 1 + n]:
        f = line.split()
        if len(f) >= 8:
            total += float(f[7])
    return total


# ---------------------------------------------------------------------------
# DCD reading (no third-party dependency)
# ---------------------------------------------------------------------------

def read_dcd_header(path):
    """Parse a CHARMM/NAMD DCD header.

    Returns:
        dict with n_frames, n_atoms, has_cell, n_fixed, and the byte offset at
        which the first frame's records begin.
    """
    with open(path, "rb") as f:
        blk = struct.unpack("<i", f.read(4))[0]
        if blk != 84:
            raise ValueError(f"{path}: not a little-endian DCD (first block {blk}, expected 84)")
        magic = f.read(4)
        if magic != b"CORD":
            raise ValueError(f"{path}: missing CORD magic, got {magic!r}")
        icntrl = struct.unpack("<20i", f.read(80))
        f.read(4)                                   # trailing record marker

        n_title = struct.unpack("<i", f.read(4))[0]  # title record
        f.read(n_title)
        f.read(4)

        f.read(4)
        n_atoms = struct.unpack("<i", f.read(4))[0]
        f.read(4)

        return {
            "n_frames": icntrl[0],
            "n_atoms": n_atoms,
            "has_cell": icntrl[10] == 1,
            "n_fixed": icntrl[8],
            "first_frame_offset": f.tell(),
        }


def read_frame_cell(path, frame):
    """Return (a, b, c) in Angstrom for `frame` (0-based) of a DCD.

    Raises if the file carries no unit-cell record, since the caller then has no
    basis for choosing a box and must not silently invent one.
    """
    h = read_dcd_header(path)
    if not h["has_cell"]:
        raise ValueError(
            f"{path} carries no unit-cell record, so the box volume of each frame is "
            f"unknown. Supply --box explicitly.")
    if h["n_fixed"]:
        raise ValueError(f"{path}: fixed-atom DCDs are not supported (n_fixed={h['n_fixed']})")
    if not 0 <= frame < h["n_frames"]:
        raise ValueError(f"frame {frame} out of range (file has {h['n_frames']} frames)")

    cell_bytes = 4 + 48 + 4
    coord_bytes = 3 * (4 + 4 * h["n_atoms"] + 4)
    with open(path, "rb") as f:
        f.seek(h["first_frame_offset"] + frame * (cell_bytes + coord_bytes))
        n = struct.unpack("<i", f.read(4))[0]
        if n != 48:
            raise ValueError(f"{path}: unexpected unit-cell record size {n} at frame {frame}")
        cell = struct.unpack("<6d", f.read(48))
    return cell[_CELL_A], cell[_CELL_B], cell[_CELL_C]


# ---------------------------------------------------------------------------
# VMD driver
# ---------------------------------------------------------------------------

def vmd_binary():
    exe = os.environ.get("VMD_BIN") or shutil.which("vmd")
    if not exe or not os.path.exists(exe):
        raise FileNotFoundError(
            "VMD not found. Install VMD (>= 1.9.3) and put it on PATH, or set VMD_BIN "
            "to the executable.")
    return exe


def run_vmd(script_text, log_path, timeout=3600):
    """Execute a Tcl script under `vmd -dispdev text`, returning its combined output."""
    exe = vmd_binary()
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as fh:
        fh.write(script_text)
        script_path = fh.name
    try:
        proc = subprocess.run([exe, "-dispdev", "text", "-e", script_path],
                              capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        with open(log_path, "w") as fh:
            fh.write(out)
        return out
    finally:
        os.unlink(script_path)


_TCL = r"""
package require solvate
package require autoionize
package require psfgen

mol load psf {psf} dcd {dcd}
set nf [molinfo top get numframes]
if {{ {frame} >= $nf }} {{
    puts "RESOLVATE_ERROR frame {frame} >= numframes $nf"
    quit
}}

# Write the requested frame as the solute for solvate.
set sel [atomselect top all frame {frame}]
$sel writepdb {tmp_pdb}
$sel writepsf {tmp_psf}

# Report the solute extent so the caller can confirm it fits the recorded cell.
set mm [measure minmax $sel]
puts "RESOLVATE_SOLUTE_MINMAX $mm"
set cen [measure center $sel]
puts "RESOLVATE_SOLUTE_CENTER $cen"
$sel delete
mol delete top

# Box of the recorded volume, centred on the solute.
solvate {tmp_psf} {tmp_pdb} -minmax {{ {{ {xmin} {ymin} {zmin} }} {{ {xmax} {ymax} {zmax} }} }} \
    -o {solv_prefix}

autoionize -psf {solv_prefix}.psf -pdb {solv_prefix}.pdb {ion_args} -o {out_prefix}

# Verification numbers, read back from the finished system.
mol load psf {out_prefix}.psf pdb {out_prefix}.pdb
set all [atomselect top all]
puts "RESOLVATE_NATOM [$all num]"
puts "RESOLVATE_CHARGE [format %.6f [vecsum [$all get charge]]]"
set w [atomselect top "water and name OH2"]
puts "RESOLVATE_NWATER [$w num]"
set i [atomselect top "ions"]
puts "RESOLVATE_NIONS [$i num]"
set p [atomselect top "protein"]
puts "RESOLVATE_NPROT [$p num]"
set fmm [measure minmax $all]
puts "RESOLVATE_FINAL_MINMAX $fmm"
$all delete
$w delete
$i delete
$p delete
puts "RESOLVATE_DONE"
quit
"""


def _tag(out, tag, cast=str):
    for line in out.splitlines():
        if line.startswith(tag + " "):
            val = line[len(tag) + 1:].strip()
            return cast(val)
    return None


def resolvate(clean_psf, clean_dcd, frame, out_prefix, box=None, ion_args=None,
              keep_intermediates=False, target_waters=None, exact_box=False,
              max_passes=4, tol=0.01):
    """Rebuild one solvated frame. Returns a dict of provenance and verification data.

    By default the solvation box is scaled so that the water count matches bulk
    density for the recorded cell volume (see `expected_waters`). Pass
    exact_box=True to fill the recorded cell verbatim instead, which starts the
    system a few per cent under-dense.
    """
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_dir, exist_ok=True)

    if box is None:
        a, b, c = read_frame_cell(clean_dcd, frame)
    else:
        a, b, c = box

    hdr = read_dcd_header(clean_dcd)
    solute_mass = psf_atom_mass(clean_psf)
    if target_waters is None:
        target_waters = expected_waters(a, b, c, solute_mass)

    work = tempfile.mkdtemp(prefix="resolvate_", dir=out_dir)
    try:
        tmp_pdb = os.path.join(work, "solute.pdb")
        tmp_psf = os.path.join(work, "solute.psf")
        solv_prefix = os.path.join(work, "solvated")

        # First pass: learn the solute centre so the box can be placed around it.
        probe = _TCL.format(psf=clean_psf, dcd=clean_dcd, frame=frame,
                            tmp_pdb=tmp_pdb, tmp_psf=tmp_psf,
                            xmin=0, ymin=0, zmin=0, xmax=1, ymax=1, zmax=1,
                            solv_prefix=solv_prefix, out_prefix=out_prefix,
                            ion_args="-neutralize")
        # Only the centre is needed from the probe, so stop before solvate runs.
        probe = probe.split("# Box of the recorded volume")[0] + "\nputs RESOLVATE_PROBE_DONE\nquit\n"
        pout = run_vmd(probe, out_prefix + ".probe.log")
        centre = _tag(pout, "RESOLVATE_SOLUTE_CENTER")
        minmax = _tag(pout, "RESOLVATE_SOLUTE_MINMAX")
        if centre is None:
            raise RuntimeError(
                f"VMD did not report the solute centre; see {out_prefix}.probe.log")
        cx, cy, cz = [float(v) for v in centre.split()]

        lo, hi = minmax.replace("{", " ").replace("}", " ").split()[:3], \
                 minmax.replace("{", " ").replace("}", " ").split()[3:6]
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        extent = [hi[i] - lo[i] for i in range(3)]

        # The protein must fit inside the recorded cell with room for a hydration
        # shell; otherwise solvate would clip it and the result would be nonsense.
        margins = [(a - extent[0]) / 2.0, (b - extent[1]) / 2.0, (c - extent[2]) / 2.0]
        if min(margins) < 0:
            raise RuntimeError(
                f"The protein does not fit the recorded cell at frame {frame}: extent "
                f"{[round(e,1) for e in extent]} A vs cell {a:.1f} x {b:.1f} x {c:.1f} A. "
                f"Pass --box to supply a larger cell.")

        # Scale the solvation box until the water count reaches bulk density for the
        # recorded volume. VMD's water lattice is under-dense, so filling the recorded
        # cell verbatim leaves the system ~6% short; the count is the quantity worth
        # matching, because NPT then relaxes the cell back to the recorded volume.
        scale = 1.0
        passes = []
        out = None
        for attempt in range(1 if exact_box else max_passes):
            sa, sb, sc = a * scale, b * scale, c * scale
            xmin, xmax = cx - sa / 2.0, cx + sa / 2.0
            ymin, ymax = cy - sb / 2.0, cy + sb / 2.0
            zmin, zmax = cz - sc / 2.0, cz + sc / 2.0

            script = _TCL.format(psf=clean_psf, dcd=clean_dcd, frame=frame,
                                 tmp_pdb=tmp_pdb, tmp_psf=tmp_psf,
                                 xmin=xmin, ymin=ymin, zmin=zmin,
                                 xmax=xmax, ymax=ymax, zmax=zmax,
                                 solv_prefix=solv_prefix, out_prefix=out_prefix,
                                 ion_args=ion_args or "-neutralize")
            out = run_vmd(script, out_prefix + ".log")
            if "RESOLVATE_DONE" not in out:
                raise RuntimeError(f"VMD did not finish; see {out_prefix}.log")
            n_w = _tag(out, "RESOLVATE_NWATER", int)
            if n_w is None:
                raise RuntimeError(f"VMD reported no water count; see {out_prefix}.log")
            passes.append({"pass": attempt + 1, "box_scale": round(scale, 5),
                           "cell_A": [round(sa, 3), round(sb, 3), round(sc, 3)],
                           "waters": n_w})
            if exact_box or abs(n_w - target_waters) <= tol * target_waters:
                break
            # Water count is very nearly proportional to free volume.
            scale *= (float(target_waters) / n_w) ** (1.0 / 3.0)

        final_a, final_b, final_c = a * scale, b * scale, c * scale

        n_atom = _tag(out, "RESOLVATE_NATOM", int)
        charge = _tag(out, "RESOLVATE_CHARGE", float)
        n_water = _tag(out, "RESOLVATE_NWATER", int)
        n_ions = _tag(out, "RESOLVATE_NIONS", int)
        n_prot = _tag(out, "RESOLVATE_NPROT", int)
        if None in (n_atom, charge, n_water, n_prot):
            raise RuntimeError(f"VMD output incomplete; see {out_prefix}.log")
        if abs(charge) > 1e-3:
            raise RuntimeError(
                f"rebuilt system is not neutral (net charge {charge:+.3f} e); "
                f"see {out_prefix}.log")
        if n_prot != hdr["n_atoms"]:
            raise RuntimeError(
                f"protein atom count changed: {hdr['n_atoms']} in the dry trajectory, "
                f"{n_prot} in the rebuilt system")

        # NAMD-ready cell for the system as built. When the box was scaled to reach
        # bulk density this is slightly larger than the recorded cell; NPT
        # equilibration contracts it back, which is the intended check.
        with open(out_prefix + ".xsc", "w") as fh:
            fh.write("# NAMD extended system configuration output file\n")
            fh.write("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n")
            fh.write(f"0 {final_a:.6f} 0 0 0 {final_b:.6f} 0 0 0 {final_c:.6f} "
                     f"{cx:.6f} {cy:.6f} {cz:.6f}\n")

        recorded_volume = a * b * c
        built_volume = final_a * final_b * final_c
        report = {
            "source": {
                "clean_psf": os.path.abspath(clean_psf),
                "clean_dcd": os.path.abspath(clean_dcd),
                "frame": frame,
                "dcd_frames": hdr["n_frames"],
                "dry_atoms": hdr["n_atoms"],
                "solute_mass_Da": round(solute_mass, 1),
            },
            "recorded_cell_A": {
                "a": a, "b": b, "c": c, "volume_A3": recorded_volume,
                "source": "per-frame DCD unit cell" if box is None else "user --box",
                "role": "equilibration target: NPT should relax the built cell back to this",
            },
            "built_cell_A": {"a": final_a, "b": final_b, "c": final_c,
                             "volume_A3": built_volume,
                             "box_scale_vs_recorded": round(scale, 5)},
            "solute": {"extent_A": [round(e, 3) for e in extent],
                       "centre_A": [cx, cy, cz],
                       "margin_A": [round(m, 3) for m in margins]},
            "rebuilt": {"total_atoms": n_atom, "protein_atoms": n_prot,
                        "waters": n_water, "ion_atoms": n_ions,
                        "net_charge_e": charge,
                        "target_waters": target_waters,
                        "waters_vs_target_pct": round(
                            100.0 * (n_water - target_waters) / target_waters, 2),
                        "density_in_recorded_cell_per_nm3": round(
                            n_water / (recorded_volume / 1000.0), 3),
                        "bulk_reference_per_nm3": BULK_TIP3P_PER_NM3},
            "passes": passes,
            "protocol": {
                "solvate": ("VMD solvate at the recorded cell" if exact_box else
                            "VMD solvate, box scaled to reach bulk-density water count"),
                "ions": ion_args or "-neutralize",
                "note": "solvent is rebuilt, not recovered; see the module docstring",
                "required_next_step": (
                    "minimise, then equilibrate under NPT with the protein restrained "
                    "before production use; fresh solvent is not relaxed around the solute"),
            },
        }
        with open(out_prefix + ".resolvate.json", "w") as fh:
            json.dump(report, fh, indent=2)
        return report
    finally:
        if not keep_intermediates:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"  intermediates kept in {work}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def locate_replica(data_root, complex_name, rep):
    """Find the clean PSF and DCD of one replica in a deposit tree."""
    d = os.path.join(data_root, complex_name, f"rep_{rep}", "cleanDCD")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no cleanDCD directory at {d}")
    psf = [f for f in os.listdir(d) if f.endswith(".clean.psf")]
    dcd = [f for f in os.listdir(d) if f.endswith(".clean.dcd")]
    if len(psf) != 1 or len(dcd) != 1:
        raise FileNotFoundError(
            f"expected exactly one .clean.psf and one .clean.dcd in {d}, "
            f"found {len(psf)} and {len(dcd)}")
    return os.path.join(d, psf[0]), os.path.join(d, dcd[0])


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild a solvated system from a dry trajectory frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--clean-psf", help="protein-only PSF")
    ap.add_argument("--clean-dcd", help="protein-only DCD (carries the unit cell)")
    ap.add_argument("--frame", type=int, default=0, help="0-based frame index (default: 0)")
    ap.add_argument("--out-prefix", help="output path prefix (default: alongside the DCD)")
    ap.add_argument("--data-root", help="deposit root holding <complex>/rep_<n>/cleanDCD/")
    ap.add_argument("--complex", help="complex directory name (with --data-root)")
    ap.add_argument("--rep", type=int, help="replica number (with --data-root)")
    ap.add_argument("--box", nargs=3, type=float, metavar=("A", "B", "C"),
                    help="override the cell in Angstrom instead of reading it from the DCD")
    ap.add_argument("--ion-args", default="-neutralize",
                    help="autoionize arguments (default: -neutralize, as in the original build)")
    ap.add_argument("--target-waters", type=int, default=None,
                    help="water count to aim for (default: bulk density for the recorded cell)")
    ap.add_argument("--exact-box", action="store_true",
                    help="fill the recorded cell verbatim instead of matching bulk density; "
                         "starts the system a few per cent under-dense")
    ap.add_argument("--info", action="store_true",
                    help="report the frame's cell and exit without building")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="retain the solvate/autoionize scratch directory")
    args = ap.parse_args()

    if args.data_root:
        if not (args.complex and args.rep):
            ap.error("--data-root requires --complex and --rep")
        clean_psf, clean_dcd = locate_replica(args.data_root, args.complex, args.rep)
    else:
        clean_psf, clean_dcd = args.clean_psf, args.clean_dcd
        if not clean_dcd:
            ap.error("give either --clean-dcd, or --data-root with --complex and --rep")
        if not args.info and not clean_psf:
            ap.error("--clean-psf is required unless --info or --data-root is used")

    if args.info:
        h = read_dcd_header(clean_dcd)
        print(f"  file        : {clean_dcd}")
        print(f"  frames      : {h['n_frames']}")
        print(f"  atoms       : {h['n_atoms']}")
        print(f"  unit cell   : {'present' if h['has_cell'] else 'ABSENT'}")
        if not h["has_cell"]:
            print("  the box volume per frame is unknown; --box would be required")
        elif not 0 <= args.frame < h["n_frames"]:
            print(f"  frame {args.frame}: out of range (0..{h['n_frames'] - 1})")
            return 1
        else:
            a, b, c = read_frame_cell(clean_dcd, args.frame)
            print(f"  frame {args.frame} cell: {a:.3f} x {b:.3f} x {c:.3f} A "
                  f"({a*b*c/1000.0:.1f} nm^3)")
        return 0

    out_prefix = args.out_prefix
    if not out_prefix:
        base = os.path.basename(clean_dcd).replace(".clean.dcd", "")
        out_prefix = os.path.join(os.path.dirname(os.path.abspath(clean_dcd)),
                                  f"{base}_frame{args.frame}_resolvated")

    print(f"Rebuilding {os.path.basename(clean_dcd)} frame {args.frame}")
    rep = resolvate(clean_psf, clean_dcd, args.frame, out_prefix,
                    box=tuple(args.box) if args.box else None,
                    ion_args=args.ion_args,
                    keep_intermediates=args.keep_intermediates,
                    target_waters=args.target_waters,
                    exact_box=args.exact_box)
    rc, bc, r = rep["recorded_cell_A"], rep["built_cell_A"], rep["rebuilt"]
    print(f"  recorded cell : {rc['a']:.2f} x {rc['b']:.2f} x {rc['c']:.2f} A  "
          f"(NPT target)")
    print(f"  built cell    : {bc['a']:.2f} x {bc['b']:.2f} x {bc['c']:.2f} A  "
          f"(scale {bc['box_scale_vs_recorded']:.4f}, {len(rep['passes'])} pass(es))")
    print(f"  protein       : {r['protein_atoms']} atoms (unchanged)")
    print(f"  waters        : {r['waters']}  (target {r['target_waters']}, "
          f"{r['waters_vs_target_pct']:+.2f}%)")
    print(f"  ions          : {r['ion_atoms']} atoms")
    print(f"  total         : {r['total_atoms']} atoms, net charge {r['net_charge_e']:+.4f} e")
    print(f"  wrote         : {out_prefix}.psf/.pdb/.xsc/.resolvate.json")
    print("  next          : minimise, then NPT-equilibrate with the protein restrained")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # These are the expected failure modes (missing input, bad frame, VMD
        # trouble); a traceback would only bury the message.
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)
