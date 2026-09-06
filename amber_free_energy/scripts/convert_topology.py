#!/usr/bin/env python3
"""
Phase 1: Topology Conversion
Convert CHARMM PSF/PDB to AMBER prmtop/inpcrd for MMPBSA.py.

Creates solvated, complex (dry), receptor, and ligand topology files.
Programmatically identifies receptor (chains A+B / segments APRO+BPRO)
and ligand (chain C / segment CPRO) based on segment IDs.

Usage:
    python -m scripts.convert_topology <complex_name> <rep_num>
"""
import os
import sys
import copy
import argparse

from . import config
from . import utils


def _create_window_symlinks(complex_name, rep_num, reference_workdir):
    """
    Create symlinks in all non-reference window workdirs pointing to the
    topology files in the reference (500ns) workdir.

    Args:
        complex_name: Name of the complex directory.
        rep_num: Replica number (1-based).
        reference_workdir: Absolute path to the reference window workdir
                           (where the actual topology files live).
    """
    topo_files = ["solvated.prmtop", "solvated.inpcrd",
                  "complex.prmtop", "complex.inpcrd",
                  "receptor.prmtop", "receptor.inpcrd",
                  "ligand.prmtop", "ligand.inpcrd",
                  "topology_metadata.json"]
    symlink_count = 0
    for window_name in config.ANALYSIS_WINDOWS:
        if window_name == config.REFERENCE_WINDOW:
            continue  # reference workdir has the actual files
        workdir_other = utils.get_mmpbsa_workdir(complex_name, rep_num, window_name)
        os.makedirs(workdir_other, exist_ok=True)
        for fname in topo_files:
            src = os.path.join(reference_workdir, fname)
            dst = os.path.join(workdir_other, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(src, workdir_other), dst)
        symlink_count += 1

    if symlink_count > 0:
        print(f"  Symlinks created for {symlink_count} non-{config.REFERENCE_WINDOW} workdirs")


def convert_topology(complex_name, rep_num):
    """
    Convert CHARMM topology to AMBER format for a single replica.

    Args:
        complex_name: Name of the complex directory
        rep_num: Replica number (1-based)

    Returns:
        dict: Metadata about the conversion (residue ranges, atom counts, etc.)
    """
    import parmed as pmd
    from parmed.charmm import CharmmPsfFile, CharmmParameterSet
    from parmed.tools import changeRadii

    rep_dir = utils.get_replica_dir(complex_name, rep_num)
    print(f"\n{'='*60}")
    print(f"Phase 1: Topology conversion for {complex_name} rep_{rep_num}")
    print(f"{'='*60}")

    # Find input files
    psf_path = utils.find_file(rep_dir, config.PSF_PATTERN)
    pdb_path = utils.find_file(rep_dir, config.PDB_PATTERN)
    xsc_path = utils.find_file(rep_dir, config.XSC_PATTERN)

    print(f"  PSF: {os.path.basename(psf_path)}")
    print(f"  PDB: {os.path.basename(pdb_path)}")
    print(f"  XSC: {os.path.basename(xsc_path)}")

    # Create output directory (use reference window workdir; others will symlink)
    workdir = utils.get_mmpbsa_workdir(complex_name, rep_num, config.REFERENCE_WINDOW)
    os.makedirs(workdir, exist_ok=True)

    # Check if already done (verify ALL required output files exist)
    metadata_path = os.path.join(workdir, "topology_metadata.json")
    required_outputs = [
        "complex.prmtop", "complex.inpcrd",
        "receptor.prmtop", "receptor.inpcrd",
        "ligand.prmtop", "ligand.inpcrd",
        "solvated.prmtop", "solvated.inpcrd",
        "topology_metadata.json",
    ]
    all_exist = all(
        utils.check_file_exists(os.path.join(workdir, f))
        for f in required_outputs
    )
    if all_exist:
        # Validate prmtop files have non-trivial content (>1KB rules out truncated files)
        prmtop_files = [f for f in required_outputs if f.endswith(".prmtop")]
        all_valid = all(
            os.path.getsize(os.path.join(workdir, f)) > 1024
            for f in prmtop_files
        )
        if all_valid:
            print("  Topology files already exist. Loading metadata...")
            # Ensure symlinks exist for all non-reference workdirs (may be missing
            # if the code was updated after the initial topology conversion)
            _create_window_symlinks(complex_name, rep_num, workdir)
            return utils.load_metadata(metadata_path)
        else:
            print("  WARNING: Existing prmtop files appear truncated. Re-running conversion.")

    # Parse box dimensions from XSC
    box_dims = utils.parse_xsc_file(xsc_path)
    print(f"  Box dimensions: {box_dims[0]:.2f} x {box_dims[1]:.2f} x {box_dims[2]:.2f} A")

    # Load CHARMM parameters
    print("  Loading CHARMM parameters...")
    params = CharmmParameterSet(*config.PARAM_FILES)

    # Load PSF and assign coordinates
    print("  Loading PSF and coordinates...")
    psf = CharmmPsfFile(psf_path)
    psf.coordinates = pmd.load_file(pdb_path).coordinates
    psf.box = box_dims
    psf.load_parameters(params)

    print(f"  Total atoms (solvated): {len(psf.atoms)}")
    print(f"  Total residues: {len(psf.residues)}")
    print(f"  CMAP terms: {len(psf.cmaps)}")
    assert len(psf.cmaps) > 0, "ERROR: No CMAP terms found -- check parameter files"

    # Assign PB-optimized radii (mbondi_pb3) for Poisson-Boltzmann solvation.
    # ParmEd ChamberParm sets all radii to 0 by default, which makes the PB
    # solver produce zero solvation energies. mbondi_pb3 radii are element-based
    # and work correctly with CHARMM atom types.
    print("  Assigning mbondi_pb3 PB radii...")
    action = changeRadii(psf, "mbondi_pb3")
    action.execute()
    zero_radii = sum(1 for a in psf.atoms if a.solvent_radius == 0.0)
    assert zero_radii == 0, f"ERROR: {zero_radii} atoms still have zero PB radius after changeRadii"
    print(f"  PB radii assigned: all {len(psf.atoms)} atoms have non-zero radii")

    # Save solvated topology
    print("  Saving solvated topology...")
    psf.save(os.path.join(workdir, "solvated.prmtop"), overwrite=True)
    psf.save(os.path.join(workdir, "solvated.inpcrd"), format="rst7", overwrite=True)

    # Analyze segments in solvated structure
    solvated_segments = utils.get_segment_residue_mapping(psf)
    print("  Segment analysis (solvated):")
    for seg, info in sorted(solvated_segments.items()):
        print(f"    {seg}: residues {info['start']}-{info['end']} ({info['count']} residues)")

    # Strip water and ions to create dry complex
    print("  Stripping water and ions...")
    complex_sys = copy.deepcopy(psf)
    for resname in config.STRIP_RESIDUE_NAMES:
        try:
            complex_sys.strip(f":{resname}")
        except Exception:
            pass  # Residue name not present

    # Analyze segments in dry complex to get residue ranges
    dry_segments = utils.get_segment_residue_mapping(complex_sys)
    print("  Segment analysis (dry complex):")
    for seg, info in sorted(dry_segments.items()):
        print(f"    {seg}: residues {info['start']}-{info['end']} ({info['count']} residues)")

    # Determine receptor and ligand residue ranges
    receptor_start = None
    receptor_end = None
    ligand_start = None
    ligand_end = None

    for seg in config.RECEPTOR_SEGMENTS:
        if seg in dry_segments:
            if receptor_start is None:
                receptor_start = dry_segments[seg]["start"]
            receptor_end = dry_segments[seg]["end"]

    for seg in config.LIGAND_SEGMENTS:
        if seg in dry_segments:
            if ligand_start is None:
                ligand_start = dry_segments[seg]["start"]
            ligand_end = dry_segments[seg]["end"]

    found_segments = set(dry_segments.keys())
    expected_segments = set(config.RECEPTOR_SEGMENTS + config.LIGAND_SEGMENTS)
    unexpected = found_segments - expected_segments
    if unexpected:
        print(f"  WARNING: Unexpected segments found in dry complex: {unexpected}")
        print(f"           Expected: {expected_segments}, Found: {found_segments}")

    assert receptor_start is not None, (
        f"ERROR: No receptor segments found. Expected {config.RECEPTOR_SEGMENTS}, "
        f"found segments: {list(dry_segments.keys())}"
    )
    assert ligand_start is not None, (
        f"ERROR: No ligand segments found. Expected {config.LIGAND_SEGMENTS}, "
        f"found segments: {list(dry_segments.keys())}"
    )

    receptor_mask = f":{receptor_start}-{receptor_end}"
    ligand_mask = f":{ligand_start}-{ligand_end}"

    print(f"  Receptor mask (dry complex): {receptor_mask}")
    print(f"  Ligand mask (dry complex): {ligand_mask}")

    # Save dry complex
    complex_sys.save(os.path.join(workdir, "complex.prmtop"), overwrite=True)
    complex_sys.save(os.path.join(workdir, "complex.inpcrd"), format="rst7", overwrite=True)
    print(f"  Complex atoms (dry): {len(complex_sys.atoms)}")

    # Create receptor (strip ligand residues)
    print("  Creating receptor topology...")
    receptor = copy.deepcopy(complex_sys)
    receptor.strip(f":{ligand_start}-{ligand_end}")
    receptor.save(os.path.join(workdir, "receptor.prmtop"), overwrite=True)
    receptor.save(os.path.join(workdir, "receptor.inpcrd"), format="rst7", overwrite=True)
    print(f"  Receptor atoms: {len(receptor.atoms)}")

    # Create ligand (keep only ligand residues — strip receptor residues)
    print("  Creating ligand topology...")
    ligand = copy.deepcopy(complex_sys)
    ligand.strip(f":{receptor_start}-{receptor_end}")
    ligand.save(os.path.join(workdir, "ligand.prmtop"), overwrite=True)
    ligand.save(os.path.join(workdir, "ligand.inpcrd"), format="rst7", overwrite=True)
    print(f"  Ligand atoms: {len(ligand.atoms)}")

    # Verify atom count consistency
    assert len(receptor.atoms) + len(ligand.atoms) == len(complex_sys.atoms), \
        f"Atom count mismatch: receptor ({len(receptor.atoms)}) + ligand ({len(ligand.atoms)}) != complex ({len(complex_sys.atoms)})"

    # Save metadata
    metadata = {
        "complex_name": complex_name,
        "replica": rep_num,
        "solvated_atoms": len(psf.atoms),
        "complex_atoms": len(complex_sys.atoms),
        "receptor_atoms": len(receptor.atoms),
        "ligand_atoms": len(ligand.atoms),
        "receptor_mask": receptor_mask,
        "ligand_mask": ligand_mask,
        "receptor_residue_range": f"{receptor_start}-{receptor_end}",
        "ligand_residue_range": f"{ligand_start}-{ligand_end}",
        "box_dims": box_dims,
        "cmap_terms": len(psf.cmaps),
        "dry_segments": {
            seg: {"start": info["start"], "end": info["end"], "count": info["count"]}
            for seg, info in dry_segments.items()
        },
    }
    utils.save_metadata(metadata_path, metadata)
    print(f"  Metadata saved to {os.path.basename(metadata_path)}")

    # Create symlinks for all non-reference workdirs to reuse the same topology files
    _create_window_symlinks(complex_name, rep_num, workdir)
    print("  Topology conversion complete!")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Convert CHARMM topology to AMBER format")
    parser.add_argument("complex_name", help="Complex directory name")
    parser.add_argument("rep_num", type=int, help="Replica number (1-based)")
    args = parser.parse_args()
    convert_topology(args.complex_name, args.rep_num)


if __name__ == "__main__":
    main()
