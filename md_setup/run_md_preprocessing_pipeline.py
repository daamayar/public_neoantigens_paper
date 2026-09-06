#!/usr/bin/env python3
"""
Script to run the complete MD trajectory processing pipeline:
1. Concatenate production trajectories using catdcd
2. Run wrapping/alignment using VMD

Reads directories from a list file and processes each one.

Usage:
    python run_md_preprocessing_pipeline.py <list_file> [options]

Example:
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt --concat-only
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt --wrap-only
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt --force
"""

import os
import re
import sys
import subprocess
import argparse


def get_production_dcd_files(directory):
    """
    Find all production DCD files matching the pattern *_ionized_production.<number>.dcd
    and return them sorted by the number in the filename.

    Returns:
        tuple: (prefix, sorted_list_of_dcd_files) or (None, []) if no files found
    """
    pattern = re.compile(r'^(.+)_ionized_production\.(\d+)\.dcd$')

    dcd_files = []
    prefix = None

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            if prefix is None:
                prefix = match.group(1)
            file_number = int(match.group(2))
            dcd_files.append((file_number, filename))

    # Sort by the number in the filename
    dcd_files.sort(key=lambda x: x[0])

    # Return just the filenames in sorted order
    sorted_files = [f[1] for f in dcd_files]

    return prefix, sorted_files


def get_psf_file(directory):
    """
    Find the PSF file matching the pattern *_ionized.psf

    Returns:
        str: PSF filename (without extension) or None if not found
    """
    for filename in os.listdir(directory):
        if filename.endswith('_ionized.psf'):
            return filename[:-4]  # Remove .psf extension
    return None


def get_concatenated_dcd(directory):
    """
    Find the concatenated DCD file matching the pattern *_stride3_500ns.dcd

    Returns:
        str: DCD filename (without extension) or None if not found
    """
    for filename in os.listdir(directory):
        if filename.endswith('_stride3_500ns.dcd'):
            return filename[:-4]  # Remove .dcd extension
    return None


def _is_output_fresh(output_path, source_paths):
    """True iff output_path exists and is newer than every source in source_paths."""
    if not os.path.exists(output_path):
        return False
    if not source_paths:
        return True
    out_mtime = os.path.getmtime(output_path)
    for src in source_paths:
        if os.path.exists(src) and os.path.getmtime(src) > out_mtime:
            return False
    return True


def concatenate_trajectories(directory, prefix, dcd_files, stride=3, first_frame=0,
                             last_frame=12500, force=False):
    """
    Concatenate DCD files using catdcd command.

    Args:
        directory: Path to the directory containing the DCD files
        prefix: Prefix for the output file
        dcd_files: List of DCD files to concatenate (in order)
        stride: Frame stride for catdcd (default: 3)
        first_frame: First frame to include (default: 0)
        last_frame: Last frame to include (default: 12500)
        force: If True, regenerate even if an up-to-date output exists.

    Returns:
        str: Output filename (without extension) or None if failed
    """
    output_file = f"{prefix}_stride{stride}_500ns.dcd"
    output_base = f"{prefix}_stride{stride}_500ns"

    output_path = os.path.join(directory, output_file)
    source_paths = [os.path.join(directory, f) for f in dcd_files]

    # Reuse the cached output only if it is newer than every source segment.
    if not force and _is_output_fresh(output_path, source_paths):
        print(f"    Concatenated file already exists and is up to date: {output_file}")
        return output_base

    if os.path.exists(output_path):
        if force:
            print(f"    --force: removing stale {output_file}")
        else:
            print(f"    Stale concatenated file (older than a source segment); "
                  f"regenerating: {output_file}")
        os.remove(output_path)

    # Build the catdcd command
    cmd = [
        "catdcd",
        "-o", output_file,
        "-stride", str(stride),
        "-first", str(first_frame),
        "-last", str(last_frame)
    ] + dcd_files

    print(f"    Running: {' '.join(cmd)}")

    # Run the command from the directory
    result = subprocess.run(
        cmd,
        cwd=directory,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"    ERROR: catdcd failed with return code {result.returncode}")
        print(f"    STDERR: {result.stderr}")
        return None

    print(f"    SUCCESS: Created {output_file}")
    return output_base


def run_wrapping(directory, psf_base, dcd_base, script_dir, wait_time=None,
                 source_paths=None, force=False):
    """
    Run VMD wrapping script on the trajectory.

    Args:
        directory: Path to the directory containing the files
        psf_base: PSF filename without extension
        dcd_base: DCD filename without extension
        script_dir: Directory containing wrapping.tcl
        wait_time: Time to wait for VMD to complete (seconds); None = no timeout
        source_paths: Files whose mtimes invalidate a cached wrapped output.
        force: If True, regenerate even if an up-to-date output exists.

    Returns:
        bool: True if successful, False otherwise
    """
    wrapped_file = dcd_base.replace('_stride3_', '_wrapped_') + '.dcd'
    wrapped_path = os.path.join(directory, wrapped_file)

    # Reuse the cached wrap only if it is newer than the sources that fed it.
    if not force and _is_output_fresh(wrapped_path, source_paths or []):
        print(f"    Wrapped file already exists and is up to date: {wrapped_file}")
        return True

    if os.path.exists(wrapped_path):
        if force:
            print(f"    --force: removing existing {wrapped_file}")
        else:
            print(f"    Stale wrapped file (older than a source); "
                  f"regenerating: {wrapped_file}")
        os.remove(wrapped_path)

    wrapping_script = os.path.join(script_dir, 'wrapping.tcl')

    if not os.path.exists(wrapping_script):
        print(f"    ERROR: wrapping.tcl not found at {wrapping_script}")
        return False

    # Per-replica VMD log so pbc / alignment warnings survive the run.
    vmd_log_path = os.path.join(directory, f"wrapping_{dcd_base}.log")

    # Build the VMD command
    vmd_cmd = f"source ~/.bashrc; vmd -dispdev text -e \"{wrapping_script}\" -args \"{psf_base}\" \"{dcd_base}\""
    cmd = ["/bin/bash", "-l", "-c", vmd_cmd]

    print(f"    Running via bash: {vmd_cmd}")
    print(f"    Working directory: {directory}")
    print(f"    VMD log: {vmd_log_path}")

    # Run VMD from the directory, streaming stdout+stderr to the log file so
    # the PIPE buffer can never fill up and cause VMD to stall.
    with open(vmd_log_path, "w") as log_fh:
        process = subprocess.Popen(
            cmd,
            cwd=directory,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            returncode = process.wait(timeout=wait_time)
        except subprocess.TimeoutExpired:
            print(f"    ERROR: VMD process timed out after {wait_time}s; killing.")
            process.kill()
            process.wait()
            # Do NOT trust a partially-written output. Move it aside so the
            # next run regenerates it instead of treating it as success.
            if os.path.exists(wrapped_path):
                partial_path = wrapped_path + ".partial"
                os.replace(wrapped_path, partial_path)
                print(f"    Partial output moved to: {partial_path}")
            print(f"    See {vmd_log_path} for VMD output.")
            return False

    if returncode != 0:
        print(f"    ERROR: VMD failed with return code {returncode}")
        print(f"    See {vmd_log_path} for VMD output.")
        if os.path.exists(wrapped_path):
            partial_path = wrapped_path + ".partial"
            os.replace(wrapped_path, partial_path)
            print(f"    Partial output moved to: {partial_path}")
        return False

    if not os.path.exists(wrapped_path):
        print(f"    ERROR: VMD exited 0 but {wrapped_file} was not created.")
        print(f"    See {vmd_log_path} for VMD output.")
        return False

    print(f"    SUCCESS: Created {wrapped_file}")
    return True


def process_directory(directory, script_dir, concat_only=False, wrap_only=False,
                      wait_time=None, force=False):
    """
    Process a single directory: concatenate DCD files and run wrapping.

    Args:
        directory: Path to the directory to process
        script_dir: Directory containing wrapping.tcl
        concat_only: Only run concatenation step
        wrap_only: Only run wrapping step
        wait_time: Time to wait for VMD to complete (None = no timeout)
        force: Rebuild outputs even if an up-to-date one exists

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.isdir(directory):
        print(f"  WARNING: Directory does not exist: {directory}")
        return False

    # Get PSF file
    psf_base = get_psf_file(directory)
    if not psf_base:
        print(f"  WARNING: No PSF file (*_ionized.psf) found in {directory}")
        return False

    print(f"  PSF: {psf_base}.psf")

    dcd_base = None
    # Source paths that invalidate the wrapped output (the segments that feed
    # the stride-3 concat). Discovered below in step 1.
    wrap_source_paths = []

    # Step 1: Concatenate trajectories (unless wrap_only)
    if not wrap_only:
        print("  Step 1: Concatenating trajectories...")
        prefix, dcd_files = get_production_dcd_files(directory)

        if not dcd_files:
            print(f"    WARNING: No production DCD files found")
            # Check if concatenated file already exists
            dcd_base = get_concatenated_dcd(directory)
            if not dcd_base:
                return False
        else:
            print(f"    Found {len(dcd_files)} production DCD files")
            print(f"    Prefix: {prefix}")
            print(f"    Files (in order): {', '.join(dcd_files)}")

            wrap_source_paths = [os.path.join(directory, f) for f in dcd_files]
            dcd_base = concatenate_trajectories(directory, prefix, dcd_files,
                                                force=force)
            if dcd_base is None:
                return False

    if concat_only:
        return True

    # Get the concatenated DCD if we haven't already
    if dcd_base is None:
        dcd_base = get_concatenated_dcd(directory)
        if not dcd_base:
            print(f"  WARNING: No concatenated DCD file (*_stride3_500ns.dcd) found")
            print(f"  Please run concatenation step first.")
            return False

    # The wrapped file should be newer than both the concat (if present) and
    # the raw segments; include whichever ones exist as source paths.
    concat_path = os.path.join(directory, dcd_base + ".dcd")
    if os.path.exists(concat_path):
        wrap_source_paths = [concat_path] + wrap_source_paths

    # Step 2: Run wrapping
    print("  Step 2: Running wrapping/alignment...")
    print(f"    DCD: {dcd_base}.dcd")

    success = run_wrapping(directory, psf_base, dcd_base, script_dir,
                           wait_time=wait_time,
                           source_paths=wrap_source_paths,
                           force=force)

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Run MD trajectory processing pipeline: concatenate and wrap trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt --concat-only
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt --wrap-only
    python run_md_preprocessing_pipeline.py list_dirs_MDs.txt --force
        """
    )
    parser.add_argument('list_file', help='File containing list of directories to process')
    parser.add_argument('--concat-only', action='store_true',
                        help='Only run concatenation step (catdcd)')
    parser.add_argument('--wrap-only', action='store_true',
                        help='Only run wrapping step (VMD)')
    parser.add_argument('--wait-time', type=int, default=None,
                        help='Timeout for VMD in seconds. Default: no timeout '
                             '(VMD is allowed to run to completion). A too-short '
                             'timeout silently produces partially-aligned '
                             'trajectories, so only set this if you know what '
                             'you are doing.')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate outputs even if an up-to-date one exists.')

    args = parser.parse_args()

    if args.concat_only and args.wrap_only:
        print("ERROR: Cannot use both --concat-only and --wrap-only")
        sys.exit(1)

    list_file = args.list_file

    if not os.path.exists(list_file):
        print(f"ERROR: List file not found: {list_file}")
        sys.exit(1)

    # Get the script directory (where wrapping.tcl is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the base directory (where the list file is located)
    base_dir = os.path.dirname(os.path.abspath(list_file))
    if not base_dir:
        base_dir = os.getcwd()

    # Read directories from the list file
    with open(list_file, 'r') as f:
        directories = [line.strip().rstrip('/') for line in f if line.strip()]

    mode = "full pipeline"
    if args.concat_only:
        mode = "concatenation only"
    elif args.wrap_only:
        mode = "wrapping only"

    print(f"MD Trajectory Processing Pipeline")
    print(f"Mode: {mode}")
    print(f"Processing {len(directories)} directories from {list_file}")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for rel_dir in directories:
        # Construct full path
        full_dir = os.path.join(base_dir, rel_dir)

        print(f"\nProcessing: {rel_dir}")
        print("-" * 40)

        if process_directory(full_dir, script_dir, args.concat_only,
                             args.wrap_only, args.wait_time, force=args.force):
            success_count += 1
            print(f"  COMPLETED: {rel_dir}")
        else:
            fail_count += 1
            print(f"  FAILED: {rel_dir}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success_count} successful, {fail_count} failed")


if __name__ == "__main__":
    main()
