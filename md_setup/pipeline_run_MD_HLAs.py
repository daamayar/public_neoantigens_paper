#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This script run MD simulations of HLA antigens.

USAGE: python pipeline_run_MD_HLAs.py <list_pdbs.txt>

Some warnings to bear in mind :

    - The script utilizes the directory of the PDB file as working directory, so output will be generated in that directory.

    - This script assumes that the MD calculation will be launched in an OAR-based computation grid. If another resource/task manager
    is used, please adapt the run_VMD function accordingly.

    - This script needs the VMD-python package (version 3.1.4 tested).

    - This script needs the NAMD3 executable and CHARMM36 topology and parameter files. Please change the variable "namd_dir" to
    the right directory pointing to these files (all these files must be in the same directory).

    - Format example for 'list_pdbs.txt' (1 structure per line):
      /path/to/A_0201.pdb
      /path/to/A_1101.pdb
      /path/to/A_0202.pdb

Author: {0}
Email: {1}


'''


import os
import sys
from vmd import evaltcl
from vmd import molecule
import re
import multiprocessing as mp
import traceback
from vmd import atomsel
import subprocess


# Directory holding the NAMD binary (namd3) and the CHARMM36m parameter files.
# Set NAMD_DIR; it defaults to ./namd_dir under the working directory.
namd_dir = os.path.abspath(os.path.expanduser(
    os.environ.get("NAMD_DIR") or os.path.join(os.getcwd(), "namd_dir")))

CWD = os.getcwd()

__author__ = "Diego Amaya"
__email__ = "diego.amaya-ramirez@inria.fr, diaamayaram@unal.edu.co"

USAGE = __doc__.format(__author__, __email__)


def check_input(args):
    try:
        """This function checks if to read from stdin/file and validates user input."""
        if not len(args):
            # Read from pipe
            if not sys.stdin.isatty():
                input_file = sys.stdin
            else:
                raise Exception
        elif len(args) == 1:
            if not os.path.isfile(args[0]):
                raise Exception
            input_file = open(args[0], 'r')
        else:
            raise Exception
    except:
        sys.exit(1)
    return input_file


def run_VMD(pdb_file):
    structure = f"{pdb_file.split('.')[0]}"
    env = os.environ.copy()
    # VMD's runtime directory, e.g. <vmd>/install_dir/lib/vmd. Set VMDDIR; if it
    # is already exported the existing value is used unchanged.
    vmddir = env.get("VMDDIR")
    if not vmddir:
        raise EnvironmentError(
            "VMDDIR is not set. Point it at the VMD runtime directory, "
            "for example /opt/vmd-1.9.4/lib/vmd.")
    env["VMDDIR"] = vmddir
    env["LD_LIBRARY_PATH"] = vmddir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    results = subprocess.run(f'vmd -dispdev text -e {CWD}/prep_MD_HMR.tcl -args {structure} {namd_dir}',
                             shell=True, env=env)
    pdbid = molecule.load('pdb', f'{structure}_ionized.pdb', 'psf', f'{structure}_ionized.psf')
    evaltcl(f'set all [atomselect {pdbid} all]')
    cell_size = evaltcl('measure minmax $all')
    cell_size = re.findall(r'\{[^}]*\}', cell_size)
    cell_size = [elem.strip('{}') for elem in cell_size]
    cell_size = [elem.split(' ') for elem in cell_size]
    X_size = abs(float(cell_size[0][0])) + abs(float(cell_size[1][0]))
    Y_size = abs(float(cell_size[0][1])) + abs(float(cell_size[1][1]))
    Z_size = abs(float(cell_size[0][2])) + abs(float(cell_size[1][2]))

    sel = atomsel("all")
    sel.beta = 0.0
    sel = atomsel("protein")
    sel.beta = 1.0
    molecule.write(pdbid, "pdb", f"{structure}_ionized_allConstraints.pdb")
    molecule.delete(pdbid)

    #evaltcl('resetpsf')
    pdbid = molecule.load('pdb', f'{structure}_ionized.pdb', 'psf', f'{structure}_ionized.psf')
    sel = atomsel("all")
    sel.beta = 0.0
    sel = atomsel("backbone")
    sel.beta = 1.0
    molecule.write(pdbid, "pdb", f"{structure}_ionized_bbConstraints.pdb")
    molecule.delete(pdbid)

    #evaltcl('resetpsf')
    pdbid = molecule.load('pdb', f'{structure}_ionized.pdb', 'psf', f'{structure}_ionized.psf')
    min_conf = open(f'{namd_dir}/general_min.conf', 'r')
    new_min_conf = open(f'{structure}_min.conf', 'w')
    for line in min_conf:
        if '<file_name>' in line:
            new_min_conf.write(line.replace('<file_name>', f'{structure}_ionized'))
        elif '<X_size>' in line:
            new_min_conf.write(line.replace('<X_size>', f'{X_size}'))
        elif '<Y_size>' in line:
            new_min_conf.write(line.replace('<Y_size>', f'{Y_size}'))
        elif '<Z_size>' in line:
            new_min_conf.write(line.replace('<Z_size>', f'{Z_size}'))
        elif '<center_X>   <center_Y>   <center_Z>' in line:
            new_min_conf.write(line.replace('<center_X>   <center_Y>   <center_Z>',
                    f'{(float(cell_size[0][0]) + float(cell_size[1][0]))/2}   {(float(cell_size[0][1]) + float(cell_size[1][1]))/2}   {(float(cell_size[0][2]) + float(cell_size[1][2]))/2}'))
        elif '<namd_dir>' in line:
            new_min_conf.write(line.replace('<namd_dir>', f'{namd_dir}'))
        else:
            new_min_conf.write(line)

    min_conf.close()
    new_min_conf.close()

    allConstrains_conf = open(f'{namd_dir}/general_allConstrains_HMR.conf', 'r')
    new_allConstrains_conf = open(f'{structure}_allConstrains_HMR.conf', 'w')
    for line in allConstrains_conf:
        if '<file_name>' in line:
            new_allConstrains_conf.write(line.replace('<file_name>', f'{structure}_ionized'))
        elif '<namd_dir>' in line:
            new_allConstrains_conf.write(line.replace('<namd_dir>', f'{namd_dir}'))
        else:
            new_allConstrains_conf.write(line)
    allConstrains_conf.close()
    new_allConstrains_conf.close()

    bbConstrains_conf = open(f'{namd_dir}/general_bbConstrains_HMR.conf', 'r')
    new_bbConstrains_conf = open(f'{structure}_bbConstrains_HMR.conf', 'w')
    for line in bbConstrains_conf:
        if '<file_name>' in line:
            new_bbConstrains_conf.write(line.replace('<file_name>', f'{structure}_ionized'))
        elif '<namd_dir>' in line:
            new_bbConstrains_conf.write(line.replace('<namd_dir>', f'{namd_dir}'))
        else:
            new_bbConstrains_conf.write(line)
    bbConstrains_conf.close()
    new_bbConstrains_conf.close()

    production_conf = open(f'{namd_dir}/general_production_HMR.conf', 'r')
    new_production_conf = open(f'{structure}_production_HMR.conf', 'w')
    for line in production_conf:
        if '<file_name>' in line:
            new_production_conf.write(line.replace('<file_name>', f'{structure}_ionized'))
        elif '<namd_dir>' in line:
            new_production_conf.write(line.replace('<namd_dir>', f'{namd_dir}'))
        else:
            new_production_conf.write(line)
    production_conf.close()
    new_production_conf.close()

    os.system('chmod +x *.conf')
    run = open(f'run_MD_{structure}.sh', 'w')
    run.writelines([
            '#! /bin/bash\n',
            f'bash -l -c "{namd_dir}/namd3 +idlepoll +p {int(mp.cpu_count()/4)} +setcpuaffinity +devices 0 {structure}_min.conf > {structure}_min.log"\n',
            f'bash -l -c "{namd_dir}/namd3 +idlepoll +p 1 +setcpuaffinity +devices 0 {structure}_allConstrains_HMR.conf > {structure}_allConstrains_HMR.log"\n',
            f'bash -l -c "{namd_dir}/namd3 +idlepoll +p 1 +setcpuaffinity +devices 0 {structure}_bbConstrains_HMR.conf > {structure}_bbConstrains_HMR.log"\n',
            f'bash -l -c "{namd_dir}/namd3 +idlepoll +p 1 +setcpuaffinity +devices 0 {structure}_production_HMR.conf > {structure}_production_HMR.log"'
            ])
    run.close()
    os.system(f'chmod +x run_MD_{structure}.sh')
    # Site-specific OAR submission. OAR_CLUSTERS is a comma-separated list of
    # cluster names; OAR_EMAIL, when set, adds a completion notification.
    cluster_list = [c.strip() for c in os.environ.get(
        "OAR_CLUSTERS", "graffiti,gruss,grue,grat").split(",") if c.strip()]
    oar_email = os.environ.get("OAR_EMAIL", "")
    notify = f"--notify mail:{oar_email} " if oar_email else ""
    os.system(
        f"oarsub {notify}-n {pdb_file.split('.')[0]}_HMR -q "
        f"{os.environ.get('OAR_QUEUE', 'production')} "
        f"-p 'cluster in ({', '.join(cluster_list)})' "
        f"-l gpu=1,walltime=96 ./run_MD_{structure}.sh")
    molecule.delete(pdbid)


if __name__ == "__main__":
    try:
        input_file = check_input(sys.argv[1:])

        for line in input_file:
            path_input, file = os.path.split(line)
            os.chdir(f'{path_input}')
            run_VMD(file.strip())

        input_file.close()
    except:
        sys.stderr.write(USAGE)
        traceback.print_exc()
        sys.exit(1)
