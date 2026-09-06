# USAGE : vmd -dispdev text -e prep_MD_HMR.tcl -args <pdb_file> <namd_dir>

package require psfgen 2.0
resetpsf

set pdb_file [lindex $argv 0]
set namd_dir [lindex $argv 1]


topology ${namd_dir}/top_all36_prot.rtf

set mol [ molecule new ${pdb_file}.pdb ]
set sel [atomselect $mol protein]
set water [atomselect $mol "water"]
$water delete
set his_residues [atomselect top "resname HIS"]
$his_residues set resname HSD
$sel moveby [vecinvert [measure center $sel weight mass]]
set chains [lsort -unique [$sel get chain]]

foreach chain $chains {
    set seg ${chain}PRO
    set sel [atomselect $mol "protein and chain $chain"]
    $sel set segid $seg
    $sel writepdb tmp.pdb
    segment $seg { pdb tmp.pdb }
    coordpdb tmp.pdb
}

guesscoord
hmassrepart

writepsf ${pdb_file}_HMR.psf
writepdb ${pdb_file}_HMR.pdb

package require solvate
solvate ${pdb_file}_HMR.psf ${pdb_file}_HMR.pdb -t 20 -o ${pdb_file}_solvate

package require autoionize
autoionize -psf ${pdb_file}_solvate.psf -pdb ${pdb_file}_solvate.pdb -neutralize -o ${pdb_file}_ionized

resetpsf
psfcontext reset

exit
