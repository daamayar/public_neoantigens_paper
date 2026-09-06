# USAGE : vmd -dispdev text -e wrapping.tcl -args psf_file dcd_file                                                               
# psf_file: base name of PSF file (without .psf extension)
# dcd_file: base name of DCD file (without .dcd extension)

package require pbctools

set psf_file [lindex $argv 0]
set dcd_file [lindex $argv 1]

# Create output trajectory name by replacing stride3 with wrapped
# e.g., flt3_d835y_neo_a_0201_yimsdsnyv_stride3_200ns -> flt3_d835y_neo_a_0201_yimsdsnyv_wrapped_200ns
set final_trajectory [regsub {_stride3_} ${dcd_file} {_wrapped_}]
append final_trajectory ".dcd"

# Tcl's regsub returns the subject unchanged when the pattern does not match, so a DCD whose
# name does not contain "_stride3_" would resolve the output to the input and the script would
# overwrite the trajectory it is reading. Refuse rather than destroy the input.
if {[file normalize $final_trajectory] eq [file normalize ${dcd_file}.dcd]} {
    puts "ERROR: output trajectory resolves to the input ($final_trajectory)."
    puts "       The input DCD must contain '_stride3_' in its name, or pass an explicit"
    puts "       output path. Aborting so the input is not overwritten."
    exit 1
}

mol new ${psf_file}.psf type psf
mol addfile ${dcd_file}.dcd type dcd waitfor all

# Join trajectory
pbc join res -ref "name CA"

# unwrap trajectory
pbc unwrap -sel "protein"

# Wrap trajectory
pbc wrap -centersel "protein" -center com -compound residue -sel "not protein" -all


# Set reference and compare selections
set reference [atomselect top "protein and backbone" frame 0]
set compare [atomselect top "protein and backbone"]

# Get total frames
set num_steps [molinfo top get numframes]
puts "Total number of frames: $num_steps"

# Align trajectory to the reference structure (frame 0)
puts "Aligning trajectory to reference..."
set all_frames [atomselect top "all"]
for {set frame 0} {$frame < $num_steps} {incr frame} {
    $compare frame $frame
    $all_frames frame $frame
    set trans_mat [measure fit $compare $reference]
    $all_frames move $trans_mat
    if {$frame % 100 == 0} {
        puts "Aligned frame $frame of $num_steps"
    }
}
puts "Alignment complete."

# Create the processed trajectory

puts "Writing processed trajectory to $final_trajectory..."
animate write dcd $final_trajectory beg 0 end [expr $num_steps - 1] skip 0 waitfor all sel $all_frames

# Verify and reload the trajectory
mol delete all
mol new ${psf_file}.psf type psf
mol addfile $final_trajectory type dcd waitfor all

set final_frames [molinfo top get numframes]
puts "Processing complete. Processed trajectory saved as $final_trajectory with $final_frames frames."

# Cleanup
$reference delete
$compare delete
$all_frames delete