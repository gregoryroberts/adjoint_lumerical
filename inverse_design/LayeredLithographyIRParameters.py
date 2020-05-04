#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np

#
# Files
#
project_name = 'layered_infrared_3layers_pol_insensitive_thicker_layers_and_spacers_6x6x4p32um_f4'

#
# Optical
#
# averaging between the part where we dip into the infrared/red part of the spectrum
# and where we are in the telecom ranges
#
index_silicon = 0.5 * ( 3.47 + 3.86 )
index_su8 = 1.575
index_air = 1.0
index_sio2 = 1.45
background_index = index_su8
min_device_index = index_su8
max_device_index = index_silicon

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.25

focal_length_um = 4.0
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Device
#
mesh_spacing_um = 0.04

#
# Pesky size to get the number of voxels to be a multiple of 3
#
num_vertical_layers = 3

device_size_lateral_um = 2.0 * 3.0#3.06#3.6
# device_size_verical_um = num_vertical_layers * ( 3.12 / num_vertical_layers )
device_size_verical_um = num_vertical_layers * ( 4.32 / num_vertical_layers )
# amorphous_silicon_height_per_layer_um = 0.52#0.8
# amorphous_silicon_height_per_layer_um = 0.72
amorphous_silicon_height_per_layer_um = 0.72
spacer_size_um = ( device_size_verical_um / num_vertical_layers ) - amorphous_silicon_height_per_layer_um

device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_vertical = 1 + int(device_size_verical_um / mesh_spacing_um)
spacer_size_voxels = 1 + int(spacer_size_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_verical_um
device_vertical_minimum_um = 0

#
# Spectral
#
lambda_min_um = 0.9
lambda_max_um = 1.6

num_bands = 3
num_points_per_band = 10
num_design_frequency_points = num_bands * num_points_per_band
num_eval_frequency_points = 60

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)

#
# Fabrication Constraints
#
min_feature_size_um = 3 * mesh_spacing_um
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )
# blur_half_width_voxels = 0

#
# FDTD
#
vertical_gap_size_um = 1.0
lateral_gap_size_um = 1.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_simulation_time_fs = 4 * 700

#
# Forward Source
#
lateral_aperture_um = device_size_lateral_um + 0.2
src_maximum_vertical_um = device_size_verical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 0.5, 1.0, 0.5 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# We are assuming that the data is organized in order of increasing wavelength (i.e. - blue first, red last)
spectral_focal_plane_map = [
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
	[2 * num_points_per_band, 3 * num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band]
]

#
# Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 4
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]

#
# Symmetry
#
forward_symmetry = [ 'y', 'x' ]
adjoint_symmetry_location = [ 0, 3, 2, 1 ]
adjoint_symmetry_pol = [ 'y', 'x' ]

#
# Optimization
#
start_epoch = 0#12#7#6
num_epochs = 14#8
num_iterations_per_epoch = 50#35#50#25
binarization_start_epoch = 2
max_binarize_movement = 0.01
desired_binarize_change = 3 * 3 * 0.005 / 2

epoch_start_permittivity_change_max = 0.05#0.1
epoch_end_permittivity_change_max = 0.01#0.02
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.025#0.05
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

