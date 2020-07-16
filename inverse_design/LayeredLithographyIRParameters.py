#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np

#
# Files
#
project_name = 'layered_infrared_3layers_pol_insensitive_thinner_layers_6x6x2p25um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_fix_binarization_fn_6x6x3p84um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_tio2_zep_init_6x6x3p84um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_sio2_air_init_6x6x3p84um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_thicker_high_random_mean_init_6x6x3p84um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_thicker_high_mean_init_6x6x3p84um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_thicker_layers_finer_6x6x3p84um_weighting_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_thicker_layers_6x6x4p32um_f4'
# project_name = 'layered_infrared_3layers_pol_insensitive_thicker_layers_and_spacers_6x6x4p32um_f4_v3'

#
# Optical
#
# averaging between the part where we dip into the infrared/red part of the spectrum
# and where we are in the telecom ranges
#
index_silicon = 0.5 * ( 3.47 + 3.86 )
index_su8 = 1.575
index_tio2 = 2.1
index_zep = 1.54
index_air = 1.0
index_sio2 = 1.45
background_index = index_su8
# background_index = index_tio2
min_device_index = index_su8
# min_device_index = index_sio2
# min_device_index = index_tio2
max_device_index = index_silicon
# max_device_index = index_air
# max_device_index = index_zep

# last_layer_permittivities = [ index_tio2**2, index_zep**2 ]
last_layer_permittivities = [ index_air**2, index_silicon**2 ]

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

random_seed = 6235234
init_permittivity_0_1_scale = 0.5

focal_length_um = 4.0
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Device
#
# mesh_spacing_um = 0.03
# mesh_spacing_um = 0.032
mesh_spacing_um = 0.025
design_spacing_um = 0.075
# design_spacing_um = 0.12
# design_spacing_um = 0.24

#
# Pesky size to get the number of voxels to be a multiple of 3
#
num_vertical_layers = 3

# device_size_lateral_um = 2.0 * 3.0#3.06#3.6
device_size_lateral_um = 6.0
# device_size_verical_um = num_vertical_layers * ( 3.168 / num_vertical_layers )
# device_size_verical_um = num_vertical_layers * 0.704

# device_size_verical_um = num_vertical_layers * 1.28
device_size_verical_um = num_vertical_layers * 0.75
# device_size_verical_um = num_vertical_layers * 0.72
# device_size_verical_um = num_vertical_layers * 0.9

# device_size_verical_um = num_vertical_layers * ( 3.12 / num_vertical_layers )
# device_size_verical_um = num_vertical_layers * ( 4.32 / num_vertical_layers )
# amorphous_silicon_height_per_layer_um = 0.52#0.8
# amorphous_silicon_height_per_layer_um = 0.72
# amorphous_silicon_height_per_layer_um = 0.704

amorphous_silicon_height_per_layer_um = 0.3
# amorphous_silicon_height_per_layer_um = 0.64
# amorphous_silicon_height_per_layer_um = 0.6#0.64
spacer_size_um = ( device_size_verical_um / num_vertical_layers ) - amorphous_silicon_height_per_layer_um

simulated_device_voxels_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
simulated_device_voxels_vertical = 2 + int(device_size_verical_um / mesh_spacing_um)

device_voxels_lateral = 2 + int(device_size_lateral_um / design_spacing_um)
# device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_vertical = 2 + int(device_size_verical_um / design_spacing_um)
# device_voxels_vertical = 1 + int(device_size_verical_um / mesh_spacing_um)
spacer_size_voxels = int(spacer_size_um / design_spacing_um)

device_vertical_maximum_um = device_size_verical_um
device_vertical_minimum_um = 0

#
# Spectral
#
lambda_min_um = 0.9
lambda_max_um = 1.6

num_bands = 3
num_points_per_band = 8
num_design_frequency_points = num_bands * num_points_per_band
num_eval_frequency_points = 60

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)

#
# Fabrication Constraints
#
# min_feature_size_um = 3 * mesh_spacing_um
# min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
# blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )
# blur_half_width_voxels = 0

#
# FDTD
#
vertical_gap_size_um = lambda_max_um
lateral_gap_size_um = lambda_max_um

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_simulation_time_fs = 8 * 700

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
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
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
num_epochs = 10#14
num_iterations_per_epoch = 50#35#50#25
binarization_start_epoch = 2#1#4
max_binarize_movement = 0.0025
desired_binarize_change = 0.005

# Probably need to taper this over the epochs!
design_change_start_epoch = 0.05#0.025
design_change_end_epoch = 0.02#0.01

# epoch_start_permittivity_change_max = 0.05#0.1
# epoch_end_permittivity_change_max = 0.01#0.02
# epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

# epoch_start_permittivity_change_min = 0.025#0.05
# epoch_end_permittivity_change_min = 0
# epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

