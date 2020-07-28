#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np

#
# Files
#
# project_name = 'layered_rgb_5layers_pol_insensitive_2x2x4um_weighting_f1p5'
project_name = 'layered_nir_5layers_pol_insensitive_4x4x5um_weighting_f3'

#
# Optical
#
# averaging between the part where we dip into the infrared/red part of the spectrum
# and where we are in the telecom ranges
#
index_silicon = 0.5 * ( 3.47 + 3.86 )
index_am = 2.4
index_substrate = 1.451
index_sio2 = 1.46
index_air = 1.0
background_index = index_sio2
min_device_index = index_sio2
max_device_index = index_am

# last_layer_permittivities = [ index_tio2**2, index_zep**2 ]
last_layer_permittivities = [ index_sio2**2, index_am**2 ]

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

random_seed = 6235234
init_permittivity_0_1_scale = 0.25

focal_length_um = 4.0
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Device
#
# mesh_spacing_um = 0.02
mesh_spacing_um = 0.04
# For now, we will put the blurring out to 60nm in the filter chain
# design_spacing_um = 0.02
design_spacing_um = 0.04

# device_border_um = 0.12
device_border_um = 0.24
device_border_voxels = int( device_border_um / mesh_spacing_um )

#
# Pesky size to get the number of voxels to be a multiple of 3
#
num_vertical_layers = 5

# device_size_lateral_um = 2.0
device_size_lateral_um = 2.0

# device_size_verical_um = num_vertical_layers * 0.8
device_size_verical_um = num_vertical_layers * 1.0

# device_height_per_layer_um = 0.4
device_height_per_layer_um = 0.5
spacer_size_um = ( device_size_verical_um / num_vertical_layers ) - device_height_per_layer_um

simulated_device_voxels_lateral = int(device_size_lateral_um / mesh_spacing_um)
simulated_device_voxels_vertical = int(device_size_verical_um / mesh_spacing_um)

device_voxels_lateral = int(device_size_lateral_um / design_spacing_um)
device_voxels_vertical = int(device_size_verical_um / design_spacing_um)
spacer_size_voxels = int(spacer_size_um / design_spacing_um)

device_vertical_maximum_um = device_size_verical_um
device_vertical_minimum_um = 0

#
# Spectral
#
# lambda_min_um = 0.4
# lambda_max_um = 0.7
lambda_min_um = 0.8
lambda_max_um = 1.4

num_bands = 3
num_points_per_band = 10
num_design_frequency_points = num_bands * num_points_per_band
num_eval_frequency_points = 60

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)

#
# Fabrication Constraints
#
max_blur_filter_size_um = 0.06
gaussian_blur_size_um = 0.1

# max_blur_filter_half_width = int( ( ( max_blur_filter_size_um / design_spacing_um ) - 1 ) / 2 )
max_blur_filter_half_width = 0#int( ( ( max_blur_filter_size_um / design_spacing_um ) - 1 ) / 2 )
# gaussian_blur_filter_sigma = int( ( ( gaussian_blur_size_um / design_spacing_um ) - 1 ) / 2 )
gaussian_blur_filter_sigma = ( ( ( gaussian_blur_size_um / design_spacing_um ) - 1 ) / 2 )

#
# FDTD
#
vertical_gap_size_um = lambda_max_um
lateral_gap_size_um = 0.5 * device_size_lateral_um

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

import_region_voxels_lateral = 1 + int( fdtd_region_size_vertical_um / design_spacing_um )

fdtd_simulation_time_fs = 8 * 700

#
# Forward Source
#
lateral_aperture_um = device_size_lateral_um + 0.5 * lateral_gap_size_um
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
# binarization_start_epoch = 2#1#4
# max_binarize_movement = 0.0025
# desired_binarize_change = 0.005

# Probably need to taper this over the epochs!
design_change_start_epoch = 0.05 / 5#0.025
design_change_end_epoch = 0.02 / 5#0.01

# epoch_start_permittivity_change_max = 0.05#0.1
# epoch_end_permittivity_change_max = 0.01#0.02
# epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

# epoch_start_permittivity_change_min = 0.025#0.05
# epoch_end_permittivity_change_min = 0
# epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

