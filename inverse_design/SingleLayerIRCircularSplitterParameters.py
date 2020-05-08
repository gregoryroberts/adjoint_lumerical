#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
# import sigmoid

#
# Files
#
project_name = 'single_layer_circular_pol_splitter_3p2x1p6xp64um_f9p6'

#
# Optical
#
# averaging between the part where we dip into the infrared/red part of the spectrum
# and where we are in the telecom ranges
#
index_silicon = 0.5 * ( 3.47 + 3.86 )
index_air = 1.0
index_sio2 = 1.45
background_index = index_air
min_device_index = index_air
max_device_index = index_silicon

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.25

# focal_length_um = 9.6
focal_length_um = 3.2

#
# Device
#
# mesh_spacing_um = 0.03
mesh_spacing_um = 0.032

# device_width_um = 9.6
# device_height_um = 4.8

device_width_um = 3.2
device_height_um = 1.6
# device_width_um = 2.4
# device_height_um = 1.2

device_size_vertical_um = 0.64

device_width_voxels = 2 + int( device_width_um / mesh_spacing_um )
device_height_voxels = 2 + int( device_height_um / mesh_spacing_um )
device_voxels_vertical = 2 + int( device_size_vertical_um / mesh_spacing_um )

device_vertical_minimum_um = 0
device_vertical_maximum_um = device_size_vertical_um

#
# Spectral
#
lambda_mid_um = 0.85
lambda_min_um = 0.825
lambda_max_um = 0.875

lambda_src_min_um = lambda_mid_um - 0.2
lambda_src_max_um = lambda_mid_um + 0.2

num_design_frequency_points = 5
num_eval_frequency_points = 11

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_design_frequency_points )

effective_focal_length_sq = focal_length_um**2 + ( device_width_um / 4. )**2

max_intensity_by_wavelength = ( device_width_um * device_height_um )**2 / ( effective_focal_length_sq * lambda_values_um**2 )

#
# Fabrication Constraints
#
min_feature_size_um = 3 * mesh_spacing_um
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )

#
# FDTD
#
vertical_gap_size_um = 2.0
lateral_gap_size_um = 2.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_width_um = 2 * lateral_gap_size_um + device_width_um
fdtd_region_height_um = 2 * lateral_gap_size_um + device_height_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_width_voxels = int( np.ceil(fdtd_region_width_um / mesh_spacing_um) )
fdtd_region_minimum_height_voxels = int( np.ceil(fdtd_region_height_um / mesh_spacing_um) )

fdtd_simulation_time_fs = 4 * 700

#
# Forward Source
#
lateral_aperture_width_um = device_width_um + 0.2
lateral_aperture_height_um = device_height_um + 0.2
src_maximum_vertical_um = device_size_vertical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }

jones_polarizations = [
	( 1. / np.sqrt( 2. ) ) * np.array( [ 1, 1j ] ),
	( 1. / np.sqrt( 2. ) ) * np.array( [ 1, -1j ] )
]

#
# Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 2
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [ device_width_um / 4., -device_width_um / 4. ]
adjoint_y_positions_um = [ 0, 0 ]

#
# Optimization
#
start_epoch = 0
num_epochs = 5
num_iterations_per_epoch = 60
binarization_start_epoch = 2
max_binarize_movement = 0.01
desired_binarize_change = 0.005 / 2

epoch_start_permittivity_change_max = 0.025
epoch_end_permittivity_change_max = 0.01
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.01
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

