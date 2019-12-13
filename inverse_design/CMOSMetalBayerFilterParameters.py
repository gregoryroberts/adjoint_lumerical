#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

#
# Files
#
# project_name = 'cmos_metal_etch_passivation_import_mirrored_reflective_no_feature_size_strict_layering_rgb_2x2xtsmc_um'
project_name = 'cmos_2p51p4_etch_passivation_transmissive_nofeature_nobin_nolayer_3freq_rgb_3x3xtsmcx2_um'

#
# Previous mirrored seed point import?
#
use_mirrored_seed_point = False
start_from_previous = False

if use_mirrored_seed_point and start_from_previous:
	print("Error, we can't load both of these seeds at the same time")
	sys.exit(1)

#
# Optical
#
background_index = 1.0
device_background_index = 1.4

design_index_background = 1.4
high_index_backfill = 2.5

min_real_permittivity =  design_index_background**2
# max_real_permittivity = -10
min_real_permittivity = design_index_background**2
max_real_permittivity = high_index_backfill**2

min_imag_permittivity = 0
# max_imag_permittivity = -3
max_imag_permittivity = 0

# init_permittivity_0_1_scale = 0.0
init_permittivity_0_1_scale = 0.5

#
# todo(groberts): this a pretty short focal length.  Where is it with respect to?  Possibly, we should
# push this out to around 1.5um like we have been doing in the past for visible splitters.
#
focal_length_um = 2

#
# Device
#
mesh_spacing_um = 0.025

layer_thicknesses_um = [
	0.05, # leftover from M8
	0.22, 0.095,       # M7
	# here, we are combining the two capping layers from layer N to layer (N - 1) into one since they are very thin
	0.08, 0.22, 0.095, # M6
	0.08, 0.22, 0.095, # M5
	0.08, 0.22, 0.095, # M4
	0.08, 0.22, 0.095, # M3
	0.08, 0.22, 0.095, # M2
	0.08, 0.13, 0.03,  # M1
]

layer_thicknesses_voxels = [ ( 1 + int( x / mesh_spacing_um ) ) for x in layer_thicknesses_um ]

layer_background_index = [
	1.45, # leftover from M8
	design_index_background, design_index_background,       # M7
	1.45, design_index_background, design_index_background, # M6
	1.45, design_index_background, design_index_background, # M5
	1.45, design_index_background, design_index_background, # M4
	1.45, design_index_background, design_index_background, # M3
	1.45, design_index_background, design_index_background, # M2
	1.45, design_index_background, design_index_background # part of M1 until the copper reflecting layer on M1
]

is_layer_designable = [
	False,
	True, False,        # M7
	False, True, False, # M6
	False, True, False, # M5
	False, True, False, # M4
	False, True, False, # M3
	False, True, False, # M2
	False, True, False
]


device_size_lateral_um = 3
# Metal layers from M7 down to M2 (and we will use M6 as a reflector)
# ( 6 * ( 2200 + 950 + 300 + 500 ) - 300 + 300 + 500 ) / 10000 = 2.42
designable_size_vertical_um = np.sum( layer_thicknesses_um )# 2.42
# We will assume just air below this
# bottom_metal_reflector_size_vertical_um = 0.13
# Top dielectric stack size we will not be designing for now because feature
# size is pretty large
# ( 6000 + 4000 + 2500 + 750 + 4000 + 750 + 32300 + 1100 + 7250 + 750 + 7750 + 500 + 6200 + 500 ) / 10000 = 7.435
# top_dielectric_stack_size_vertcial_um = 7.435
# for now, just one of the M8 layers
# top_dielectric_stack_size_vertcial_um = m8_stack_layer_thickness_um[ 0 ]

device_size_vertical_um = designable_size_vertical_um

device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
designable_device_voxels_vertical = 2 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0

#
# Spectral
#
src_lambda_min_um = 0.4
src_lambda_max_um = 0.7

num_bands = 3
num_points_per_band = 1
num_design_frequency_points = num_bands * num_points_per_band

band_offset_um = ( ( 1.0 / ( 1 + num_points_per_band ) ) * ( src_lambda_max_um - src_lambda_min_um ) / num_bands )
lambda_min_um = src_lambda_min_um + band_offset_um
lambda_max_um = src_lambda_max_um - band_offset_um

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)

#
# Fabrication Constraints
#
min_feature_size_um = 0.08
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( (min_feature_size_voxels - 1) / 2. ) )

# num_vertical_layers = 6

#
# FDTD
#
vertical_gap_size_um = 1.0
lateral_gap_size_um = 1.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um# + focal_length_um# - bottom_metal_reflector_size_vertical_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / mesh_spacing_um ) )


# fdtd_simulation_time_fs = 2000
fdtd_simulation_time_fs = 5

#
# Forward Source
#
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_vertical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# We are assuming that the data is organized in order of increasing wavelength (i.e. - blue first, red last)
spectral_focal_plane_map = -( 1 / ( 2 * num_points_per_band ) ) * np.ones( num_adjoint_sources, num_design_frequency_points )
spectral_focal_plane_map[ 0, 0 : num_points_per_band ] = 1 / num_points_per_band
spectral_focal_plane_map[ 1, num_points_per_band : 2 * num_points_per_band ] = 1 / num_points_per_band
spectral_focal_plane_map[ 2, 2 * num_points_per_band : 3 * num_points_per_band ] = 1 / num_points_per_band
spectral_focal_plane_map[ 3, num_points_per_band : 2 * num_points_per_band ] = 1 / num_points_per_band




#
# Adjoint sources
#
adjoint_vertical_um = designable_device_vertical_minimum_um - focal_length_um
num_focal_spots = 4
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]

#
# Optimization
#
num_epochs = 1
num_iterations_per_epoch = 100
start_epoch = 0

use_fixed_step_size = True
fixed_step_size =  0.01 * 3 / 2

epoch_start_permittivity_change_max = 0.1
epoch_end_permittivity_change_max = 0.02
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.05
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

