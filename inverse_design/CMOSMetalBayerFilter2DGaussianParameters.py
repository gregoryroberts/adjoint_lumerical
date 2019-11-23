#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

#
# Files
#
project_name = 'cmos_metal_gaussian_single_band_etch_passivation_2d_reflective_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_2focal_8pts_focal1p5um_etch_passivation_2d_reflective_no_feature_size_strict_layering_rgb_4p5xtsmc_um'
# project_name = 'cmos_dielectric_reflective_no_toplayers_3focal_30pts_focal1p5um_etch_passivation_2d_no_feature_size_strict_layering_rgb_3xtsmc_um'

#
# Random Seed
#
# np.random.seed( 123123 )

#
# Optical
#
background_index = 1.0
device_background_index = 1.4
design_index_background = 1.4

min_real_permittivity = design_index_background**2
# max_real_permittivity = -10
# max_real_permittivity = -10
# max_real_permittivity = 2.5
# max_real_permittivity = -18.5
max_real_permittivity = 1

min_imag_permittivity = 0
# max_imag_permittivity = -3
# max_imag_permittivity = -12
# max_imag_permittivity = -12.5
# max_imag_permittivity = -8
max_imag_permittivity = 0

reflector_real_permittivity = 2.5
reflector_imag_permittivity = -12.5

init_permittivity_0_1_scale = 0.5
# init_max_random_0_1_scale = 0.02
# init_permittivity_0_1_scale = 0.5

init_from_old = False
init_from_random = False

if init_from_random:
	np.random.seed( 867886 )

#
# todo(groberts): this a pretty short focal length.  Where is it with respect to?  Possibly, we should
# push this out to around 1.5um like we have been doing in the past for visible splitters.
#
focal_length_um = 1.5

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
	# False,
	False, True, False
]

fix_layer_permittivity_to_reflective = [
	False,
	False, False,        # M7
	False, False, False, # M6
	False, False, False, # M5
	False, False, False, # M4
	False, False, False, # M3
	False, False, False, # M2
	False, False, False   # M1
]

m8_stack_layer_thickness_um = [ 0 ]#[ 0.62 ]
m8_stack_layer_refractive_index = [ 1.45 ]


device_size_lateral_um = 2
# device_size_lateral_um = 3.0
# Metal layers from M7 down to M2 (and we will use M6 as a reflector)
# ( 6 * ( 2200 + 950 + 300 + 500 ) - 300 + 300 + 500 ) / 10000 = 2.42
designable_size_vertical_um = np.sum( layer_thicknesses_um )# 2.42
# We will assume just air below this
# bottom_metal_absorber_size_vertical_um = 1.0
bottom_metal_absorber_size_vertical_um = 0
# Top dielectric stack size we will not be designing for now because feature
# size is pretty large
# ( 6000 + 4000 + 2500 + 750 + 4000 + 750 + 32300 + 1100 + 7250 + 750 + 7750 + 500 + 6200 + 500 ) / 10000 = 7.435
# top_dielectric_stack_size_vertcial_um = 7.435
# for now, just one of the M8 layers
top_dielectric_stack_size_vertcial_um = m8_stack_layer_thickness_um[ 0 ]

device_size_verical_um = top_dielectric_stack_size_vertcial_um + designable_size_vertical_um# + bottom_metal_reflector_size_vertical_um

bottom_metal_absorber_size_vertical_voxels = 1 + int( bottom_metal_absorber_size_vertical_um / mesh_spacing_um )

device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
designable_device_voxels_vertical = 2 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0



#
# copy so the extend calls following do not modify the memory for the m8 stack variable
#
top_dielectric_layer_thickness_um = m8_stack_layer_thickness_um.copy()

top_dielectric_layer_refractice_index = m8_stack_layer_refractive_index.copy()


bottom_metal_absorber_start_um = -bottom_metal_absorber_size_vertical_um
bottom_metal_absorber_end_um = 0

m8_stack_start_um = designable_size_vertical_um# bottom_metal_reflector_end_um + designable_size_vertical_um
m8_stack_end_um = m8_stack_start_um + np.sum( m8_stack_layer_thickness_um )

dielectric_stack_start_um = m8_stack_start_um
dielectric_stack_end_um = m8_stack_end_um

#
# Spectral
#
# lambda_min_um = 0.4
# lambda_max_um = 0.7


num_bands = 2
# num_bands = 2
num_points_per_band = 10

# lambda_min_low_um = 0.45
# lambda_max_low_um = 0.55

# lambda_min_high_um = 0.55
# lambda_max_high_um = 0.65

lambda_min_um = 0.5
lambda_max_um = 0.6

num_design_frequency_points = num_bands * num_points_per_band
num_wavelengths = num_design_frequency_points
num_eval_frequency_points = 1 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
# lambda_values_um_low = np.linspace(lambda_min_low_um, lambda_max_low_um, num_points_per_band)
# lambda_values_um_high = np.linspace(lambda_min_high_um, lambda_max_high_um, num_points_per_band)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)
# max_intensity_by_wavelength_low = (device_size_lateral_um**2) / (lambda_values_um_low**2)
# max_intensity_by_wavelength_high = (device_size_lateral_um**2) / (lambda_values_um_high**2)

#
# Fabrication Constraints
#
min_feature_size_um = 0.08
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( (min_feature_size_voxels - 1) / 2. ) )

#
# FDTD
#
vertical_gap_size_um = 1.0
lateral_gap_size_um = 1.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + 2 * focal_length_um + bottom_metal_absorber_size_vertical_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
# fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um + focal_length_um - bottom_metal_reflector_size_vertical_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um + focal_length_um# + focal_length_um# - bottom_metal_reflector_size_vertical_um
# fdtd_region_minimum_vertical_um = bottom_metal_reflector_start_um - vertical_gap_size_um
fdtd_region_minimum_vertical_um = -bottom_metal_absorber_size_vertical_um - vertical_gap_size_um - focal_length_um# - focal_length_um# bottom_metal_reflector_start_um - vertical_gap_size_um - focal_length_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / mesh_spacing_um ) )


fdtd_simulation_time_fs = 2000

#
# Forward Source
#
lateral_aperture_um = 1.5 * device_size_lateral_um
# src_maximum_vertical_um = m8_stack_end_um + 0.5 * vertical_gap_size_um# + 0.5 * focal_length_um + 0.5 * vertical_gap_size_um
# src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um# bottom_metal_reflector_start_um - 0.5 * vertical_gap_size_um
src_maximum_vertical_um = m8_stack_end_um + 0.5 * vertical_gap_size_um# + 0.75 * focal_length_um# + 0.5 * vertical_gap_size_um# + 0.5 * focal_length_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -bottom_metal_absorber_size_vertical_um - focal_length_um - 0.5 * vertical_gap_size_um# bottom_metal_reflector_start_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# We are assuming that the data is organized in order of increasing wavelength (i.e. - blue first, red last)
# spectral_focal_plane_map = [
# 	[0, num_points_per_band],
# 	[num_points_per_band, 2 * num_points_per_band],
# 	[2 * num_points_per_band, 3 * num_points_per_band],
# ]
spectral_focal_plane_map = [
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
	# [2 * num_points_per_band, 3 * num_points_per_band],
]
#
# Adjoint sources
#
# This seems like a long focal length
adjoint_vertical_um = -focal_length_um# m8_stack_end_um + focal_length_um# + 0.5 * vertical_gap_size_um
# adjoint_vertical_um = -focal_length_um
# num_focal_spots = 3
num_focal_spots = 2
num_adjoint_sources = num_focal_spots
# adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
# adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]

# adjoint_x_positions_um = [ -device_size_lateral_um / 3., 0.0, device_size_lateral_um / 3. ]
adjoint_x_positions_um = [ 0, 0 ]
adjoint_y_positions_um = [ -focal_length_um, designable_device_vertical_maximum_um + focal_length_um ]

#
# Optimization
#
num_epochs = 1
num_iterations_per_epoch = 100
start_epoch = 0

#
# Figure of merit regularization
#
regularization = 0.1
alpha = 5

use_simulated_annealing = False
simulated_annealing_cutoff = 0.025
num_gradient_iterations_fraction = 0.1
num_gradient_iterations = int( num_gradient_iterations_fraction * num_iterations_per_epoch )
simulated_annealing_cutoff_iteration = num_iterations_per_epoch - num_gradient_iterations
temperature_scaling = -np.log( simulated_annealing_cutoff_iteration + 2 ) / ( 2 * np.log( simulated_annealing_cutoff ) )

if ( num_epochs is not 1 ) and use_simulated_annealing:
	print("Error: We are only setting parameters to use simulated annealing for single epoch optimizations right now.")
	sys.exit(1)


use_fixed_step_size = False
# fixed_step_size = 12
fixed_step_size =  8

use_adaptive_step_size = True
desired_max_max_design_change = 0.05
desired_min_max_design_change = 0.005
adaptive_step_size = 5
# adaptive_step_size = 50 * 0.01 * 3 / 2
# adaptive_step_size = 15 * 0.01 * 3 / 2
# adaptive_step_size = .25 * 7.5 * 15 * 0.01 * 3 / 2


epoch_start_permittivity_change_max = 0.1
epoch_end_permittivity_change_max = 0.02
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.05
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

