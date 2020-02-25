#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

#
# Files
#
# project_name = 'cmos_dielectric_2d_3focal_layered_highindex_p22layers_gbr_dual_pol_force_binarize_andsigmoid_fullopt_denser_mesh_3xtsmc_um'
# project_name = 'cmos_dielectric_2d_3focal_layered_highindex_p22layers_grb_dual_pol_force_binarize_andsigmoid_fullopt_denser_mesh_3xtsmc_um'
# project_name = 'cmos_dielectric_2d_3focal_layered_highindex_p22layers_rgb_dual_pol_force_binarize_andsigmoid_fullopt_denser_mesh_3xtsmc_um'
# project_name = 'cmos_dielectric_2d_3focal_layered_highindex_p22layers_rgb_ez_pol_force_binarize_andsigmoid_fullopt_denser_mesh_3xtsmc_um'
project_name = 'cmos_dielectric_2d_3focal_layered_highindex_p22layers_rbg_dual_pol_force_binarize_k_space_offset_andsigmoid_fullopt_denser_mesh_3xtsmc_um'

#
# Random Seed
#
# np.random.seed( 123123 )

#
# Optical
#
background_index = 1.0
device_background_index = 1.35
design_index_background = 1.35
high_index_backfill = 2.5

min_real_permittivity = design_index_background**2
max_real_permittivity = high_index_backfill**2

min_imag_permittivity = 0
max_imag_permittivity = 0

init_max_random_0_1_scale = 0.25
init_permittivity_0_1_scale = 0.5

init_from_old = False
eval_binarized = False
init_from_random = False
eval_rebinarized = False
init_binarized_seed = False
optimize_via_lsf = True
optimize_only_fabrication = False
optimize_fom_and_fab = True

fom_start_weight = 1.0
fom_end_weight = 0.1

lsf_start_step_size = 0.1#0.08
lsf_end_step_size = 0.05#0.01


if init_from_random:
	np.random.seed( 867886 )

focal_length_um = 1.5

#
# Device
#
mesh_spacing_um = 0.025
lsf_mesh_spacing_um = 0.005

design_layer_thickness_um = 0.22

layer_thicknesses_um = [
	# 0.05, # leftover from M8
	design_layer_thickness_um, 0.095,       # M7
	# here, we are combining the two capping layers from layer N to layer (N - 1) into one since they are very thin
	0.08, design_layer_thickness_um, 0.095, # M6
	0.08, design_layer_thickness_um, 0.095, # M5
	0.08, design_layer_thickness_um, 0.095, # M4
	0.08, design_layer_thickness_um, 0.095, # M3
	0.08, design_layer_thickness_um, 0.095, # M2
	0.08, 0.13, 0.03,  # M1
]

layer_thicknesses_voxels = [ ( 1 + int( x / mesh_spacing_um ) ) for x in layer_thicknesses_um ]

layer_background_index = [
	# 1.45, # leftover from M8
	design_index_background, design_index_background,       # M7
	1.45, design_index_background, design_index_background, # M6
	1.45, design_index_background, design_index_background, # M5
	1.45, design_index_background, design_index_background, # M4
	1.45, design_index_background, design_index_background, # M3
	1.45, design_index_background, design_index_background, # M2
	1.45, design_index_background, design_index_background # part of M1 until the copper reflecting layer on M1
]

is_layer_designable = [
	# False,
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
	# False,
	False, False,        # M7
	False, False, False, # M6
	False, False, False, # M5
	False, False, False, # M4
	False, False, False, # M3
	False, False, False, # M2
	False, False, False   # M1
]

device_size_lateral_um = 3.0
designable_size_vertical_um = np.sum( layer_thicknesses_um )
bottom_metal_absorber_size_vertical_um = 0

device_size_verical_um = designable_size_vertical_um

bottom_metal_absorber_size_vertical_voxels = 1 + int( bottom_metal_absorber_size_vertical_um / mesh_spacing_um )

opt_device_voxels_lateral = 1 + int( device_size_lateral_um / lsf_mesh_spacing_um )
device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
designable_device_voxels_vertical = 2 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0

#
# Spectral
#
num_bands = 3
num_points_per_band = 15

lambda_min_um = 0.45
lambda_max_um = 0.65

num_design_frequency_points = num_bands * num_points_per_band
num_wavelengths = num_design_frequency_points
num_eval_frequency_points = 1 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um * 1.02)**2 / (focal_length_um**2 * lambda_values_um**2)

#
# Fabrication Constraints
#
min_feature_size_um = 0.08
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( (min_feature_size_voxels - 1) / 2. ) )

# Set at feature size requirement and use for gap now (we can hopefully touch-up with level set at the end)
control_size_k_vectors_um = 0.075
control_size_k_vectors_voxels = int( control_size_k_vectors_um / lsf_mesh_spacing_um )

control_feature_size_m1_um = 1.25 * 0.09
control_feature_size_m1_voxels = int( control_feature_size_m1_um / lsf_mesh_spacing_um )

control_fetaure_size_um = 1.25 * 0.10#1.25 * 0.09
control_fetaure_size_voxels = int( control_fetaure_size_um / lsf_mesh_spacing_um )

control_gap_size_m1_um = 1.25 * 0.09
control_gap_size_m1_voxels = int( control_gap_size_m1_um / lsf_mesh_spacing_um )

control_gap_size_um = 1.25 * 0.10#1.25 * 0.110
control_gap_size_voxels = int( control_gap_size_um / lsf_mesh_spacing_um )

# from top to bottom
metal_layer_feature_sizes = [
	control_fetaure_size_voxels,   # M7
	control_fetaure_size_voxels,   # M6
	control_fetaure_size_voxels,   # M5
	control_fetaure_size_voxels,   # M4
	control_fetaure_size_voxels,   # M3
	control_fetaure_size_voxels,   # M2
	control_feature_size_m1_voxels # M1
]

metal_layer_gap_sizes = [
	control_gap_size_voxels,   # M7
	control_gap_size_voxels,   # M6
	control_gap_size_voxels,   # M5
	control_gap_size_voxels,   # M4
	control_gap_size_voxels,   # M3
	control_gap_size_voxels,   # M2
	control_gap_size_m1_voxels # M1	
]


enforce_layering = True

#
# FDTD
#
vertical_gap_size_um = 1.0
lateral_gap_size_um = 1.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um + bottom_metal_absorber_size_vertical_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -bottom_metal_absorber_size_vertical_um - vertical_gap_size_um - focal_length_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / mesh_spacing_um ) )


fdtd_simulation_time_fs = 2000

#
# Forward Source
#
num_polarizations = 2

lateral_aperture_um = 1.1 * device_size_lateral_um
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
	[2 * num_points_per_band, 3 * num_points_per_band],
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
]
#
# Adjoint sources
#
# This seems like a long focal length
adjoint_vertical_um = -focal_length_um
num_focal_spots = 3
num_adjoint_sources = num_focal_spots

# adjoint_x_positions_um = [ -3 * device_size_lateral_um / 8., -device_size_lateral_um / 8., device_size_lateral_um / 8., 3 * device_size_lateral_um / 8. ]
adjoint_x_positions_um = [ -device_size_lateral_um / 3., 0.0, device_size_lateral_um / 3. ]

#
# Optimization
#
num_epochs = 1#14
num_iterations_per_epoch = 80#400#200# 50
start_epoch = 0
start_binarization_epoch = 4

num_iterations_free_optimize = 100#150
num_iterations_level_set = num_iterations_free_optimize + 60#150#50
num_epochs_level_set = 1#5
# level_set_starting_epoch = 10

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


use_fixed_step_size = True
fixed_step_size = 10 * 1 / 5
max_density_change_epoch_start = 0.05 / ( max_real_permittivity - min_real_permittivity )
max_density_change_epoch_end = 0.005 / ( max_real_permittivity - min_real_permittivity )

use_adaptive_step_size = False
desired_max_max_design_change = 0.05
desired_min_max_design_change = 0.001
adaptive_step_size = 0.3 * 0.01 * 3 / 5

epoch_start_permittivity_change_max = 0.1
epoch_end_permittivity_change_max = 0.02
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.05
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

