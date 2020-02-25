#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

import FreeOptimization
import OptimizationLayersSpacers
import OptimizationLayersSpacersGlobalBinarization2D

import FreeBayerFilter2D

#
# Files
#
project_name = 'cmos_dielectric_2d_3focal_layered_higherindex_p22layers_rbg_expl_reject_dual_pol_multistage_force_bin_design_mesh_longer_focal_4xtsmc_um'

#
# Optical
#
background_index = 1.0
# device_background_index = 1.35
design_index_background = 1.0#1.35
device_background_index = 2.5
high_index_backfill = 2.5

min_real_permittivity = design_index_background**2
max_real_permittivity = high_index_backfill**2

min_imag_permittivity = 0
max_imag_permittivity = 0

min_real_index = np.sqrt( min_real_permittivity )
max_real_index = np.sqrt( max_real_permittivity )

permittivity_bounds = np.array( [ min_real_permittivity + 1j * min_imag_permittivity, max_real_permittivity + 1j * max_imag_permittivity ], dtype=np.complex )

init_permittivity_0_1_scale = 0.5

lsf_start_step_size = 0.1
lsf_end_step_size = 0.05

focal_length_um = 2 * 1.5


#
# Device
#
mesh_spacing_um = 0.025
lsf_mesh_spacing_um = 0.005

design_layer_thickness_um = 0.22

layer_thicknesses_um = [
	design_layer_thickness_um, 0.095,       # M7
	# here, we are combining the two capping layers from layer N to layer (N - 1) into one since they are very thin
	0.08, design_layer_thickness_um, 0.095, # M6
	0.08, design_layer_thickness_um, 0.095, # M5
	0.08, design_layer_thickness_um, 0.095, # M4
	0.08, design_layer_thickness_um, 0.095, # M3
	0.08, design_layer_thickness_um, 0.095, # M2
	0.08, 0.13, 0.03,                       # M1
]

layer_thicknesses_voxels = [ ( 1 + int( x / mesh_spacing_um ) ) for x in layer_thicknesses_um ]

layer_background_index = [
	design_index_background, design_index_background,       # M7
	1.45, design_index_background, design_index_background, # M6
	1.45, design_index_background, design_index_background, # M5
	1.45, design_index_background, design_index_background, # M4
	1.45, design_index_background, design_index_background, # M3
	1.45, design_index_background, design_index_background, # M2
	1.45, design_index_background, design_index_background # part of M1 until the copper reflecting layer on M1
]

is_layer_designable = [
	True, False,        # M7
	False, True, False, # M6
	False, True, False, # M5
	False, True, False, # M4
	False, True, False, # M3
	False, True, False, # M2
	False, True, False  # M1
]

device_size_lateral_um = 4.0
designable_size_vertical_um = np.sum( layer_thicknesses_um )
bottom_metal_absorber_size_vertical_um = 0

device_size_verical_um = designable_size_vertical_um

bottom_metal_absorber_size_vertical_voxels = 1 + int( bottom_metal_absorber_size_vertical_um / mesh_spacing_um )

opt_device_voxels_lateral = 1 + int( device_size_lateral_um / lsf_mesh_spacing_um )
device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
designable_device_voxels_vertical = 2 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0


layer_start_coordinates_um = np.zeros( len( layer_thicknesses_um ) )
layer_start_coordinates_um[ 0 ] = designable_device_vertical_maximum_um - layer_thicknesses_um[ 0 ]
for layer_idx in range( 1, len( layer_thicknesses_um ) ):
    layer_start_coordinates_um[ layer_idx ] = layer_start_coordinates_um[ layer_idx - 1 ] - layer_thicknesses_um[ layer_idx ]


#
# Fabrication Constraints
#

# Set at feature size requirement and use for gap now (we can hopefully touch-up with level set at the end)
control_size_k_vectors_um = 0.10
control_size_k_vectors_voxels = int( control_size_k_vectors_um / lsf_mesh_spacing_um )

control_feature_size_m1_um = 1.2 * 0.09
control_feature_size_m1_voxels = int( control_feature_size_m1_um / lsf_mesh_spacing_um )

control_fetaure_size_um = 1.2 * 0.10
control_fetaure_size_voxels = int( control_fetaure_size_um / lsf_mesh_spacing_um )

control_gap_size_m1_um = 1.2 * 0.09
control_gap_size_m1_voxels = int( control_gap_size_m1_um / lsf_mesh_spacing_um )

control_gap_size_um = 1.2 * 0.10
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



#
# Optimization Stages
#

device_size_um = np.array( [ device_size_lateral_um, device_size_verical_um ] )

num_free_iterations = 25
num_free_epochs = 1
max_change_per_iter_free = 0.015
free_optimization_seed = init_permittivity_0_1_scale * np.ones( ( 2, 2 ) )
free_optimization_file_prefix = "free_"

free_optimization = FreeOptimization.FreeOptimization(
    num_free_iterations, num_free_epochs, max_change_per_iter_free,
    device_size_um, permittivity_bounds,
    lsf_mesh_spacing_um, free_optimization_seed,
    free_optimization_file_prefix )


num_free_in_layers_iterations = 25
num_free_in_layers_epochs = 1
max_change_per_iter_free_in_layers = 0.015
free_in_layers_file_prefix = "free_in_layers_"

def free_in_layers_bayer_create( layer_size_voxels, num_internal_layers, bayer_idx ):
    return FreeBayerFilter2D.FreeBayerFilter2D(
        layer_size_voxels, permittivity_bounds, init_permittivity_0_1_scale, num_internal_layers )

def density_filter_update( bayer_filter, gradient_real, gradient_imag, lsf_gradient_real, lsf_gradient_imag, epoch, iteration, max_abs_gradient_step, max_abs_lsf_gradient_step, max_fab_gradient_step, step_size ):
    adjust_step_size = step_size / ( max_abs_gradient_step + np.finfo(np.float).eps )
    bayer_filter.step( gradient_real, gradient_imag, adjust_step_size )

free_in_layers = OptimizationLayersSpacers.OptimizationLayersSpacers(
    num_free_in_layers_iterations, num_free_in_layers_epochs, max_change_per_iter_free_in_layers,
    device_size_um, lsf_mesh_spacing_um,
    layer_start_coordinates_um, layer_thicknesses_um, is_layer_designable, layer_background_index, background_index,
    free_in_layers_bayer_create, density_filter_update,
    free_in_layers_file_prefix )


num_density_layered_iterations = 50
num_density_layered_epochs = 1
max_change_per_iter_in_density_layered = 0.01
density_layered_file_prefix = "density_layered_"

def density_layered_bayer_create( layer_size_voxels, num_internal_layers, bayer_idx ):
    single_layer = 1
    return FreeBayerFilter2D.FreeBayerFilter2D(
        layer_size_voxels, permittivity_bounds, init_permittivity_0_1_scale, single_layer )

density_layered = OptimizationLayersSpacers.OptimizationLayersSpacers(
    num_density_layered_iterations, num_density_layered_epochs, max_change_per_iter_in_density_layered,
    device_size_um, lsf_mesh_spacing_um,
    layer_start_coordinates_um, layer_thicknesses_um, is_layer_designable, layer_background_index, background_index,
    density_layered_bayer_create, density_filter_update,
    density_layered_file_prefix )

#
# Things to try:
# Shorten focal length
# Robustness based design (instead of abrupt feature/gap removal - make sure that is operating how you want it to also)
# Change lower index to air for the backfilled design where we etch oxide out first
#

num_density_layered_binarize_iterations = 200#400
num_density_layered_binarize_epochs = 2#3
iteration_erode_dilate_start = 0
iteration_erode_dilate_gap = 30#40
control_feature = 0.08 / lsf_mesh_spacing_um
num_iterations_of_erosion_dilation = int( np.floor( ( control_feature - ( 1 - ( control_feature % 2 ) ) ) / 2. ) )
print( "num iterations erosion dilation = " + str( num_iterations_of_erosion_dilation ) )
binarize_max_movement = 0.005 * 4
binarize_desired_change = 0.0025 * 3
binarization_cutoff = 0.98
density_layered_binarize_file_prefix = "density_layered_binarize_"

def density_layered_binarize_bayer_create( layer_size_voxels, num_internal_layers, bayer_idx ):
    single_layer = 1
    return FreeBayerFilter2D.FreeBayerFilter2D(
        layer_size_voxels, permittivity_bounds, init_permittivity_0_1_scale, single_layer )

density_layered_binarize = OptimizationLayersSpacersGlobalBinarization2D.OptimizationLayersSpacersGlobalBinarization2D(
    num_density_layered_binarize_iterations, num_density_layered_binarize_epochs, binarize_max_movement, binarize_desired_change,
    iteration_erode_dilate_start, iteration_erode_dilate_gap, num_iterations_of_erosion_dilation,
    device_size_um, lsf_mesh_spacing_um,
    layer_start_coordinates_um, layer_thicknesses_um, is_layer_designable, layer_background_index, background_index,
    density_layered_binarize_bayer_create, None,
    density_layered_binarize_file_prefix )



optimization_stages = [
    free_optimization,
    free_in_layers,
    density_layered,
    density_layered_binarize
]

def convert_free_to_free_in_layers( filebase ):
    free_in_layers.convert_permittivity_to_density(
        free_optimization.assemble_index(),
        min_real_index, max_real_index
    )

def convert_free_in_layers_to_density_layered( filebase ):
    density_layered.convert_permittivity_to_layered_density(
        free_in_layers.assemble_index(),
        min_real_index, max_real_index
    )

def convert_density_layered_to_density_layered_binarize( filebase ):
    density_layered_binarize.load_other_design( density_layered.filename_prefix, filebase, density_layered.num_epochs - 1 )


optimization_conversion_functions = [
    None,
    convert_free_to_free_in_layers,
    convert_free_in_layers_to_density_layered,
    convert_density_layered_to_density_layered_binarize
]

#
# Optimization Starting/Restarting Point
#
init_optimization_state = 0#3
init_optimization_epoch = 0#1
num_optimization_states = len( optimization_stages )

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


fdtd_simulation_time_fs = 3 * 2000

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

# spectral_focal_plane_map = [
# 	[0 * num_points_per_band, 2 * num_points_per_band],
# 	[0, 3 * num_points_per_band],
# 	[num_points_per_band, 3 * num_points_per_band],
# ]
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

