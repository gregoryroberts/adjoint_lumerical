#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

import FreeOptimizationMultiDevice
import OptimizationLayersSpacersMultiDevice
import OptimizationLayersSpacersGlobalBinarization2DMultiDevice

import FreeBayerFilter2D

# 1. Shorten focal length?
# 2. Strehl ratio? Or at least divide by the transmission total on the focal plane to normalize?
# 3. Antireflection coating on top of GSST?
# 4. Illuminate from the bottom so that the GSST is on the bottom of the structure?
# 5. Check the optimization to make sure the computed gradient is correct!

# todo( gdroberts ): should we also save out the optimization file along with the parameters?


#
# Files
#
# project_name = 'cmos_dielectric_2d_switchable_color_polarization_lossless_gsst_8xtsmc_um_focal_6p75um'
# project_name = 'cmos_dielectric_2d_switchable_color_polarization_8xtsmc_um_focal_6p75um'
# project_name = 'cmos_dielectric_2d_switchable_color_polarization_lossless_high_index_layer_spectral_8xtsmc_um_focal_6p75um_test'
project_name = 'cmos_dielectric_2d_switchable_color_dual_pol_6xtsmc_um_normalized_overlaps_transmit_focus_single_angle_isolated_addarc_v3'

is_lumerical_version_2020a = False#True

#
# Spectral
#
num_bands = 2
num_points_per_band = 10

lambda_min_um = 0.9#1.2
lambda_max_um = 1.4#1.7

num_design_frequency_points = num_bands * num_points_per_band
num_wavelengths = num_design_frequency_points
num_eval_frequency_points = 1 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)


#
# Optical
#
# todo: the side background index shouldn't be TiO2 because that part will not be etched away!  Go back to oxide!
# Maybe force binarize more slowly.  Consider just optimizing for Strell ratio if we aren't going to get high transmisison anyway.
# Check binarized device and look at feature sizes!
# For now, let's let the high index be TiO2.  It looks like it is transparent out into the infrared
background_index = 1.0
design_index_background = 1.35
device_background_index = 1.35
air_backfill = 1.0
high_index_backfill = 2.5

min_real_permittivity = design_index_background**2
# max_real_permittivity = high_index_backfill**2
max_real_permittivity = air_backfill**2

min_imag_permittivity = 0
max_imag_permittivity = 0

min_real_index = np.sqrt( min_real_permittivity )
max_real_index = np.sqrt( max_real_permittivity )

permittivity_bounds = np.array( [ min_real_permittivity + 1j * min_imag_permittivity, max_real_permittivity + 1j * max_imag_permittivity ], dtype=np.complex )

init_permittivity_0_1_scale = 0.5

focal_length_um = 0 * 6.75

#
# Device
#
# We should override the mesh in the GSST though because that has very high index.  Or, potentially,
# let's just override the mesh in our device region that we can control.
mesh_spacing_um = 0.05#0.1
# Set the design mesh spacing to that of the design rules
lsf_mesh_spacing_um = 0.05#0.1

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

device_size_lateral_um = 4 * lambda_max_um
designable_size_vertical_um = np.sum( layer_thicknesses_um )
bottom_metal_absorber_size_vertical_um = 0

device_size_verical_um = designable_size_vertical_um

bottom_metal_absorber_size_vertical_voxels = 1 + int( bottom_metal_absorber_size_vertical_um / mesh_spacing_um )

opt_device_voxels_lateral = 1 + int( device_size_lateral_um / lsf_mesh_spacing_um )
device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
designable_device_voxels_vertical = 1 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0


layer_start_coordinates_um = np.zeros( len( layer_thicknesses_um ) )
layer_start_coordinates_um[ 0 ] = designable_device_vertical_maximum_um - layer_thicknesses_um[ 0 ]
for layer_idx in range( 1, len( layer_thicknesses_um ) ):
    layer_start_coordinates_um[ layer_idx ] = layer_start_coordinates_um[ layer_idx - 1 ] - layer_thicknesses_um[ layer_idx ]


#
# GSST Information
#
# Should we explicitly re-mesh this part?

# amorphous
gsst_min_n = 3.3
gsst_min_k = 0.0

# crystalline
gsst_max_n = 5.0
gsst_max_k = 0.2

gsst_num_states = 2
gsst_n_states =[ gsst_min_n, gsst_max_n ]
gsst_k_states = [ gsst_min_k, gsst_max_k ]

gsst_thickness_um = 0.2
gsst_min_y_um = designable_device_vertical_maximum_um
gsst_max_y_um = gsst_min_y_um + gsst_thickness_um


arc_thickness_um = 0.2
arc_index_mgf2 = 1.37


#
# Put small spacer here
#
# putting the cavity there really enhanced the output (at least for one frequency)
# decently broadband too.. interesting, wonder if we can use/understand this effect!
# also, investigate why the higher index with the mesh override region seemed to be
# giving inaccurate gradients or simulations?  Need to investigate if this was related
# to an interpolation or some weird simulation thing
# also, why did it only split one color and then reject the other ones when you had the
# small cavity
# gsst_max_y_um = designable_device_vertical_minimum_um# - 0.2
# gsst_min_y_um = gsst_max_y_um - gsst_thickness_um

#
# Fabrication Constraints
#
# We have set the design (and simulation) mesh to be at the feature size requirement so for now, we don't need to
# explicitly add any additional feature size requirements.

#
# Optimization Stages
#

device_size_um = np.array( [ device_size_lateral_um, device_size_verical_um ] )

num_free_iterations = 150#25
num_free_epochs = 1
max_change_per_iter_free = 0.015 * 2 / 3
free_optimization_seed = init_permittivity_0_1_scale * np.ones( ( 2, 2 ) )
free_optimization_file_prefix = "free_"

def generate_full_bayer_with_blurs():
    def bayer_filter_create( size_voxels, num_layers ):
        return FreeBayerFilter2D.FreeBayerFilter2D(
            size_voxels, permittivity_bounds, 0.0, num_layers )

    return bayer_filter_create

free_optimization_creator_fns = [
    generate_full_bayer_with_blurs()
]

# np.random.seed( 123123 )
# free_optimization_seed = np.random.random( ( 10, 10 ) )

free_optimization = FreeOptimizationMultiDevice.FreeOptimizationMultiDevice(
    num_free_iterations, num_free_epochs, max_change_per_iter_free,
    device_size_um, permittivity_bounds,
    lsf_mesh_spacing_um, free_optimization_seed,
    free_optimization_creator_fns,
    free_optimization_file_prefix,
    len( free_optimization_creator_fns ) )

num_free_in_layers_iterations = 25
num_free_in_layers_epochs = 1
max_change_per_iter_free_in_layers = 0.015
free_in_layers_file_prefix = "free_in_layers_"


def generate_free_layers_bayer_with_blurs():
    def bayer_filter_create( layer_size_voxels, num_internal_layers, bayer_idx ):
        return FreeBayerFilter2D.FreeBayerFilter2D(
            layer_size_voxels, permittivity_bounds, init_permittivity_0_1_scale, num_internal_layers )

    return bayer_filter_create

free_in_layers_creator_fns = [
    generate_free_layers_bayer_with_blurs()
]

def density_filter_update( bayer_filter, gradient_real, gradient_imag, lsf_gradient_real, lsf_gradient_imag, epoch, iteration, max_abs_gradient_step, max_abs_lsf_gradient_step, step_size ):
    adjust_step_size = step_size / ( max_abs_gradient_step + np.finfo(np.float).eps )
    bayer_filter.step( gradient_real, gradient_imag, adjust_step_size )

free_in_layers = OptimizationLayersSpacersMultiDevice.OptimizationLayersSpacersMultiDevice(
    num_free_in_layers_iterations, num_free_in_layers_epochs, max_change_per_iter_free_in_layers,
    device_size_um, lsf_mesh_spacing_um,
    layer_start_coordinates_um, layer_thicknesses_um, is_layer_designable, layer_background_index, background_index,
    free_in_layers_creator_fns,
    free_in_layers_file_prefix, len( free_in_layers_creator_fns ) )


num_density_layered_iterations = 100
num_density_layered_epochs = 1
max_change_per_iter_in_density_layered = 0.01
density_layered_file_prefix = "density_layered_"

def generate_layered_bayer_with_blurs():
    def bayer_filter_create( layer_size_voxels, num_internal_layers, bayer_idx ):
        single_layer = 1
        return FreeBayerFilter2D.FreeBayerFilter2D(
            layer_size_voxels, permittivity_bounds, init_permittivity_0_1_scale, single_layer )

    return bayer_filter_create


layered_creator_fns = [
    generate_layered_bayer_with_blurs()
]

density_layered = OptimizationLayersSpacersMultiDevice.OptimizationLayersSpacersMultiDevice(
    num_density_layered_iterations, num_density_layered_epochs, max_change_per_iter_in_density_layered,
    device_size_um, lsf_mesh_spacing_um,
    layer_start_coordinates_um, layer_thicknesses_um, is_layer_designable, layer_background_index, background_index,
    layered_creator_fns,
    density_layered_file_prefix,
    len( layered_creator_fns ) )

#
# Things to try:
# Shorten focal length
# Robustness based design (instead of abrupt feature/gap removal - make sure that is operating how you want it to also)
# Change lower index to air for the backfilled design where we etch oxide out first
#

num_density_layered_binarize_iterations = 100
num_density_layered_binarize_epochs = 1
binarize_max_movement = 3 * 0.005 * 3
binarize_desired_change = 0.0025 * 4
# I don't think this does anything right now
# binarization_cutoff = 0.98
density_layered_binarize_file_prefix = "density_layered_binarize_"

binarize_middle_device = 0#1

density_layered_binarize = OptimizationLayersSpacersGlobalBinarization2DMultiDevice.OptimizationLayersSpacersGlobalBinarization2DMultiDevice(
    num_density_layered_binarize_iterations, num_density_layered_binarize_epochs, binarize_max_movement, binarize_desired_change,
    device_size_um, lsf_mesh_spacing_um,
    layer_start_coordinates_um, layer_thicknesses_um, is_layer_designable, layer_background_index, background_index,
    layered_creator_fns,
    density_layered_binarize_file_prefix,
    len( layered_creator_fns ), binarize_middle_device )


optimization_stages = [
    free_optimization,
    free_in_layers,
    density_layered,
    density_layered_binarize
]

def convert_free_to_free_in_layers( filebase ):
    free_in_layers.convert_permittivity_to_density(
        free_optimization.assemble_index( binarize_middle_device ),
        min_real_index, max_real_index
    )

def convert_free_in_layers_to_density_layered( filebase ):
    density_layered.convert_permittivity_to_layered_density(
        free_in_layers.assemble_index( binarize_middle_device ),
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
init_optimization_state = 0
init_optimization_epoch = 0
num_optimization_states = len( optimization_stages )

eval_optimization_state = 3
eval_optimization_epoch = 8#7#5
eval_device_idx = 0


#
# FDTD
#
vertical_gap_size_um = 5.0 * lambda_max_um
lateral_gap_size_um = 8.0 * lambda_max_um

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um + bottom_metal_absorber_size_vertical_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -bottom_metal_absorber_size_vertical_um - vertical_gap_size_um - focal_length_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / mesh_spacing_um ) )

fdtd_simulation_time_fs = 10 * 2000

reflected_focal_length_um = 0.5 * vertical_gap_size_um
max_intensity_by_wavelength = (device_size_lateral_um * 1.02)**2 / (reflected_focal_length_um**2 * lambda_values_um**2)


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
# We are assuming that the data is organized in order of increasing wavelength (i.e. - 'blue' first, 'red' last)
spectral_focal_plane_map = [
    [ 0, num_points_per_band ],
    [ num_points_per_band, 2 * num_points_per_band ]
]

spectral_map = [ 0, num_points_per_band ]

#
# Adjoint sources
#
# This seems like a long focal length
adjoint_vertical_um = -focal_length_um
num_focal_spots = 2
num_adjoint_sources = num_focal_spots

adjoint_x_positions_um = [ -device_size_lateral_um / 4., device_size_lateral_um / 4. ]

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

reflection_monitor_aperture_um = 1.1 * device_size_lateral_um

#
# As a first pass, we will not correct for the spread of angles associated with the broadband
# source and angled injection - each frequency will have different angels they are optimized for
# and a different spread because they will all coincide at normal incidence
#
# optimization_angles_mid_frequency_degrees = [ 0, 4, 8, 12 ]
optimization_angles_mid_frequency_degrees = [ 0 ]
num_optimization_angles = len( optimization_angles_mid_frequency_degrees )
#
# This should only be selected if you start with a symmetry device and care about the positive and
# negative angles the same
#
use_y_reflective_symmetry = True

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

