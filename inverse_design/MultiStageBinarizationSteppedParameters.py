#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

import FreeOptimizationMultiDevice
import OptimizationLayersSpacersMultiDevice
import OptimizationLayersSpacersGlobalBinarization2DMultiDevice

# import FreeBayerFilterWithBlur2D
# import FreeBayerFilterDualPhaseBlur2D

#
# Files
#
# project_name = 'cmos_dielectric_2d_3focal_layered_higherindex_p22layers_rbg_dual_pol_multistage_force_bin_feature_sensitivity_design_mesh_4xtsmc_um_focal_3um'
# project_name = 'cmos_dielectric_2d_3focal_layered_higherindex_p22layers_rbg_dual_pol_multistage_coarse_mesh_80nm_4xtsmc_um_focal_3um'
# project_name = 'cmos_dielectric_2d_3focal_layered_higherindex_p22layers_rbg_dual_pol_multistage_dual_phase_blur_4xtsmc_um_focal_3um'
project_name = 'cmos_dielectric_2d_3focal_layered_higherindex_p22layers_rbg_dual_pol_multistage_blocked_4xtsmc_um_focal_3um'

#
# Optical
#
# todo: the side background index shouldn't be TiO2 because that part will not be etched away!  Go back to oxide!
# Maybe force binarize more slowly.  Consider just optimizing for Strell ratio if we aren't going to get high transmisison anyway.
# Check binarized device and look at feature sizes!
background_index = 1.0
design_index_background = 2.4
device_background_index = 2.4
high_index_backfill = 1.0
substrate_index = 1.45

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

# For the focal length, should we put these things in a near field regime.  In the nature photonics, with very
# simple scatterers made of SiN, they can get fairly impressive splitting.  The "focal length" seems to be smaller
# there (not sure if it would be considered near field or not, but it would be right on the edge).
focal_length_um = 2 * 1.5


#
# Device
#
mesh_spacing_um = 0.025
# This will change when we LSF optimize at the end at 5nm but for now, get a coarse starting point!
lsf_mesh_spacing_um = 0.1#0.005
blocked_feature_spacing_um = 0.1
# lsf_mesh_spacing_um = 0.08
lsf_step_size_factor = lsf_mesh_spacing_um / 0.005

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
	design_index_background, design_index_background, design_index_background, # M6
	design_index_background, design_index_background, design_index_background, # M5
	design_index_background, design_index_background, design_index_background, # M4
	design_index_background, design_index_background, design_index_background, # M3
	design_index_background, design_index_background, design_index_background, # M2
	design_index_background, design_index_background, design_index_background # part of M1 until the copper reflecting layer on M1
]

# is_layer_designable = [
# 	True, False,        # M7
# 	False, True, False, # M6
# 	False, True, False, # M5
# 	False, True, False, # M4
# 	False, True, False, # M3
# 	False, True, False, # M2
# 	False, True, False  # M1
# ]

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


#
# Things to try:
# Shorten focal length
# Robustness based design (instead of abrupt feature/gap removal - make sure that is operating how you want it to also)
# Change lower index to air for the backfilled design where we etch oxide out first
#

device_size_um = np.array( [ device_size_lateral_um, device_size_verical_um ] )

optimization_stages = []
optimization_conversion_functions = []

num_density_layered_binarize_iterations = 50
num_density_layered_binarize_epochs = 4
binarize_max_movement = 0.005
binarize_desired_change = 0.005
binarize_middle_device = 0

num_opt_layers = 7
for opt_layer_idx in range( 0, num_opt_layers ):
    opt_layer_designability = [
        False, False,        # M7
        False, False, False, # M6
        False, False, False, # M5
        False, False, False, # M4
        False, False, False, # M3
        False, False, False, # M2
        False, False, False  # M1
    ]

    opt_layer_designability[ 3 * opt_layer_idx ] = True

    density_layered_binarize_file_prefix = "density_layered_binarize_" + str( opt_layer_idx ) + "_"

    density_layered_binarize = OptimizationLayersSpacersGlobalBinarization2DMultiDevice.OptimizationLayersSpacersGlobalBinarization2DMultiDevice(
        num_density_layered_binarize_iterations, num_density_layered_binarize_epochs, binarize_max_movement, binarize_desired_change,
        device_size_um, lsf_mesh_spacing_um,
        layer_start_coordinates_um, layer_thicknesses_um, opt_layer_designability, layer_background_index, background_index,
        layered_creator_fns,
        density_layered_binarize_file_prefix,
        len( layered_creator_fns ), binarize_middle_device )

    optimization_stages.append( density_layered_binarize )

    def convert_density_layered_binarize_to_density_layered_binarize( filebase ):
        density_layered_binarize.load_other_design( density_layered_binarize.filename_prefix, filebase, density_layered_binarize.num_epochs - 1 )

    optimization_conversion_functions.append( convert_density_layered_binarize_to_density_layered_binarize )



#
# Optimization Starting/Restarting Point
#
init_optimization_state = 0
init_optimization_epoch = 0
num_optimization_states = len( optimization_stages )

eval_optimization_state = num_opt_layers - 1
eval_optimization_epoch = num_density_layered_binarize_epochs - 1
eval_device_idx = 0

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

