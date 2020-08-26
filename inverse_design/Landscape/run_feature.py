#
# Housekeeping
#
import os
import shutil
import sys

#
# Math
#
import numpy as np
import math

#
# Plotting
#
import matplotlib as mpl
import matplotlib.pylab as plt

#
# Optimizer
#
import ColorSplittingOptimization2D

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )

if ( max_index > 3.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

mesh_size_nm = 8
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.55
num_lambda_values = 8

bandwidth_um = lambda_max_um - lambda_min_um
exclusion_um = 0.030
modified_bandwidth_um = bandwidth_um - exclusion_um
left_middle_bound_um = lambda_min_um + 0.5 * modified_bandwidth_um
right_middle_bound_um = left_middle_bound_um + exclusion_um

num_left_lambda = int( 0.5 * num_lambda_values )
num_right_lambda = num_lambda_values - num_left_lambda
lambda_left = np.linspace( lambda_min_um, left_middle_bound_um, num_left_lambda )
lambda_right = np.linspace( right_middle_bound_um, lambda_max_um, num_right_lambda )

min_relative_permittivity = 1.0**2
max_relative_permittivity = max_index**2


def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

lambda_values_um = np.array( list( lambda_left ) + list( lambda_right ) )

device_width_voxels = 12 * 10
device_height_voxels = 12 * 6
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 12 * 10
focal_points_x_relative = [ 0.25, 0.75 ]

num_layers = int( device_height_voxels / density_coarsen_factor )
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ idx ] = 1

mean_density = 0.5
sigma_density = 0.2
init_from_old = False#True
binarize_set_point = 0.5

blur_fields_size_voxels = 0#4
blur_fields = False#True

# num_iterations = 450#150#300
# num_iterations_nominal = 150
num_iterations_nominal = 300
num_iterations = int( np.ceil(
	num_iterations_nominal * ( max_relative_permittivity - min_relative_permittivity ) / ( 1.5**2 - min_relative_permittivity ) ) )

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

design_width = int( device_width_voxels / density_coarsen_factor )
design_height = int( device_height_voxels / density_coarsen_factor )
num_design_voxels = design_width * design_height

use_pairings = False
if use_pairings:
	assert ( num_design_voxels % 2 ) == 0, 'Design voxel pairings cannot be made'

	num_pairings = int( 0.5 * num_design_voxels )

	used_voxels = np.zeros( num_design_voxels, dtype=np.uint )

	pairings = np.zeros( ( num_pairings, 4 ), dtype=np.uint )
	np.random.seed( random_seed )

	for pair_idx in range( 0, num_pairings ):
		num_left = num_design_voxels - 2 * pair_idx

		rand_idx1 = np.random.randint( 0, num_left )
		rand_idx2 = np.random.randint( 0, num_left - 1 )

		counter1 = 0
		linear_idx1 = 0
		for scan in range( 0, num_design_voxels ):
			if used_voxels[ scan ] == 0:
				if counter1 == rand_idx1:
					used_voxels[ scan ] += 1
					linear_idx1 = scan
					break
				counter1 += 1

		counter2 = 0
		linear_idx2 = 0
		for scan in range( 0, num_design_voxels ):
			if used_voxels[ scan ] == 0:
				if counter2 == rand_idx2:
					used_voxels[ scan ] += 1
					linear_idx2 = scan
					break
				counter2 += 1

		areal1_idx0 = int( linear_idx1 / design_height )
		areal1_idx1 = linear_idx1 % design_height
		areal2_idx0 = int( linear_idx2 / design_height )
		areal2_idx1 = linear_idx2 % design_height

		pairings[ pair_idx ] = np.array( [ areal1_idx0, areal1_idx1, areal2_idx0, areal2_idx1 ] )


# sys.exit(0)

dense_plot_freq_iters = 50#10
num_dense_wls = 4 * num_lambda_values
dense_plot_wls = np.linspace( lambda_min_um, lambda_max_um, num_dense_wls )

dense_focal_map = [ 0 for idx in range( 0, num_dense_wls ) ]
for idx in range( int( 0.5 * num_dense_wls ), num_dense_wls ):
	dense_focal_map[ idx ] = 1


make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )



make_optimizer.init_density_with_uniform( mean_density )
# make_optimizer.init_density_with_random( mean_density, sigma_density )
# np.save( save_folder + "/opt_random_seed.npy", make_optimizer.random_seed )
# np.save( save_folder + "/opt_init_random_density.npy", make_optimizer.design_density )

# density = np.load( save_folder + "/opt_optimized_density.npy" )
# make_optimizer.init_density_directly( density )


binarize = True
# binarize_movement_per_step = 0.005
# binarize_max_movement_per_voxel = 0.005
binarize_movement_per_step_nominal = 0.0075
binarize_max_movement_per_voxel_nominal = 0.0075

rho_delta_scaling = ( 1.5**2 - np.real( min_relative_permittivity ) ) / np.real( max_relative_permittivity - min_relative_permittivity )
binarize_movement_per_step = binarize_movement_per_step_nominal * rho_delta_scaling
binarize_max_movement_per_voxel = binarize_max_movement_per_voxel_nominal * rho_delta_scaling

dropout_start = 0
dropout_end = 0#151
dropout_p = 0.1

use_log_fom = False

wavelength_adversary = False#True
adversary_update_iters = 10

dual_opt = False

make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )


num_design_steps = design_height
# num_design_steps = 1

assert ( design_height % num_design_steps ) == 0, "Expected an even number of blocks in design!"

blocks_per_step = int( design_height / num_design_steps )

make_optimizer.init_density_with_uniform( mean_density )

for design_step_idx in range( 0, num_design_steps ):
	opt_mask_design_step = np.zeros( ( design_width, design_height ) )
	start = design_step_idx * blocks_per_step
	opt_mask_design_step[ :, start : ( start + blocks_per_step ) ] = 1

	num_iterations_design_step = int( num_iterations / num_design_steps )
	binarize_movement_per_step_design_step = binarize_movement_per_step / num_design_steps
	binarize_max_movement_per_voxel_design_step = binarize_max_movement_per_voxel / num_design_steps

	make_optimizer.optimize(
		int( num_iterations ),
		save_folder + "/opt",
		False, 20, 20, 0.95,
		opt_mask_design_step,
		use_log_fom,
		wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
		binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
		dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map )

	make_optimizer.save_optimization_data( save_folder + "/opt_step_" + str( design_step_idx ) + "_" )
