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
import ColorSplittingOptimizationDeep2D

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )

if ( max_index > 3.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

mesh_size_nm = 8#6#4#8#6#8
density_coarsen_factor = 4#3#4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.55
num_lambda_values = 2#8

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
# min_index = ( max_index ) * ( 1.0 / 2.25 )
# min_index = ( max_index ) * ( 1.0 / 1.5 )
# min_relative_permittivity = min_index**2
max_relative_permittivity = max_index**2


def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

# lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )
# lambda_values_um = np.array( list( lambda_left ) + list( lambda_right ) )
lambda_values_um = np.array( [ lambda_min_um, lambda_max_um ] )

feature_test = False#True

if feature_test:

	device_width_voxels = 240
	device_height_voxels = 168
	device_voxels_total = device_width_voxels * device_height_voxels
	focal_length_voxels = 200
	focal_points_x_relative = [ 0.25, 0.75 ]

else:
	#
	# todo: focal spot sizes might be a touch big here...
	#
	# device_width_voxels = 120#160#120
	# device_width_voxels = 162
	device_width_voxels = 192
	device_height_voxels = 128#135#102#100#104#100#132#100
	# device_height_voxels = 72#100#72
	# device_height_voxels = #52#64#52
	# device_height_voxels = 48#32
	# device_height_voxels = 32#24
	device_voxels_total = device_width_voxels * device_height_voxels
	focal_length_voxels = 96#135#100#132#100
	focal_points_x_relative = [ 0.25, 0.75 ]

num_layers = int( device_height_voxels / density_coarsen_factor )
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ idx ] = 1

# for idx in range( 0, num_lambda_values ):
# 	if ( idx % 2 ) == 0:
# 		focal_map[ idx ] = 1

mean_density = 0.5
sigma_density = 0.2
init_from_old = False#True
binarize_set_point = 0.5

blur_fields_size_voxels = 0#4
blur_fields = False#True

# num_iterations = 450#150#300
# num_iterations_nominal = 150
num_iterations_nominal = 300
# num_iterations_nominal = 200
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


make_optimizer = ColorSplittingOptimizationDeep2D.ColorSplittingOptimizationDeep2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )


# make_optimizer = ColorSplittingOptimizationDeep2D.ColorSplittingOptimizationDeep2D(
# 	[ device_width_voxels, device_height_voxels ],
# 	density_coarsen_factor, mesh_size_nm,
# 	[ min_relative_permittivity, max_relative_permittivity ],
# 	focal_points_x_relative, focal_length_voxels,
# 	lambda_values_um, focal_map, random_seed,
# 	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
# 	blur_fields, blur_fields_size_voxels, pairings )

if init_from_old:
	density = np.load( save_folder + "/opt_optimized_density.npy" )
	make_optimizer.init_density_directly( density )

	fields, permittivity = make_optimizer.get_device_efields( 0 )

	normalize_fields = np.abs( fields )**2 / np.max( np.abs( fields )**2 )
	normalize_permittivity = permittivity / max_relative_permittivity
	print( "Maximum field = " + str( np.max( np.abs( fields )**2 ) ) )
	print( "Average field in material FOM = " + str( np.sum( np.abs( fields )**2 / permittivity**2 ) ) )
	print( "Normalized field in material FOM = " + str( np.sum( normalize_fields / permittivity**2 ) ) )
	print( "Double Normalized field in material FOM = " + str( np.sum( normalize_fields / normalize_permittivity**2 ) ) )

	init_fom = make_optimizer.compute_fom(
		make_optimizer.omega_values[ 0 ],
		make_optimizer.get_device_permittivity(),
		make_optimizer.focal_spots_x_voxels[ 0 ] )

	print( 'init fom = ' + str( init_fom ) )

	index_low = 0.9 * max_index
	index_high = 1.1 * max_index

	num_index = 15

	index_sweep = np.linspace( index_low, index_high, num_index )

	plot_fom_sweep = np.zeros( num_index )
	plot_fom_diff_sweep = np.zeros( num_index )

	for index_idx in range( 0, num_index ):
		make_optimizer.max_relative_permittivity = index_sweep[ index_idx ]**2

		calc_fom = make_optimizer.compute_fom(
			make_optimizer.omega_values[ 0 ],
			make_optimizer.get_device_permittivity(),
			make_optimizer.focal_spots_x_voxels[ 0 ] )
		print( 'calc fom = ' + str( calc_fom ) )
		print( 'ratio = ' + str( calc_fom / init_fom ) )
		print()

		plot_fom_sweep[ index_idx ] = calc_fom / init_fom
		plot_fom_diff_sweep[ index_idx ] = ( calc_fom - init_fom ) / init_fom

	plt.subplot( 1, 2, 1 )
	plt.plot( 100. * ( index_sweep - max_index ) / max_index, plot_fom_sweep, color='r', linewidth=2 )
	plt.subplot( 1, 2, 2 )
	plt.plot( 100. * ( index_sweep - max_index ) / max_index, plot_fom_diff_sweep, color='g', linewidth=2 )
	plt.show()



	# make_optimizer.verify_adjoint_against_finite_difference()

	# make_optimizer.plot_fields( 0 )
	# make_optimizer.plot_subcell_gradient_variations( 0, 5 )
else:
	make_optimizer.init_density_with_uniform( mean_density )
	# make_optimizer.init_density_with_random( mean_density, sigma_density )
	# np.save( save_folder + "/opt_random_seed.npy", make_optimizer.random_seed )
	# np.save( save_folder + "/opt_init_random_density.npy", make_optimizer.design_density )

	# density = np.load( save_folder + "/opt_optimized_density.npy" )
	# make_optimizer.init_density_directly( density )

	include_loss = False
	if include_loss:
		single_pass_transmittance = 0.9
		device_height_m = device_height_voxels * mesh_size_nm * 1e-9
		lambda_min_m = lambda_min_um * 1e-6
		loss_index = -lambda_min_m * np.log( single_pass_transmittance ) / ( device_height_m * 2 * np.pi )
		real_permittivity = max_index**2 - loss_index**2
		imag_permittivity = 2 * np.sqrt( real_permittivity ) * loss_index
		max_relative_permittivity = real_permittivity + 1j * imag_permittivity


	binarize = True
	# binarize_movement_per_step = 0.005
	# binarize_max_movement_per_voxel = 0.005
	binarize_movement_per_step_nominal = 0.0075
	binarize_max_movement_per_voxel_nominal = 0.0075# / 10.

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

	make_optimizer = ColorSplittingOptimizationDeep2D.ColorSplittingOptimizationDeep2D(
		[ device_width_voxels, device_height_voxels ],
		density_coarsen_factor, mesh_size_nm,
		[ min_relative_permittivity, max_relative_permittivity ],
		focal_points_x_relative, focal_length_voxels,
		lambda_values_um, focal_map, random_seed,
		num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
		blur_fields, blur_fields_size_voxels, None, binarize_set_point )

	if dual_opt:
		opt_mask_mid_focus = np.zeros( ( design_width, design_height ) )
		opt_mask_splitter = np.zeros( ( design_width, design_height ) )

		device_layer_voxels_focuser = int( device_height_voxels * 2. / 7. )
		design_layer_voxels_focuser = int( device_layer_voxels_focuser / density_coarsen_factor )

		opt_mask_mid_focus[ :, ( design_height - design_layer_voxels_focuser ) : ] = 1
		opt_mask_splitter[ :, 0 : design_layer_voxels_focuser ] = 1

		focal_points_x_relative_mid_focus = [ 0.5, 0.5 ]
		focal_length_voxels_mid_focus = -device_layer_voxels_focuser

		make_optimizer_mid_focus = ColorSplittingOptimizationDeep2D.ColorSplittingOptimizationDeep2D(
			[ device_width_voxels, device_height_voxels ],
			density_coarsen_factor, mesh_size_nm,
			[ min_relative_permittivity, max_relative_permittivity ],
			focal_points_x_relative_mid_focus, focal_length_voxels_mid_focus,
			lambda_values_um, focal_map, random_seed,
			num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
			blur_fields, blur_fields_size_voxels, None, binarize_set_point )

		uniform_density_with_spacer = mean_density * np.ones( ( design_width, design_height ) )
		uniform_density_with_spacer *= np.maximum( opt_mask_mid_focus, opt_mask_splitter )
		make_optimizer_mid_focus.init_density_directly( uniform_density_with_spacer )

		make_optimizer_mid_focus.optimize(
			int( num_iterations / 1.5 ),
			save_folder + "/opt",
			False, 20, 20, 0.95,
			opt_mask_mid_focus,
			use_log_fom,
			wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
			binarize, 1.5 * binarize_movement_per_step, 1.5 * binarize_max_movement_per_voxel,
			dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map )

		make_optimizer_mid_focus.save_optimization_data( save_folder + "/opt_focuser" )

		# Note: with the mask, the binarization binarizes everything, so you should reinitialize the mean
		# density here
		init_density_part2 = make_optimizer_mid_focus.design_density
		init_density_part2[ :, 0 : design_layer_voxels_focuser ] = mean_density

		make_optimizer.init_density_directly( init_density_part2 )

		make_optimizer.optimize(
			int( num_iterations / 1.5 ),
			save_folder + "/opt",
			False, 20, 20, 0.95,
			opt_mask_splitter,
			use_log_fom,
			wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
			binarize, 1.5 * binarize_movement_per_step, 1.5 * binarize_max_movement_per_voxel,
			dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map )

		make_optimizer.save_optimization_data( save_folder + "/opt" )

	else:
		make_optimizer.init_density_with_uniform( mean_density )

		index_regularization = False#True
		downsample_abs_max = False#True

		# folder_to_plot = './bin_rate_down_avg_wider_v3'
		# final_density = np.load(  folder_to_plot + '/opt_2p25/opt_optimized_density.npy' )

		# make_optimizer.optimize(
		# 	1000,
		# 	# int( num_iterations ),
		# 	save_folder + "/opt" )
		make_optimizer.optimize_vote(
			1000,
			save_folder + "/opt" )

		# make_optimizer.init_density_directly( final_density )
		# make_optimizer.plot_fields( 2 )
		# make_optimizer.optimize_with_level_set( 10 )


		make_optimizer.save_optimization_data( save_folder + "/opt" )
