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

import scipy.optimize


#
# Plotting
#
import matplotlib as mpl
import matplotlib.pylab as plt

#
# Optimizer
#
import ColorSplittingOptimization2D


def compute_binarization( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )
# def compute_binarization_gradient( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 1. / total_shape ) * ( input_variable - 0.5 ) / np.sum( np.sqrt( ( input_variable - 0.5 )**2 )	)

def compute_binarization_gradient( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sign( input_variable - set_point )


if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )

index_to_name = {}
index_to_name[ 1.5 ] = '1p5'
index_to_name[ 1.75 ] = '1p75'
index_to_name[ 2.0 ] = '2'
index_to_name[ 2.25 ] = '2p25'
index_to_name[ 2.5 ] = '2p5'
index_to_name[ 2.75 ] = '2p75'
index_to_name[ 3.0 ] = '3'
index_to_name[ 3.25 ] = '3p25'
index_to_name[ 3.5 ] = '3p5'

if ( max_index > 1.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

# mesh_size_nm = 25
mesh_size_nm = 20
# mesh_size_nm = 8#6#4#8#6#8
# density_coarsen_factor = 16#20
density_coarsen_factor = 4
# density_coarsen_factor = 4#3#4
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
# min_index = ( max_index ) * ( 1.0 / 2.25 )
# min_index = ( max_index ) * ( 1.0 / 1.5 )
# min_relative_permittivity = min_index**2
max_relative_permittivity = max_index**2


def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

# lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )
lambda_values_um = np.array( list( lambda_left ) + list( lambda_right ) )

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
	# device_width_voxels = 80#200
	# device_width_voxels = 320
	# device_width_voxels = 50 * 6
	device_width_voxels = 51 * 4
	# device_height_voxels = 80 * 5
	# device_height_voxels = 120
	# device_height_voxels = 800
	# device_height_voxels = 600
	# device_height_voxels = 400
	# device_height_voxels = 50 * 6
	device_height_voxels = 51 * 4
	# device_height_voxels = 200
	# spacing_device_height_voxels = 40
	# device_height_voxels = 72#100#72
	# device_height_voxels = #52#64#52
	# device_height_voxels = 48#32
	# device_height_voxels = 32#24
	device_voxels_total = device_width_voxels * device_height_voxels
	# focal_length_voxels = 50#135#100#132#100
	# focal_length_voxels = 200
	# focal_length_voxels = 50 * 5
	focal_length_voxels = 50 * 4
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
# num_iterations_nominal = 300
# num_iterations_nominal = 300#360
# num_iterations_nominal = 320
num_iterations_nominal = 4 * 350
# num_iterations_nominal = 80
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


make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )


# make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
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
	binarize_movement_per_step_nominal = 0.005 * .3
	binarize_max_movement_per_voxel_nominal = 0.005 * .3# / 10.

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

	if dual_opt:
		opt_mask_mid_focus = np.zeros( ( design_width, design_height ) )
		opt_mask_splitter = np.zeros( ( design_width, design_height ) )

		device_layer_voxels_focuser = int( device_height_voxels * 2. / 7. )
		design_layer_voxels_focuser = int( device_layer_voxels_focuser / density_coarsen_factor )

		opt_mask_mid_focus[ :, ( design_height - design_layer_voxels_focuser ) : ] = 1
		opt_mask_splitter[ :, 0 : design_layer_voxels_focuser ] = 1

		focal_points_x_relative_mid_focus = [ 0.5, 0.5 ]
		focal_length_voxels_mid_focus = -device_layer_voxels_focuser

		make_optimizer_mid_focus = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
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

	# todo: did you make sure each frequency has same amount of energy going into simulation?
	# todo: are devices rearranging or not?
	else:
		make_optimizer.init_density_with_uniform( mean_density )

		index_regularization = False#True
		downsample_abs_max = False#True

		# old_density = np.load(
		# 	'/central/groups/Faraon_Computing/projects/binarize_bin_rate_down_avg_wider_save_v6_' +
		# 	index_to_name[ max_index ] +
		# 	'/opt_optimized_density.npy' )

		# make_optimizer.init_density_directly( old_density )

		# todo: try to run dropout for several iterations with a given mask instead of switching it every iteration
		# (or run it over a certain amount of binarization)

		viz_opt = False#True

		if not viz_opt:

			# make_optimizer.design_density[ :, 5:20 ] = 1
			# opt_mask = np.zeros( make_optimizer.design_density.shape )

			# opt_mask[ :, 20:25 ] = 1

			# make_optimizer.optimize(
			# 	int( 0.5 * num_iterations ),
			# 	save_folder + "/opt",
			# 	False, 20, 20, 0.95,
			# 	# None,
			# 	opt_mask,
			# 	use_log_fom,
			# 	wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
			# 	binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
			# 	dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map,
			# 	index_regularization,
			# 	downsample_abs_max )

			# opt_mask = np.zeros( make_optimizer.design_density.shape )
			# opt_mask[ :, 0:5 ] = 1

			# make_optimizer.optimize(
			# 	int( 0.5 * num_iterations ),
			# 	save_folder + "/opt",
			# 	False, 20, 20, 0.95,
			# 	# None,
			# 	opt_mask,
			# 	use_log_fom,
			# 	wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
			# 	binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
			# 	dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map,
			# 	index_regularization,
			# 	downsample_abs_max )

			binarize_v2 = 1
			eps_movement_per_voxel = 0.02 / 4.
			# eps_movement_per_voxel = 0.02 / 2.
			if binarize_v2 == 1:
				binarize_max_movement_per_voxel = eps_movement_per_voxel / ( max_relative_permittivity - min_relative_permittivity )

				# old_density = np.load(
				# 	'/central/groups/Faraon_Computing/projects/very_thick_bin2_v3_' +
				# 	index_to_name[ max_index ] +
				# 	'/opt_optimized_density.npy' )

				# make_optimizer.init_density_directly( old_density )

				dropout_start = 0
				dropout_end = 0#num_iterations#int( 0.75 * num_iterations )#0#num_iterations# int( 0.75 * num_iterations )
				dropout_p = 0.25#0.1#0.25#0.5#0.75#0.9#0.75
				dropout_bin_freq = 0.01
				binarize = True#False
				fom_ratio = True
				fom_simple_sum = True

				mask_generator = np.random.random( ( design_width, design_height ) ) 
				half_opt_mask = mask_generator >= 0.5
				negative_half_opt_mask = mask_generator < 0.5

				use_half_opt_mask = False#True
				if use_half_opt_mask:
					make_optimizer.optimize(
						int( 0.5 * num_iterations ),
						save_folder + "/opt",
						False, 20, 20, 0.95,
						# None,
						half_opt_mask,
						use_log_fom,
						wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
						binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
						dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map,
						index_regularization,
						downsample_abs_max, binarize_v2, 0.1, fom_ratio, fom_simple_sum )
					
					make_optimizer.save_optimization_data( save_folder + "/opt0_" )


					make_optimizer.optimize(
						int( 0.5 * num_iterations ),
						save_folder + "/opt",
						False, 20, 20, 0.95,
						# None,
						negative_half_opt_mask,
						use_log_fom,
						wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
						binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
						dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map,
						index_regularization,
						downsample_abs_max, binarize_v2, 0.1, fom_ratio, fom_simple_sum )
				else:		
					dilation_erosion = False#True
					dilation_erosion_amt = 2
					# dilation_erosion_binarization_freq = 0.025
					dilation_erosion_binarization_freq = 0.1
					# dilation_erosion_binarization_freq = 0.05

					fourier_fab_penalty = True
					fourier_fab_penalty_opt_headstart_iters = 100
					fourier_fab_penalty_feature_size = 5
					fourier_fab_penalty_relative_weight = 0.5


					# depth_sectioned_opt_mask = np.zeros( make_optimizer.design_density.shape )
					# num_voxels_material = 6
					# num_voxels_spacer = 7

					# voxel_idx = 0
					# in_material = True
					# num_remaining = num_voxels_material
					# while voxel_idx < depth_sectioned_opt_mask.shape[ 1 ]:
					# 	if in_material:
					# 		depth_sectioned_opt_mask[ :, voxel_idx ] = 1
					# 		num_remaining -= 1

					# 		if num_remaining == 0:
					# 			in_material = False
					# 			num_remaining = num_voxels_spacer
					# 	else:
					# 		depth_sectioned_opt_mask[ :, voxel_idx ] = 0
					# 		make_optimizer.design_density[ :, voxel_idx ] = 0

					# 		num_remaining -= 1

					# 		if num_remaining == 0:
					# 			in_material = True
					# 			num_remaining = num_voxels_material

					# 	voxel_idx += 1

					make_optimizer.optimize(
						num_iterations,
						save_folder + "/opt",
						False, 20, 20, 0.95,
						None,
						# opt_mask,
						# depth_sectioned_opt_mask,
						use_log_fom,
						wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
						binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
						dropout_start, dropout_end, dropout_p, dropout_bin_freq,
						dense_plot_freq_iters, dense_plot_wls, dense_focal_map,
						index_regularization,
						downsample_abs_max, binarize_v2, 0.1, fom_ratio, fom_simple_sum,
						dilation_erosion, dilation_erosion_amt, dilation_erosion_binarization_freq,
						fourier_fab_penalty, fourier_fab_penalty_opt_headstart_iters, fourier_fab_penalty_feature_size, fourier_fab_penalty_relative_weight )



			else:
				make_optimizer.optimize(
					num_iterations,
					save_folder + "/opt",
					False, 20, 20, 0.95,
					None,
					# opt_mask,
					use_log_fom,
					wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
					binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
					dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map,
					index_regularization,
					downsample_abs_max, binarize_v2, 0.1 )


			make_optimizer.save_optimization_data( save_folder + "/opt" )
		else:

			#
			# 1. Random dipoles
			# 2. Loss function
			# 3. Train in pieces
			# 4. Currently have dropout
			# 5. Change around wavelengths for generalization - i.e. different wavelength batches
			# 6. Go even thicker?
			# 7. Loss function regularization?
			# 8. Explicit feature size pressures? Homogenization penalty function for when voxels want to be too inhomogenous locally
			# 9. Field and/or gradient blurring
			# 10. Modal averaging (weight against a mode profile for a given supervoxel size)
			# 11. Lateral focal shift invariance in the loss function?
			#

			# mask_generator = np.random.random( ( design_width, design_height ) ) 
			# half_opt_mask = mask_generator >= 0.5
			# negative_half_opt_mask = mask_generator < 0.5

			# plt.imshow( half_opt_mask )
			# plt.show()
			# plt.imshow( negative_half_opt_mask )
			# plt.show()
			# sys.exit(0)

			# final_density1 = np.load( '/Users/gregory/Downloads/erode_dilate_v1.npy' )
			final_density1 = np.load( '/Users/gregory/Downloads/four_um_erode_dilate_final_v2_v1.npy' )
			final_density2 = np.load( '/Users/gregory/Downloads/four_um_v2_erode_dilate_100_v1.npy' )
			final_density3 = np.load( '/Users/gregory/Downloads/four_um_erode_dilate_200_v1.npy' )
			final_density4 = np.load( '/Users/gregory/Downloads/four_um_v2_erode_dilate_200_v1.npy' )
			final_density5 = np.load( '/Users/gregory/Downloads/four_um_erode_dilate_100_v1.npy' )


			np.random.seed( 12123 )

			from scipy.ndimage import grey_dilation
			from scipy.ndimage import grey_erosion
			from scipy.ndimage import gaussian_filter
			# test_density = np.random.random( ( 51, 51 ) )
			# test_density = np.random.random( ( 41, 41 ) )
			test_density = np.random.random( ( 21, 21 ) )
			raw_density = test_density.copy()
			test_density = gaussian_filter( test_density, 7 )
			test_density -= np.min( test_density )
			test_density /= np.max( test_density )

			print( np.mean( test_density ) )

			# test_density = np.zeros( ( 41, 41 ) )
			# test_density[ 0:5, : ] = 1
			# test_density[ 10:15, : ] = 1
			# test_density[ 20:25, : ] = 1
			# test_density[ 30:35, : ] = 1

			# spacing = 1
			# counter = 0
			# cur = 1

			# while counter < 41:
			# 	test_density[ counter : np.minimum( counter + spacing, 41 ), : ] = cur
			# 	test_density[ :, counter : np.minimum( counter + spacing, 41 ) ] = 0.75 * ( 1 - cur )
			# 	cur = 1 - cur
			# 	counter += spacing



			def step_binarize_v2( density, gradient, binarize_amount_factor, binarize_max_movement, opt_mask ):

				if opt_mask is None:
					opt_mask = np.ones( density.shape )

				density_for_binarizing = density.flatten()
				flatten_gradient = gradient.flatten()

				# flatten_design_cuts = density_for_binarizing.copy()
				# extract_binarization_gradient_full = compute_binarization_gradient( density_for_binarizing, self.binarization_set_point )
				# flatten_fom_gradients = flatten_gradient.copy()
				flatten_opt_mask = opt_mask.flatten()


				flatten_design_cuts = []
				flatten_fom_gradients = []
				extract_binarization_gradient = []

				for idx in range( 0, len( flatten_opt_mask ) ):
					if flatten_opt_mask[ idx ] > 0:
						flatten_design_cuts.append( density_for_binarizing[ idx ] )
						flatten_fom_gradients.append( flatten_gradient[ idx ] )
						# extract_binarization_gradient.append( extract_binarization_gradient_full[ idx ] )

				flatten_design_cuts = np.array( flatten_design_cuts )
				flatten_fom_gradients = np.array( flatten_fom_gradients )
				# extract_binarization_gradient = np.array( extract_binarization_gradient )
				extract_binarization_gradient = compute_binarization_gradient( flatten_design_cuts, 0.5 )

				beta = binarize_max_movement
				projected_binarization_increase = 0

				c = flatten_fom_gradients

				initial_binarization = compute_binarization( flatten_design_cuts, 0.5 )

				b = np.real( extract_binarization_gradient )

				lower_bounds = np.zeros( len( c ) )
				upper_bounds = np.zeros( len( c ) )

				for idx in range( 0, len( c ) ):
					upper_bounds[ idx ] = np.maximum( np.minimum( beta, 1 - flatten_design_cuts[ idx ] ), 0 )
					lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -flatten_design_cuts[ idx ] ), 0 )

				max_possible_binarization_change = 0
				for idx in range( 0, len( c ) ):
					if b[ idx ] > 0:
						max_possible_binarization_change += b[ idx ] * upper_bounds[ idx ]
					else:
						max_possible_binarization_change += b[ idx ] * lower_bounds[ idx ]
				
				# Try this! Not sure how well it will work
				# if initial_binarization < 0.1:
				# 	alpha = binarize_amount
				# else:
					# alpha = np.minimum( initial_binarization * max_possible_binarization_change, binarize_amount )
				# alpha = np.minimum( max_possible_binarization_change, binarize_amount )
				alpha = binarize_amount_factor * max_possible_binarization_change

				def ramp( x ):
					return np.maximum( x, 0 )

				def opt_function( nu ):
					lambda_1 = ramp( nu * b - c )
					lambda_2 = c + lambda_1 - nu * b

					return -( -np.dot( lambda_1, upper_bounds ) + np.dot( lambda_2, lower_bounds ) + nu * alpha )

				tolerance = 1e-12
				# optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance )
				# optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance, bounds=[ [ 0.0, np.inf ] ] )

				bin_constraint = scipy.optimize.LinearConstraint(A = 1, lb = 0, ub = np.inf)
				optimization_solution_nu = scipy.optimize.minimize(
					opt_function, 0,
					method='trust-constr',
					constraints=[bin_constraint] ,
					options={'xtol': 1E-16, 'gtol': 1E-16, 'barrier_tol': 1E-16})


				nu_star = optimization_solution_nu.x
				lambda_1_star = ramp( nu_star * b - c )
				lambda_2_star = c + lambda_1_star - nu_star * b
				x_star = np.zeros( len( c ) )

				for idx in range( 0, len( c ) ):
					if lambda_1_star[ idx ] > 0:
						x_star[ idx ] = upper_bounds[ idx ]
					else:
						x_star[ idx ] = lower_bounds[ idx ]


				proposed_design_variable = flatten_design_cuts + x_star
				proposed_design_variable = np.minimum( np.maximum( proposed_design_variable, 0 ), 1 )

				refill_idx = 0
				refill_design_variable = density_for_binarizing.copy()
				for idx in range( 0, len( flatten_opt_mask ) ):
					if flatten_opt_mask[ idx ] > 0:
						refill_design_variable[ idx ] = proposed_design_variable[ refill_idx ]
						refill_idx += 1

				return np.reshape( refill_design_variable, density.shape )






			def compute_fab_penalty_fourier( rho, feature_size ):

				pad_size = 10
				decay_width = 5
				input_rho = np.pad( rho, ( ( pad_size, pad_size ), ( pad_size, pad_size ) ) )

				for x_idx in range( 0, pad_size ):
					input_rho[ x_idx, : ] = input_rho[ pad_size, : ] * np.exp( -( x_idx - pad_size )**2 / decay_width**2 )
					input_rho[ input_rho.shape[ 0 ] - x_idx - 1, : ] = input_rho[ input_rho.shape[ 0 ] - pad_size - 1, : ] * np.exp( -( x_idx - pad_size )**2 / decay_width**2 )
				for y_idx in range( 0, pad_size ):
					input_rho[ :, y_idx ] = input_rho[ :, pad_size ] * np.exp( -( y_idx - pad_size )**2 / decay_width**2 )
					input_rho[ :, input_rho.shape[ 1 ] - y_idx - 1 ] = input_rho[ :, input_rho.shape[ 1 ] - pad_size - 1 ] * np.exp( -( y_idx - pad_size )**2 / decay_width**2 )


				mid_pt = input_rho.shape[ 0 ] // 2
				feature_cutoff = mid_pt / ( 1.0 * feature_size )


				input_rho -= np.mean( input_rho )
				# input_rho /= ( np.max( input_rho ) - np.min( input_rho ) )

				fft_rho = np.fft.fftshift( np.fft.fft2( input_rho ) )

				# fft_rho = np.fft.fftshift( np.fft.fft2( input_rho - np.mean( input_rho ) ) )
				# fft_rho = np.fft.fftshift( np.fft.fft2( input_rho ) )
				# plt.subplot( 1, 2, 1 )
				# plt.imshow( input_rho )
				# plt.subplot( 1, 2, 2 )
				# plt.imshow( np.abs( fft_rho ) )
				# plt.subplot( 1, 3, 3 )
				# plt.imshow( np.imag( fft_rho ) )

				# fig = plt.gcf()
				# ax = fig.gca()

				# feature_circle = plt.Circle((mid_pt, mid_pt), feature_cutoff, color='r', fill=False)
				# ax.add_patch(feature_circle)

				# plt.show()

				inside = 0
				outside = 0
				for x_idx in range( 0, input_rho.shape[ 0 ] ):
					for y_idx in range( 0, input_rho.shape[ 1 ] ):
						radius = np.sqrt( ( x_idx - mid_pt )**2 + ( y_idx - mid_pt )**2 )

						if radius > feature_cutoff:
							outside += np.abs( fft_rho[ x_idx, y_idx ] )**2
						else:
							inside += np.abs( fft_rho[ x_idx, y_idx ] )**2

				# print( outside + inside )
				return outside / ( outside + inside )
				# return outside

			def compute_fab_penalty( input_rho, feature_size, do_sum=True ):
				d = feature_size
				beta = 1. / 3.

				rho = input_rho.copy()

				rho -= 0.5
				rho /= 0.5

				pad_rho = np.pad( rho, ( ( 1, 1 ), ( 1, 1 ) ), mode='edge' )

				pad_rho_plus_x = pad_rho[ 2 :, 1 : -1 ]
				pad_rho_minus_x = pad_rho[ 0 : -2, 1 : -1 ]

				pad_rho_plus_y = pad_rho[ 1 : -1 , 2 : ]
				pad_rho_minus_y = pad_rho[ 1 : -1, 0 : -2 ]

				d_dx = 0.5 * ( pad_rho_plus_x - pad_rho_minus_x )
				d_dy = 0.5 * ( pad_rho_plus_y - pad_rho_minus_y )

				norm_gradient_direction = np.sqrt( d_dx**2 + d_dy**2 )
				normalized_gradient_direction = [ d_dx / norm_gradient_direction, d_dy / norm_gradient_direction ]

				project_first_order = ( d_dx * normalized_gradient_direction[ 0 ] + d_dy * normalized_gradient_direction[ 1 ] )

				get_plus_projected = normalized_gradient_direction[ 0 ] * pad_rho_plus_x + normalized_gradient_direction[ 1 ] * pad_rho_plus_y
				get_minus_projected = normalized_gradient_direction[ 0 ] * pad_rho_minus_x + normalized_gradient_direction[ 1 ] * pad_rho_minus_y
				
				d2_dv2 = get_plus_projected + get_minus_projected - 2 * rho
				d_dv = 0.5 * ( get_plus_projected - get_minus_projected )


				term0 = np.abs( d2_dv2 ) / ( ( np.pi / d ) * np.abs( rho ) + beta * d_dv )
				term1 = np.pi / d
				combine_terms = np.maximum( term0 - term1, 0 )

				if do_sum:
					total = np.sum( combine_terms )

					return total
				else:
					return combine_terms

			def compute_fab_gradient( input_rho, feature_size, method ):

				h = 1e-3
				deriv = np.zeros( input_rho.shape )

				middle = method( input_rho, test_feature_size )

				for x in range( 0, input_rho.shape[ 0 ] ):
					for y in range( 0, input_rho.shape[ 1 ] ):
						input_rho_copy = input_rho.copy()
						input_rho_copy[ x, y ] += h
						up = method( input_rho_copy, test_feature_size )

						# input_rho_copy = input_rho.copy()
						# input_rho_copy[ x, y ] -= h
						# down = method( input_rho_copy, test_feature_size )

						# deriv[ x, y ] = ( up - down ) / ( 2 * h )
						deriv[ x, y ] = ( up  - middle ) / h


				return deriv

			feature_sweep = np.linspace( 1, 20, 100 )
			penalties = []
			for feature in feature_sweep:
				penalties.append( compute_fab_penalty_fourier( test_density, feature ) )

			plt.plot( feature_sweep, penalties, linewidth=2, color='r' )
			# plt.show()

			feature_sweep = np.linspace( 1, 9, 100 )
			penalties = []
			for feature in feature_sweep:
				test_density = gaussian_filter( raw_density, feature )
				test_density -= np.min( test_density )
				test_density /= np.max( test_density )

				penalties.append( compute_fab_penalty_fourier( test_density, 3 ) )

			plt.plot( feature_sweep, penalties, linewidth=2, color='g' )
			plt.show()

			# test_feature_size = 9
			# print( test_density.shape )
			# print( compute_fab_penalty_fourier( test_density, test_feature_size ) )


			test_density = gaussian_filter( raw_density, 2 )
			test_density -= np.min( test_density )
			test_density /= np.max( test_density )
			test_feature_size = 4


			# sys.exit( 0 )




			start = test_density.copy()
			num_steps = 30#50#10

			print( 'before = ' + str( compute_fab_penalty_fourier( test_density, test_feature_size ) ) )
			print( np.mean( test_density ) )
			print( np.std( test_density ) )

			last = 0
			for step in range( 0, num_steps ):
				print( 'step = ' + str( step ) )
				cur_penalty = compute_fab_penalty_fourier( test_density, test_feature_size )
				cur_binarization = compute_binarization( test_density.flatten(), 0.5 )
				print( 'cur binarization = ' + str(  cur_binarization ) )
				print( 'cur penalty = ' + str(  cur_penalty ) )
				print( cur_penalty - last )
				last = cur_penalty
				# print()

				# import time
				# import sys
				# start_time = time.time()
				get_fab_deriv = compute_fab_gradient( test_density, test_feature_size, compute_fab_penalty_fourier )
				# elapsed_time = time.time() - start_time
				# print( 'Gradient took ' + str( elapsed_time ) + ' seconds' )
				# sys.exit(0)
				# plt.subplot( 1, 2, 1 )
				# plt.imshow( test_density, cmap='hot' )
				# plt.colorbar()
				# plt.subplot( 1, 2, 2 )
				# plt.imshow( get_fab_deriv, cmap='hot' )
				# plt.colorbar()
				# plt.show()

				norm = np.max( np.abs( get_fab_deriv ) )
				get_fab_deriv /= norm

				# test_density -= 0.5 * get_fab_deriv# * 1e-5
				# test_density -= 0.02 * get_fab_deriv# * 1e-5

				change_deriv = get_fab_deriv.copy()
				# change_deriv = np.zeros( test_density.shape )
				# change_deriv[ 10, 10 ] = get_fab_deriv[ 10, 10 ]


		# def step_binarize_v2( density, gradient, binarize_amount_factor, binarize_max_movement, opt_mask ):

				# test_density = step_binarize_v2( test_density, 0.02 * change_deriv, 0.1, 0.02, None )
				test_density -= 0.02 * change_deriv

				expected_change = np.sum( -norm * get_fab_deriv * 0.02 * change_deriv )
				print( 'expected... = ' + str( expected_change ) )
				print()


				# test_density = np.minimum( 1.0, np.maximum( 0.0, test_density ) )


			print( 'after = ' + str( compute_fab_penalty_fourier( test_density, test_feature_size ) ) )
			print( np.mean( test_density ) )
			print( np.std( test_density ) )

			plt.subplot( 1, 2, 1 )
			plt.imshow( start )
			plt.clim([ 0, 1 ])
			plt.colorbar()
			plt.subplot( 1, 2, 2 )
			plt.imshow( test_density )
			plt.clim([ 0, 1 ])
			plt.colorbar()
			plt.show()

			plt.plot( start[ 10, : ], color='g', linewidth=2 )
			plt.plot( test_density[ 10, : ], color='r', linestyle='--', linewidth=2 )

			plt.ylim([ 0, 1 ])
			plt.show()

			# plt.plot( start[ :, 7 ], color='g', linewidth=2 )
			# plt.plot( test_density[ :, 7 ], color='r', linestyle='--', linewidth=2 )
			# plt.ylim([ 0, 1 ])
			# plt.show()

			plt.plot( 1.0 * np.greater_equal( start[ 10, : ], 0.5 ), color='g', linewidth=2 )
			plt.plot( 1.0 * np.greater_equal( test_density[ 10, : ], 0.5 ), color='r', linestyle='--', linewidth=2 )
			plt.ylim([ -0.25, 1.25 ])
			plt.show()


			# print( compute_fab_penalty( test_density, test_feature_size ) )

			# print( compute_fab_penalty( test_density, 3 ) )
			# print( compute_fab_penalty( test_density, 5 ) )
			# print( compute_fab_penalty( test_density, 7 ) )
			# print( compute_fab_penalty( test_density, 15 ) )
			sys.exit(0)


			test_density *= 0
			test_density[ 10:15, 10:15 ] = 1
			test_density[ 10:15, 18:23 ] = 1

			test_density = final_density4.copy()

			dilation_erosion_size = 5

			cur_density = grey_dilation( test_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )
			cur_density = grey_erosion( cur_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )
			cur_density = grey_erosion( cur_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )
			cur_density = grey_dilation( cur_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )

			# cur_density = grey_erosion( test_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )
			# cur_density = grey_dilation( cur_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )
			# cur_density = grey_dilation( cur_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )
			# cur_density = grey_erosion( cur_density, ( dilation_erosion_size, dilation_erosion_size ), mode='nearest' )


			cur_density = np.maximum( 0.0, np.minimum( 1.0, cur_density ) )

			print( np.mean( test_density ) )
			print( np.mean( cur_density ) )
			print()

			plt.subplot( 1, 2, 1 )
			plt.imshow( test_density )
			plt.colorbar()
			plt.subplot( 1, 2, 2 )
			plt.imshow( cur_density )
			plt.colorbar()
			plt.show()

			# plt.subplot( 1, 2, 1 )
			plt.plot( test_density[ :, 5 ], linewidth=2, color='r' )
			# plt.plot( test_density[ :, 9 ], linewidth=2, color='r' )
			# plt.plot( test_density[ :, 11 ], linewidth=2, color='r' )
			# plt.plot( test_density[ :, 8 ], linewidth=2, color='r' )
			# plt.plot( test_density[ :, 12 ], linewidth=2, color='r' )
			# plt.subplot( 1, 2, 2 )
			# plt.plot( cur_density[ :, 8 ], linewidth=2, color='g', linestyle='--' )
			# plt.plot( cur_density[ :, 9 ], linewidth=2, color='g', linestyle='--' )
			plt.plot( cur_density[ :, 5 ], linewidth=2, color='g', linestyle='--' )
			# plt.plot( cur_density[ :, 11 ], linewidth=2, color='g', linestyle='--' )
			# plt.plot( cur_density[ :, 12 ], linewidth=2, color='g', linestyle='--' )
			plt.show()

			plt.plot( final_density1[ :, 10 ], linewidth=2, color='r' )
			# plt.subplot( 1, 2, 2 )
			plt.plot( final_density2[ :, 10 ], linewidth=2, color='g', linestyle='--' )
			plt.plot( final_density3[ :, 10 ], linewidth=2, color='b', linestyle='--' )
			plt.plot( final_density4[ :, 10 ], linewidth=2, color='m', linestyle='--' )
			plt.plot( final_density5[ :, 10 ], linewidth=2, color='c', linestyle='--' )
			plt.show()


			# final_density1 = np.load( '/Users/gregory/Downloads/ten_um_bin_sum_fom_ratio_wide_longer_f_v1.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/ten_um_bin_sum_fom_ratio_wide_longer_f_v2.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/ten_um_bin_sum_fom_ratio_wide_longer_f_v3.npy' )


			# final_density1 = np.load( '/Users/gregory/Downloads/ten_um_bin_sum_fom_ratio_wide_v1.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/ten_um_bin_sum_fom_ratio_wide_v2.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/ten_um_bin_sum_fom_ratio_wide_v3.npy' )


			# final_density1 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_mask_v2_v1_density.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_mask_v2_v2_density.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_mask_v2_v3_density.npy' )


			# final_density1 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_v1_density.npy' )
			# # final_density2 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_v2_density.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_mask_v1_density.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/twenty_um_bin_sum_fom_ratio_v3_density.npy' )

			# print( np.sum( np.abs( final_density1 - final_density2 ) ) )

			# final_density1 = np.load( '/Users/gregory/Downloads/five_um_bin_v1_density.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/five_um_bin_v2_density.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/mid_thick_bin_v2_v3_density.npy' )
			# final_density4 = np.load( '/Users/gregory/Downloads/mid_thick_bin_v2_v4_density.npy' )


			# final_density1 = np.load( '/Users/gregory/Downloads/mid_thick_bin_v2_v1_density.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/mid_thick_bin_v2_v2_density.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/mid_thick_bin_v2_v3_density.npy' )
			# final_density4 = np.load( '/Users/gregory/Downloads/mid_thick_bin_v2_v4_density.npy' )

			# final_density1 = np.load( '/Users/gregory/Downloads/mid_thick_density_v1.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/mid_thick_density_v2.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/mid_thick_density_v3.npy' )
			# final_density4 = np.load( '/Users/gregory/Downloads/mid_thick_density_v4.npy' )


			# final_density1 = np.load( '/Users/gregory/Downloads/very_thick_density_v1.npy' )
			# final_density2 = np.load( '/Users/gregory/Downloads/very_thick_density_v2.npy' )
			# final_density3 = np.load( '/Users/gregory/Downloads/very_thick_density_v3.npy' )

			plt.subplot( 1, 5, 1 )
			plt.imshow( np.swapaxes( final_density1, 0, 1 ), cmap='Blues' )
			plt.subplot( 1, 5, 2 )
			plt.imshow( np.swapaxes( final_density2, 0, 1 ), cmap='Blues' )
			plt.subplot( 1, 5, 3 )
			plt.imshow( np.swapaxes( final_density3, 0, 1 ), cmap='Blues' )
			plt.subplot( 1, 5, 4 )
			plt.imshow( np.swapaxes( final_density4, 0, 1 ), cmap='Blues' )
			plt.subplot( 1, 5, 5 )
			plt.imshow( np.swapaxes( final_density5, 0, 1 ), cmap='Blues' )
			plt.show()

			sys.exit( 0 )

			# make_optimizer.init_density_with_uniform( 0.5 )

			# final_density = np.load( '/Users/gregory/Downloads/thick_2_density_v2.npy' )

			final_density = final_density2
			bin_final_density = 1.0 * np.greater_equal( final_density, 0.5 )

	# def compute_fom_and_gradient( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0, dropout_mask=None ):

			import_density = ColorSplittingOptimization2D.upsample( final_density1, make_optimizer.coarsen_factor )
			# import_density = ColorSplittingOptimization2D.upsample( make_optimizer.design_density, make_optimizer.coarsen_factor )
			device_permittivity = make_optimizer.density_to_permittivity( import_density )

			# make_optimizer.compute_fom_and_gradient(
			# 	make_optimizer.omega_values[ 2 ],
			# 	device_permittivity,
			# 	make_optimizer.focal_spots_x_voxels[ 0 ],
			# 	1.0,
			# 	1.0 * ( np.random.random( device_permittivity.shape ) >= 0.25 ) )

			# sys.exit( 0 )

			make_optimizer.init_density_directly( final_density )
			# make_optimizer.init_density_directly( bin_final_density )
			# make_optimizer.init_density_with_uniform( 0.5 )
			Ez_dev = make_optimizer.plot_fields( 6 )
			I_dev = np.abs( Ez_dev )**2
			I_dev = I_dev[ make_optimizer.device_width_start : make_optimizer.device_width_end, make_optimizer.focal_point_y ]

			make_optimizer.init_density_with_uniform( 1 * 0.5 )
			Ez_flat = make_optimizer.plot_fields( 6 )
			I_flat = np.abs( Ez_flat )**2
			I_flat = I_flat[ make_optimizer.device_width_start : make_optimizer.device_width_end, make_optimizer.focal_point_y ]

			sum_flat = np.sum( I_flat )
			sum_dev = np.sum( I_dev )
			left_t = np.sum( I_dev[ 0 : int( 0.5 * len( I_dev ) ) ] ) / sum_flat
			right_t = np.sum( I_dev[ int( 0.5 * len( I_dev ) ) : -1 ] ) / sum_flat

			print( 'Net transmission ~ ' + str( sum_dev / sum_flat ) )
			print( 'Left transmission ~ ' + str( left_t ) )
			print( 'Right transmission ~ ' + str( right_t ) )

			fp_dev = (
				np.abs( I_dev )
			)
			fp_flat = (
				np.abs( I_flat )
			)

			plt.plot( fp_dev, color='r', linewidth=2 )
			plt.plot( fp_flat, color='g', linewidth=2 )
			plt.show()


			sys.exit( 0 )
			print( make_optimizer.compute_net_fom() )
			# make_optimizer.optimize_with_level_set( 10 )
