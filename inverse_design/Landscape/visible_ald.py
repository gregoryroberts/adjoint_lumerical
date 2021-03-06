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
import ColorSplittingOptimizationALD2D

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )

if ( max_index > 3.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

mesh_size_nm = 30#25
density_coarsen_factor = 10
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.50
lambda_max_um = 0.65
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

device_width_voxels = 100
device_height_voxels = 200#100
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 120#80
focal_points_x_relative = [ 0.25, 0.75 ]

# num_layers = int( device_height_voxels / density_coarsen_factor )
num_layers = 10#5
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
for layer_idx in range( 0, num_layers ):
	if ( layer_idx % 2 ) == 0:
		designable_layer_indicators[ layer_idx ] = True
	else:
		designable_layer_indicators[ layer_idx ] = False
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ idx ] = 1

mean_density = 0.5
sigma_density = 0.2
init_from_old = False#True#False
binarize_set_point = 0.5

blur_fields_size_voxels = 0
blur_fields = False

num_iterations_nominal = 350#450
num_iterations = int( np.ceil(
	num_iterations_nominal * ( max_relative_permittivity - min_relative_permittivity ) / ( 1.5**2 - min_relative_permittivity ) ) )

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

design_width = int( device_width_voxels / density_coarsen_factor )
design_height = int( device_height_voxels / density_coarsen_factor )
num_design_voxels = design_width * design_height
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


make_optimizer = ColorSplittingOptimizationALD2D.ColorSplittingOptimizationALD2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )


if init_from_old:
	density = np.load( save_folder + "/opt_optimized_density.npy" )
	permittivity = np.load( save_folder + "/opt_optimized_permittivity.npy" )
	make_optimizer.init_density_directly( density )

	fom = np.load( save_folder + "/opt_fom_evolution.npy" )
	binarization = np.load( save_folder + "/opt_binarization_evolution.npy" )

	plt.subplot( 1, 2, 1 )
	plt.plot( np.log10( fom ), linewidth=2, color='g' )
	plt.subplot( 1, 2, 2 )
	plt.plot( binarization, linewidth=2, color='r' )
	plt.show()

	plt.imshow( np.sqrt( permittivity ), cmap='Greens' )
	plt.colorbar()
	plt.show()

	# sys.exit(0)


	fields, permittivity = make_optimizer.get_device_efields( num_lambda_values - 1 )

	normalize_fields = np.abs( fields )**2 / np.max( np.abs( fields )**2 )

	plt.imshow( np.abs( fields ) )
	plt.show()

	make_optimizer.plot_fields( num_lambda_values - 1 )

	sys.exit( 0 )


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

else:
	make_optimizer.init_density_with_uniform( mean_density )

	binarize = True
	binarize_movement_per_step_nominal = 0.005
	binarize_max_movement_per_voxel_nominal = 0.005

	rho_delta_scaling = ( 1.5**2 - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )
	binarize_movement_per_step = binarize_movement_per_step_nominal * rho_delta_scaling
	binarize_max_movement_per_voxel = binarize_max_movement_per_voxel_nominal * rho_delta_scaling

	dropout_start = 0
	dropout_end = 0
	dropout_p = 0.1

	use_log_fom = False

	wavelength_adversary = False
	adversary_update_iters = 10

	opt_mask = np.ones( ( design_width, design_height ) )

	opt_mask[ :, 10:15 ] = 0

	opt_mask[ 6:12, 0:10 ] = 0
	opt_mask[ 18:24, 0:10 ] = 0

	opt_mask[ 0:6, 15:25 ] = 0
	opt_mask[ 12:18, 15:25 ] = 0
	opt_mask[ 24:30, 15:25 ] = 0


	make_optimizer.optimize(
		num_iterations,
		None, # opt_mask,
		use_log_fom,
		wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
		binarize, binarize_movement_per_step, binarize_max_movement_per_voxel,
		dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map )

	make_optimizer.save_optimization_data( save_folder + "/opt" )
