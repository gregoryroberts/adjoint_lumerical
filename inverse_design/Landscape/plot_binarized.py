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
import matplotlib.patches as patches

#
# Optimizer
#
import ColorSplittingOptimization2D



mesh_size_nm = 8
# mesh_size_nm = 6
density_coarsen_factor = 8#4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.55
num_lambda_values = 8

def density_bound_from_eps( eps_val, min_perm, max_perm ):
	return ( eps_val - min_perm ) / ( max_perm - min_perm )

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 120#160#120
# device_width_voxels = 200
device_height_voxels = 104#100#132#100
# device_height_voxels = 72
# device_height_voxels = 52
# device_height_voxels = 32
# device_height_voxels = 24
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 100#132#100
focal_points_x_relative = [ 0.25, 0.75 ]

num_layers = int( device_height_voxels / density_coarsen_factor )
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ idx ] = 1


# folder_to_plot = './bin_v5_blurred/'
# folder_to_plot = './bin_v2/'
# folder_to_plot = './bin_pairs_v2'
# folder_to_plot = './bin_finer_mesh_v1'
# folder_to_plot = './bin_unif_v2'
# folder_to_plot = './bin_unif_push_v1'
# folder_to_plot = './bin_unif_wider_v1'
# folder_to_plot = './bin_switch_focal_v1'
# folder_to_plot = './bin_contrast_v2'
# folder_to_plot = './bin_unif_v3'
# folder_to_plot = './bin_dropout_p5_v1'
# folder_to_plot = './bin_log_fom_v1'
# folder_to_plot = './bin_unif_excl_v1'
# folder_to_plot = './bin_adv_excl_v4'
# folder_to_plot = './bin_unif_thinner_excl_v3'
# folder_to_plot = './bin_rate_v6'
# folder_to_plot = './bin_rate_v6'
folder_to_plot = './bin_rate_excl_thick_v1'

device_suffixes = [ '1p5', '1p75', '2', '2p25', '2p5', '2p75', '3', '3p25', '3p5' ]
device_legend = [ 'n=1.5', 'n=1.75', 'n=2', 'n=2.25', 'n=2.5', 'n=2.75', 'n=3', 'n=3.25', 'n=3.5' ]
colors = [ 'r', 'g', 'b', 'm', 'c', 'k', 'y', 'purple', 'orange' ]
# device_suffixes = [ '3', '3p25', '3p5' ]
# colors = [ 'y', 'purple', 'orange' ]

fom_compare = 0

for device_idx in range( 0, len( device_suffixes ) ):
	device_suffix = device_suffixes[ device_idx ]
	device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"
	figure_of_merit = np.load( device_file_prefactor + 'opt_fom_evolution.npy' )
	fom_compare += ( 1. / len( device_suffixes ) ) * figure_of_merit[ 0 ]

legend = []

max_indices = [ 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5 ]
# min_indices = [ ( max_index ) * ( 1.0 / 2.25 ) for max_index in max_indices ]
min_indices = [ ( max_index ) * ( 1.0 / 1.5 ) for max_index in max_indices ]

for opt_idx in range( 0, len( max_indices ) ):
	device_suffix = device_suffixes[ opt_idx ]
	device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"

	dense_plots = np.load( device_file_prefactor + "/opt_dense_plots.npy" )
	dense_plot_idxs = np.load( device_file_prefactor + "/opt_dense_plot_idxs.npy" )

	num_to_look_at = 5
	spots = np.linspace( 0, len( dense_plots ) - 1, num_to_look_at )

	# colors = [ 'r', 'g', 'b', 'm', 'k' ]

	figure_of_merit = np.load( device_file_prefactor + 'opt_fom_evolution.npy' )
	binarization_evolution = np.load( device_file_prefactor + 'opt_binarization_evolution.npy' )

	legend_data = []
	plt.subplot( 1, 3, 1 )
	for spot_idx in range( 0, num_to_look_at ):
		look_spot = int( spots[ spot_idx ] )
		opt_loc = dense_plot_idxs[ look_spot ]
		legend_data.append( str( opt_loc ) )
		plt.plot(
			np.linspace( lambda_min_um, lambda_max_um, len( dense_plots[ look_spot ] ) ),
			dense_plots[ look_spot ], color=colors[ spot_idx % len( colors ) ], linewidth=2 )
	plt.title( "Index = " + str( max_indices[ opt_idx ] ) )
	plt.legend( legend_data )
	plt.subplot( 1, 3, 2 )
	plt.plot( figure_of_merit, color='b', linewidth=2 )
	plt.subplot( 1, 3, 3 )
	plt.plot( binarization_evolution, color='b', linewidth=2 )
	plt.show()

# sys.exit(0)


# for opt_idx in range( 0, len( max_indices ) ):
# 	min_index = min_indices[ opt_idx ]
# 	max_index = max_indices[ opt_idx ]
# 	min_relative_permittivity = min_index**2
# 	max_relative_permittivity = max_index**2

# 	device_suffix = device_suffixes[ opt_idx ]
# 	device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"

# 	make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
# 		[ device_width_voxels, device_height_voxels ],
# 		density_coarsen_factor, mesh_size_nm,
# 		[ min_relative_permittivity, max_relative_permittivity ],
# 		focal_points_x_relative, focal_length_voxels,
# 		lambda_values_um, focal_map, 0,
# 		num_layers, designable_layer_indicators, non_designable_permittivity, "",
# 		False, 0, None, 0.5 )

# 	make_optimizer.init_density_directly( np.load( device_file_prefactor + "/opt_optimized_density.npy" ) )

# 	plot_perm = make_optimizer.density_to_permittivity( make_optimizer.design_density )

# 	plt.imshow( plot_perm, cmap='Reds' )
# 	plt.title( str( min_index ) + ' ' + str( max_index ) + ' ' + str( opt_idx ) )
# 	plt.colorbar()
# 	plt.show()



# def compute_binarization( input_variable, set_point=0.5 ):
# 	total_shape = np.product( input_variable.shape )
# 	# return ( 2. / total_shape ) * np.sqrt( np.sum( ( input_variable - set_point )**2 ) )
# 	return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )
# # def compute_binarization_gradient( input_variable ):
# # 	total_shape = np.product( input_variable.shape )
# # 	return ( 1. / total_shape ) * ( input_variable - 0.5 ) / np.sum( np.sqrt( ( input_variable - 0.5 )**2 )	)

# def compute_binarization_gradient( input_variable, set_point=0.5 ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 2. / total_shape ) * np.sign( input_variable - set_point )
# 	# return ( 2. / total_shape ) * ( input_variable - set_point ) / np.sqrt( np.sum( ( input_variable - set_point )**2 ) )

# # def compute_binarization( input_variable ):
# # 	total_shape = np.product( input_variable.shape )
# # 	return ( 2 / np.sqrt( total_shape ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
# # def compute_binarization_gradient( input_variable ):
# # 	total_shape = np.product( input_variable.shape )
# # 	return ( 4 / total_shape ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )


# # def compute_binarization( input_variable ):
# # 	total_shape = np.product( input_variable.shape )
# # 	return ( 4. / total_shape ) * np.sum( ( input_variable - 0.5 )**2 )

# # def compute_binarization_gradient( input_variable ):
# # 	total_shape = np.product( input_variable.shape )
# # 	return ( 8. / total_shape ) * ( input_variable - 0.5 )

# x = np.linspace( 0, 1, 1000 )
# fd_grad = np.zeros( 1000 )
# for fd_idx in range( 0, 1000 ):
# 	start = compute_binarization( x )
# 	p = x.copy()
# 	p[ fd_idx ] += 1e-6
# 	up = compute_binarization( p )

# 	fd_grad[ fd_idx ] = ( up - start ) / 1e-6


# y = [ compute_binarization( val ) for val in x ]
# # y = compute_binarization( x )
# z = compute_binarization_gradient( x )
# # plt.plot( x, y, linewidth=2, color='r' )
# plt.plot( x, z, linewidth=2, color='g', linestyle='--' )
# plt.plot( x, fd_grad, linewidth=2, color='m', linestyle=':' )
# plt.show()
# sys.exit(0)

# device_suffix_3p5 = device_suffixes[ -1 ]
# device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"
# lsf_devices = np.load( device_file_prefactor + "/lsf_device_evolution.npy" )

# num = 1
# for device_idx in range( 0, lsf_devices.shape[ 0 ] ):
# 	if ( device_idx % 15 ) == 0:
# 		plt.subplot( 5, 4, num )
# 		num += 1
# 		plt.imshow( lsf_devices[ device_idx ], cmap='Reds' )
# plt.show()


# index_high = [ 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5 ]
# device_left_suffix = device_suffixes[ 0 ]
cmaps = ['Reds', 'Greens', 'Blues']
# device_suffixes_line = [ '1p5', '1p75', '2', '2p25', '2p5', '2p75', '3', '3p25', '3p5_lsf' ]

# for device_right_idx in range( 1, len( device_suffixes_line ) ):
# 	min_relative_permittivity = 1.0**2
# 	max_relative_permittivity1 = 1.5**2
# 	max_relative_permittivity2 = index_high[device_right_idx]**2

# 	device_right_suffix = device_suffixes_line[ device_right_idx ]
# 	device_right_suffix_compare = device_suffixes[ device_right_idx ]

# 	load_line_cut = np.load( folder_to_plot + '/line_' + device_left_suffix + '_' + device_right_suffix + '/fom_line.npy' )
# 	load_line_cut_compare = np.load( './bin_v2/line_' + device_left_suffix + '_' + device_right_suffix_compare + '/fom_line.npy' )

# 	get_densities = np.load( folder_to_plot + '/line_' + device_left_suffix + '_' + device_right_suffix + '/densities_line.npy' )
# 	get_permittivities = np.load( folder_to_plot + '/line_' + device_left_suffix + '_' + device_right_suffix + '/permittivities_line.npy' )

# 	alpha = np.linspace( 0.0, 1.0, len( load_line_cut ) )
# 	print( len( alpha ) )

# 	plt.subplot( 2, 2, 1 )
# 	plt.plot( alpha, np.log10( load_line_cut / load_line_cut[ 0 ] ), color='k', linewidth=2 )
# 	plt.plot( alpha, np.log10( load_line_cut_compare / load_line_cut[ 0 ] ), color='k', linewidth=2, linestyle='--' )
# 	plt.ylabel( 'Log( f / f(n=1.5) )', fontsize=14 )
# 	plt.xlabel( 'alpha', fontsize=14 )
# 	plt.title( 'n=' + str( index_high[0] ) + ' to n=' + str( index_high[ device_right_idx ] ), fontsize=14 )

# 	num_density_sweep = 3
# 	show_densities = np.linspace( 0, len( get_densities ) - 1, num_density_sweep )

# 	for density_idx in range( 0, num_density_sweep ):
# 		choose_density = int( show_densities[ density_idx ] )

# 		plt.subplot( 2, 2, density_idx + 2 )

# 		plt.imshow( get_permittivities[ choose_density ], cmap=cmaps[ density_idx ] )#cmap='Blues' )
# 		if density_idx == 0:
# 			plt.title( 'Device Density', fontsize=12 )

# 	plt.show()


for device_idx in range( 0, len( device_suffixes ) ):
	device_suffix = device_suffixes[ device_idx ]
	device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"
	device_file_prefactor_part2 = folder_to_plot + '/opt_' + device_suffix + "_part2/"
	# init_density = np.load( device_file_prefactor + 'opt_init_random_density.npy' )
	final_density = np.load( device_file_prefactor + 'opt_optimized_density.npy' )
	figure_of_merit = np.load( device_file_prefactor + 'opt_fom_evolution.npy' )
	# figure_of_merit_part2 = np.load( device_file_prefactor_part2 + 'opt_fom_evolution.npy' )
	binarization_evolution = np.load( device_file_prefactor + 'opt_binarization_evolution.npy' )
	# binarization_evolution_part2 = np.load( device_file_prefactor_part2 + 'opt_binarization_evolution.npy' )

	print( final_density.shape )

	# lsf_fom_evolution = None
	# if device_idx == ( len( device_suffixes ) - 1 ):
	# 	lsf_fom_evolution = np.load( device_file_prefactor + '/lsf_fom_evolution.npy' )

	# print( len( figure_of_merit ) )
	# print( len( figure_of_merit_part2))

	figure_of_merit = np.array( list( figure_of_merit ) )# + list( figure_of_merit_part2 ) )
	binarization = np.array( list( binarization_evolution ) )#  + list( binarization_evolution_part2 ) )


	# device_file_prefactor2 = './bin_dropout_p25_v1/opt_' + device_suffix + "/"
	# device_file_prefactor2 = './bin_unif_thinner_excl_v1/opt_' + device_suffix + "/"
	device_file_prefactor2 = './bin_v2/opt_' + device_suffix + "/"
	# device_file_prefactor2_part2 = './bin_v2/opt_' + device_suffix + "_part2/"
	figure_of_merit2 = np.load( device_file_prefactor2 + 'opt_fom_evolution.npy' )
	# figure_of_merit2_part2 = np.load( device_file_prefactor2_part2 + 'opt_fom_evolution.npy' )
	# figure_of_merit2_total = np.array( list( figure_of_merit2 ) + list( figure_of_merit2_part2 ) )

	binarization_evolution2 = np.load( device_file_prefactor2 + 'opt_binarization_evolution.npy' )
	# binarization_evolution2_part2 = np.load( device_file_prefactor2_part2 + 'opt_binarization_evolution.npy' )
	# binarization_evolution2_total = np.array( list( binarization_evolution2 ) + list( binarization_evolution2_part2 ) )

	final_density_compare = np.load( device_file_prefactor2 + '/opt_optimized_density.npy' )

	plt.subplot( 1, 2, 1 )
	plt.imshow( final_density, cmap='Greens' )
	plt.subplot( 1, 2, 2 )
	plt.imshow( final_density_compare, cmap='Greens' )
	plt.show()



for device_idx in range( 0, len( device_suffixes ) ):
	device_suffix = device_suffixes[ device_idx ]
	device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"
	device_file_prefactor_part2 = folder_to_plot + '/opt_' + device_suffix + "_part2/"
	# init_density = np.load( device_file_prefactor + 'opt_init_random_density.npy' )
	final_density = np.load( device_file_prefactor + 'opt_optimized_density.npy' )
	figure_of_merit = np.load( device_file_prefactor + 'opt_fom_evolution.npy' )
	# figure_of_merit_part2 = np.load( device_file_prefactor_part2 + 'opt_fom_evolution.npy' )
	binarization_evolution = np.load( device_file_prefactor + 'opt_binarization_evolution.npy' )
	# binarization_evolution_part2 = np.load( device_file_prefactor_part2 + 'opt_binarization_evolution.npy' )

	# lsf_fom_evolution = None
	# if device_idx == ( len( device_suffixes ) - 1 ):
	# 	lsf_fom_evolution = np.load( device_file_prefactor + '/lsf_fom_evolution.npy' )

	# print( len( figure_of_merit ) )
	# print( len( figure_of_merit_part2))

	figure_of_merit = np.array( list( figure_of_merit ) )# + list( figure_of_merit_part2 ) )
	binarization = np.array( list( binarization_evolution ) )#  + list( binarization_evolution_part2 ) )


	# device_file_prefactor2 = './bin_unif_thinner_excl_v1/opt_' + device_suffix + "/"
	device_file_prefactor2 = './bin_unif_v2/opt_' + device_suffix + "/"
	# device_file_prefactor2_part2 = './bin_v2/opt_' + device_suffix + "_part2/"
	figure_of_merit2 = np.load( device_file_prefactor2 + 'opt_fom_evolution.npy' )
	# figure_of_merit2_part2 = np.load( device_file_prefactor2_part2 + 'opt_fom_evolution.npy' )
	# figure_of_merit2_total = np.array( list( figure_of_merit2 ) + list( figure_of_merit2_part2 ) )

	binarization_evolution2 = np.load( device_file_prefactor2 + 'opt_binarization_evolution.npy' )
	# binarization_evolution2_part2 = np.load( device_file_prefactor2_part2 + 'opt_binarization_evolution.npy' )
	# binarization_evolution2_total = np.array( list( binarization_evolution2 ) + list( binarization_evolution2_part2 ) )


	# device_file_prefactor3 = './bin_adv_excl_v4/opt_' + device_suffix + "/"
	device_file_prefactor3 = './bin_unif_thinner_excl_v3/opt_' + device_suffix + "/"
	# device_file_prefactor3_part2 = './bin_v2/opt_' + device_suffix + "_part2/"
	figure_of_merit3 = np.load( device_file_prefactor3 + 'opt_fom_evolution.npy' )
	# figure_of_merit3_part2 = np.load( device_file_prefactor3_part2 + 'opt_fom_evolution.npy' )
	# figure_of_merit3_total = np.array( list( figure_of_merit3 ) + list( figure_of_merit3_part2 ) )

	binarization_evolution3 = np.load( device_file_prefactor3 + 'opt_binarization_evolution.npy' )
	# binarization_evolution3_part2 = np.load( device_file_prefactor3_part2 + 'opt_binarization_evolution.npy' )
	# binarization_evolution3_total = np.array( list( binarization_evolution3 ) + list( binarization_evolution3_part2 ) )



	# figure_of_merit_part2 = np.load( device_file_prefactor_part2 + 'opt_fom_evolution.npy' )
	# binarization_evolution_part2 = np.load( device_file_prefactor_part2 + 'opt_binarization_evolution.npy' )

	plot_fom = np.array( list( figure_of_merit ) )#+ list( figure_of_merit_part2 ) )
	plot_binarization = np.array( list( binarization ) )#+ list( binarization_evolution_part2 ) )

	print( len( plot_fom ) )

	# print( figure_of_merit[ 0 ] )
	print( figure_of_merit[ -1 ] )
	print( fom_compare )
	print()

	# fom_compare = figure_of_merit[ 0 ]

	# plt.subplot( 2, 2, 1 )
	# plt.imshow( init_density, cmap='gray' )
	# plt.colorbar()
	# plt.subplot( 2, 2, 2 )
	# plt.imshow( final_density, cmap='gray' )
	# plt.colorbar()
	# plt.subplot( 2, 2, 3 )
	# plt.plot( binarization_evolution, color='g', linewidth=2 )
	# plt.subplot( 2, 2, 4 )

	# plt.subplot( 1, 2, 1 )
	# plt.plot( np.log10( figure_of_merit / fom_compare ), color=colors[ device_idx ], linewidth=2 )
	# plt.subplot( 1, 2, 2 )
	# plt.plot( binarization_evolution, color=colors[ device_idx ], linewidth=2 )
	

	plt.subplot( 1, 2, 1 )
	plt.plot( np.log10( plot_fom / fom_compare ), color=colors[ device_idx ], linewidth=2 )
	# plt.plot( plot_fom, color=colors[ device_idx ], linewidth=2 )
	plt.legend( device_legend )
	plt.ylabel( 'Log( f / f_0 )', fontsize=16 )
	plt.xlabel( 'Iteration', fontsize=16 )
	# plt.plot( np.log10( figure_of_merit2_total / fom_compare ), color=colors[ device_idx ], linewidth=2, linestyle='--' )
	# plt.plot( np.log10( figure_of_merit2 / fom_compare ), color=colors[ device_idx ], linewidth=2, linestyle='--' )
	# plt.plot( np.log10( figure_of_merit3 / fom_compare ), color=colors[ device_idx ], linewidth=2, linestyle=':' )
	# plt.plot( figure_of_merit2, color=colors[ device_idx ], linewidth=2, linestyle='--' )

	# if device_idx == ( len( device_suffixes ) - 1 ):

	# 	total_with_lsf_fom = np.array( list( figure_of_merit2_total ) + list( lsf_fom_evolution ) )

	# 	plt.plot( np.log10( total_with_lsf_fom / fom_compare ), color=colors[ device_idx ], linewidth=2 )


	# plt.plot( np.log10( figure_of_merit3_total / fom_compare ), color=colors[ device_idx ], linewidth=2, linestyle=':' )
	plt.subplot( 1, 2, 2 )
	plt.plot( plot_binarization, color=colors[ device_idx ], linewidth=2 )
	plt.ylabel( 'Binarization', fontsize=16 )
	plt.xlabel( 'Iteration', fontsize=16 )
	plt.legend( device_legend )
	# plt.plot( binarization_evolution2_total, color=colors[ device_idx ], linewidth=2, linestyle='--' )
	# plt.plot( binarization_evolution2, color=colors[ device_idx ], linewidth=2, linestyle='--' )
	# plt.plot( binarization_evolution3, color=colors[ device_idx ], linewidth=2, linestyle=':' )
	

	# plt.show()

plt.show()
# sys.exit(0)

# for device_idx in range( 0, len( device_suffixes ) ):
# 	device_suffix = device_suffixes[ device_idx ]
# 	device_file_prefactor = folder_to_plot + '/opt_' + device_suffix + "/"
# 	device_file_prefactor_part2 = folder_to_plot + '/opt_' + device_suffix + "_part2/"
# 	init_density = np.load( device_file_prefactor + 'opt_init_random_density.npy' )
# 	final_density = np.load( device_file_prefactor + 'opt_optimized_density.npy' )
# 	directions = np.load( device_file_prefactor + 'opt_gradient_directions.npy' )

# 	plt_idx = 1
# 	for k in [ 0, 50, 100, 150, 200, 250, 300, 350, 400 ]:
# 		plt.subplot( 3, 3, plt_idx )
# 		plt_idx += 1
# 		get_direction = directions[ k ]
# 		fft_direction = np.fft.fftshift( np.fft.fft2( get_direction ) )
# 		# plt.imshow( directions[ k ], cmap='Blues' )
# 		plt.imshow( np.real( fft_direction ), cmap='Reds' )
# 		plt.colorbar()
# 	plt.show()

# 	plt.subplot( 1, 2, 1 )
# 	plt.imshow( init_density, cmap='Greens' )
# 	plt.clim( [ 0, 1 ] )
# 	plt.colorbar()
# 	plt.subplot( 1, 2, 2 )
# 	plt.imshow( final_density, cmap='Greens' )
# 	plt.clim( [ 0, 1 ] )
# 	plt.colorbar()
# 	plt.show()

# sys.exit(0)

device_left_suffix = device_suffixes[ 0 ]
# get_density_1 = np.load( folder_to_plot + '/opt_' + device_suffixes[ 0 ] + "_part2/opt_optimized_density.npy" )

num_alpha = 200
alpha = np.linspace( 0.0, 1.0, num_alpha )


index_high = [ 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5 ]

for device_right_idx in range( 0, len( device_suffixes ) ):
	if device_right_idx == 0:
		continue
	min_relative_permittivity = 1.0**2
	# max_relative_permittivity1 = 1.5**2
	max_relative_permittivity1 = index_high[ 4 ]**2
	max_relative_permittivity2 = index_high[device_right_idx]**2

	device_right_suffix = device_suffixes[ device_right_idx ]

	load_line_cut = np.load( folder_to_plot + '/line_' + device_left_suffix + '_' + device_right_suffix + '/fom_line.npy' )
	# load_line_cut_compare = np.load( './bin_v2/line_' + device_left_suffix + '_' + device_right_suffix + '/fom_line.npy' )
	# print(load_line_cut)

	get_densities = np.load( folder_to_plot + '/line_' + device_left_suffix + '_' + device_right_suffix + '/densities_line.npy' )
	get_permittivities = np.load( folder_to_plot + '/line_' + device_left_suffix + '_' + device_right_suffix + '/permittivities_line.npy' )

	alpha = np.linspace( 0.0, 1.0, len( load_line_cut ) )
	print( len( alpha ) )

	plt.subplot( 2, 2, 1 )
	plt.plot( alpha, np.log10( load_line_cut / load_line_cut[ 0 ] ), color='k', linewidth=2 )
	plt.ylabel( 'Log( f / f(n=1.5) )', fontsize=14 )
	plt.xlabel( 'alpha', fontsize=14 )
	plt.title( 'n=' + str( index_high[0] ) + ' to n=' + str( index_high[ device_right_idx ] ), fontsize=14 )
	# plt.plot( alpha, np.log10( load_line_cut_compare / load_line_cut[ 0 ] ), color='g', linewidth=1 )

	num_density_sweep = 3
	show_densities = np.linspace( 0, len( get_densities ) - 1, num_density_sweep )

	for density_idx in range( 0, num_density_sweep ):
		choose_density = int( show_densities[ density_idx ] )
		# density = get_densities[ choose_density ]

		# get_density_2 = np.load( folder_to_plot + '/opt_' + device_right_suffix + "_part2/opt_optimized_density.npy" )
		# get_density_2 = np.load( folder_to_plot + '/opt_' + device_right_suffix + "_part2/opt_optimized_density.npy" )

		# get_permittivity_1 = min_relative_permittivity + ( max_relative_permittivity1 - min_relative_permittivity ) * get_density_1
		# get_permittivity_2 = min_relative_permittivity + ( max_relative_permittivity2 - min_relative_permittivity ) * get_density_2
		
		# middle_permittivity = ( 1. - alpha[ choose_density ] ) * get_permittivity_1 + alpha[ choose_density ] * get_permittivity_2


		# max_relative_permittivity_both_opts = np.maximum( max_relative_permittivity1, max_relative_permittivity2 )

		# make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
		# 	[ device_width_voxels, device_height_voxels ],
		# 	density_coarsen_factor, mesh_size_nm,
		# 	[ min_relative_permittivity, max_relative_permittivity_both_opts ],
		# 	focal_points_x_relative, focal_length_voxels,
		# 	lambda_values_um, focal_map, 0,
		# 	num_layers, designable_layer_indicators, non_designable_permittivity,
		# 	None )

		# middle_density = density_bound_from_eps( middle_permittivity, min_relative_permittivity, max_relative_permittivity_both_opts )
		# make_optimizer.init_density_directly( middle_density )

		# omega_max = make_optimizer.omega_values[ 0 ]
		# fwd_Ez = make_optimizer.compute_forward_fields( omega_max, make_optimizer.get_device_permittivity() )

		plt.subplot( 2, 2, density_idx + 2 )
		# rect = patches.Rectangle(
		# 	( make_optimizer.device_height_start, make_optimizer.device_width_start ),
		# 	make_optimizer.device_height_voxels, make_optimizer.device_width_voxels, 
		# 	linewidth=3, edgecolor='k', facecolor='none' )

		# for focal_pt in range( 0, 2 ):
		# 	get_fx = make_optimizer.focal_spots_x_voxels[ focal_pt ]
		# 	get_fy = make_optimizer.focal_point_y

		# 	focal_circle = patches.Circle(
		# 		( get_fy, get_fx ),
		# 		2, color='r', facecolor='r' )
		# 	plt.gca().add_patch( focal_circle )

		# plt.gca().add_patch( rect )

		plt.imshow( get_permittivities[ choose_density ], cmap=cmaps[ density_idx ] )#cmap='Blues' )
		if density_idx == 0:
			plt.title( 'Device Density', fontsize=12 )
		# plt.imshow( make_optimizer.get_device_permittivity(), cmap='Blues' )
		# plt.imshow( np.abs( fwd_Ez ), cmap='gray' )
		# plt.colorbar()

	plt.show()


# mesh_size_nm = 8
# density_coarsen_factor = 4
# mesh_size_m = mesh_size_nm * 1e-9
# lambda_min_um = 0.45
# lambda_max_um = 0.55
# num_lambda_values = 8

# max_index = 3.0
# min_relative_permittivity = 1.0**2
# max_relative_permittivity = max_index**2

# def density_bound_from_eps( eps_val ):
# 	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

# lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

# device_width_voxels = 120
# device_height_voxels = 100
# device_voxels_total = device_width_voxels * device_height_voxels
# focal_length_voxels = 100
# focal_points_x_relative = [ 0.25, 0.75 ]

# num_layers = int( device_height_voxels / density_coarsen_factor )
# spacer_permittivity = 1.0**2
# designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
# non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

# focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
# for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
# 	focal_map[ idx ] = 1


# make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
# 	[ device_width_voxels, device_height_voxels ],
# 	density_coarsen_factor, mesh_size_nm,
# 	[ min_relative_permittivity, max_relative_permittivity ],
# 	focal_points_x_relative, focal_length_voxels,
# 	lambda_values_um, focal_map, 0,
# 	num_layers, designable_layer_indicators, non_designable_permittivity )

# make_optimizer.init_density_directly( np.load( './bin_v1/binarize_opt_' + device_suffixes[ 3 ] + "/opt_optimized_density.npy" ))

# make_optimizer.compute_net_fom()



