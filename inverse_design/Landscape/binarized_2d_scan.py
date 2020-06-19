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

if len( sys.argv ) < 5:
	print( "Usage: python " + sys.argv[ 0 ] + " { optimization folder base } { max index } { optimization name } { random_seed }" )
	sys.exit( 1 )

opt_folder_base = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )
opt_name = sys.argv[ 3 ]
random_seed = int( sys.argv[ 4 ] )

np.random.seed( random_seed )

save_folder = opt_folder_base + "/" + opt_name + "_2d_scan/"

np.save( save_folder + "/random_seed.npy", np.array( random_seed ) )

if not os.path.isdir(save_folder):
	os.mkdir(save_folder)

mesh_size_nm = 8
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.55
num_lambda_values = 8

def density_bound_from_eps( eps_val, min_perm, max_perm ):
	return ( eps_val - min_perm ) / ( max_perm - min_perm )

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 120
device_height_voxels = 100
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 100
focal_points_x_relative = [ 0.25, 0.75 ]

num_layers = int( device_height_voxels / density_coarsen_factor )
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ idx ] = 1

get_density = np.load( opt_folder_base + "/" + opt_name + "/opt_optimized_density.npy" )

min_relative_permittivity = 1.0**2
max_relative_permittivity = max_index**2

search_dim = 15

def pick_search_direction( binary_density ):
	flatten_density = binary_density.flatten()

	search_direction = np.random.normal( 1.0, 0.25, len( flatten_density ) )
	search_direction -= np.min( search_direction )

	for idx in range( 0, len( flatten_density ) ):
		if flatten_density[ idx ] > 0.5:
			search_direction[ idx ] *= -1.0

	search_direction /= np.sqrt( np.sum( search_direction**2 ) )

	return np.reshape( search_direction, binary_density.shape )

search_delta = pick_search_direction( get_density )
search_eta = pick_search_direction( get_density )

max_abs_direction = np.max( np.abs( search_delta ), np.abs( search_eta ) )

search_limits = [ 0, 0.01 * search_dim / max_abs_direction ]
search_weights = np.linspace( search_limits[ 0 ], search_limits[ 1 ], search_dim )

make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, 0,
	num_layers, designable_layer_indicators, non_designable_permittivity )

fom_values = np.zeros( ( search_dim, search_dim ) )

for delta_idx in range( 0, search_dim ):
	for eta_idx in range( 0, search_dim ):
		test_density = get_density + search_weights[ delta_idx ] * search_delta + search_weights[ eta_idx ] * search_eta

		make_optimizer.init_density_directly( middle_density )

		fom_values[ delta_idx, eta_idx ] = make_optimizer.compute_net_fom()
		np.save( save_folder + "/fom_2d.npy", fom_values )
