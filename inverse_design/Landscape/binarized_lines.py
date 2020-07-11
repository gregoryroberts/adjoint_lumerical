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

if len( sys.argv ) < 6:
	print( "Usage: python " + sys.argv[ 0 ] + " { optimization folder base } { max index 1 } { max_index 2 } { optimiation 1 name } { optimiation 2 name }" )
	sys.exit( 1 )

opt_folder_base = sys.argv[ 1 ]
max_index1 = float( sys.argv[ 2 ] )
max_index2 = float( sys.argv[ 3 ] )
opt1_name = sys.argv[ 4 ]
opt2_name = sys.argv[ 5 ]

save_folder = opt_folder_base + "/line_" + opt1_name + "__" + opt2_name + "/"

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

get_density_1 = np.load( opt_folder_base + "/" + opt1_name + "/opt_optimized_density.npy" )
# get_density_2 = np.load( opt_folder_base + "/" + opt2_name + "/opt_optimized_density.npy" )
get_densities_2 = np.load( opt_folder_base + "/" + opt2_name + "/lsf_device_evolution.npy" )
get_density_2 = get_densities_2[ get_densities_2.shape[ 0 ] - 1 ]

min_relative_permittivity = 1.0**2
max_relative_permittivity1 = max_index1**2
max_relative_permittivity2 = max_index2**2

max_relative_permittivity_both_opts = np.maximum( max_relative_permittivity1, max_relative_permittivity2 )

get_permittivity_1 = min_relative_permittivity + ( max_relative_permittivity1 - min_relative_permittivity ) * get_density_1
get_permittivity_2 = min_relative_permittivity + ( max_relative_permittivity2 - min_relative_permittivity ) * get_density_2

num_alpha = 200
alpha = np.linspace( 0.0, 1.0, num_alpha )
fom_values = np.zeros( num_alpha )

permittivities = np.zeros( [ num_alpha ] + list( get_permittivity_1.shape ) )
densities = np.zeros( [ num_alpha ] + list( get_permittivity_1.shape ) )

for alpha_idx in range( 0, num_alpha ):
	middle_permittivity = ( 1. - alpha[ alpha_idx ] ) * get_permittivity_1 + alpha[ alpha_idx ] * get_permittivity_2

	# make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	# 	[ device_width_voxels, device_height_voxels ],
	# 	density_coarsen_factor, mesh_size_nm,
	# 	[ min_relative_permittivity, max_relative_permittivity_both_opts ],
	# 	focal_points_x_relative, focal_length_voxels,
	# 	lambda_values_um, focal_map, 0,
	# 	num_layers, designable_layer_indicators, non_designable_permittivity,
	# 	save_folder )

	make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
		[ device_width_voxels, device_height_voxels ],
		density_coarsen_factor, mesh_size_nm,
		[ min_relative_permittivity, max_relative_permittivity ],
		focal_points_x_relative, focal_length_voxels,
		lambda_values_um, focal_map, random_seed,
		num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
		False, 0, None, 0.5 )


	middle_density = density_bound_from_eps( middle_permittivity, min_relative_permittivity, max_relative_permittivity_both_opts )
	make_optimizer.init_density_directly( middle_density )

	permittivities[ alpha_idx ] = middle_permittivity
	densities[ alpha_idx ] = middle_density

	fom_values[ alpha_idx ] = make_optimizer.compute_net_fom()

	np.save( save_folder + "/fom_line.npy", fom_values )
	np.save( save_folder + "/permittivities_line.npy", permittivities )
	np.save( save_folder + "/densities_line.npy", densities )

