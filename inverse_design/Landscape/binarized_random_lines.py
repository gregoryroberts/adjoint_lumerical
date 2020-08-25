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

bandwidth_um = lambda_max_um - lambda_min_um
exclusion_um = 0.030
modified_bandwidth_um = bandwidth_um - exclusion_um
left_middle_bound_um = lambda_min_um + 0.5 * modified_bandwidth_um
right_middle_bound_um = left_middle_bound_um + exclusion_um

num_left_lambda = int( 0.5 * num_lambda_values )
num_right_lambda = num_lambda_values - num_left_lambda
lambda_left = np.linspace( lambda_min_um, left_middle_bound_um, num_left_lambda )
lambda_right = np.linspace( right_middle_bound_um, lambda_max_um, num_right_lambda )

#
# todo: REDO LINE SEARCHES FOR EXCLUSION OPTIMIZATIONS BECAUSE YOU WERE USING DIFFERENT
# LAMBDA ARRAYS! in general, check to make sure FOM matches up on either side.  Also
# make sure this file matches up with other file.  These need to be put together in a
# better way to ensure the same parameters are being used in both places!
#

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )
# lambda_values_um = np.array( list( lambda_left ) + list( lambda_right ) )

device_width_voxels = 120
# device_height_voxels = 104#100
device_height_voxels = 100#16#32


single_pass_transmittance = 0.9
device_height_m = device_height_voxels * mesh_size_nm * 1e-9
lambda_min_m = lambda_min_um * 1e-6
loss_index = -lambda_min_m * np.log( single_pass_transmittance ) / ( device_height_m * 2 * np.pi )

real_permittivity_1 = max_index1**2 - loss_index**2
imag_permittivity_1 = 2 * np.sqrt( real_permittivity_1 ) * loss_index

real_permittivity_2 = max_index2**2 - loss_index**2
imag_permittivity_2 = 2 * np.sqrt( real_permittivity_2 ) * loss_index







design_width_voxels = int( device_width_voxels / density_coarsen_factor )
design_height_voxels = int( device_height_voxels / density_coarsen_factor )

# device_height_voxels = 72
# device_height_voxels = 52
# device_height_voxels = 32
# device_height_voxels = 24
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

np.random.seed( 5234234 )
get_density_1 = 1.0 * np.greater( np.random.normal( 0.5, 0.5, ( design_width_voxels, design_height_voxels ) ), 0.5 )
get_density_2 = 1.0 * np.greater( np.random.normal( 0.5, 0.5, ( design_width_voxels, design_height_voxels ) ), 0.5 )

# get_density_1 = np.load( opt_folder_base + "/" + opt1_name + "/opt_optimized_density.npy" )
# get_density_2 = np.load( opt_folder_base + "/" + opt2_name + "/opt_optimized_density.npy" )

# get_densities_2 = np.load( opt_folder_base + "/" + opt2_name + "/level_set_device_evolution.npy" )
# get_density_2 = get_densities_2[ get_densities_2.shape[ 0 ] - 1 ]

min_relative_permittivity = 1.0**2
max_relative_permittivity1 = real_permittivity_1 + 1j * imag_permittivity_1
max_relative_permittivity2 = real_permittivity_2 + 1j * imag_permittivity_2

max_relative_permittivity_both_opts = np.maximum( max_relative_permittivity1, max_relative_permittivity2 )

get_permittivity_1 = min_relative_permittivity + ( max_relative_permittivity1 - min_relative_permittivity ) * get_density_1
get_permittivity_2 = min_relative_permittivity + ( max_relative_permittivity2 - min_relative_permittivity ) * get_density_2

num_alpha = 200
alpha = np.linspace( 0.0, 1.0, num_alpha )
fom_values = np.zeros( num_alpha )
binarization_values = np.zeros( num_alpha )

permittivities = np.zeros( [ num_alpha ] + list( get_permittivity_1.shape ) )
densities = np.zeros( [ num_alpha ] + list( get_permittivity_1.shape ) )


for alpha_idx in range( 0, num_alpha ):
	middle_permittivity = ( 1. - alpha[ alpha_idx ] ) * get_permittivity_1 + alpha[ alpha_idx ] * get_permittivity_2

	make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
		[ device_width_voxels, device_height_voxels ],
		density_coarsen_factor, mesh_size_nm,
		[ min_relative_permittivity, max_relative_permittivity_both_opts ],
		focal_points_x_relative, focal_length_voxels,
		lambda_values_um, focal_map, 0,
		num_layers, designable_layer_indicators, non_designable_permittivity,
		save_folder,
		False, 0, None, 0.5 )


	middle_density = density_bound_from_eps( middle_permittivity, min_relative_permittivity, max_relative_permittivity_both_opts )
	make_optimizer.init_density_directly( middle_density )

	permittivities[ alpha_idx ] = middle_permittivity
	densities[ alpha_idx ] = middle_density

	fom_values[ alpha_idx ] = make_optimizer.compute_net_fom()

	np.save( save_folder + "/fom_line.npy", fom_values )
	np.save( save_folder + "/permittivities_line.npy", permittivities )
	np.save( save_folder + "/densities_line.npy", densities )

