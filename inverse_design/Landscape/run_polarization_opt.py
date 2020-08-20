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
density_coarsen_factor = 24#1#12#24#12#32#24#48#12#48
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.55
num_lambda_values = 1

# bandwidth_um = lambda_max_um - lambda_min_um
# exclusion_um = 0.030
# modified_bandwidth_um = bandwidth_um - exclusion_um
# left_middle_bound_um = lambda_min_um + 0.5 * modified_bandwidth_um
# right_middle_bound_um = left_middle_bound_um + exclusion_um

# num_left_lambda = int( 0.5 * num_lambda_values )
# num_right_lambda = num_lambda_values - num_left_lambda
# lambda_left = np.linspace( lambda_min_um, left_middle_bound_um, num_left_lambda )
# lambda_right = np.linspace( right_middle_bound_um, lambda_max_um, num_right_lambda )

lambda_left = np.array( [ lambda_min_um ] )
lambda_right = np.array( [ lambda_max_um ] )

min_relative_permittivity = 1.0**2
max_relative_permittivity = max_index**2

def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

# lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )
# lambda_values_um = np.array( list( lambda_left ) + list( lambda_right ) )
lambda_values_um = np.array( list( lambda_left ) )# + list( lambda_right ) )

device_width_voxels = 10 * density_coarsen_factor
device_height_voxels = 4 * density_coarsen_factor
# device_width_voxels = 8#24#12#24#12#64#8#48
# device_height_voxels = 8#24#24#12#64#8#48
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

mean_density = 0.5
sigma_density = 0.2
init_from_old = False
binarize_set_point = 0.5

blur_fields_size_voxels = 0
blur_fields = False

num_iterations_nominal = 100#60#300
num_iterations = int( np.ceil(
	num_iterations_nominal * ( max_relative_permittivity - min_relative_permittivity ) / ( 1.5**2 - min_relative_permittivity ) ) )

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

design_width = int( device_width_voxels / density_coarsen_factor )
design_height = int( device_height_voxels / density_coarsen_factor )
num_design_voxels = design_width * design_height

use_pairings = False

dense_plot_freq_iters = 10
num_dense_wls = 4 * num_lambda_values
dense_plot_wls = np.linspace( lambda_min_um, lambda_max_um, num_dense_wls )

dense_focal_map = [ 0 for idx in range( 0, num_dense_wls ) ]
for idx in range( int( 0.5 * num_dense_wls ), num_dense_wls ):
	dense_focal_map[ idx ] = 1


binarize = False#True
binarize_movement_per_step_nominal = 0.0075
binarize_max_movement_per_voxel_nominal = 0.0075

rho_delta_scaling = ( 1.5**2 - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )
binarize_movement_per_step = binarize_movement_per_step_nominal * rho_delta_scaling
binarize_max_movement_per_voxel = binarize_max_movement_per_voxel_nominal * rho_delta_scaling

dropout_start = 0
dropout_end = 0
dropout_p = 0.1

use_log_fom = False

wavelength_adversary = False
adversary_update_iters = 10

# dual_opt = True

make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )

# test_density = np.load( 'opt_with_pol_v1/opt_1p5/opt_optimized_density.npy' )

# random_density = np.random.random( ( design_width, design_height ) )

# make_optimizer.init_density_directly( random_density )

make_optimizer.init_density_with_uniform( 0.5 )

# make_optimizer.optimize(
# 	int( num_iterations ),
# 	save_folder + "/opt",
# 	True,
# 	# False,
# 	False, 20, 20, 0.95,
# 	None,
# 	use_log_fom,
# 	wavelength_adversary, adversary_update_iters, lambda_left, lambda_right,
# 	binarize, 1.5 * binarize_movement_per_step, 1.5 * binarize_max_movement_per_voxel,
# 	dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map )

# make_optimizer.verify_adjoint_against_finite_difference_lambda_design( save_folder + "/opt" )
make_optimizer.verify_adjoint_against_finite_difference_lambda_design_anisotropic( save_folder + "/opt" )

# make_optimizer.save_optimization_data( save_folder + "/opt" )
