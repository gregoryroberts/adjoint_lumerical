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


if ( max_index > 1.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

mesh_size_nm = 25
# mesh_size_nm = 8#6#4#8#6#8
density_coarsen_factor = 16#20
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


#
# todo: focal spot sizes might be a touch big here...
#
# device_width_voxels = 120#160#120
# device_width_voxels = 162
# device_width_voxels = 80#200
device_width_voxels = 320
# device_height_voxels = 80 * 5
# device_height_voxels = 120
# device_height_voxels = 800
# device_height_voxels = 600
device_height_voxels = 400
# device_height_voxels = 200
# spacing_device_height_voxels = 40
# device_height_voxels = 72#100#72
# device_height_voxels = #52#64#52
# device_height_voxels = 48#32
# device_height_voxels = 32#24
device_voxels_total = device_width_voxels * device_height_voxels
# focal_length_voxels = 50#135#100#132#100
focal_length_voxels = 200
focal_points_x_relative = [ 0.25, 0.75 ]


num_layers = int( device_height_voxels / density_coarsen_factor )
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ idx ] = 1

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

design_width = int( device_width_voxels / density_coarsen_factor )
design_height = int( device_height_voxels / density_coarsen_factor )
num_design_voxels = design_width * design_height


make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )


loc0 = '/central/groups/Faraon_Computing/projects/ten_um_bin_sum_fom_ratio_wide_longer_f_v1_1p5/'
loc1 = '/central/groups/Faraon_Computing/projects/ten_um_bin_sum_fom_ratio_wide_longer_f_v2_1p5/'

density0 = np.load( loc0 + 'opt_optimized_density.npy' )
density1 = np.load( loc1 + 'opt_optimized_density.npy' )

num_alpha = 10
alphas = np.linspace( 0, 1, num_alpha )

fom_line = np.zeros( num_alpha )

for alpha_idx in range( 0, num_alpha ):
	alpha = alphas[ alpha_idx ]

	mix_density = alpha * density1 + ( 1. - alpha ) * density0

	net_fom = make_optimizer.compute_net_ratio_sum_fom_from_density( mix_density )

	fom_line[ alpha_idx ] = net_fom

np.save( '/central/groups/Faraon_Computing/projects/ten_um_bin_sum_fom_ratio_wide_longer_f_v1_1p5/fom_line_v1_v2.npy', fom_line )
np.save( '/central/groups/Faraon_Computing/projects/ten_um_bin_sum_fom_ratio_wide_longer_f_v2_1p5/fom_line_v1_v2.npy', fom_line )





