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

min_relative_permittivity = 1.0**2
max_relative_permittivity = max_index**2

def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 120
# device_width_voxels = 200
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

mean_density = 0.5
sigma_density = 0.2
init_from_old = False#True
binarize_set_point = 0.5

blur_fields_size_voxels = 0#4
blur_fields = False#True

num_density_iterations = 100#150#300
num_lsf_iterations = 250

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

design_width = int( device_width_voxels / density_coarsen_factor )
design_height = int( device_height_voxels / density_coarsen_factor )
num_design_voxels = design_width * design_height
assert ( num_design_voxels % 2 ) == 0, 'Design voxel pairings cannot be made'

make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )

if init_from_old:
	density = np.load( save_folder + "/opt_optimized_density.npy" )
	make_optimizer.init_density_directly( density )
else:
	make_optimizer.init_density_with_uniform( 0.5 )
	make_optimizer.optimize( num_density_iterations )
	make_optimizer.save_optimization_data( save_folder + "/opt" )

make_optimizer.optimize_with_level_set( num_lsf_iterations )

np.save( save_folder + "/lsf_fom_evolution.npy", make_optimizer.lsf_fom_evolution )
np.save( save_folder + "/lsf_fom_by_wl_evolution.npy", make_optimizer.lsf_fom_by_wl_evolution )
np.save( save_folder + "/level_set_device_evolution.npy", make_optimizer.level_set_device_evolution )

# for iter_idx in range( 0, 5 ):
# 	plt.subplot( 3, 3, iter_idx + 1 )
# 	plt.imshow( make_optimizer.level_set_device_evolution[ iter_idx], cmap='Greens' )
# 	plt.colorbar()
# plt.show()
