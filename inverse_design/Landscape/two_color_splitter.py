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
import ColorSplittingOptimization2DSigmoid

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )

if ( max_index > 3.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

mesh_size_nm = 150
density_coarsen_factor = 5
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 3.5
lambda_max_um = 4.5
num_lambda_values = 2

min_relative_permittivity = 1.0**2
max_relative_permittivity = max_index**2

def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

lambda_values_um = np.array( [ lambda_min_um, lambda_max_um ] )

device_width_voxels = 240
device_height_voxels = 240

device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 160
focal_points_x_relative = [ 0.25, 0.75 ]

# num_layers = int( device_height_voxels / density_coarsen_factor )
num_layers = 16
spacer_permittivity = 1.0**2
designable_layer_indicators = [ ( ( idx % 2 ) == 0 ) for idx in range( 0, num_layers ) ]

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

num_iterations_nominal = 300
num_iterations = int( np.ceil(
	num_iterations_nominal * ( max_relative_permittivity - min_relative_permittivity ) / ( 1.5**2 - min_relative_permittivity ) ) )

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

design_width = int( device_width_voxels / density_coarsen_factor )
design_height = int( device_height_voxels / density_coarsen_factor )
num_design_voxels = design_width * design_height

use_log_fom = False

dense_plot_freq_iters = 50#10
num_dense_wls = 4 * num_lambda_values
dense_plot_wls = np.linspace( lambda_min_um, lambda_max_um, num_dense_wls )

dense_focal_map = [ 0 for idx in range( 0, num_dense_wls ) ]
for idx in range( int( 0.5 * num_dense_wls ), num_dense_wls ):
	dense_focal_map[ idx ] = 1

binarize = True
binarize_movement_per_step_nominal = 0.0075 / 3.
binarize_max_movement_per_voxel_nominal = 0.0075 / 9.# / 10.

rho_delta_scaling = ( 1.5**2 - np.real( min_relative_permittivity ) ) / np.real( max_relative_permittivity - min_relative_permittivity )
binarize_movement_per_step = binarize_movement_per_step_nominal * rho_delta_scaling
binarize_max_movement_per_voxel = binarize_max_movement_per_voxel_nominal * rho_delta_scaling

wavelength_adversary = False
adversary_update_iters = 10

dropout_start = 0
dropout_end = 0
dropout_p = 0.1

make_optimizer = ColorSplittingOptimization2DSigmoid.ColorSplittingOptimization2DSigmoid(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder,
	blur_fields, blur_fields_size_voxels, None, binarize_set_point )

make_optimizer.init_density_with_uniform( mean_density )

# make_optimizer.plot_geometry()

# opt_density = np.load( '/Users/gregory/Downloads/opt_mwir_density.npy' )
# opt_density = 1.0 * np.greater_equal( opt_density, 0.5 )
# opt_density = np.load( '/Users/gregory/Development/Photonics/adjoint_lumerical/inverse_design/Landscape/test/opt_optimized_density.npy' )

# plt.plot( opt_density[ :, 8 ] )
# plt.show()

# make_optimizer.init_density_directly( opt_density )
# import_density = ColorSplittingOptimization2DSigmoid.upsample( make_optimizer.design_density, density_coarsen_factor )

# device_permittivity = make_optimizer.density_to_permittivity( import_density )

# fwd_Ez = make_optimizer.compute_forward_fields( 2 * np.pi * 3.0 * 1e8 / ( lambda_min_um * 1e-6 ), device_permittivity )

# plt.imshow( np.abs( fwd_Ez ) )
# plt.show()

# fwd_Ez = make_optimizer.compute_forward_fields( 2 * np.pi * 3.0 * 1e8 / ( lambda_max_um * 1e-6 ), device_permittivity )

# plt.imshow( np.abs( fwd_Ez ) )
# plt.show()


# plt.imshow( opt_density )
# plt.colorbar()
# plt.show()

make_optimizer.optimize(
	int( num_iterations ),
	save_folder + "/opt",
	False, 20, 20, 0.95,
	None,
	use_log_fom,
	wavelength_adversary, adversary_update_iters, np.array( [ lambda_min_um ] ), np.array( [ lambda_max_um ] ),
	False,
	# binarize,
	binarize_movement_per_step, binarize_max_movement_per_voxel,
	dropout_start, dropout_end, dropout_p, dense_plot_freq_iters, dense_plot_wls, dense_focal_map )

make_optimizer.save_optimization_data( save_folder + "/opt" )
