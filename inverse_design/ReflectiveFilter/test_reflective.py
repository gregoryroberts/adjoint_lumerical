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
import ColorReflectorOptimization2D

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )

if ( max_index > 3.5 ):
	print( "This index is a bit too high for the simulation mesh" )

random_seed = np.random.randint( 0, 2**32 - 1 )

# mesh_size_nm = 8
mesh_size_nm = 40#16
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 3.0
lambda_max_um = 4.0
num_lambda_values = 6#8

test_omega = 2 * np.pi * 3.0 * 1e8 / ( 0.55 * 1e-6 )

min_relative_permittivity = 1.0**2
max_relative_permittivity = max_index**2

def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 2 * 152#60#120
device_height_voxels = 2 * 120#60#100
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 120#60#100
focal_points_x_relative = [ 0.25, 0.75 ]

num_layers = int( device_height_voxels / density_coarsen_factor )
spacer_permittivity = 1.0**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

focal_map = np.zeros( ( 2, num_lambda_values ) )
# focal_map = [ 0 for idx in range( 0, num_lambda_values ) ]
for idx in range( int( 0.5 * num_lambda_values ), num_lambda_values ):
	focal_map[ 0, idx ] = 1

mean_density = 0.5
sigma_density = 0.2
# init_from_old = False#True

# blur_fields_size_voxels = 4
# blur_fields = True

num_iterations = 450#150#300

log_file = open( save_folder + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

make_optimizer = ColorReflectorOptimization2D.ColorReflectorOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	lambda_values_um, focal_map, random_seed,
	num_layers, designable_layer_indicators, non_designable_permittivity, save_folder )

# make_optimizer.load_optimization_data( save_folder + "/output" )

# plt.subplot( 1, 2, 1 )
# plt.imshow( make_optimizer.design_density, cmap='Greens' )
# plt.colorbar()
# plt.subplot( 1, 2, 2 )
# plt.imshow( np.load( save_folder + '/output_optimized_density.npy' ), cmap='Greens' )
# plt.colorbar()
# plt.savefig( save_folder + '/figures/plot_density.png' )
# plt.show()

# for state in range( 0, 2 ):
# 	make_optimizer.set_gsst_state( state )
# 	for idx in range( 0, num_lambda_values ):
# 		omega_value = 3.0 * 1e8 / ( lambda_values_um[ idx ] * 1e-6 )
# 		Hx, Hy, Ez = make_optimizer.get_current_fields_ez( idx )#make_optimizer.omega_values[ 1 ] )
# 		Hx, Hy, Ez = make_optimizer.get_source_subtracted_fields_ez( idx )#make_optimizer.omega_values[ 1 ] )

# 		get_fom = make_optimizer.get_current_fom_ez( idx )
# 		if not focal_map[ state, idx ]:
# 			get_fom = np.maximum( 1 - get_fom, 0 )

# 		print( 'fom = ' + str( get_fom ) )
# 		print( 'state = ' + str( state ) )

# 		plt.clf()
# 		plt.subplot( 1, 2, 1 )
# 		plt.imshow( np.abs( Ez )**2, cmap='Reds' )
# 		plt.colorbar()
# 		plt.subplot( 1, 2, 2 )
# 		plt.imshow( np.real( Ez ), cmap='Blues' )
# 		plt.savefig( save_folder + '/figures/plot_' + str( state ) + '_' + str( idx ) + '_Ez.png' )

# 		plt.show()

# sys.exit( 0 )

# make_optimizer.verify_adjoint_against_finite_difference()
make_optimizer.init_density_with_random( 0.5, 0.2 )
# make_optimizer.init_density_with_uniform( 0.4 )
make_optimizer.optimize( 80 )
make_optimizer.save_optimization_data( save_folder + "/output" )

