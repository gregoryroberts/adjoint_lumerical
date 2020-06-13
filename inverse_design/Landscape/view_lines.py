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

if len( sys.argv ) < 2:
	print( "Usage: python " + sys.argv[ 0 ] + " { data folder }" )
	sys.exit( 1 )

data_folder = sys.argv[ 1 ]
data_folder_compare = sys.argv[ 2 ]

number_of_optimizations = 4

random_seeds = np.zeros( number_of_optimizations, dtype=np.uint32 )

mesh_size_nm = 15
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.60
num_lambda_values = 2

# min_relative_permittivity = 1.0**2
# max_relative_permittivity = 2.0**2

min_relative_permittivity = 1.5**2
max_relative_permittivity = 2.5**2


lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 140
# device_height_voxels = 60
device_height_voxels = 100
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 100
focal_points_x_relative = [ 0.25, 0.75 ]


# num_layers = int( device_height_voxels / density_coarsen_factor )
# num_layers = 3
num_layers = 5
spacer_permittivity = 1.5**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
for idx in range( 0, num_layers ):
	if ( idx % 2 ) == 0:
		designable_layer_indicators[ idx ] = True
	else:
		designable_layer_indicators[ idx ] = False
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]



mean_densities = np.linspace( 0.2, 0.8, number_of_optimizations )
sigma_density = 0.2

optimizers = []

num_iterations = 60

ref_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, [ 0, 1 ], 0,
	num_layers, designable_layer_indicators, non_designable_permittivity )

ref_optimizer.init_density_with_uniform( 0.5 )

ref_fom = ref_optimizer.compute_net_fom()

for opt_idx in range( 0, number_of_optimizations ):
	random_seeds[ opt_idx ] = np.random.randint( 0, 2**32 - 1 )

	load_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
		[ device_width_voxels, device_height_voxels ],
		density_coarsen_factor, mesh_size_nm,
		[ min_relative_permittivity, max_relative_permittivity ],
		focal_points_x_relative, focal_length_voxels,
		lambda_values_um, [ 0, 1 ], random_seeds[ opt_idx ],
		num_layers, designable_layer_indicators, non_designable_permittivity )

	init_density = np.load( data_folder + "/opt_" + str( opt_idx ) + "_init_random_density.npy" )
	optimized_density = np.load( data_folder + "/opt_" + str( opt_idx ) + "_optimized_density.npy" )
	fom = np.load( data_folder + "/opt_" + str( opt_idx ) + "_fom_evolution.npy" )

	upsampled = ColorSplittingOptimization2D.upsample( optimized_density, 4 )
	# plt.subplot( 2, 2, opt_idx + 1 )
	# plt.imshow( upsampled, cmap='gray' )
	# plt.colorbar()

	plt.subplot( 2, 3, 1 )
	plt.imshow( init_density )
	plt.colorbar()
	plt.subplot( 2, 3, 2 )
	plt.imshow( optimized_density )
	plt.colorbar()
	plt.subplot( 2, 3, 3 )
	plt.plot( fom, color='g', linewidth=2 )
	plt.show()

	# init_density_compare = np.load( data_folder_compare + "/opt_" + str( opt_idx ) + "_init_random_density.npy" )
	# optimized_density_compare = np.load( data_folder_compare + "/opt_" + str( opt_idx ) + "_optimized_density.npy" )
	# fom_compare = np.load( data_folder_compare + "/opt_" + str( opt_idx ) + "_fom_evolution.npy" )


	# plt.subplot( 2, 3, 4 )
	# plt.imshow( init_density_compare )
	# plt.colorbar()
	# plt.subplot( 2, 3, 5 )
	# plt.imshow( optimized_density_compare )
	# plt.colorbar()
	# plt.subplot( 2, 3, 6 )
	# plt.plot( fom_compare, color='g', linewidth=2 )

	# plt.show()

	load_optimizer.init_density_directly( optimized_density )

	get_fom = load_optimizer.compute_net_fom()
	# print( "Fom is " + str( get_fom ) )

	optimizers.append( load_optimizer )

plt.show()

fom_line_searches = np.load( data_folder + "/fom_line_searches.npy" )

num_searches = fom_line_searches.shape[ 0 ]
num_alpha = fom_line_searches.shape[ 1 ]
alpha = np.linspace( -0.5, 1.5, num_alpha )

colors = [ 'r', 'g', 'b', 'm', 'c', 'y' ]

for search_idx in range( 0, num_searches ):
	start_data_pt = 0
	end_data_pt = len( alpha ) - 1

	while fom_line_searches[ search_idx, start_data_pt ] == -1:
		start_data_pt += 1

	while fom_line_searches[ search_idx, end_data_pt ] == -1:
		end_data_pt -= 1
	end_data_pt += 1

	line_data = fom_line_searches[ search_idx, start_data_pt : end_data_pt ] / ref_fom
	plt.subplot( 3, 2, search_idx + 1 )
	plt.plot( alpha[ start_data_pt : end_data_pt ], line_data, linewidth=2, color=colors[ search_idx % len( colors ) ] )
plt.show()




