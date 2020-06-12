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

number_of_optimizations = 4

random_seeds = np.zeros( number_of_optimizations, dtype=np.uint32 )

mesh_size_nm = 15
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.60
num_lambda_values = 2

min_relative_permittivity = 1.0**2
max_relative_permittivity = 2.0**2

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 140
device_height_voxels = 60
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 100
focal_points_x_relative = [ 0.25, 0.75 ]

mean_densities = np.linspace( 0.2, 0.8, number_of_optimizations )
sigma_density = 0.2

optimizers = []

num_iterations = 60

for opt_idx in range( 0, number_of_optimizations ):
	random_seeds[ opt_idx ] = np.random.randint( 0, 2**32 - 1 )

	load_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
		[ device_width_voxels, device_height_voxels ],
		density_coarsen_factor, mesh_size_nm,
		[ min_relative_permittivity, max_relative_permittivity ],
		focal_points_x_relative, focal_length_voxels,
		lambda_values_um, [ 0, 1 ], random_seeds[ opt_idx ] )

	init_density = np.load( data_folder + "/opt_" + str( opt_idx ) + "_init_random_density.npy" )
	optimized_density = np.load( data_folder + "/opt_" + str( opt_idx ) + "_optimized_density.npy" )
	fom = np.load( data_folder + "/opt_" + str( opt_idx ) + "_fom_evolution.npy" )

	# plt.subplot( 1, 3, 1 )
	# plt.imshow( init_density )
	# plt.colorbar()
	# plt.subplot( 1, 3, 2 )
	# plt.imshow( optimized_density )
	# plt.colorbar()
	# plt.subplot( 1, 3, 3 )
	# plt.plot( fom, color='g', linewidth=2 )
	# plt.show()

	load_optimizer.init_density_directly( optimized_density )

	# get_fom = load_optimizer.compute_net_fom()

	optimizers.append( load_optimizer )

fom_line_searches = np.load( data_folder + "/fom_line_searches.npy" )

num_searches = fom_line_searches.shape[ 0 ]
num_alpha = fom_line_searches.shape[ 1 ]
alpha = np.linspace( 0.0, 1.0, num_alpha )

for search_idx in range( 0, num_searches ):
	line_data = fom_line_searches[ search_idx, : ]
	plt.plot( alpha, line_data, linewidth=2, color='r' )
	plt.show()


# test_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
# 	[ device_width_voxels, device_height_voxels ],
# 	density_coarsen_factor, mesh_size_nm,
# 	[ min_relative_permittivity, max_relative_permittivity ],
# 	focal_points_x_relative, focal_length_voxels,
# 	lambda_values_um, [ 0, 1 ], random_seeds[ opt_idx ] )

# num_searches = int( math.factorial( number_of_optimizations ) / ( math.factorial( 2 ) * math.factorial( number_of_optimizations - 2 ) ) )
# fom_line_searches = np.zeros( ( num_searches, num_alpha ) )

# line_search_idx = 0
# for opt1_idx in range( 0, number_of_optimizations ):
# 	get_opt1 = optimizers[ opt1_idx ]
# 	opt1_density = get_opt1.design_density

# 	for opt2_idx in range( opt1_idx + 1, number_of_optimizations ):
# 		get_opt2 = optimizers[ opt2_idx ]
# 		opt2_density = get_opt2.design_density

# 		for alpha_idx in range( 0, num_alpha ):

# 			alpha_weight = alpha[ alpha_idx ]

# 			weighted_density = opt1_density + alpha_weight * ( opt2_density - opt1_density )

# 			test_optimizer.init_density_directly( weighted_density )
# 			net_fom = test_optimizer.compute_net_fom()

# 			fom_line_searches[ line_search_idx, alpha_idx ] = net_fom

# 		line_search_idx += 1


# np.save( data_folder + "/fom_line_searches.npy", fom_line_searches )






