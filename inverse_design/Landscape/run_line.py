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
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]

number_of_optimizations = 4

random_seeds = np.zeros( number_of_optimizations, dtype=np.uint32 )

mesh_size_nm = 15
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_um = 0.45
lambda_max_um = 0.60
num_lambda_values = 2


min_relative_permittivity = 1.5**2
max_relative_permittivity = 2.5**2

def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )

eps_max_landscape = 3.0**2
eps_min_landscape = 1.0**2

density_max_landscape = density_bound_from_eps( eps_max_landscape )
density_min_landscape = density_bound_from_eps( eps_min_landscape )

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_lambda_values )

device_width_voxels = 140
device_height_voxels = 60
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 100
focal_points_x_relative = [ 0.25, 0.75 ]

num_layers = 3
spacer_permittivity = 1.5**2
designable_layer_indicators = [ True for idx in range( 0, num_layers ) ]
non_designable_permittivity = [ spacer_permittivity for idx in range( 0, num_layers ) ]

mean_densities = np.linspace( 0.2, 0.8, number_of_optimizations )
sigma_density = 0.2

optimizers = []

num_iterations = 75

for opt_idx in range( 0, number_of_optimizations ):
	random_seeds[ opt_idx ] = np.random.randint( 0, 2**32 - 1 )

	make_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
		[ device_width_voxels, device_height_voxels ],
		density_coarsen_factor, mesh_size_nm,
		[ min_relative_permittivity, max_relative_permittivity ],
		focal_points_x_relative, focal_length_voxels,
		lambda_values_um, [ 0, 1 ], random_seeds[ opt_idx ],
		num_layers, designable_layer_indicators, non_designable_permittivity )

	make_optimizer.init_density_with_random( mean_densities[ opt_idx ], sigma_density )

	np.save( save_folder + "/opt_" + str( opt_idx ) + "_init_random_density.npy", make_optimizer.design_density )
	np.save( save_folder + "/opt_" + str( opt_idx ) + "_random_seed.npy", make_optimizer.random_seed )

	make_optimizer.optimize( num_iterations )

	make_optimizer.save_optimization_data( save_folder + "/opt_" + str( opt_idx ) )

	optimizers.append( make_optimizer )

num_alpha = 60
alpha = np.linspace( -0.5, 1.5, num_alpha )

test_optimizer = ColorSplittingOptimization2D.ColorSplittingOptimization2D(
	[ device_width_voxels, device_height_voxels ],
	density_coarsen_factor, mesh_size_nm,
	[ min_relative_permittivity, max_relative_permittivity ],
	focal_points_x_relative, focal_length_voxels,
	lambda_values_um, [ 0, 1 ], random_seeds[ opt_idx ],
	num_layers, designable_layer_indicators, non_designable_permittivity )

num_searches = int( math.factorial( number_of_optimizations ) / ( math.factorial( 2 ) * math.factorial( number_of_optimizations - 2 ) ) )
fom_line_searches = np.zeros( ( num_searches, num_alpha ) )
fom_line_search_valid = np.ones( ( num_searches, num_alpha ) )

line_search_idx = 0
for opt1_idx in range( 0, number_of_optimizations ):
	get_opt1 = optimizers[ opt1_idx ]
	opt1_density = get_opt1.design_density

	for opt2_idx in range( opt1_idx + 1, number_of_optimizations ):
		get_opt2 = optimizers[ opt2_idx ]
		opt2_density = get_opt2.design_density

		for alpha_idx in range( 0, num_alpha ):

			alpha_weight = alpha[ alpha_idx ]

			weighted_density = opt1_density + alpha_weight * ( opt2_density - opt1_density )

			if np.min( weighted_density ) < density_min_landscape:
				fom_line_search_valid[ line_search_idx, alpha_idx ] = 0
				fom_line_searches[ line_search_idx, alpha_idx ] = -1
				continue
			if np.max( weighted_density ) > density_max_landscape:
				fom_line_search_valid[ line_search_idx, alpha_idx ] = 0
				fom_line_searches[ line_search_idx, alpha_idx ] = -1
				continue

			test_optimizer.init_density_directly( weighted_density )
			net_fom = test_optimizer.compute_net_fom()

			fom_line_searches[ line_search_idx, alpha_idx ] = net_fom

		line_search_idx += 1


np.save( save_folder + "/fom_line_searches.npy", fom_line_searches )






