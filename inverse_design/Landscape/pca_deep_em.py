from sklearn.decomposition import PCA

#
# Housekeeping
#
import os
import shutil
import sys
import time

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
import DeepEM

if len( sys.argv ) < 6:
	print( "Usage: python " + sys.argv[ 0 ] + " { save folder } { max index } { loss percentage } { random generator seed } { number of hours to run }" )
	sys.exit( 1 )

save_folder = sys.argv[ 1 ]
max_index = float( sys.argv[ 2 ] )
loss_percentage = float( sys.argv[ 3 ] )
random_seed = int( sys.argv[ 4 ] )
number_of_hours_to_run = float( sys.argv[ 5 ] )
number_of_seconds_to_run = 60. * 60. * number_of_hours_to_run

if ( max_index > 3.5 ):
	print( "This index is a bit too high for the simulation mesh" )


mesh_size_nm = 9
density_coarsen_factor = 10
mesh_size_m = mesh_size_nm * 1e-9
lambda_um = 0.532
# num_lambda_values = 1
num_layers = 2

device_width_voxels = 120
device_height_voxels = 120
device_voxels_total = device_width_voxels * device_height_voxels
focal_length_voxels = 120

min_relative_permittivity = 1.0**2

single_pass_transmittance = 1 - ( loss_percentage / 100. )
device_height_m = device_height_voxels * mesh_size_nm * 1e-9
lambda_m = lambda_um * 1e-6
loss_index = -lambda_m * np.log( single_pass_transmittance ) / ( device_height_m * 2 * np.pi )

real_permittivity = max_index**2 - loss_index**2
imag_permittivity = 2 * np.sqrt( real_permittivity ) * loss_index
max_relative_permittivity = real_permittivity + 1j * imag_permittivity


densities = []
focal_abs = []
num_before_saving = 500

# num_to_load = 10 * num_before_saving
num_load_epochs = 10

for epoch_num in range( 0, num_load_epochs ):
	densities += list( np.load( save_folder + "/generated_densities_" + str( epoch_num ) + ".npy" ) )
	focal_abs += list( np.abs( np.load( save_folder + "/generated_focal_fields_" + str( epoch_num ) + ".npy" ) ) )


pca = PCA()
pca.fit_transform( focal_abs )

n_components_variance_max = 120
ratio_variance = np.zeros( ratio_variance )

for n_idx in range( 0, n_components_variance_max ):
	ratio_variance.append( pca.explained_variance_ratio_( n_idx ) )

np.save( save_folder + "/ratio_variance.npy" )


# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
