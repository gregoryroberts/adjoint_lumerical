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

#
# Fitting
#
from sklearn.decomposition import PCA

if len( sys.argv ) < 5:
	print( "Usage: python " + sys.argv[ 0 ] + " { base folder } { simulation id start } { simulation id end } { number of simulations per block }" )
	sys.exit( 1 )

base_folder = sys.argv[ 1 ]
sim_start_id = int( sys.argv[ 2 ] )
sim_end_id_non_inclusive = int( sys.argv[ 3 ] ) + 1
sims_per_block = int( sys.argv[ 4 ] )

save_location_base = base_folder + '/simulations_'

#
# Let's take 80% of the data as training, 10% for validation, 10% for testing
#
num_blocks = sim_end_id_non_inclusive - sim_start_id
num_total_data_points = num_blocks * sims_per_block
fraction_data_training = 0.80

fraction_data_validation = 0.10

num_data_points_for_training = int( fraction_data_training * num_total_data_points )
num_data_points_for_validation = int( fraction_data_validation * num_total_data_points )

num_collected_data_points_training = 0
num_collected_data_points_validation = 0
collection_block = sim_start_id
collection_simulation = 0

training_permittivities = []
training_gradients = []
training_foms = []

while num_collected_data_points_training < num_data_points_for_training:

	if collection_simulation >= sims_per_block:
		collection_block += 1
		collection_simulation = 0
		continue

	training_permittivities.append(
		np.load( save_location_base + str( collection_block ) + '/device_permittivity_' + str( collection_simulation ) + '.npy' ) )

	training_gradients.append(
		np.load( save_location_base + str( collection_block ) + '/gradients_' + str( collection_simulation ) + '.npy' ) )

	training_foms.append(
		np.load( save_location_base + str( collection_block ) + '/figures_of_merit_' + str( collection_simulation ) + '.npy' ) )

	collection_simulation += 1
	num_collected_data_points_training += 1


validation_permittivities = []
validation_gradients = []
validation_foms = []

while num_collected_data_points_validation < num_total_data_points:

	if collection_simulation >= sims_per_block:
		collection_block += 1
		collection_simulation = 0
		continue

	validation_permittivities.append(
		np.load( save_location_base + str( collection_block ) + '/device_permittivity_' + str( collection_simulation ) + '.npy' ) )

	validation_gradients.append(
		np.load( save_location_base + str( collection_block ) + '/gradients_' + str( collection_simulation ) + '.npy' ) )

	validation_foms.append(
		np.load( save_location_base + str( collection_block ) + '/figures_of_merit_' + str( collection_simulation ) + '.npy' ) )

	collection_simulation += 1
	num_collected_data_points_validation += 1


def list_to_directions( list_in ):
	directions = []
	for idx in range( 0, len( list_in ) ):
		matrix = list_in[ idx ]

		flatten_matrix = matrix.flatten()
		matrix_direction = flatten_matrix / np.sqrt( np.sum( flatten_matrix**2 ) )

		directions.append( matrix_direction )

	return directions


training_permittivity_directions = list_to_directions( training_permittivities )
training_gradient_directions = list_to_directions( training_gradients )

validation_permittivity_directions = list_to_directions( validation_permittivities )
validation_gradient_directions = list_to_directions( validation_gradients )

num_components = 100
pca = PCA( n_components=num_components )
pca.fit( training_gradient_directions )

component_directions = pca.components_
normalized_directions = []

for direction in component_directions:
	normalized_directions.append( direction / np.sqrt( np.sum( direction**2 ) ) )

validation_remaining_lengths = []
for validation_idx in range( 0, len( validation_gradient_directions ) ):
	get_direction = validation_gradient_directions[ validation_idx ]

	for component_idx in range( 0, num_components ):
		get_direction -= np.dot( component_directions[ component_idx ], get_direction ) * component_directions[ component_idx ]

	validation_remaining_lengths.append( np.sqrt( np.sum( get_direction**2 ) ) )

np.save( base_folder + '/validation_remaining_lengths.npy', validation_remaining_lengths )




