#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
# import sigmoid

#
# Files
#
project_name = 'layered_mwir_2d_lithography_bridges_polarization_10layers_4p5to5um_fixed_step_addcage_25x25x25um'


optimized_focal_spots = [ 0, 1, 2, 3 ]


init_permittivity_0_1_scale = 0.5

focal_length_um = 20
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Device
#
mesh_spacing_um = 0.25

device_size_lateral_um = 25
device_size_verical_um = 25

device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_vertical = 1 + int(device_size_verical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_verical_um
device_vertical_minimum_um = 0

#
# Spectral
#
lambda_min_um = 4.5
lambda_max_um = 5

num_design_frequency_points = 10
num_eval_frequency_points = 60

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)

effective_focal_length_sq = focal_length_um**2 + ( device_size_lateral_um / 4. )**2 + ( device_size_lateral_um / 4. )**2

max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (effective_focal_length_sq * lambda_values_um**2)

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)

#
# Fabrication Constraints
#
min_feature_size_um = 0.75
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( (min_feature_size_voxels - 1) / 2. ) )

num_vertical_layers = 10

#
# FDTD
#
vertical_gap_size_um = 12
lateral_gap_size_um = 4

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_simulation_time_fs = 3000

#
# Forward Source
#
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_verical_um + 0.8 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# We are assuming that the data is organized in order of increasing wavelength (i.e. - blue first, red last)
# spectral_focal_plane_map = [
# 	[0, num_points_per_band],
# 	[num_points_per_band, 2 * num_points_per_band],
# 	[2 * num_points_per_band, 3 * num_points_per_band],
# 	[num_points_per_band, 2 * num_points_per_band]
# ]

#
# Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 4
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]

#
# Optimization
#
num_epochs = 10
num_iterations_per_epoch = 30

use_fixed_step_size = False
fixed_step_size = 2.0

epoch_start_permittivity_change_max = 0.15
epoch_end_permittivity_change_max = 0.05
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.05
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

#
# Topology
#
topology_num_free_iterations_between_patches = 8


background_index = 1.0
min_device_index = 1.0
max_device_index = 1.5

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2



#
# Forward Source
#
lateral_aperture_um = device_size_lateral_um + 0.2
src_maximum_vertical_um = device_size_verical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um


jones_sorting_vectors = [
	np.array( [ 1. / np.sqrt( 2 ), -1j / np.sqrt( 2 ) ] ),
	np.array( [ 0.514, 0.794 + 0.324j ] ),
	np.array( [ 0.986, 0.169j ] ),
	np.array( [ 0.514, -0.794 + 0.324j ] )
]


def find_orthogonal( v ):
	if np.abs( v[ 0 ] ) == 1:
		return np.array( [ 0, 1 ] )
	elif np.abs( v[ 1 ] ) == 1:
		return np.array( [ 1, 0 ] )

	a = v[ 0 ]
	b = v[ 1 ]

	d = 1 / np.sqrt( 1 + np.abs( b )**2 / np.abs( a )**2 )
	c = -np.conj( b ) * d / np.conj( a )

	return np.array( [ c, d ] )

jones_orthogonal_vectors = []
for jones_idx in range( 0, len( jones_sorting_vectors ) ):
	jones_orthogonal_vectors.append( find_orthogonal( jones_sorting_vectors[ jones_idx ] ) )


expected_parallel_max_efficiency = 0.8
parallel_fom_bound = 0.5 * expected_parallel_max_efficiency





