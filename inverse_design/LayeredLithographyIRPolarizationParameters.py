#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
# import sigmoid

#
# Files
#
# project_name = 'layered_infrared_3layers_pol_splitter_parallel_fom_rcp_45_lcp_m45_v1_4p2x4p2x3p12um_f2p6'
# project_name = 'layered_infrared_3layers_pol_splitter_parallel_fom_v1_4p2x4p2x3p12um_f2p6'
# project_name = 'layered_infrared_3layers_pol_splitter_parallel_fom_v1_finer_reduce_bw_only_circular_4p8x4p8x3p12um_f2p72'
# project_name = 'layered_infrared_3layers_pol_splitter_parallel_fom_4layer_v1_4p2x4p2x3p12um_f2p6'
project_name = 'layered_infrared_3layers_pol_splitter_parallel_fom_v1_20nm_reduce_bw_4p8x4p8x3p12um_f2p72'

optimized_focal_spots = [ 0, 1, 2, 3 ]

#
# Optical
#
# averaging between the part where we dip into the infrared/red part of the spectrum
# and where we are in the telecom ranges
#
index_silicon = 0.5 * ( 3.47 + 3.68 )
index_su8 = 1.575
index_air = 1.0
index_sio2 = 1.45
background_index = index_su8
min_device_index = index_su8
max_device_index = index_silicon

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.25

# focal_length_um = 2.6
focal_length_um = 2.72
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Device
#
# mesh_spacing_um = 0.04
# mesh_spacing_um = 0.032
mesh_spacing_um = 0.02

#
# Pesky size to get the number of voxels to be a multiple of 3
#
num_vertical_layers = 3#4

device_size_lateral_um = 4.8#4.2#3.6
# device_size_verical_um = num_vertical_layers * ( 3.12 / num_vertical_layers )
# device_size_verical_um = num_vertical_layers * ( 3.168 / num_vertical_layers )
device_size_verical_um = num_vertical_layers * ( 3.6 / num_vertical_layers )
# amorphous_silicon_height_per_layer_um = 0.52#0.8
# amorphous_silicon_height_per_layer_um = 0.704
# amorphous_silicon_height_per_layer_um = 0.72
amorphous_silicon_height_per_layer_um = 0.6
spacer_size_um = ( device_size_verical_um / num_vertical_layers ) - amorphous_silicon_height_per_layer_um

device_voxels_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_vertical = 2 + int(device_size_verical_um / mesh_spacing_um)
spacer_size_voxels = 1 + int(spacer_size_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_verical_um
device_vertical_minimum_um = 0

#
# Spectral
#
lambda_min_um = 0.95#0.9
lambda_max_um = 1.05#1.1

# num_bands = 3
# num_points_per_band = 10
num_design_frequency_points = 10#10#num_bands * num_points_per_band
num_eval_frequency_points = 60

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)

effective_focal_length_sq = focal_length_um**2 + ( device_size_lateral_um / 4. )**2 + ( device_size_lateral_um / 4. )**2

max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (effective_focal_length_sq * lambda_values_um**2)

#
# Fabrication Constraints
#
min_feature_size_um = 3 * mesh_spacing_um
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )
# blur_half_width_voxels = 0

#
# FDTD
#
vertical_gap_size_um = 1.0
lateral_gap_size_um = 1.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_simulation_time_fs = 4 * 700

#
# Forward Source
#
lateral_aperture_um = device_size_lateral_um + 0.2
src_maximum_vertical_um = device_size_verical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 0.5, 1.0, 0.5 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# We are assuming that the data is organized in order of increasing wavelength (i.e. - blue first, red last)
# spectral_focal_plane_map = [
# 	[0, num_points_per_band],
# 	[num_points_per_band, 2 * num_points_per_band],
# 	[2 * num_points_per_band, 3 * num_points_per_band],
# 	[num_points_per_band, 2 * num_points_per_band]
# ]

jones_sorting_vectors = [
	np.array( [ 1. / np.sqrt( 2 ), -1j / np.sqrt( 2 ) ] ),
	np.array( [ 0.514, 0.794 + 0.324j ] ),
	np.array( [ 0.986, 0.169j ] ),
	np.array( [ 0.514, -0.794 + 0.324j ] )
]

# jones_sorting_vectors = [
# 	np.array( [ 1. / np.sqrt( 2 ), 1. / np.sqrt( 2 ) ] ),
# 	np.array( [ 1. / np.sqrt( 2 ), 1j / np.sqrt( 2 ) ] ),
# 	np.array( [ 1. / np.sqrt( 2 ), -1. / np.sqrt( 2 ) ] ),
# 	np.array( [ 1. / np.sqrt( 2 ), -1j / np.sqrt( 2 ) ] )
# ]

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
	
# fom_sigmoid_beta = 5.0
# fom_sigmoid_eta = 0.25
# fom_sigmoid = sigmoid.Sigmoid( fom_sigmoid_beta, fom_sigmoid_eta )

expected_parallel_max_efficiency = 0.8
parallel_fom_bound = 0.5 * expected_parallel_max_efficiency

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
start_epoch = 7
num_epochs = 12
num_iterations_per_epoch = 60
binarization_start_epoch = 9#1
max_binarize_movement = 0.01
desired_binarize_change = 3 * 0.005 / 2

design_change_start_epoch = 0.025
design_change_end_epoch = 0.01

# epoch_start_permittivity_change_max = 0.025
# epoch_end_permittivity_change_max = 0.01
# epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

# epoch_start_permittivity_change_min = 0.01
# epoch_end_permittivity_change_min = 0
# epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

