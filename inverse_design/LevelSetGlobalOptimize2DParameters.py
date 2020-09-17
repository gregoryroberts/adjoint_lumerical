#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

# import FreeOptimizationMultiDevice
# import OptimizationLayersSpacersMultiDevice
# import OptimizationLayersSpacersGlobalBinarization2DMultiDevice

# import FreeBayerFilterWithBlur2D
# import FreeBayerFilterDualPhaseBlur2D

#
# Files
#
project_name = 'cmos_dielectric_2d_3focal_layered_higherindex_p22layers_rbg_lsf_global_3xtsmc_um_focal_1p5um'

#
# Optical
#
# todo: the side background index shouldn't be TiO2 because that part will not be etched away!  Go back to oxide!
# Maybe force binarize more slowly.  Consider just optimizing for Strell ratio if we aren't going to get high transmisison anyway.
# Check binarized device and look at feature sizes!
background_index = 1.0
design_index_background = 1.0#1.35
device_background_index = 1.0#1.35# 2.5
high_index_backfill = 2.5
substrate_index = 1.0

min_real_permittivity = design_index_background**2
max_real_permittivity = high_index_backfill**2

min_imag_permittivity = 0
max_imag_permittivity = 0

min_real_index = np.sqrt( min_real_permittivity )
max_real_index = np.sqrt( max_real_permittivity )

permittivity_bounds = np.array( [ min_real_permittivity + 1j * min_imag_permittivity, max_real_permittivity + 1j * max_imag_permittivity ], dtype=np.complex )

init_permittivity_0_1_scale = 0.5

lsf_start_step_size = 0.1
lsf_end_step_size = 0.05

# For the focal length, should we put these things in a near field regime.  In the nature photonics, with very
# simple scatterers made of SiN, they can get fairly impressive splitting.  The "focal length" seems to be smaller
# there (not sure if it would be considered near field or not, but it would be right on the edge).
focal_length_um = 1.5


#
# Device
#
mesh_spacing_um = 0.025
lsf_mesh_spacing_um = 0.005

device_layer_thicknesses_um = np.array( [
	0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.13
] )
device_spacer_thicknesses_um = np.array( [
	0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.0
] )

feature_size_voxels_by_profiles = [ int( 0.1 / lsf_mesh_spacing_um ) for idx in range( 0, 6 ) ]
feature_size_voxels_by_profiles.append( int( 0.09 / lsf_mesh_spacing_um ) )


device_size_lateral_um = 3.0
designable_size_vertical_um = np.sum( device_layer_thicknesses_um ) + np.sum( device_spacer_thicknesses_um )

device_size_verical_um = designable_size_vertical_um

opt_device_voxels_lateral = 1 + int( device_size_lateral_um / lsf_mesh_spacing_um )
device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
designable_device_voxels_vertical = 1 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0

#
# Spectral
#
num_bands = 3
num_points_per_band = 5#10#15

# band_weights = [ 1, 1, 1 ]
band_weights = [ 0, 0, 1 ]
# pol_weights = [ 1, 1 ]
pol_weights = [ 0, 1 ]

lambda_min_um = 0.4
lambda_max_um = 0.7

num_design_frequency_points = num_bands * num_points_per_band
num_wavelengths = num_design_frequency_points
num_eval_frequency_points = 1 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um * 1.02)**2 / (focal_length_um**2 * lambda_values_um**2)



#
# FDTD
#
vertical_gap_size_um = 1.0
lateral_gap_size_um = 1.0

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -vertical_gap_size_um - focal_length_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / mesh_spacing_um ) )


fdtd_simulation_time_fs = 3 * 2000

#
# Forward Source
#
num_polarizations = 2

lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_verical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# We are assuming that the data is organized in order of increasing wavelength (i.e. - blue first, red last)
spectral_focal_plane_map = [
	[2 * num_points_per_band, 3 * num_points_per_band],
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
]

# spectral_focal_plane_map = [
# 	[0 * num_points_per_band, 2 * num_points_per_band],
# 	[0, 3 * num_points_per_band],
# 	[num_points_per_band, 3 * num_points_per_band],
# ]
#
# Adjoint sources
#
# This seems like a long focal length
adjoint_vertical_um = -focal_length_um
num_focal_spots = 3
num_adjoint_sources = num_focal_spots

# adjoint_x_positions_um = [ -3 * device_size_lateral_um / 8., -device_size_lateral_um / 8., device_size_lateral_um / 8., 3 * device_size_lateral_um / 8. ]
adjoint_x_positions_um = [ -device_size_lateral_um / 3., 0.0, device_size_lateral_um / 3. ]

#
# Optimization
#
num_epochs = 1
num_iterations_per_epoch = 4#80
init_optimization_epoch = 0

