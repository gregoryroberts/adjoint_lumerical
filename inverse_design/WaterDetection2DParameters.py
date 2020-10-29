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
project_name = 'water_detector_36x24um_f40um'

#
# Optical
#
# todo: the side background index shouldn't be TiO2 because that part will not be etched away!  Go back to oxide!
# Maybe force binarize more slowly.  Consider just optimizing for Strell ratio if we aren't going to get high transmisison anyway.
# Check binarized device and look at feature sizes!
background_index = 1.0
design_index_background = 1.0
device_background_index = 1.5
device_lateral_background_density = 0.0
high_index_backfill = 1.5
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
focal_length_um = 40


#
# Device
#
# mesh_spacing_um = 0.25
mesh_spacing_um = 0.2

device_size_lateral_um = 36
designable_size_vertical_um = 24

device_size_verical_um = designable_size_vertical_um

device_voxels_lateral = 1 + int( ( device_size_lateral_um + np.finfo(np.float64).eps ) / mesh_spacing_um )
designable_device_voxels_vertical = 1 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = designable_size_vertical_um
designable_device_vertical_minimum_um = 0

#
# Spectral
#
num_bands = 3
# num_points_per_band = 5
num_points_per_band = 11

band_centers_um = [
	2.65, 2.8, 3.1
]

# band_ranges_um = [
# 	[ 2.5, 2.8 ],
# 	[ 2.65, 2.95 ],
# 	[ 2.95, 3.25 ]
# ]

band_ranges_um = [
	[ 2.3, 3.0 ],
	[ 2.45, 3.15 ],
	[ 2.75, 3.45 ]
]


band_weights = [ 1, 1, 1 ]
# pol_weights = [ 0, 1 ]
pol_weights = [ 1, 0 ]

lambda_min_um = 2
lambda_max_um = 4

# num_design_frequency_points = num_bands * num_points_per_band
# num_wavelengths = num_design_frequency_points
num_eval_frequency_points = 10 * num_bands * num_points_per_band

lambda_values_by_band_um = np.zeros( ( num_bands, num_points_per_band ) )
max_intensity_by_band_by_wavelength = np.zeros( ( num_bands, num_points_per_band ) )
for band_idx in range( 0, num_bands ):
	lambda_values_by_band_um[ band_idx ] = np.linspace( band_ranges_um[ band_idx ][ 0 ], band_ranges_um[ band_idx ][ 1 ], num_points_per_band )
	max_intensity_by_band_by_wavelength[ band_idx ] = (device_size_lateral_um * 1.02)**2 / (focal_length_um**2 * lambda_values_by_band_um[ band_idx ]**2)


#
# FDTD
#
vertical_gap_size_um = lambda_max_um
lateral_gap_size_um = lambda_max_um

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

spectral_focal_plane_map = [
	0, 1, 2
]


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
num_iterations_per_epoch = 25
init_optimization_epoch = 0

fom_quotient_regularization = 0.1

