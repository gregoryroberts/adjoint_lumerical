#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

#
# Files
#
project_name = '2d_rbg_save_directions'

#
# Random Seed
#
# np.random.seed( 123123 )

#
# Optical
#
background_index = 1.0
device_min_index = 1.0
device_max_index = 1.5
device_min_permittivity = device_min_index**2
device_max_permittivity = device_max_index**2

device_permittivity_bounds = [ device_min_permittivity, device_max_permittivity ]

device_mid_permittivity = 0.5 * ( device_min_permittivity + device_max_permittivity )

init_max_random_0_1_scale = 0.25
init_permittivity_0_1_scale = 0.5

focal_length_um = 2.0

#
# Device
#
computation_mesh_um = 0.016
feature_size_meshes_um = [ 0.016, 0.02, 0.032, 0.04, 0.08, 0.16, 0.32 ]

num_feature_size_optimizations = len( feature_size_meshes_um )

device_size_lateral_um = 3.2
device_size_vertical_um = 1.6

designable_device_vertical_maximum_um = device_size_vertical_um
designable_device_vertical_minimum_um = 0

#
# Spectral
#
num_bands = 3
num_points_per_band = 10

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

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -vertical_gap_size_um - focal_length_um

fdtd_region_minimum_vertical_voxels = int( np.ceil( fdtd_region_size_vertical_um / computation_mesh_um ) )
fdtd_region_minimum_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / computation_mesh_um ) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / computation_mesh_um ) )

fdtd_simulation_time_fs = 2000

#
# Forward Source
#
num_polarizations = 2

lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_vertical_um + 0.5 * vertical_gap_size_um
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
#
# Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 3
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [ -device_size_lateral_um / 3., 0.0, device_size_lateral_um / 3. ]

#
# Optimization
#
num_epochs = 8
num_iterations_per_epoch = 30

use_fixed_step_size = True
fixed_step_size = 10 * 1 / 5

use_adaptive_step_size = False
desired_max_max_design_change = 0.05
desired_min_max_design_change = 0.001
adaptive_step_size = 0.3 * 0.01 * 3 / 5

epoch_start_permittivity_change_max = 0.1
epoch_end_permittivity_change_max = 0.02
epoch_range_permittivity_change_max = epoch_start_permittivity_change_max - epoch_end_permittivity_change_max

epoch_start_permittivity_change_min = 0.05
epoch_end_permittivity_change_min = 0
epoch_range_permittivity_change_min = epoch_start_permittivity_change_min - epoch_end_permittivity_change_min

