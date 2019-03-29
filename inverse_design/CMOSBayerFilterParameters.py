#
# Parameter file for the Bayer Filter CMOS optimization
#

import numpy as np

#
# Optical
#
background_index = 1.0
min_device_index = 1.0
max_device_index = 1.5

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.25

focal_length_um = 1.5
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Spectral
#
lambda_min_um = 0.4
lambda_max_um = 0.7

num_bands = 3
num_points_per_band = 10
num_design_frequency_points = num_bands * num_points_per_band

#
# Device
#
mesh_spacing_um = 0.02

device_size_lateral_um = 2
device_size_verical_um = 2

device_voxels_lateral = int(device_size_lateral_um / mesh_spacing_um)
device_voxels_vertical = int(device_size_verical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_verical_um
device_vertical_minimum_um = 0

#
# Fabrication Constraints
#
min_feature_size_um = 0.1
min_feature_size_voxels = min_feature_size_um / mesh_spacing_um
blur_half_width_voxels = int( np.ceil( (min_feature_size_voxels - 1) / 2. ) )

num_vertical_layers = 10

#
# FDTD
#
vertical_gap_size_um = 0.5
lateral_gap_size_um = 0.2

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_verical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_simulation_time_fs = 700

#
# Forward Source
#
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_verical_um + 0.5 * vertical_gap_size_um
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

#
# Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_adjoint_sources = 4
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]
