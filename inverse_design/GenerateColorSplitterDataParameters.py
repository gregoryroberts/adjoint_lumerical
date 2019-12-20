#
# Parameter file for the Bayer Filter Layered Lithography optimization
#

import numpy as np
import sys

#
# Files
#
data_prefix = 'neural_data_'


#
# Materials
#
background_index = 1.0
index_low_bounds = [ 1.0, 1.5 ]
# delta_n / index_low
index_contrast_ratio_bounds = [ 0.3, 1.5 ]

#
# Wavelengths
#
# two wavelength average
lambda_plus_um = 0.6
# two wavelength half splitting
lambda_minus_bounds_um = [ 0.05, 0.2 ]

#
# Device Geometry
#
aperture_size_lambda_plus_units = 4
device_depth_bounds_lambda_plus_units = [ 1, 4 ]
# we will measure focal length from the end of the device
focal_length_bounds_lambda_plus_units = [ 1, 6 ]
# we will use rough formula of:
# NA ~ n * W / ( 2 * f ), where n = 1.0 (imaging in air), W is the aperture size, f is the focal distance (which we will take to be from the end of the device)
# so in order to pick focal length, we compute it via:
# f = NA * W / 2
numerical_aperture_bounds = [ 0.25, 1.5 ]

aperture_size_um = lambda_plus_um * aperture_size_lambda_plus_units

#
# Simulation Region
#
lateral_gap_lambda_plus = 2
vertical_gap_lambda_plus = 2

lateral_gap_um = lambda_plus_um * lateral_gap_lambda_plus
vertical_gap_um = lambda_plus_um * vertical_gap_lambda_plus

mesh_spacing_lambda_plus = 1. / 12.
mesh_spacing_um = lambda_plus_um * mesh_spacing_lambda_plus

fdtd_simulation_time_fs = 10000

#
# Focal spots/Adjoint sources
#
num_adjoint_sources = 2

#
# Optimization
#
min_iterations = 40
max_iterations = 50
gradient_norm_dropoff = 0.05

num_collect_step_size_iterations = 5

start_design_change_max = 0.03
step_design_change_max = 0.03

