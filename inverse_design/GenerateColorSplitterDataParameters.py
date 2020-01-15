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
bandwidth_fraction = 0.1
num_bandwidth_spacings_min = 3

lambda_minus_bounds_um = [ num_bandwidth_spacings_min * 0.5 * bandwidth_fraction, 0.2 ]

num_colors = 2
bandwidth_um = 0.1 * lambda_plus_um
num_half_bandwidth_points = 5
num_freq_points_per_color = 2 * num_half_bandwidth_points + 1
num_eval_points = 55

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
# f = W / ( 2 * NA )
numerical_aperture_bounds = [ 0.25, 1.5 ]

aperture_size_um = lambda_plus_um * aperture_size_lambda_plus_units

#
# Simulation Region
#
lateral_gap_lambda_plus = 2
vertical_gap_lambda_plus = 2

lateral_gap_um = lambda_plus_um * lateral_gap_lambda_plus
vertical_gap_um = lambda_plus_um * vertical_gap_lambda_plus

mesh_spacing_lambda_plus = 1. / 18.
mesh_spacing_um = lambda_plus_um * mesh_spacing_lambda_plus

fdtd_simulation_time_fs = 10000

#
# Focal spots/Adjoint sources
#
num_adjoint_sources = 2

#
# Optimization
#
min_iterations = 60
# num_free_iterations = int( 0.5 * min_iterations )
max_iterations = 130
fom_empirical_gradient_dropoff = 0.05

num_collect_step_size_iterations = 5

start_design_change_max = 0.03
step_design_change_max = 0.03

