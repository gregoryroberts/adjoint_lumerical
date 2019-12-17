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
aperture_size_bounds_lambda_plus_units = [ 2, 8 ]
device_depth_bounds_lambda_plus_units = [ 2, 8 ]
# we will measure focal length from the end of the device
focal_length_bounds_lambda_plus_units = [ 1, 6 ]

#
# Simulation Region
#
lateral_gap_lambda_plus = 3
vertical_gap_lambda_plus = 3

lateral_gap_um = lambda_plus_um * lateral_gap_lambda_plus
vertical_gap_um = lambda_plus_um * vertical_gap_lambda_plus

mesh_spacing_lambda_plus = 1. / 15.
mesh_spacing_um = lambda_plus_um * mesh_spacing_lambda_plus

#
# Device
#
mesh_spacing_um = 0.025

c_eps_nought = ( 3.0 * 1e8 ) * ( 8.854 * 1e-12 )



fdtd_simulation_time_fs = 10000

#
# Focal spots/Adjoint sources
#
num_adjoint_sources = 2

#
# Optimization
#
num_iterations = 0

start_design_change_max = 0.05
end_design_change_max = 0.005
