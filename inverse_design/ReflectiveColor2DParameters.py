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
# project_name = 'cmos_dielectric_2d_refl_p22layers_rbg_lsf_contrast_3p6xtsmc_um'
# project_name = 'cmos_dielectric_2d_refl_p22layers_tio2_bin_lsf_blue_contrast_si_10p8xtsmc_um'
project_name = 'cmos_dielectric_2d_refl_p22layers_sio2_bin_lsf_blue_contrast_si_10p8xtsmc_um'

#
# Optical
#
# todo: the side background index shouldn't be TiO2 because that part will not be etched away!  Go back to oxide!
# Maybe force binarize more slowly.  Consider just optimizing for Strell ratio if we aren't going to get high transmisison anyway.
# Check binarized device and look at feature sizes!
background_index = 1.0
design_index_background = 1.0#1.35
device_background_index = 1.46#2.1#1.46#2.1#1.46#2.1#1.46#2.1#1.46#2.1#1.46#1.35# 2.5
# device_background_index = 2.2#1.46#2.1#1.46#2.1#1.46#2.1#1.46#2.1#1.46#2.1#1.46#1.35# 2.5
device_lateral_background_density = 1.0#0.0#1.0
# high_index_backfill = 2.5
high_index_backfill = 1.46#2.1#1.46#2.1#1.46#2.1#1.46#2.1#1.46#2.1#1.46
# high_index_backfill = 2.2
# high_index_backfill = 2.1
# substrate_index = 1.0

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
# focal_length_um = 1.5
# focal_length_um = 1.8
# focal_length_um = 2.0
# focal_length_um = 3.0

device_to_mode_match_um = 0.75 * 6




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

# device_layer_thicknesses_um = np.array( [
# 	0.13, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22
# ] )
# device_spacer_thicknesses_um = np.array( [
# 	0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.0
# ] )


feature_size_um_by_profiles = [ 0.1 for idx in range( 0, 6 ) ]
feature_size_um_by_profiles.append( 0.09 )
feature_size_voxels_by_profiles = [ int( 0.1 / lsf_mesh_spacing_um ) for idx in range( 0, 6 ) ]
feature_size_voxels_by_profiles.append( int( 0.09 / lsf_mesh_spacing_um ) )


# device_size_lateral_um = 3.0
# device_size_lateral_um = 3.6
device_size_lateral_um = 3.6 * 3
# device_size_lateral_um = 4.0
designable_size_vertical_um = np.sum( device_layer_thicknesses_um ) + np.sum( device_spacer_thicknesses_um )

device_size_verical_um = designable_size_vertical_um

opt_device_voxels_lateral = 1 + int( device_size_lateral_um / lsf_mesh_spacing_um )
# device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_lateral = 1 + int( ( device_size_lateral_um + np.finfo(np.float64).eps ) / mesh_spacing_um )
designable_device_voxels_vertical = 1 + int(designable_size_vertical_um / mesh_spacing_um)

designable_device_vertical_maximum_um = 0.5 * device_size_verical_um
designable_device_vertical_minimum_um = -0.5 * device_size_verical_um

#
# Spectral
#
num_bands = 3
num_points_per_band = 10#10#15

# band_weights = [ 1, 1, 1 ]
# band_weights = [ 1, 0, 1 ]
# band_weights = [ 0, 0, 1 ]
# band_weights = [ 0, 1, 1 ]
# pol_weights = [ 1, 1 ]
# pol_weights = [ 0, 1 ]
pol_weights = [ 1, 0 ]

lambda_min_um = 0.45
lambda_max_um = 0.75

num_design_frequency_points = num_bands * num_points_per_band
num_wavelengths = num_design_frequency_points
num_eval_frequency_points = 1 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
# max_intensity_by_wavelength = (device_size_lateral_um * 1.02)**2 / (focal_length_um**2 * lambda_values_um**2)



#
# FDTD
#
vertical_gap_size_um = 1.5
lateral_gap_size_um = 0.75 * device_size_lateral_um
# lateral_gap_size_um = 3.0 * device_size_lateral_um

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_verical_um + 2 * device_to_mode_match_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = 0.5 * device_size_verical_um + vertical_gap_size_um + device_to_mode_match_um
fdtd_region_minimum_vertical_um = -0.5 * device_size_verical_um - vertical_gap_size_um - device_to_mode_match_um

fdtd_region_minimum_vertical_voxels = int( np.ceil(fdtd_region_size_vertical_um / mesh_spacing_um) )
fdtd_region_minimum_lateral_voxels = int( np.ceil(fdtd_region_size_lateral_um / mesh_spacing_um) )

fdtd_region_size_lateral_voxels = int( np.ceil( fdtd_region_size_lateral_um / mesh_spacing_um ) )


fdtd_simulation_time_fs = 3 * 2000

#
# Forward Source
#
num_polarizations = 2

# lateral_aperture_um = 1.5 * device_size_lateral_um
lateral_aperture_um = 1.0 * device_size_lateral_um
adjoint_aperture_um = 2.0 * device_size_lateral_um
# adjoint_aperture_um = 5.0 * device_size_lateral_um
src_maximum_vertical_um = 0.5 * device_size_verical_um + 0.75 * device_to_mode_match_um
src_minimum_vertical_um = -0.5 * device_to_mode_match_um - 0.5 * vertical_gap_size_um

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }

# spectral_focal_plane_map = [
# 	[0, num_points_per_band],
# 	[2 * num_points_per_band, 3 * num_points_per_band],
# 	[num_points_per_band, 2 * num_points_per_band],
# ]

optimize_reflection_band = [ 0, num_points_per_band ]
# optimize_reflection_band = [ num_points_per_band, 2 * num_points_per_band ]
# optimize_reflection_band = [ 2 * num_points_per_band, 3 * num_points_per_band ]

# normal_reflection_weights = np.zeros( num_design_frequency_points )
# normal_reflection_weights[ optimize_reflection_band[ 0 ] : optimize_reflection_band[ 1 ] ] = 1.0

# angled_reflection_weights = np.zeros( num_design_frequency_points )

#
# We could also try to explicitly send it off to an angled mode instead of reducing the reflection mode overlap

# angled_reflection_weights = np.zeros( num_design_frequency_points )
# angled_reflection_weights[ optimize_reflection_band[ 0 ] : optimize_reflection_band[ 1 ] ] = 1.0 / num_points_per_band

# angled_reflection_weights[ 0 : optimize_reflection_band[ 0 ] ] = -1.0 / ( 2.0 * num_points_per_band )
# angled_reflection_weights[ optimize_reflection_band[ 1 ] : num_design_frequency_points ] = -1.0 / ( 2.0 * num_points_per_band )


# angled_transmission_weights = np.zeros( num_design_frequency_points )

# normal_reflection_weights = np.zeros( num_design_frequency_points )
# normal_reflection_weights[ optimize_reflection_band[ 0 ] : optimize_reflection_band[ 1 ] ] = -1.0 / num_points_per_band

# normal_reflection_weights[ 0 : optimize_reflection_band[ 0 ] ] = -1.0 / ( 2.0 * num_points_per_band )
# normal_reflection_weights[ optimize_reflection_band[ 1 ] : num_design_frequency_points ] = -1.0 / ( 2.0 * num_points_per_band )


# normal_transmission_weights = angled_transmission_weights.copy()



# plus_redirect_weights = ( 1.0 / num_design_frequency_points ) * np.ones( num_design_frequency_points )
plus_redirect_weights = np.zeros( num_design_frequency_points )
plus_redirect_weights[ 0 : optimize_reflection_band[ 0 ] ] = 1.0 / num_design_frequency_points
plus_redirect_weights[ optimize_reflection_band[ 1 ] : num_design_frequency_points ] = 1.0 / num_design_frequency_points

plus_direct_weights = np.zeros( num_design_frequency_points )
plus_direct_weights[ optimize_reflection_band[ 0 ] : optimize_reflection_band[ 1 ] ] = num_bands * ( 1.0 / num_design_frequency_points )


# plus_redirect_weights = np.zeros( num_design_frequency_points )
# plus_redirect_weights[ 9 ] = 1
# plus_redirect_weights[ 4 ] = 1
# plus_redirect_weights[ 23 ] = 1

# minus_redirect_weights = np.zeros( num_design_frequency_points )
minus_redirect_weights = ( 1.0 / num_design_frequency_points ) * np.ones( num_design_frequency_points )

# minus_redirect_weights[ 11 ] = 1
# minus_redirect_weights[ 0 : optimize_reflection_band[ 0 ] ] = 1.0 / num_design_frequency_points
# minus_redirect_weights[ optimize_reflection_band[ 1 ] : num_design_frequency_points ] = 1.0 / num_design_frequency_points

minus_direct_weights = np.zeros( num_design_frequency_points )
# minus_direct_weights[ optimize_reflection_band[ 0 ] : optimize_reflection_band[ 1 ] ] = 1.0 / num_design_frequency_points
minus_direct_weights[ optimize_reflection_band[ 0 ] : optimize_reflection_band[ 1 ] ] = num_bands * ( 1.0 / num_design_frequency_points )
# minus_direct_weights[ 4 ] = 1

print( plus_redirect_weights )
print( plus_direct_weights )
print( minus_redirect_weights )
print( minus_direct_weights )



device_rotation_angle_degrees = 12.0
device_rotation_angle_radians = device_rotation_angle_degrees * np.pi / 180.

# spectral_focal_plane_map = [
# 	[0 * num_points_per_band, 2 * num_points_per_band],
# 	[0, 3 * num_points_per_band],
# 	[num_points_per_band, 3 * num_points_per_band],
# ]
#
# Adjoint sources
#
# This seems like a long focal length
# adjoint_vertical_um = -focal_length_um
# num_focal_spots = 3
# num_adjoint_sources = num_focal_spots

# adjoint_x_positions_um = [ -3 * device_size_lateral_um / 8., -device_size_lateral_um / 8., device_size_lateral_um / 8., 3 * device_size_lateral_um / 8. ]
# adjoint_x_positions_um = [ -device_size_lateral_um / 3., 0.0, device_size_lateral_um / 3. ]

adjoint_reflection_position_y_um = 0.5 * designable_size_vertical_um + device_to_mode_match_um
adjoint_transmission_position_y_um = -0.5 * designable_size_vertical_um - device_to_mode_match_um

adjoint_beam_radius_um = 1.0#2.5
adjoint_beam_lateral_offset_um = 0.5 * device_size_lateral_um

#
# Optimization
#
num_epochs = 1
num_iterations_per_epoch = 25#10#80
init_optimization_epoch = 0

fom_quotient_regularization = 0.1

