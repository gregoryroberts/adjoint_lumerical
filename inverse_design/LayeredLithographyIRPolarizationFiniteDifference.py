import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyIRPolarizationParameters import *
import LayeredLithographyIRPolarizationDevice

# import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

#
# Qij is the the electric field j'th polarization component at a focal point due to an input beam polarized along the i'th direction
#
# This function expects the data to be organized first by the focal spot (corresponding to the given Jones analyzer vector laid out in the 
# parameters file) and then by wavelength.  It will optimize for broadband performance.
#
# The figure of merit here is intended to be minimized.  The function will also provide information for performance-based gradient weightings,
# but the figure of merit here will be computed as an evenly weighted average.
#
# def figure_of_merit( Qxx, Qxy, Qyx, Qyy ):

# 	total_fom = 0
# 	fom_by_focal_spot_by_type_by_wavelength = np.zeros( ( num_focal_spots, 3, num_design_frequency_points ) )

# 	for focal_spot_idx in range( 0, num_focal_spots ):
# 		alpha = jones_sorting_vectors[ focal_spot_idx ][ 0 ]
# 		beta = jones_sorting_vectors[ focal_spot_idx ][ 1 ]
# 		alpha_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 0 ]
# 		beta_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 1 ]

# 		Qxx_focal_spot = Qxx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
# 		Qxy_focal_spot = Qxy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
# 		Qyx_focal_spot = Qyx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
# 		Qyy_focal_spot = Qyy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )

# 		orthogonal_cancel_x = np.abs( Qyx_focal_spot + ( alpha_prime / beta_prime ) * Qxx_focal_spot )**2
# 		orthogonal_cancel_y = np.abs( Qxy_focal_spot + ( beta_prime / alpha_prime ) * Qyy_focal_spot )**2

# 		parallel = np.minimum( np.abs( alpha )**2, np.abs( beta )**2 ) * np.maximum(
# 			parallel_fom_bound - np.abs( Qxx_focal_spot / alpha )**2 - np.abs( Qyy_focal_spot / beta )**2,
# 			0 )

# 		total_fom += ( 1 / num_focal_spots ) * np.mean( orthogonal_cancel_x + orthogonal_cancel_y + parallel )

# 		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 0, : ] = orthogonal_cancel_x
# 		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 1, : ] = orthogonal_cancel_y
# 		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 2, : ] = parallel

# 	return total_fom, fom_by_focal_spot_by_type_by_wavelength

def figure_of_merit( Qxx, Qxy, Qyx, Qyy ):

	total_fom = 0
	fom_by_focal_spot_by_type_by_wavelength = np.zeros( ( num_focal_spots, 3, num_design_frequency_points ) )

	for focal_spot_idx in range( 0, num_focal_spots ):
		alpha = jones_sorting_vectors[ focal_spot_idx ][ 0 ]
		beta = jones_sorting_vectors[ focal_spot_idx ][ 1 ]
		alpha_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 0 ]
		beta_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 1 ]

		Qxx_focal_spot = Qxx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
		Qxy_focal_spot = Qxy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
		Qyx_focal_spot = Qyx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
		Qyy_focal_spot = Qyy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )

		orthogonal_cancel_x = np.abs( Qyx_focal_spot + ( alpha_prime / beta_prime ) * Qxx_focal_spot )**2
		orthogonal_cancel_y = np.abs( Qxy_focal_spot + ( beta_prime / alpha_prime ) * Qyy_focal_spot )**2

		parallel_x_term = np.abs( alpha * Qxx_focal_spot + beta * Qyx_focal_spot )**2
		parallel_y_term = np.abs( alpha * Qxy_focal_spot + beta * Qyy_focal_spot )**2
		parallel = np.maximum( parallel_fom_bound - parallel_x_term - parallel_y_term, 0 )

		total_fom += ( 1 / num_focal_spots ) * np.mean( orthogonal_cancel_x + orthogonal_cancel_y + parallel )

		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 0, : ] = orthogonal_cancel_x
		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 1, : ] = orthogonal_cancel_y
		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 2, : ] = parallel

	return total_fom, fom_by_focal_spot_by_type_by_wavelength

# def gradient(
# 	fom_by_focal_spot_by_type_by_wavelength,
# 	Ex_forward_fields, Ey_forward_fields,
# 	Ex_adjoint_fields_by_focal_spot, Ey_adjoint_fields_by_focal_spot,
# 	Qxx, Qxy, Qyx, Qyy ):

# 	num_total_fom = num_focal_spots * 3 * num_design_frequency_points
# 	rearrange_figures_of_merit = np.zeros( num_total_fom )

# 	for focal_spot_idx in range( 0, num_focal_spots ):
# 		for fom_type_idx in range( 0, 3 ):
# 			for wl_idx in range( 0, num_design_frequency_points ):
# 				rearrange_figures_of_merit[
# 					focal_spot_idx * 3 * num_design_frequency_points +
# 					fom_type_idx * num_design_frequency_points +
# 					wl_idx
# 				] = fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, fom_type_idx, wl_idx ]


# 	fom_weightings = ( 2. / num_total_fom ) - rearrange_figures_of_merit**2 / np.sum( rearrange_figures_of_merit )
# 	fom_weightings = np.maximum( fom_weightings, 0 )
# 	fom_weightings /= np.sum( fom_weightings )

# 	gradient_shape = Ex_forward_fields[ 0, 0 ].shape
# 	gradient = np.zeros( gradient_shape )

# 	for focal_spot_idx in range( 0, num_focal_spots ):

# 		alpha = jones_sorting_vectors[ focal_spot_idx ][ 0 ]
# 		beta = jones_sorting_vectors[ focal_spot_idx ][ 1 ]
# 		alpha_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 0 ]
# 		beta_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 1 ]

# 		Qxx_focal_spot = Qxx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
# 		Qxy_focal_spot = Qxy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
# 		Qyx_focal_spot = Qyx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
# 		Qyy_focal_spot = Qyy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )


# 		for fom_type_idx in range( 0, 3 ):
# 			weighting_start_idx = focal_spot_idx * 3 * num_design_frequency_points + fom_type_idx * num_design_frequency_points
# 			weighting_end_idx = weighting_start_idx + num_design_frequency_points

# 			get_weightings = fom_weightings[ weighting_start_idx : weighting_end_idx ]

# 			if fom_type_idx == 0:
# 				d_dQxx_0 = np.abs( alpha_prime / beta_prime )**2 * np.conj( Qxx_focal_spot ) + ( alpha_prime / beta_prime ) * np.conj( Qyx_focal_spot )
# 				d_dQyx_0 = np.conj( Qyx_focal_spot ) + np.conj( alpha_prime / beta_prime ) * np.conj( Qxx_focal_spot )

# 				for wl_idx in range( 0, num_design_frequency_points ):
# 					gradient_component_0_xx = 2 * np.real(
# 						np.sum(
# 							get_weightings[ wl_idx ] *
# 							d_dQxx_0[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
# 							axis=0
# 						)
# 					)

# 					gradient_component_0_yx = 2 * np.real(
# 						np.sum(
# 							get_weightings[ wl_idx ] *
# 							d_dQyx_0[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
# 							axis=0
# 						)
# 					)

# 					gradient += ( gradient_component_0_xx + gradient_component_0_yx )

# 			elif fom_type_idx == 1:
# 				d_dQyy_1 = np.abs( beta_prime / alpha_prime )**2 * np.conj( Qyy_focal_spot ) + ( beta_prime / alpha_prime ) * np.conj( Qxy_focal_spot )
# 				d_dQxy_1 = np.conj( Qxy_focal_spot ) + np.conj( beta_prime / alpha_prime ) * np.conj( Qyy_focal_spot )

# 				for wl_idx in range( 0, num_design_frequency_points ):
# 					gradient_component_1_yy = 2 * np.real(
# 						np.sum(
# 							get_weightings[ wl_idx ] *
# 							d_dQyy_1[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
# 							axis=0
# 						)
# 					)

# 					gradient_component_1_xy = 2 * np.real(
# 						np.sum(
# 							get_weightings[ wl_idx ] *
# 							d_dQxy_1[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
# 							axis=0
# 						)
# 					)

# 					gradient += ( gradient_component_1_yy + gradient_component_1_xy )

# 			else:
# 				d_dQxx_2 = -np.minimum( np.abs( alpha )**2, np.abs( beta )**2 ) * np.conj( Qxx_focal_spot ) / np.abs( alpha )**2
# 				d_dQyy_2 = -np.minimum( np.abs( alpha )**2, np.abs( beta )**2 ) * np.conj( Qyy_focal_spot ) / np.abs( beta )**2
# 				# d_dQxx_2 = -np.conj( Qxx_focal_spot ) / np.abs( alpha )**2
# 				# d_dQyy_2 = -np.conj( Qyy_focal_spot ) / np.abs( beta )**2

# 				for wl_idx in range( 0, num_design_frequency_points ):
# 					gradient_component_2_xx = 2 * np.real(
# 						np.sum(
# 							get_weightings[ wl_idx ] *
# 							d_dQxx_2[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
# 							axis=0
# 						)
# 					)

# 					gradient_component_2_yy = 2 * np.real(
# 						np.sum(
# 							get_weightings[ wl_idx ] *
# 							d_dQyy_2[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
# 							axis=0
# 						)
# 					)

# 					gradient += ( gradient_component_2_xx + gradient_component_2_yy )

# 	return gradient

def gradient(
	fom_by_focal_spot_by_type_by_wavelength,
	Ex_forward_fields, Ey_forward_fields,
	Ex_adjoint_fields_by_focal_spot, Ey_adjoint_fields_by_focal_spot,
	Qxx, Qxy, Qyx, Qyy ):

	num_total_fom = num_focal_spots * 3 * num_design_frequency_points
	rearrange_figures_of_merit = np.zeros( num_total_fom )

	for focal_spot_idx in range( 0, num_focal_spots ):
		for fom_type_idx in range( 0, 3 ):
			for wl_idx in range( 0, num_design_frequency_points ):
				rearrange_figures_of_merit[
					focal_spot_idx * 3 * num_design_frequency_points +
					fom_type_idx * num_design_frequency_points +
					wl_idx
				] = fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, fom_type_idx, wl_idx ]


	fom_weightings = ( 2. / num_total_fom ) - rearrange_figures_of_merit**2 / np.sum( rearrange_figures_of_merit )
	fom_weightings = np.maximum( fom_weightings, 0 )
	fom_weightings /= np.sum( fom_weightings )

	gradient_shape = Ex_forward_fields[ 0, 0 ].shape
	gradient = np.zeros( gradient_shape )

	for focal_spot_idx in range( 0, num_focal_spots ):

		alpha = jones_sorting_vectors[ focal_spot_idx ][ 0 ]
		beta = jones_sorting_vectors[ focal_spot_idx ][ 1 ]
		alpha_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 0 ]
		beta_prime = jones_orthogonal_vectors[ focal_spot_idx ][ 1 ]

		Qxx_focal_spot = Qxx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
		Qxy_focal_spot = Qxy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
		Qyx_focal_spot = Qyx[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )
		Qyy_focal_spot = Qyy[ focal_spot_idx, : ] / np.sqrt( max_intensity_by_wavelength )


		for fom_type_idx in range( 0, 3 ):
			weighting_start_idx = focal_spot_idx * 3 * num_design_frequency_points + fom_type_idx * num_design_frequency_points
			weighting_end_idx = weighting_start_idx + num_design_frequency_points

			get_weightings = fom_weightings[ weighting_start_idx : weighting_end_idx ]

			if fom_type_idx == 0:
				d_dQxx_0 = np.abs( alpha_prime / beta_prime )**2 * np.conj( Qxx_focal_spot ) + ( alpha_prime / beta_prime ) * np.conj( Qyx_focal_spot )
				d_dQyx_0 = np.conj( Qyx_focal_spot ) + np.conj( alpha_prime / beta_prime ) * np.conj( Qxx_focal_spot )

				for wl_idx in range( 0, num_design_frequency_points ):
					gradient_component_0_xx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxx_0[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient_component_0_yx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyx_0[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient += ( gradient_component_0_xx + gradient_component_0_yx )

			elif fom_type_idx == 1:
				d_dQyy_1 = np.abs( beta_prime / alpha_prime )**2 * np.conj( Qyy_focal_spot ) + ( beta_prime / alpha_prime ) * np.conj( Qxy_focal_spot )
				d_dQxy_1 = np.conj( Qxy_focal_spot ) + np.conj( beta_prime / alpha_prime ) * np.conj( Qyy_focal_spot )

				for wl_idx in range( 0, num_design_frequency_points ):
					gradient_component_1_yy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyy_1[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient_component_1_xy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxy_1[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient += ( gradient_component_1_yy + gradient_component_1_xy )

			else:
				d_dQxx_2 = -( np.abs( alpha )**2 * np.conj( Qxx_focal_spot ) + alpha * np.conj( beta ) * np.conj( Qyx_focal_spot ) )
				d_dQyx_2 = -( np.abs( beta )**2 * np.conj( Qyx_focal_spot ) + np.conj( alpha ) * beta * np.conj( Qxx_focal_spot ) )

				d_dQxy_2 = -( np.abs( alpha )**2 * np.conj( Qxy_focal_spot ) + alpha * np.conj( beta  ) * np.conj( Qyy_focal_spot ) )
				d_dQyy_2 = -( np.abs( beta )**2 * np.conj( Qyy_focal_spot ) + np.conj( alpha ) * beta * np.conj( Qxy_focal_spot ) )

				for wl_idx in range( 0, num_design_frequency_points ):
					gradient_component_2_xx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxx_2[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient_component_2_yx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyx_2[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient_component_2_xy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxy_2[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					gradient_component_2_yy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyy_2[ wl_idx ] * Ey_forward_fields[ :, wl_idx, :, :, : ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
							axis=0
						)
					)

					if fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, fom_type_idx, wl_idx ] > 0:
						gradient += ( gradient_component_2_xx + gradient_component_2_yx + gradient_component_2_xy + gradient_component_2_yy )

	return gradient


#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
projects_directory_location += "/" + project_name + "_fd1em3"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredLithographyIRPolarizationParameters.py", projects_directory_location + "/LayeredLithographyIRPolarizationParameters.py")

#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['z max'] = fdtd_region_maximum_vertical_um * 1e-6
fdtd['z min'] = fdtd_region_minimum_vertical_um * 1e-6
fdtd['mesh type'] = 'uniform'
fdtd['mesh refinement'] = 'staircase'
fdtd['define x mesh by'] = 'number of mesh cells'
fdtd['define y mesh by'] = 'number of mesh cells'
fdtd['define z mesh by'] = 'number of mesh cells'
fdtd['mesh cells x'] = fdtd_region_minimum_lateral_voxels
fdtd['mesh cells y'] = fdtd_region_minimum_lateral_voxels
fdtd['mesh cells z'] = fdtd_region_minimum_vertical_voxels
fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
fdtd['background index'] = background_index

#
# General polarized source information
#
xy_phi_rotations = [0, 90]
xy_names = ['x', 'y']


#
# Add a TFSF plane wave forward source at normal incidence
#
forward_sources = []

for xy_idx in range(0, 2):
	forward_src = fdtd_hook.addtfsf()
	forward_src['name'] = 'forward_src_' + xy_names[xy_idx]
	forward_src['angle phi'] = xy_phi_rotations[xy_idx]
	forward_src['direction'] = 'Backward'
	forward_src['x span'] = lateral_aperture_um * 1e-6
	forward_src['y span'] = lateral_aperture_um * 1e-6
	forward_src['z max'] = src_maximum_vertical_um * 1e-6
	forward_src['z min'] = src_minimum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_max_um * 1e-6

	forward_sources.append(forward_src)

#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = []

for adj_src_idx in range(0, num_adjoint_sources):
	adjoint_sources.append([])
	for xy_idx in range(0, 2):
		adj_src = fdtd_hook.adddipole()
		adj_src['name'] = 'adj_src_' + str(adj_src_idx) + xy_names[xy_idx]
		adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
		adj_src['y'] = adjoint_y_positions_um[adj_src_idx] * 1e-6
		adj_src['z'] = adjoint_vertical_um * 1e-6
		adj_src['theta'] = 90
		adj_src['phi'] = xy_phi_rotations[xy_idx]
		adj_src['wavelength start'] = lambda_min_um * 1e-6
		adj_src['wavelength stop'] = lambda_max_um * 1e-6

		adjoint_sources[adj_src_idx].append(adj_src)

#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['monitor type'] = '3D'
design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
design_efield_monitor['y span'] = device_size_lateral_um * 1e-6
design_efield_monitor['z max'] = device_vertical_maximum_um * 1e-6
design_efield_monitor['z min'] = device_vertical_minimum_um * 1e-6
design_efield_monitor['override global monitor settings'] = 1
design_efield_monitor['use wavelength spacing'] = 1
design_efield_monitor['use source limits'] = 1
design_efield_monitor['frequency points'] = num_design_frequency_points
design_efield_monitor['output Hx'] = 0
design_efield_monitor['output Hy'] = 0
design_efield_monitor['output Hz'] = 0

#
# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
# gradient.
#
focal_monitors = []

for adj_src in range(0, num_adjoint_sources):
	focal_monitor = fdtd_hook.addpower()
	focal_monitor['name'] = 'focal_monitor_' + str(adj_src)
	focal_monitor['monitor type'] = 'point'
	focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	focal_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
	focal_monitor['z'] = adjoint_vertical_um * 1e-6
	focal_monitor['override global monitor settings'] = 1
	focal_monitor['use wavelength spacing'] = 1
	focal_monitor['use source limits'] = 1
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)


#
# Add SiO2 at the top
#
sio2_top = fdtd_hook.addrect()
sio2_top['name'] = 'sio2_top'
sio2_top['x span'] = fdtd_region_size_lateral_um * 1e-6
sio2_top['y span'] = fdtd_region_size_lateral_um * 1e-6
sio2_top['z min'] = device_vertical_maximum_um * 1e-6
sio2_top['z max'] = fdtd_region_maximum_vertical_um * 1e-6
sio2_top['index'] = index_sio2

air_bottom = fdtd_hook.addrect()
air_bottom['name'] = 'air_bottom'
air_bottom['x span'] = fdtd_region_size_lateral_um * 1e-6
air_bottom['y span'] = fdtd_region_size_lateral_um * 1e-6
air_bottom['z min'] = fdtd_region_minimum_vertical_um * 1e-6
air_bottom['z max'] = device_vertical_minimum_um * 1e-6
air_bottom['index'] = index_air


#
# Add device region and create device permittivity
#
design_import = fdtd_hook.addimport()
design_import['name'] = 'design_import'
design_import['x span'] = device_size_lateral_um * 1e-6
design_import['y span'] = device_size_lateral_um * 1e-6
design_import['z max'] = device_vertical_maximum_um * 1e-6
design_import['z min'] = device_vertical_minimum_um * 1e-6

bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
bayer_filter = LayeredLithographyIRPolarizationDevice.LayeredLithographyIRPolarizationDevice(
	bayer_filter_size_voxels,
	[min_device_permittivity, max_device_permittivity],
	init_permittivity_0_1_scale,
	num_vertical_layers,
	spacer_size_voxels,
	[index_air**2, index_silicon**2],
	max_binarize_movement,
	desired_binarize_change)


bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	fdtd_hook.switchtolayout()

	for xy_idx in range(0, 2):
		(forward_sources[xy_idx]).enabled = 0

	for adj_src_idx in range(0, num_adjoint_sources):
		for xy_idx in range(0, 2):
			(adjoint_sources[adj_src_idx][xy_idx]).enabled = 0

#
# Consolidate the data transfer functionality for getting data from Lumerical FDTD process to
# python process.  This is much faster than going through Lumerical's interop library
#
def get_monitor_data(monitor_name, monitor_field):
	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	extracted_data_name = lumerical_data_name + "_data"
	data_transfer_filename = projects_directory_location + "/data_transfer_" + monitor_name + "_" + monitor_field

	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command_extract_data = extracted_data_name + " = " + lumerical_data_name + "." + monitor_field + ";"
	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + extracted_data_name + ");"

	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)
	lumapi.evalScript(fdtd_hook.handle, command_extract_data)

	# start_time = time.time()

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat")

	monitor_data = np.array(load_file[extracted_data_name])

	# end_time = time.time()

	# print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	data = get_monitor_data(monitor_name, monitor_field)
	return (data['real'] + np.complex(0, 1) * data['imag'])

#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

fix_fd_x = int( 0.5 * device_voxels_lateral )
fix_fd_y = int( 0.5 * device_voxels_lateral )
num_fd_z = device_voxels_vertical

cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )
cur_permittivity = (1.5**2) * np.ones( cur_permittivity.shape )

fdtd_hook.switchtolayout()
fdtd_hook.select("design_import")
fdtd_hook.importnk2(np.sqrt(cur_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)

delta = 1e-3

Qxx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
Qxy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
Qyx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
Qyy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )

for xy_idx in range(0, 2):
	disable_all_sources()
	(forward_sources[xy_idx]).enabled = 1
	fdtd_hook.run()

	forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	focal_data[xy_names[xy_idx]] = []
	for focal_idx in range( 0, num_focal_spots ):
		focal_monitor_data = get_complex_monitor_data( focal_monitors[ focal_idx ][ 'name' ], 'E' )

		if xy_idx == 0:
			Qxx[ focal_idx, : ] = focal_monitor_data[ 0, :, 0, 0, 0 ]
			Qxy[ focal_idx, : ] = focal_monitor_data[ 1, :, 0, 0, 0 ]
		else:
			Qyy[ focal_idx, : ] = focal_monitor_data[ 1, :, 0, 0, 0 ]
			Qyx[ focal_idx, : ] = focal_monitor_data[ 0, :, 0, 0, 0 ]


current_figure_of_merit_0, fom_by_focal_spot_by_type_by_wavelength_0 = figure_of_merit( Qxx, Qxy, Qyx, Qyy )


adjoint_ex_fields = []
adjoint_ey_fields = []
for adj_src_idx in range(0, num_adjoint_sources):
    for xy_idx in range(0, 2):
        disable_all_sources()
        (adjoint_sources[adj_src_idx][xy_idx]).enabled = 1
        fdtd_hook.run()

        if xy_idx == 0:
            adjoint_ex_fields.append(
                get_complex_monitor_data(design_efield_monitor['name'], 'E'))
        else:
            adjoint_ey_fields.append(
                get_complex_monitor_data(design_efield_monitor['name'], 'E'))

np.save( projects_directory_location + "/fom_by_focal_spot_by_type_by_wavelength_check_fd.npy", fom_by_focal_spot_by_type_by_wavelength_0 )
np.save( projects_directory_location + "/forward_ex_fields_check_fd.npy", forward_e_fields[ 'x' ] )
np.save( projects_directory_location + "/forward_ey_fields_check_fd.npy", forward_e_fields[ 'y' ] )
np.save( projects_directory_location + "/adjoint_ex_by_focal_spot_fields_check_fd.npy", adjoint_ex_fields )
np.save( projects_directory_location + "/adjoint_ey_by_focal_spot_fields_check_fd.npy", adjoint_ey_fields )
np.save( projects_directory_location + "/Qxx_check_fd.npy", Qxx )
np.save( projects_directory_location + "/Qxy_check_fd.npy", Qxy )
np.save( projects_directory_location + "/Qyx_check_fd.npy", Qyx )
np.save( projects_directory_location + "/Qyy_check_fd.npy", Qyy )

minimization_gradient = gradient(
 	fom_by_focal_spot_by_type_by_wavelength_0,
 	forward_e_fields[ 'x' ], forward_e_fields[ 'y' ],
 	adjoint_ex_fields, adjoint_ey_fields,
 	Qxx, Qxy, Qyx, Qyy )

np.save( projects_directory_location + "/fom_lower.npy", fom_by_focal_spot_by_type_by_wavelength_0 )

finite_difference_gradient = np.zeros( num_fd_z )
finite_difference_gradient_all = np.zeros( ( num_fd_z, num_focal_spots, 3, num_design_frequency_points ) )
fom_upper_all = np.zeros( ( num_fd_z, num_focal_spots, 3, num_design_frequency_points ) )

finite_difference_gradient_single_sided = np.zeros( finite_difference_gradient.shape )
finite_difference_gradient_all_single_sided = np.zeros( finite_difference_gradient_all.shape )

for fd_z_idx in range( 0, num_fd_z ):
	print( "Currently working on finite difference iteration " + str( fd_z_idx ) + " out of " + str( num_fd_z ) + " total!" )

	fd_permittivity = cur_permittivity.copy()
	fd_permittivity[ ( fix_fd_x - 3 ) : ( fix_fd_x + 3 ), ( fix_fd_y - 3 ) : ( fix_fd_y + 3 ), fd_z_idx ] += delta

	fdtd_hook.switchtolayout()
	fdtd_hook.select("design_import")
	fdtd_hook.importnk2(np.sqrt(fd_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)

	Qxx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
	Qxy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
	Qyx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
	Qyy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )

	for xy_idx in range(0, 2):
		disable_all_sources()
		(forward_sources[xy_idx]).enabled = 1
		fdtd_hook.run()

		forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')

		focal_data[xy_names[xy_idx]] = []
		for focal_idx in range( 0, num_focal_spots ):
			focal_monitor_data = get_complex_monitor_data( focal_monitors[ focal_idx ][ 'name' ], 'E' )

			if xy_idx == 0:
				Qxx[ focal_idx, : ] = focal_monitor_data[ 0, :, 0, 0, 0 ]
				Qxy[ focal_idx, : ] = focal_monitor_data[ 1, :, 0, 0, 0 ]
			else:
				Qyy[ focal_idx, : ] = focal_monitor_data[ 1, :, 0, 0, 0 ]
				Qyx[ focal_idx, : ] = focal_monitor_data[ 0, :, 0, 0, 0 ]


	current_figure_of_merit_up, fom_by_focal_spot_by_type_by_wavelength_up = figure_of_merit( Qxx, Qxy, Qyx, Qyy )

	fd_permittivity = cur_permittivity.copy()
	fd_permittivity[ ( fix_fd_x - 3 ) : ( fix_fd_x + 3 ), ( fix_fd_y - 3 ) : ( fix_fd_y + 3 ), fd_z_idx ] -= delta

	fdtd_hook.switchtolayout()
	fdtd_hook.select("design_import")
	fdtd_hook.importnk2(np.sqrt(fd_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)

	Qxx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
	Qxy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
	Qyx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
	Qyy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )

	for xy_idx in range(0, 2):
		disable_all_sources()
		(forward_sources[xy_idx]).enabled = 1
		fdtd_hook.run()

		forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')

		focal_data[xy_names[xy_idx]] = []
		for focal_idx in range( 0, num_focal_spots ):
			focal_monitor_data = get_complex_monitor_data( focal_monitors[ focal_idx ][ 'name' ], 'E' )

			if xy_idx == 0:
				Qxx[ focal_idx, : ] = focal_monitor_data[ 0, :, 0, 0, 0 ]
				Qxy[ focal_idx, : ] = focal_monitor_data[ 1, :, 0, 0, 0 ]
			else:
				Qyy[ focal_idx, : ] = focal_monitor_data[ 1, :, 0, 0, 0 ]
				Qyx[ focal_idx, : ] = focal_monitor_data[ 0, :, 0, 0, 0 ]


	current_figure_of_merit_down, fom_by_focal_spot_by_type_by_wavelength_down = figure_of_merit( Qxx, Qxy, Qyx, Qyy )


	finite_difference_gradient[ fd_z_idx ] = ( current_figure_of_merit_up - current_figure_of_merit_down ) / ( 2 * delta )
	finite_difference_gradient_all[ fd_z_idx ] = ( fom_by_focal_spot_by_type_by_wavelength_up - fom_by_focal_spot_by_type_by_wavelength_down ) / ( 2 * delta )
	fom_upper_all[ fd_z_idx ] = fom_by_focal_spot_by_type_by_wavelength_up
	finite_difference_gradient_single_sided[ fd_z_idx ] = ( current_figure_of_merit_up - current_figure_of_merit_0 ) / ( delta )
	finite_difference_gradient_all_single_sided[ fd_z_idx ] = ( fom_by_focal_spot_by_type_by_wavelength_up - fom_by_focal_spot_by_type_by_wavelength_0 ) / ( delta )


	np.save( projects_directory_location + "/finite_difference_gradient.npy", finite_difference_gradient )
	np.save( projects_directory_location + "/finite_difference_gradient_all.npy", finite_difference_gradient_all )
	np.save( projects_directory_location + "/fom_upper_all.npy", fom_upper_all )
	np.save( projects_directory_location + "/finite_difference_gradient_single_sided.npy", finite_difference_gradient_single_sided )
	np.save( projects_directory_location + "/finite_difference_gradient_all_single_sided.npy", finite_difference_gradient_all_single_sided )

