import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyIRPolarizationParameters import *
import LayeredLithographyIRPolarizationDevice

import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )


import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

import queue

import subprocess

import platform

import re



def get_slurm_node_list( slurm_job_env_variable=None ):
	if slurm_job_env_variable is None:
		slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
	if slurm_job_env_variable is None:
		raise ValueError('Environment variable does not exist.')

	solo_node_pattern = r'hpc-\d\d-[\w]+'
	cluster_node_pattern = r'hpc-\d\d-\[.*?\]'
	solo_nodes = re.findall(solo_node_pattern, slurm_job_env_variable)
	cluster_nodes = re.findall(cluster_node_pattern, slurm_job_env_variable)
	inner_bracket_pattern = r'\[(.*?)\]'

	output_arr = solo_nodes
	for cluster_node in cluster_nodes:
		prefix = cluster_node.split('[')[0]
		inside_brackets = re.findall(inner_bracket_pattern, cluster_node)[0]
		# Split at commas and iterate through results
		for group in inside_brackets.split(','):
			# Split at hypen. Get first and last number. Create string in range
			# from first to last.
			node_clump_split = group.split('-')
			starting_number = int(node_clump_split[0])
			try:
				ending_number = int(node_clump_split[1])
			except IndexError:
				ending_number = starting_number
			for i in range(starting_number, ending_number+1):
				# Can use print("{:02d}".format(1)) to turn a 1 into a '01'
				# string. 111 -> 111 still, in case nodes hit triple-digits.
				output_arr.append(prefix + "{:02d}".format(i))
	return output_arr



num_nodes_available = int( sys.argv[ 1 ] )
num_cpus_per_node = 8
cluster_hostnames = get_slurm_node_list()

#
# Qij is the the electric field j'th polarization component at a focal point due to an input beam polarized along the i'th direction
#
# This function expects the data to be organized first by the focal spot (corresponding to the given Jones analyzer vector laid out in the 
# parameters file) and then by wavelength.  It will optimize for broadband performance.
#
# The figure of merit here is intended to be minimized.  The function will also provide information for performance-based gradient weightings,
# but the figure of merit here will be computed as an evenly weighted average.
#
'''
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

		parallel = np.minimum( np.abs( alpha )**2, np.abs( beta )**2 ) * np.maximum(
			parallel_fom_bound - np.abs( Qxx_focal_spot / alpha )**2 - np.abs( Qyy_focal_spot / beta )**2,
			0 )

		total_fom += ( 1 / num_focal_spots ) * np.mean( orthogonal_cancel_x + orthogonal_cancel_y + parallel )

		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 0, : ] = orthogonal_cancel_x
		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 1, : ] = orthogonal_cancel_y
		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 2, : ] = parallel

	return total_fom, fom_by_focal_spot_by_type_by_wavelength
'''

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
		# parallel = np.maximum( parallel_fom_bound - parallel_x_term - parallel_y_term, 0 )

		orthogonal_x_term = np.abs( alpha_prime * Qxx_focal_spot + beta_prime * Qyx_focal_spot )**2
		orthogonal_y_term = np.abs( alpha_prime * Qxy_focal_spot + beta_prime * Qyy_focal_spot )**2

		parallel_intensity = parallel_x_term + parallel_y_term
		orthogonal_intensity = orthogonal_x_term + orthogonal_y_term

		parallel_fom = np.minimum( parallel_fom_bound, parallel_intensity )
		orthogonal_fom = np.maximum( 1 - orthogonal_intensity, 0 )
		product_fom = parallel_fom * orthogonal_fom

		# total_fom += ( 1 / num_focal_spots ) * np.mean( orthogonal_cancel_x + orthogonal_cancel_y + parallel )
		# total_fom += ( 1 / num_focal_spots ) * np.mean( parallel )
		total_fom += ( 1. / num_focal_spots ) * np.mean( product_fom )

		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 0, : ] = orthogonal_cancel_x
		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 1, : ] = orthogonal_cancel_y
		# fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 2, : ] = parallel
		fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, 2, : ] = product_fom

	return total_fom, fom_by_focal_spot_by_type_by_wavelength

'''

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
				d_dQxx_2 = -np.minimum( np.abs( alpha )**2, np.abs( beta )**2 ) * np.conj( Qxx_focal_spot ) / np.abs( alpha )**2
				d_dQyy_2 = -np.minimum( np.abs( alpha )**2, np.abs( beta )**2 ) * np.conj( Qyy_focal_spot ) / np.abs( beta )**2

				for wl_idx in range( 0, num_design_frequency_points ):
					gradient_component_2_xx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxx_2[ wl_idx ] * Ex_forward_fields[ :, wl_idx, :, :, : ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, wl_idx, :, :, : ],
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
						gradient += ( gradient_component_2_xx + gradient_component_2_yy )

	return gradient

'''

def gradient(
	fom_by_focal_spot_by_type_by_wavelength,
	Ex_forward_fields, Ey_forward_fields,
	Ex_adjoint_fields_by_focal_spot, Ey_adjoint_fields_by_focal_spot,
	Qxx, Qxy, Qyx, Qyy ):

	num_total_fom = num_focal_spots * 3 * num_design_frequency_points
	rearrange_figures_of_merit = np.zeros( num_total_fom )
	weighting_mask = np.zeros( num_total_fom )

	for focal_spot_idx in range( 0, num_focal_spots ):
		for fom_type_idx in range( 0, 3 ):
			for wl_idx in range( 0, num_design_frequency_points ):
				rearrange_figures_of_merit[
					focal_spot_idx * 3 * num_design_frequency_points +
					fom_type_idx * num_design_frequency_points +
					wl_idx
				] = fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, fom_type_idx, wl_idx ]

				if fom_type_idx == 2:
					weighting_mask[ focal_spot_idx * 3 * num_design_frequency_points +
						fom_type_idx * num_design_frequency_points +
						wl_idx ] = 1


	fom_weightings = ( 2. / num_total_fom ) - rearrange_figures_of_merit**2 / np.sum( rearrange_figures_of_merit )
	fom_weightings = np.maximum( fom_weightings, 0 )
	fom_weightings /= np.sum( fom_weightings )

	#
	# This is because we are actually minimizing all three figures of merit, so we would like to flip the orientation
	# of the weightings (i.e. - a small figure of merit means you are doing well in this optimization)
	#
	# fom_weightings = 1 - fom_weightings
	# fom_weightings *= weighting_mask
	# Renormalize, so they add to 1
	# fom_weightings /= np.sum( fom_weightings )

	gradient_shape = Ex_forward_fields[ 0, :, :, :, 0 ].shape
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

		parallel_x_term = np.abs( alpha * Qxx_focal_spot + beta * Qyx_focal_spot )**2
		parallel_y_term = np.abs( alpha * Qxy_focal_spot + beta * Qyy_focal_spot )**2

		orthogonal_x_term = np.abs( alpha_prime * Qxx_focal_spot + beta_prime * Qyx_focal_spot )**2
		orthogonal_y_term = np.abs( alpha_prime * Qxy_focal_spot + beta_prime * Qyy_focal_spot )**2

		parallel_intensity = parallel_x_term + parallel_y_term
		orthogonal_intensity = orthogonal_x_term + orthogonal_y_term

		parallel_fom = np.minimum( parallel_fom_bound, parallel_intensity )
		orthogonal_fom = np.maximum( 1 - orthogonal_intensity, 0 )

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
							d_dQxx_0[ wl_idx ] * Ex_forward_fields[ :, :, :, :, wl_idx ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_0_yx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyx_0[ wl_idx ] * Ey_forward_fields[ :, :, :, :, wl_idx ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
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
							d_dQyy_1[ wl_idx ] * Ey_forward_fields[ :, :, :, :, wl_idx ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_1_xy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxy_1[ wl_idx ] * Ex_forward_fields[ :, :, :, :, wl_idx ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient += ( gradient_component_1_yy + gradient_component_1_xy )

			else:
				d_dQxx_2 = -( np.abs( alpha )**2 * np.conj( Qxx_focal_spot ) + alpha * np.conj( beta ) * np.conj( Qyx_focal_spot ) )
				d_dQyx_2 = -( np.abs( beta )**2 * np.conj( Qyx_focal_spot ) + np.conj( alpha ) * beta * np.conj( Qxx_focal_spot ) )

				d_dQxy_2 = -( np.abs( alpha )**2 * np.conj( Qxy_focal_spot ) + alpha * np.conj( beta  ) * np.conj( Qyy_focal_spot ) )
				d_dQyy_2 = -( np.abs( beta )**2 * np.conj( Qyy_focal_spot ) + np.conj( alpha ) * beta * np.conj( Qxy_focal_spot ) )


				d_dQxx_2_prime = -( np.abs( alpha_prime )**2 * np.conj( Qxx_focal_spot ) + alpha_prime * np.conj( beta_prime ) * np.conj( Qyx_focal_spot ) )
				d_dQyx_2_prime = -( np.abs( beta_prime )**2 * np.conj( Qyx_focal_spot ) + np.conj( alpha_prime ) * beta_prime * np.conj( Qxx_focal_spot ) )

				d_dQxy_2_prime = -( np.abs( alpha_prime )**2 * np.conj( Qxy_focal_spot ) + alpha_prime * np.conj( beta_prime  ) * np.conj( Qyy_focal_spot ) )
				d_dQyy_2_prime = -( np.abs( beta_prime )**2 * np.conj( Qyy_focal_spot ) + np.conj( alpha_prime ) * beta_prime * np.conj( Qxy_focal_spot ) )



				for wl_idx in range( 0, num_design_frequency_points ):
					gradient_component_2_xx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxx_2[ wl_idx ] * Ex_forward_fields[ :, :, :, :, wl_idx ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_2_yx = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyx_2[ wl_idx ] * Ey_forward_fields[ :, :, :, :, wl_idx ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_2_xy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxy_2[ wl_idx ] * Ex_forward_fields[ :, :, :, :, wl_idx ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_2_yy = 2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyy_2[ wl_idx ] * Ey_forward_fields[ :, :, :, :, wl_idx ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)


					gradient_component_2_xx_prime = -2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxx_2_prime[ wl_idx ] * Ex_forward_fields[ :, :, :, :, wl_idx ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_2_yx_prime = -2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyx_2_prime[ wl_idx ] * Ey_forward_fields[ :, :, :, :, wl_idx ] * Ex_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_2_xy_prime = -2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQxy_2_prime[ wl_idx ] * Ex_forward_fields[ :, :, :, :, wl_idx ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)

					gradient_component_2_yy_prime = -2 * np.real(
						np.sum(
							get_weightings[ wl_idx ] *
							d_dQyy_2_prime[ wl_idx ] * Ey_forward_fields[ :, :, :, :, wl_idx ] * Ey_adjoint_fields_by_focal_spot[ focal_spot_idx ][ :, :, :, :, wl_idx ],
							axis=0
						)
					)




					if parallel_fom[ wl_idx ] < parallel_fom_bound:
						gradient += orthogonal_fom[ wl_idx ] * ( gradient_component_2_xx + gradient_component_2_yx + gradient_component_2_xy + gradient_component_2_yy )

					gradient += parallel_fom[ wl_idx ] * ( gradient_component_2_xx_prime + gradient_component_2_yx_prime + gradient_component_2_xy_prime + gradient_component_2_yy_prime )


					# if fom_by_focal_spot_by_type_by_wavelength[ focal_spot_idx, fom_type_idx, wl_idx ] > 0:
					# 	gradient += ( gradient_component_2_xx + gradient_component_2_yx + gradient_component_2_xy + gradient_component_2_yy )

	return gradient



#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
# projects_directory_location += "/" + project_name + "_parallel"

projects_directory_location = "/central/groups/Faraon_Computing/projects" 
projects_directory_location += "/" + project_name + '_parallel'

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


transmission_focal_monitors = []

for adj_src in range(0, num_adjoint_sources):
	transmission_focal_monitor = fdtd_hook.addpower()
	transmission_focal_monitor['name'] = 'transmission_focal_monitor_' + str(adj_src)
	transmission_focal_monitor['monitor type'] = '2D Z-Normal'
	transmission_focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_focal_monitor['x span'] = 0.5 * device_size_lateral_um * 1e-6
	transmission_focal_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
	transmission_focal_monitor['y span'] = 0.5 * device_size_lateral_um * 1e-6
	transmission_focal_monitor['z'] = adjoint_vertical_um * 1e-6
	transmission_focal_monitor['override global monitor settings'] = 1
	transmission_focal_monitor['use wavelength spacing'] = 1
	transmission_focal_monitor['use source limits'] = 1
	transmission_focal_monitor['frequency points'] = num_eval_frequency_points
	transmission_focal_monitor.enabled = 0

	transmission_focal_monitors.append(transmission_focal_monitor)

transmission_focal = fdtd_hook.addpower()
transmission_focal['name'] = 'transmission_focal'
transmission_focal['monitor type'] = '2D Z-Normal'
transmission_focal['x'] = 0 * 1e-6
transmission_focal['x span'] = device_size_lateral_um * 1e-6
transmission_focal['y'] = 0 * 1e-6
transmission_focal['y span'] = device_size_lateral_um * 1e-6
transmission_focal['z'] = adjoint_vertical_um * 1e-6
transmission_focal['override global monitor settings'] = 1
transmission_focal['use wavelength spacing'] = 1
transmission_focal['use source limits'] = 1
transmission_focal['frequency points'] = num_eval_frequency_points
transmission_focal.enabled = 0


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


# bayer_filter.set_design_variable( np.random.random( bayer_filter.get_design_variable().shape ) )
# bayer_filter.step(
# 	np.random.random( bayer_filter.get_design_variable().shape ),
# 	0.01,
# 	True,
# 	projects_directory_location
# )
# sys.exit(0)

# bayer_filter.set_design_variable( np.load(projects_directory_location + "/cur_design_variable.npy") )

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	for xy_idx in range(0, 2):
		fdtd_hook.select( forward_sources[xy_idx]['name'] )
		fdtd_hook.set( 'enabled', 0 )

	for adj_src_idx in range(0, num_adjoint_sources):
		for xy_idx in range(0, 2):
			fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
			fdtd_hook.set( 'enabled', 0 )


#
# Consolidate the data transfer functionality for getting data from Lumerical FDTD process to
# python process.  This is much faster than going through Lumerical's interop library
#
# def get_monitor_data(monitor_name, monitor_field):
# 	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
# 	extracted_data_name = lumerical_data_name + "_data"
# 	data_transfer_filename = projects_directory_location + "/data_transfer_" + monitor_name + "_" + monitor_field

# 	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
# 	command_extract_data = extracted_data_name + " = " + lumerical_data_name + "." + monitor_field + ";"
# 	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + extracted_data_name + ");"

# 	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)
# 	lumapi.evalScript(fdtd_hook.handle, command_extract_data)

# 	# start_time = time.time()

# 	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
# 	monitor_data = {}
# 	load_file = h5py.File(data_transfer_filename + ".mat")

# 	monitor_data = np.array(load_file[extracted_data_name])

# 	# end_time = time.time()

# 	# print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

# 	return monitor_data

# def get_complex_monitor_data(monitor_name, monitor_field):
# 	data = get_monitor_data(monitor_name, monitor_field)
# 	return (data['real'] + np.complex(0, 1) * data['imag'])

def get_efield( monitor_name ):
	field_polariations = [ 'Ex', 'Ey', 'Ez' ]
	data_xfer_size_MB = 0

	start = time.time()

	Epol_0 = fdtd_hook.getdata( monitor_name, field_polariations[ 0 ] )
	data_xfer_size_MB += Epol_0.nbytes / ( 1024. * 1024. )

	total_efield = np.zeros( [ len (field_polariations ) ] + list( Epol_0.shape ), dtype=np.complex )
	total_efield[ 0 ] = Epol_0

	for pol_idx in range( 1, len( field_polariations ) ):
		Epol = fdtd_hook.getdata( monitor_name, field_polariations[ pol_idx ] )
		data_xfer_size_MB += Epol.nbytes / ( 1024. * 1024. )

		total_efield[ pol_idx ] = Epol

	elapsed = time.time() - start

	date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
	log_file.write( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )
	log_file.close()

	return total_efield

#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_by_focal_spot_by_type_by_wavelength_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_focal_spots, 3, num_design_frequency_points))
contrast_per_focal_spot_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_focal_spots, num_design_frequency_points))
parallel_intensity_per_focal_spot_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_focal_spots, num_design_frequency_points))
orthogonal_intensity_per_focal_spot_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_focal_spots, num_design_frequency_points))

step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = 0.001

if start_epoch > 0:
	design_variable_reload = np.load( projects_directory_location + '/cur_design_variable_' + str( start_epoch - 1 ) + '.npy' )
	bayer_filter.set_design_variable( design_variable_reload )




jobs_queue = queue.Queue()

def add_job( job_name, queue_in ):
	full_name = projects_directory_location + "/" + job_name
	fdtd_hook.save( full_name )
	queue_in.put( full_name )

	return full_name

def run_jobs( queue_in ):
	small_queue = queue.Queue()

	while not queue_in.empty():
		for node_idx in range( 0, num_nodes_available ):
			small_queue.put( queue_in.get() )

		run_jobs_inner( small_queue )

def run_jobs_inner( queue_in ):
	processes = []
	# for job_idx in range( 0, len( queue_in ) ):
	# really should protect against number of available engines here
	job_idx = 0
	while not queue_in.empty():
		get_job_path = queue_in.get()

		process = subprocess.Popen(
			[
				'/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
				cluster_hostnames[ job_idx ],
				get_job_path
			]
		)
		processes.append( process )

		job_idx += 1
	
	completed_jobs = [ 0 for i in range( 0, len( processes ) ) ]
	while np.sum( completed_jobs ) < len( processes ):
		for job_idx in range( 0, len( processes ) ):
			if completed_jobs[ job_idx ] == 0:

				poll_result = processes[ job_idx ].poll()
				if not( poll_result is None ):
					completed_jobs[ job_idx ] = 1

		time.sleep( 1 )


fdtd_hook.save( projects_directory_location + "/optimization.fsp" )

#
# Run the optimization
#
for epoch in range(start_epoch, num_epochs):
	bayer_filter.update_filters(epoch)

	for iteration in range(0, num_iterations_per_epoch):
		print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

		job_names = {}

		fdtd_hook.switchtolayout()
		cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )
		fdtd_hook.select("design_import")
		fdtd_hook.importnk2(np.sqrt(cur_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)


		#
		# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
		#
		Qxx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
		Qxy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
		Qyx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
		Qyy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )

		for xy_idx in range(0, 2):
			disable_all_sources()

			fdtd_hook.select( forward_sources[xy_idx]['name'] )
			fdtd_hook.set( 'enabled', 1 )

			job_name = 'forward_job_' + str( xy_idx ) + '.fsp'
			fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
			job_names[ ( 'forward', xy_idx ) ] = add_job( job_name, jobs_queue )



		for adj_src_idx in range(0, num_adjoint_sources):
			for xy_idx in range(0, 2):
				disable_all_sources()
				fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
				fdtd_hook.set( 'enabled', 1 )

				job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '.fsp'

				fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
				job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] = add_job( job_name, jobs_queue )



		run_jobs( jobs_queue )



####
		for xy_idx in range(0, 2):
			fdtd_hook.load( job_names[ ( 'forward', xy_idx ) ] )

			forward_e_fields[xy_names[xy_idx]] = get_efield(design_efield_monitor['name'] )

			focal_data[xy_names[xy_idx]] = []
			for focal_idx in range( 0, num_focal_spots ):
				focal_monitor_data = get_efield( focal_monitors[ focal_idx ][ 'name' ] )

				if xy_idx == 0:
					Qxx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]
					Qxy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
				else:
					Qyy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
					Qyx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]
####


		
		for focal_idx in range( 0, num_focal_spots ):
			analyzer_vector = jones_sorting_vectors[ focal_idx ]
			orthogonal_vector = jones_orthogonal_vectors[ focal_idx ]

			create_forward_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ focal_idx, : ]
			create_forward_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ focal_idx, : ]

			create_forward_orthogonal_response_x = orthogonal_vector[ 0 ] * Qxx[ focal_idx, : ] + orthogonal_vector[ 1 ] * Qyx[ focal_idx, : ]
			create_forward_orthogonal_response_y = orthogonal_vector[ 0 ] * Qxy[ focal_idx, : ] + orthogonal_vector[ 1 ] * Qyy[ focal_idx, : ]

			parallel_intensity = np.abs( create_forward_parallel_response_x )**2 + np.abs( create_forward_parallel_response_y )**2
			orthogonal_intensity = np.abs( create_forward_orthogonal_response_x )**2 + np.abs( create_forward_orthogonal_response_y )**2
			contrast = ( parallel_intensity - orthogonal_intensity ) / ( parallel_intensity + orthogonal_intensity )

			parallel_intensity_per_focal_spot_evolution[ epoch, iteration, focal_idx, : ] = parallel_intensity / max_intensity_by_wavelength
			orthogonal_intensity_per_focal_spot_evolution[ epoch, iteration, focal_idx, : ] = orthogonal_intensity / max_intensity_by_wavelength
			contrast_per_focal_spot_evolution[ epoch, iteration, focal_idx, : ] = contrast


		current_figure_of_merit, fom_by_focal_spot_by_type_by_wavelength = figure_of_merit( Qxx, Qxy, Qyx, Qyy )

		figure_of_merit_by_focal_spot_by_type_by_wavelength_evolution[ epoch, iteration ] = fom_by_focal_spot_by_type_by_wavelength
		figure_of_merit_evolution[ epoch, iteration ] = current_figure_of_merit

		print( 'The current figure of merit = ' + str( current_figure_of_merit ) )

		np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)
		np.save(projects_directory_location + "/figure_of_merit_by_focal_spot_by_type_by_wavelength.npy", figure_of_merit_by_focal_spot_by_type_by_wavelength_evolution)
		np.save(projects_directory_location + "/contrast_per_focal_spot.npy", contrast_per_focal_spot_evolution)
		np.save(projects_directory_location + "/parallel_intensity.npy", parallel_intensity_per_focal_spot_evolution)
		np.save(projects_directory_location + "/orthogonal_intensity.npy", orthogonal_intensity_per_focal_spot_evolution)

		#
		# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
		# gradients for x- and y-polarized forward sources.
		#
		adjoint_ex_fields = []
		adjoint_ey_fields = []
		for adj_src_idx in range(0, num_adjoint_sources):
			for xy_idx in range(0, 2):
				fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] )

				if xy_idx == 0:
					adjoint_ex_fields.append(
						get_efield( design_efield_monitor['name'] ) )
				else:
					adjoint_ey_fields.append(
						get_efield( design_efield_monitor['name'] ) )

		minimization_gradient = gradient(
			fom_by_focal_spot_by_type_by_wavelength,
			forward_e_fields[ 'x' ], forward_e_fields[ 'y' ],
			adjoint_ex_fields, adjoint_ey_fields,
			Qxx, Qxy, Qyx, Qyy )


		#
		# Step 4: Step the design variable.
		#
		device_gradient = minimization_gradient
		# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
		# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
		# device_gradient = np.swapaxes(device_gradient, 0, 2)

		design_gradient = bayer_filter.backpropagate(device_gradient)

		max_change_design = epoch_start_permittivity_change_max
		min_change_design = epoch_start_permittivity_change_min

		if num_iterations_per_epoch > 1:

			max_change_design = (
				epoch_end_permittivity_change_max +
				(num_iterations_per_epoch - 1 - iteration) * (epoch_range_permittivity_change_max / (num_iterations_per_epoch - 1))
			)

			min_change_design = (
				epoch_end_permittivity_change_min +
				(num_iterations_per_epoch - 1 - iteration) * (epoch_range_permittivity_change_min / (num_iterations_per_epoch - 1))
			)


		cur_design_variable = bayer_filter.get_design_variable()

		step_size = step_size_start

		check_last = False
		last = 0

		while True:
			proposed_design_variable = cur_design_variable - step_size * design_gradient
			proposed_design_variable = np.maximum(
										np.minimum(
											proposed_design_variable,
											1.0),
										0.0)

			difference = np.abs(proposed_design_variable - cur_design_variable)
			max_difference = np.max(difference)

			if (max_difference <= max_change_design) and (max_difference >= min_change_design):
				break
			elif (max_difference <= max_change_design):
				step_size *= 2
				if (last ^ 1) and check_last:
					break
				check_last = True
				last = 1
			else:
				step_size /= 2
				if (last ^ 0) and check_last:
					break
				check_last = True
				last = 0

		step_size_start = step_size

		last_design_variable = cur_design_variable.copy()
		#
		# todo: fix this in other files! the step already does the backpropagation so you shouldn't
		# pass it an already backpropagated gradient!  Sloppy, these files need some TLC and cleanup!
		#
		enforce_binarization = False
		if epoch >= binarization_start_epoch:
			enforce_binarization = True
		device_gradient = np.flip( device_gradient, axis=2 )
		bayer_filter.step(device_gradient, step_size, enforce_binarization, projects_directory_location)
		cur_design_variable = bayer_filter.get_design_variable()

		average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
		max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

		step_size_evolution[epoch][iteration] = step_size
		average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
		max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

		fdtd_hook.switchtolayout()
		fdtd_hook.save()
		shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/optimization_start_epoch_" + str( epoch ) + ".fsp" )
		np.save(projects_directory_location + '/device_gradient.npy', device_gradient)
		np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
		np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
		np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
		np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
		np.save(projects_directory_location + "/cur_design_variable.npy", cur_design_variable)
		np.save(projects_directory_location + "/cur_design_variable_" + str( epoch ) + ".npy", cur_design_variable)

	fdtd_hook.switchtolayout()
	fdtd_hook.save()
	shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/optimization_end_epoch_" + str( epoch ) + ".fsp" )


