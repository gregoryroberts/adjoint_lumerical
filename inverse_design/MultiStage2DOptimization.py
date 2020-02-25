import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from MultiStage2DParameters import *

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )


def get_non_struct_data(monitor_name, monitor_field):
	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	data_transfer_filename = projects_directory_location + "/data_transfer_" + monitor_name + "_" + monitor_field

	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + lumerical_data_name + ");"

	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[lumerical_data_name])

	return monitor_data['real']

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

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[extracted_data_name])

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	data = get_monitor_data(monitor_name, monitor_field)
	return (data['real'] + np.complex(0, 1) * data['imag'])

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/MultiStage2DParameters.py", projects_directory_location + "/ArchiveMultiStage2DParameters.py")



#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['dimension'] = '2D'
fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['y max'] = fdtd_region_maximum_vertical_um * 1e-6
fdtd['y min'] = fdtd_region_minimum_vertical_um * 1e-6
fdtd['mesh type'] = 'uniform'
fdtd['define x mesh by'] = 'number of mesh cells'
fdtd['define y mesh by'] = 'number of mesh cells'
fdtd['mesh cells x'] = fdtd_region_minimum_lateral_voxels
fdtd['mesh cells y'] = fdtd_region_minimum_vertical_voxels
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
source_polarization_angles = [ 90, 0 ]
affected_coords_by_polarization = [ [ 2 ], [ 0, 1 ] ]

for pol_idx in range( 0, num_polarizations ):
	forward_src = fdtd_hook.addtfsf()
	forward_src['name'] = 'forward_src_' + str( pol_idx )
	forward_src['polarization angle'] = source_polarization_angles[ pol_idx ]
	forward_src['direction'] = 'Backward'
	forward_src['x span'] = lateral_aperture_um * 1e-6
	forward_src['y max'] = src_maximum_vertical_um * 1e-6
	forward_src['y min'] = src_minimum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_max_um * 1e-6

	forward_sources.append( forward_src )


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	fdtd_hook.switchtolayout()

	for pol_idx in range( 0, num_polarizations ):
		forward_sources[ pol_idx ].enabled = 0

	for coord_idx in range( 0, 3 ):
		for adj_src_idx in range(0, num_adjoint_sources):
			(adjoint_sources[ coord_idx ][ adj_src_idx ]).enabled = 0


#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = [ [] for i in range( 0, 3 ) ]
coord_to_phi = [ 0, 90, 0 ]
coord_to_theta = [ 90, 90, 0 ]

for coord_idx in range( 0, 3 ):
	for adj_src_idx in range(0, num_adjoint_sources):
		adj_src = fdtd_hook.adddipole()
		adj_src['name'] = 'adj_src_' + str(adj_src_idx) + "_" + str( coord_idx )
		adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
		adj_src['y'] = adjoint_vertical_um * 1e-6
		adj_src['theta'] = coord_to_theta[ coord_idx ]
		adj_src['phi'] = coord_to_phi[ coord_idx ]
		adj_src['wavelength start'] = lambda_min_um * 1e-6
		adj_src['wavelength stop'] = lambda_max_um * 1e-6

		adjoint_sources[ coord_idx ].append( adj_src )


#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
design_efield_monitor['y min'] = designable_device_vertical_minimum_um * 1e-6
design_efield_monitor['y max'] = designable_device_vertical_maximum_um * 1e-6
design_efield_monitor['override global monitor settings'] = 1
design_efield_monitor['use linear wavelength spacing'] = 1
design_efield_monitor['use source limits'] = 0
design_efield_monitor['minimum wavelength'] = lambda_min_um * 1e-6
design_efield_monitor['maximum wavelength'] = lambda_max_um * 1e-6
design_efield_monitor['frequency points'] = num_design_frequency_points
design_efield_monitor['output Hx'] = 0
design_efield_monitor['output Hy'] = 0
design_efield_monitor['output Hz'] = 0

design_index_monitor = fdtd_hook.addindex()
design_index_monitor['name'] = 'design_index_monitor'
design_index_monitor['x span'] = device_size_lateral_um * 1e-6
design_index_monitor['y min'] = designable_device_vertical_minimum_um * 1e-6
design_index_monitor['y max'] = designable_device_vertical_maximum_um * 1e-6


#
# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
# gradient.
#
focal_monitors = []

for adj_src in range(0, num_adjoint_sources):
	focal_monitor = fdtd_hook.addpower()
	focal_monitor['name'] = 'focal_monitor' + str(adj_src)
	focal_monitor['monitor type'] = 'point'
	focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	focal_monitor['y'] = adjoint_vertical_um * 1e-6
	focal_monitor['override global monitor settings'] = 1
	focal_monitor['use linear wavelength spacing'] = 1
	focal_monitor['use source limits'] = 0
	focal_monitor['minimum wavelength'] = lambda_min_um * 1e-6
	focal_monitor['maximum wavelength'] = lambda_max_um * 1e-6
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)


for adj_src in range(0, num_adjoint_sources):
	transmission_monitor = fdtd_hook.addpower()
	transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
	transmission_monitor['monitor type'] = 'Linear X'
	transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_monitor['x span'] = ( 1.0 / num_focal_spots ) * device_size_lateral_um * 1e-6
	transmission_monitor['y'] = adjoint_vertical_um * 1e-6
	transmission_monitor['override global monitor settings'] = 1
	transmission_monitor['use linear wavelength spacing'] = 1
	transmission_monitor['use source limits'] = 1
	transmission_monitor['frequency points'] = num_eval_frequency_points

#
# Add device region and create device permittivity
#

device_import = fdtd_hook.addimport()
device_import['name'] = 'device_import'
device_import['x span'] = device_size_lateral_um * 1e-6
device_import['y min'] = designable_device_vertical_minimum_um * 1e-6
device_import['y max'] = designable_device_vertical_maximum_um * 1e-6
device_import['z min'] = -0.51 * 1e-6
device_import['z max'] = 0.51 * 1e-6

#
# Add blocks of dielectric on the side of the designable region because we will be leaving those as blank, unpatterned dielectric material
# Would it be better to have this be a part of the bayer filter material inmport and jus tnot modify it (i.e. - mask out any changes to it).  I'm
# thinking of how the mesh behaves here between these interacting objects.  For now, we will add the blocks around the side because this will make
# it simpler at first and then we can move towards making a more sophisticated device class or subclass.
# Further, in reality, this may be a stack of material in general.  However, it will be mostly the low-K dielctric background material so we will assume
# this is not a stratified stack and is actaully just a single piece of low index material background.  In fact, actually, we may not have this at all.
# These types of more exact parameters are somehwat undetermined at this point.
#
extra_lateral_space_per_side_um = 0.5 * ( fdtd_region_size_lateral_um - device_size_lateral_um )
extra_lateral_space_offset_um = 0.5 * ( device_size_lateral_um + extra_lateral_space_per_side_um )

def side_to_string( side_number ):
	side_integer = int( side_number )
	if side_integer < 0:
		return ( "n" + str( np.abs( side_integer ) ) )
	else:
		return str( side_integer )

device_background_side_x = [ -1, 1 ]#, 0, 0 ]
# device_background_side_y = [ 0, 0, -1, 1 ]

for device_background_side_idx in range( 0, 2 ):
	side_x = device_background_side_x[ device_background_side_idx ]

	side_block = fdtd_hook.addrect()

	side_block['name'] = 'device_background_' + side_to_string( side_x )
	side_block['y min'] = designable_device_vertical_minimum_um * 1e-6
	side_block['y max'] = designable_device_vertical_maximum_um * 1e-6
	side_block['x'] = side_x * extra_lateral_space_offset_um * 1e-6
	side_block['x span'] = (
		np.abs( side_x ) * extra_lateral_space_per_side_um +
		( 1 - np.abs( side_x ) ) * fdtd_region_size_lateral_um ) * 1e-6
	side_block['z span'] = 1.02 * 1e-6

	side_block['index'] = device_background_index


gaussian_normalization = np.zeros( num_points_per_band )
middle_point = num_points_per_band / 2.
spacing = 1. / ( num_points_per_band - 1 )
half_bandwidth = 0.4 * num_points_per_band

for wl_idx in range( 0, num_points_per_band ):
	gaussian_normalization[ wl_idx ] =  ( 1. / half_bandwidth ) * np.sqrt( 1 / ( 2 * np.pi ) ) * np.exp( -0.5 * ( wl_idx - middle_point )**2 / ( half_bandwidth**2 ) )
	
gaussian_normalization /= np.sum( gaussian_normalization )
gaussian_normalization_all = np.array( [ gaussian_normalization for i in range( 0, num_bands ) ] ).flatten()


reversed_field_shape = [1, designable_device_voxels_vertical, device_voxels_lateral]
reversed_field_shape_with_pol = [num_polarizations, 1, designable_device_voxels_vertical, device_voxels_lateral]


start_epoch = init_optimization_epoch

for optimization_state_idx in range( init_optimization_state, num_optimization_states ):
	my_optimization_state = optimization_stages[ optimization_state_idx ]

	if start_epoch > 0:
		#
		# Then, we will load our current optimization state with this epoch
		#
		my_optimization_state.load( projects_directory_location, start_epoch - 1 )
	else:
		#
		# We need to transfer the optimization state from the previous optimization
		# if there is one or else we need to start from scratch
		#
		if optimization_state_idx > 0:
			previous_optimization_state = optimization_stages[ optimization_state_idx - 1 ]
			previous_optimization_state.load( projects_directory_location, previous_optimization_state.num_epochs - 1 )

			optimization_conversion_functions[ optimization_state_idx ]( projects_directory_location )


	num_epochs = my_optimization_state.num_epochs
	num_iterations_per_epoch = my_optimization_state.num_iterations

	get_index = my_optimization_state.assemble_index()
	device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, get_index.shape[ 0 ] )
	device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, get_index.shape[ 1 ] )
	device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )

	for epoch in range( start_epoch, num_epochs ):

		my_optimization_state.update_epoch( epoch )

		for iteration in range( 0, num_iterations_per_epoch ):
			print("Working on optimization state " + str( optimization_state_idx ) + " and epoch " + str(epoch) + " and iteration " + str(iteration))

			fdtd_hook.switchtolayout()
			get_index = my_optimization_state.assemble_index()
			inflate_index = np.zeros( ( get_index.shape[ 0 ], get_index.shape[ 1 ], 2 ), dtype=np.complex )
			inflate_index[ :, :, 0 ] = get_index
			inflate_index[ :, :, 1 ] = get_index

			fdtd_hook.select( device_import[ 'name' ] )
			fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

			xy_polarized_gradients_by_pol = np.zeros(reversed_field_shape_with_pol, dtype=np.complex)
			xy_polarized_gradients = np.zeros(reversed_field_shape, dtype=np.complex)
			xy_polarized_gradients_by_pol_lsf = np.zeros(reversed_field_shape_with_pol, dtype=np.complex)
			xy_polarized_gradients_lsf = np.zeros(reversed_field_shape, dtype=np.complex)
			figure_of_merit_by_pol = np.zeros( num_polarizations )

			figure_of_merit = 0
			for pol_idx in range( 0, num_polarizations ):

				affected_coords = affected_coords_by_polarization[ pol_idx ]

				#
				# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
				#
				disable_all_sources()
				forward_sources[ pol_idx ].enabled = 1
				fdtd_hook.run()

				forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

				focal_data = []
				for adj_src_idx in range(0, num_adjoint_sources):
					focal_data.append(
						get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )

				#
				# Step 2: Compute the figure of merit
				#
				normalized_intensity_focal_point_wavelength = np.zeros( ( num_focal_spots, num_design_frequency_points ) )
				conjugate_weighting_focal_point_wavelength = np.zeros( ( 3, num_focal_spots, num_design_frequency_points ), dtype=np.complex )

				figure_of_merit_total = np.zeros( num_design_frequency_points )
				conjugate_weighting_wavelength = np.zeros( ( 3, num_design_frequency_points ), dtype=np.complex )

				for focal_idx in range(0, num_focal_spots):
					spectral_indices = spectral_focal_plane_map[ focal_idx ]

					for wl_idx in range( 0, num_points_per_band ):

						for coord_idx in range( 0, len( affected_coords ) ):
							current_coord = affected_coords[ coord_idx ]

							figure_of_merit_total[ wl_idx + spectral_indices[ 0 ] ] += (
								np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx + spectral_indices[ 0 ], 0, 0, 0 ])**2 )
							) / max_intensity_by_wavelength[ wl_idx + spectral_indices[ 0 ] ]

							conjugate_weighting_wavelength[ current_coord, wl_idx + spectral_indices[ 0 ] ] = np.conj(
								focal_data[ focal_idx ][ current_coord, wl_idx + spectral_indices[ 0 ], 0, 0, 0 ] / max_intensity_by_wavelength[ wl_idx + spectral_indices[ 0 ] ] )
			
				# todo: make sure this figure of merit weighting makes sense the way it is done across wavelengths and focal points
				fom_weighting = ( 2. / len( figure_of_merit_total ) ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				fom_weighting = np.maximum( fom_weighting, 0 )
				fom_weighting /= np.sum( fom_weighting )

				figure_of_merit_by_pol[ pol_idx ] = np.sum( gaussian_normalization_all * figure_of_merit_total )
				figure_of_merit += ( 1. / num_polarizations ) * figure_of_merit_by_pol[ pol_idx ]

				#
				# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
				# gradients for x- and y-polarized forward sources.
				#
				polarized_gradient = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)
				polarized_gradient_lsf = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)

				current_index = np.real( get_non_struct_data( design_index_monitor[ 'name' ], 'index_x' ) )
				current_permittivity = np.sqrt( np.squeeze( current_index ) )

				for coord_idx in range( 0, len( affected_coords ) ):
					current_coord = affected_coords[ coord_idx ]
					for adj_src_idx in range( 0, num_adjoint_sources ):
						disable_all_sources()
						(adjoint_sources[current_coord][adj_src_idx]).enabled = 1
						fdtd_hook.run()

						adjoint_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

						spectral_indices = spectral_focal_plane_map[ adj_src_idx ]

						for spectral_idx in range(0, num_points_per_band):
							for polarization_idx in range( 0, 3 ):
								# x-coordinate is the perpendicular coorinate to the level set boundary that can be actually changed
								if polarization_idx == 0:
									polarized_gradient_lsf += ( ( ( 1. / min_real_permittivity ) - ( 1. / max_real_permittivity ) ) *
										gaussian_normalization_all[ spectral_idx % num_points_per_band ] * ( conjugate_weighting_wavelength[current_coord, spectral_indices[0] + spectral_idx] * fom_weighting[spectral_indices[0] + spectral_idx] ) *
										current_permittivity * adjoint_e_fields[polarization_idx, spectral_indices[0] + spectral_idx, :, :, :] *
										current_permittivity * forward_e_fields[polarization_idx, spectral_indices[0] + spectral_idx, :, :, :]
									)
								else:
									polarized_gradient_lsf += ( ( max_real_permittivity - min_real_permittivity ) *
										gaussian_normalization_all[ spectral_idx % num_points_per_band ] * (conjugate_weighting_wavelength[current_coord, spectral_indices[0] + spectral_idx] * fom_weighting[spectral_indices[0] + spectral_idx]) *
										current_permittivity * adjoint_e_fields[polarization_idx, spectral_indices[0] + spectral_idx, :, :, :] *
										current_permittivity * forward_e_fields[polarization_idx, spectral_indices[0] + spectral_idx, :, :, :]
									)
						for spectral_idx in range(0, num_points_per_band):
							polarized_gradient += np.sum(
								gaussian_normalization_all[ spectral_idx % num_points_per_band ] * (conjugate_weighting_wavelength[current_coord, spectral_indices[0] + spectral_idx] * fom_weighting[spectral_indices[0] + spectral_idx]) *
								adjoint_e_fields[:, spectral_indices[0] + spectral_idx, :, :, :] *
								forward_e_fields[:, spectral_indices[0] + spectral_idx, :, :, :],
								axis=0)


				xy_polarized_gradients_by_pol[ pol_idx ] = polarized_gradient
				xy_polarized_gradients_by_pol_lsf[ pol_idx ] = polarized_gradient

			my_optimization_state.submit_figure_of_merit( figure_of_merit, iteration, epoch )

			weight_grad_by_pol = ( 2. / num_polarizations ) - figure_of_merit_by_pol**2 / np.sum( figure_of_merit_by_pol**2 )
			weight_grad_by_pol = np.maximum( weight_grad_by_pol, 0 )
			weight_grad_by_pol /= np.sum( weight_grad_by_pol )

			print(figure_of_merit_by_pol)
			print(weight_grad_by_pol)
			print()


			for pol_idx in range( 0, num_polarizations ):
				xy_polarized_gradients += weight_grad_by_pol[ pol_idx ] * xy_polarized_gradients_by_pol[ pol_idx ]
				xy_polarized_gradients_lsf += weight_grad_by_pol[ pol_idx ] * xy_polarized_gradients_by_pol_lsf[ pol_idx ]

			#
			# Step 4: Step the design variable.
			#
			device_gradient_real = 2 * np.real( xy_polarized_gradients )
			device_gradient_imag = 2 * np.imag( xy_polarized_gradients )
			# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
			# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
			device_gradient_real = np.swapaxes(device_gradient_real, 0, 2)
			device_gradient_imag = np.swapaxes(device_gradient_imag, 0, 2)

			device_gradient_real_lsf = 2 * np.real( xy_polarized_gradients_lsf )
			device_gradient_imag_lsf = 2 * np.imag( xy_polarized_gradients_lsf )
			# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
			# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
			device_gradient_real_lsf = np.swapaxes(device_gradient_real_lsf, 0, 2)
			device_gradient_imag_lsf = np.swapaxes(device_gradient_imag_lsf, 0, 2)

			my_optimization_state.update( -device_gradient_real, -device_gradient_imag, -device_gradient_real_lsf, -device_gradient_imag_lsf, epoch, iteration )


		#
		# At the end of every epoch, we need to save everything we have
		#
		my_optimization_state.save( projects_directory_location, epoch )

		shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/" + my_optimization_state.filename_prefix + "optimization_" + str( epoch ) + ".fsp" )

	#
	# We must start from the 0th epoch on every stage past the initial stage
	#
	start_epoch = 0
