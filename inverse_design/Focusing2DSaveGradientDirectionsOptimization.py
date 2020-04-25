import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Focusing2DSaveGradientDirectionsParameters import *
import Focusing2DFilter

import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

import scipy.signal

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
project_subfolder = ""
if len(sys.argv) > 1:
	project_subfolder = "/" + sys.argv[1] + "/"

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/')) + project_subfolder
projects_directory_location = "/central/groups/Faraon_Computing/projects/"

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

shutil.copy2(python_src_directory + "/Focusing2DSaveGradientDirectionsParameters.py", projects_directory_location + "/Focusing2DSaveGradientDirectionsParameters.py")

def get_non_struct_data(monitor_name, monitor_field):
	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	data_transfer_filename = projects_directory_location + "/data_transfer_" + monitor_name + "_" + monitor_field

	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + lumerical_data_name + ");"

	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)

	# start_time = time.time()

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[lumerical_data_name])

	# end_time = time.time()

	# print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

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

	# start_time = time.time()

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[extracted_data_name])

	# end_time = time.time()

	# print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	data = get_monitor_data(monitor_name, monitor_field)
	return (data['real'] + np.complex(0, 1) * data['imag'])

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
design_efield_monitor['use wavelength spacing'] = 1
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
	focal_monitor['use wavelength spacing'] = 1
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
	transmission_monitor['use wavelength spacing'] = 1
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
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

def reinterpolate( input_array, output_shape ):
    input_shape = input_array.shape

    assert len( input_shape ) == len( output_shape ), "Reinterpolate: expected the input and output to have same number of dimensions"

    output_array = input_array.copy()

    for axis_idx in range( 0, len( input_shape ) ):
        output_array = scipy.signal.resample( output_array, output_shape[ axis_idx ], axis=axis_idx )

    return output_array    

def import_device_permittivity( permittivity ):
	permittivity_shape = permittivity.shape
	x_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, permittivity_shape[ 0 ] )
	y_range = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, permittivity_shape[ 1 ] )
	z_range = 1e-6 * np.linspace( -0.51, 0.51, 2 )

	fdtd_hook.switchtolayout()
	fdtd_hook.select( 'device_import' )
	fdtd_hook.importnk2( np.sqrt( permittivity ), x_range, y_range, z_range )

for feature_size_optimization_idx in range( 0, num_feature_size_optimizations ):
	min_feature_size_um = feature_size_meshes_um[ feature_size_optimization_idx ]
	device_size_voxels_lateral = int( device_size_lateral_um / min_feature_size_um )
	device_size_voxels_vertical = int( device_size_vertical_um / min_feature_size_um )

	device_permittivity = ( device_mid_permittivity ) * np.ones( ( device_size_voxels_lateral, device_size_voxels_vertical ) )

	bayer_filter = Focusing2DFilter.Focusing2DFilter( [ device_size_voxels_lateral, device_size_voxels_vertical ], device_permittivity_bounds, 0.5 )

	binarization_evolution = np.zeros( ( num_epochs, num_iterations_per_epoch ) )
	figure_of_merit_evolution = np.zeros( ( num_epochs, num_iterations_per_epoch ) )

	reversed_field_shape = [ device_size_voxels_vertical, device_size_voxels_lateral ]
	reversed_field_shape_with_pol = [ num_polarizations, 1, designable_device_voxels_vertical, device_voxels_lateral ]
	xy_polarized_gradients_by_pol = np.zeros( reversed_field_shape_with_pol, dtype=np.complex )
	xy_polarized_gradients = np.zeros( reversed_field_shape, dtype=np.complex )
	figure_of_merit_by_pol = np.zeros( num_polarizations )

	net_gradient_by_epoch_by_iteration = np.zeros(
		( num_epochs, num_iterations_per_epoch, device_voxels_lateral, device_size_voxels_vertical ) )
	gradient_by_epoch_by_iteration_by_pol_by_wavelength = np.zeros(
		( num_epochs, num_iterations_per_epoch, num_polarizations, num_design_frequency_points, device_voxels_lateral, device_size_voxels_vertical ) )
	fom_by_epoch_by_iteration_by_pol_by_wavelength = np.zeros(
		( num_epochs, num_iterations_per_epoch, num_polarizations, num_design_frequency_points ) )

	for epoch in range( 0, num_epochs ):

		bayer_filter.update_filters( epoch )

		log_file = open(projects_directory_location + "/log.txt", 'a')
		log_file.write(
			"Working on optimization " + str( feature_size_optimization_idx ) + " and epoch " + str( epoch ) +
			" out of ( " + str( num_feature_size_optimizations - 1 ) + ", " + str( num_epochs - 1 ) + " )" )
		log_file.close()

		for iteration in range( 0, num_iterations_per_epoch ):

			device_density = bayer_filter.get_design_variable()
			device_permittivity = bayer_filter.get_permittivity()
			import_device_permittivity( device_permittivity )

			binarization = np.mean( np.abs( device_density - 0.5 ) ) / 0.5
			binarization_evolution[ epoch, iteration ] = binarization


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

						fom_by_epoch_by_iteration_by_pol_by_wavelength[ epoch, iteration, pol_idx, wl_idx + spectral_indices[ 0 ] ] = figure_of_merit_total[ wl_idx + spectral_indices[ 0 ] ]

				fom_weighting = ( 2. / len( figure_of_merit_total ) ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				fom_weighting = np.maximum( fom_weighting, 0 )
				fom_weighting /= np.sum( fom_weighting )

				figure_of_merit = np.mean( figure_of_merit_total )

				figure_of_merit_by_pol[ pol_idx ] = figure_of_merit

				figure_of_merit_evolution[epoch, iteration] += ( 1. / num_polarizations ) * figure_of_merit

				#
				# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
				# gradients for x- and y-polarized forward sources.
				#
				polarized_gradient = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)

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
							# Since we are in 2D, we will just take the 0th point in the z-direction
							pull_adjoint_efields = np.squeeze( adjoint_e_fields[ :, spectral_indices[ 0 ] + spectral_idx, 0, :, : ] )
							pull_forward_efields = np.squeeze( forward_e_fields[ :, spectral_indices[ 0 ] + spectral_idx, 0, :, : ] )

							reinterpolate_adjoint_efields = reinterpolate( pull_adjoint_efields, reversed_field_shape )
							reinterpolate_forward_efields = reinterpolate( pull_forward_efields, reversed_field_shape )

							gradient_by_epoch_by_iteration_by_pol_by_wavelength[
								epoch, iteration, pol_idx, spectral_indices[ 0 ] + spectral_idx ] = 2 * np.real(
									np.sum(
										conjugate_weighting_wavelength[ current_coord, spectral_indices[ 0 ] + spectral_idx ] *
										reinterpolate_adjoint_efields *
										reinterpolate_forward_efields,
									axis=0 )
								)

							polarized_gradient += np.sum(
								( conjugate_weighting_wavelength[current_coord, spectral_indices[0] + spectral_idx] * fom_weighting[spectral_indices[0] + spectral_idx] ) *
								reinterpolate_adjoint_efields *
								reinterpolate_forward_efields,
								axis=0 )
					
				xy_polarized_gradients_by_pol[ pol_idx ] = polarized_gradient
			
			weight_grad_by_pol = ( 2. / num_polarizations ) - figure_of_merit_by_pol**2 / np.sum( figure_of_merit_by_pol**2 )
			weight_grad_by_pol = np.maximum( weight_grad_by_pol, 0 )
			weight_grad_by_pol /= np.sum( weight_grad_by_pol )

			for pol_idx in range( 0, num_polarizations ):
				xy_polarized_gradients += weight_grad_by_pol[ pol_idx ] * xy_polarized_gradients_by_pol[ pol_idx ]

			#
			# Step 4: Step the design variable.
			#
			device_gradient_real = 2 * np.real( xy_polarized_gradients )

			net_gradient_by_epoch_by_iteration[ epoch, iteration ] = device_gradient_real
			# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
			# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
			device_gradient_real = np.swapaxes( device_gradient_real, 0, 1 )
			design_gradient = bayer_filter.backpropagate( device_gradient_real )

			step_size = 0.01 / np.max( np.abs( design_gradient ) )
			step_size_evolution[ epoch ][ iteration ] = step_size

		# Save once per epoch in case we want to check on it!
		np.save( projects_directory_location + "/net_gradient_direction_" + str( feature_size_optimization_idx ) + ".npy", net_gradient_by_epoch_by_iteration )
		np.save( projects_directory_location + "/gradient_directions_" + str( feature_size_optimization_idx ) + ".npy", gradient_by_epoch_by_iteration_by_pol_by_wavelength )
		np.save( projects_directory_location + "/fom_all_objectives_" + str( feature_size_optimization_idx ) + ".npy", fom_by_epoch_by_iteration_by_pol_by_wavelength )
		np.save( projects_directory_location + "/figure_of_merit_" + str( feature_size_optimization_idx ) + ".npy", figure_of_merit_evolution )
		np.save( projects_directory_location + "/binarization_" + str( feature_size_optimization_idx ) + ".npy", binarization_evolution )

