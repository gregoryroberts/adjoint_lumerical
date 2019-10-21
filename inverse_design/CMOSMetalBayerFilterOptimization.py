import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSMetalBayerFilterParameters import *
import CMOSMetalBayerFilter

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

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/CMOSMetalBayerFilterParameters.py", projects_directory_location + "/ArchiveCMOSMetalBayerFilter.py")

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
design_efield_monitor['z min'] = designable_device_vertical_minimum_um * 1e-6
design_efield_monitor['z max'] = designable_device_vertical_maximum_um * 1e-6
design_efield_monitor['override global monitor settings'] = 1
design_efield_monitor['use linear wavelength spacing'] = 1
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
	focal_monitor['use linear wavelength spacing'] = 1
	focal_monitor['use source limits'] = 1
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)


#
# Build the dielectric stack on top of the metal
#
num_dielectric_layers = len( top_dielectric_layer_thickness_um )
for dielectric_layer_idx in range( 0, num_dielectric_layers ):
	dielectric_layer = fdtd_hook.addrect()
	dielectric_layer['name'] = 'dielectric_layer_' + str( dielectric_layer_idx )
	dielectric_layer['x span'] = fdtd_region_size_lateral_um * 1e-6
	dielectric_layer['y span'] = fdtd_region_size_lateral_um * 1e-6

	dielectric_layer_z_min_um = dielectric_stack_start_um + np.sum( top_dielectric_layer_thickness_um[ 0 : dielectric_layer_idx ] )
	dielectric_layer_z_max_um = dielectric_layer_z_min_um + top_dielectric_layer_thickness_um[ dielectric_layer_idx ]

	dielectric_layer['z min'] = dielectric_layer_z_min_um * 1e-6
	dielectric_layer['z max'] = dielectric_layer_z_max_um * 1e-6

	dielectric_layer['index'] = top_dielectric_layer_refractice_index[ dielectric_layer_idx ]


#
# Make bottom layer reflective for a reflective device.  We will do this with an nk2 material that we import the maximum
# real and imaginary permittivity parts in for that we are using for the design.  Thus, it will reflect and account for
# metallic loss
#
metal_reflector_import = fdtd_hook.addimport()
metal_reflector_import['name'] = 'bottom_metal_reflector'
metal_reflector_import['x span'] = fdtd_region_size_lateral_um * 1e-6
metal_reflector_import['y span'] = fdtd_region_size_lateral_um * 1e-6
metal_reflector_import['z min'] = bottom_metal_reflector_start_um * 1e-6
metal_reflector_import['z max'] = bottom_metal_reflector_end_um * 1e-6

metal_reflector_permittivity = (
		( max_real_permittivity + 1j * max_imag_permittivity ) *
		np.ones( ( fdtd_region_size_lateral_voxels, fdtd_region_size_lateral_voxels, bottom_metal_reflector_size_vertical_voxels ) )
	)
metal_reflector_index = permittivity_to_index( metal_reflector_permittivity )

metal_reflector_region_x = 1e-6 * np.linspace(-0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, fdtd_region_size_lateral_voxels)
metal_reflector_region_y = 1e-6 * np.linspace(-0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, fdtd_region_size_lateral_voxels)
metal_reflector_region_z = 1e-6 * np.linspace(bottom_metal_reflector_start_um, bottom_metal_reflector_end_um, bottom_metal_reflector_size_vertical_voxels)

fdtd_hook.select('bottom_metal_reflector')
fdtd_hook.importnk2(metal_reflector_index, metal_reflector_region_x, metal_reflector_region_y, metal_reflector_region_z)


#
# Add device region and create device permittivity
#
design_import = fdtd_hook.addimport()
design_import['name'] = 'design_import'
design_import['x span'] = device_size_lateral_um * 1e-6
design_import['y span'] = device_size_lateral_um * 1e-6
design_import['z min'] = designable_device_vertical_minimum_um * 1e-6
design_import['z max'] = designable_device_vertical_maximum_um * 1e-6

min_device_permittivity = min_real_permittivity + 1j * min_imag_permittivity
max_device_permittivity = max_real_permittivity + 1j * max_imag_permittivity

bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, designable_device_voxels_vertical])
bayer_filter = CMOSMetalBayerFilter.CMOSMetalBayerFilter(
	bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_permittivity_0_1_scale, num_vertical_layers)

bayer_filter.set_design_variable( np.load( projects_directory_location + "/cur_design_variable.npy" ) )

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, designable_device_voxels_vertical)


#
# Add blocks of dielectric on the side of the designable region because we will be leaving those as blank, unpatterned dielectric material
# Would it be better to have this be a part of the bayer filter material inmport and jus tnot modify it (i.e. - mask out any changes to it).  I'm
# thinking of how the mesh behaves here between these interacting objects.  For now, we will add the blocks around the side because this will make
# it simpler at first and then we can move towards making a more sophisticated device class or subclass.
# Further, in reality, this may be a stack of material in general.  However, it will be mostly the low-K dielctric background material so we will assume
# this is not a stratified stack and is actaully just a single piece of low index material background.
#
extra_lateral_space_per_side_um = 0.5 * ( fdtd_region_size_lateral_um - device_size_lateral_um )
extra_lateral_space_offset_um = 0.5 * ( device_size_lateral_um + extra_lateral_space_per_side_um )

def side_to_string( side_number ):
	side_integer = int( side_number )
	if side_integer < 0:
		return ( "n" + str( np.abs( side_integer ) ) )
	else:
		return str( side_integer )

device_background_side_x = [ -1, 1, 0, 0 ]
device_background_side_y = [ 0, 0, -1, 1 ]

for device_background_side_idx in range( 0, 4 ):
	side_x = device_background_side_x[ device_background_side_idx ]
	side_y = device_background_side_y[ device_background_side_idx ]

	side_block = fdtd_hook.addrect()

	side_block['name'] = 'device_background_' + side_to_string( side_x ) + "_" + side_to_string( side_y )
	side_block['z min'] = designable_device_vertical_minimum_um * 1e-6
	side_block['z max'] = designable_device_vertical_maximum_um * 1e-6
	side_block['x'] = side_x * extra_lateral_space_offset_um * 1e-6
	side_block['x span'] = (
		np.abs( side_x ) * extra_lateral_space_per_side_um +
		( 1 - np.abs( side_x ) ) * fdtd_region_size_lateral_um ) * 1e-6
	side_block['y'] = side_y * extra_lateral_space_offset_um * 1e-6
	side_block['y span'] = (
		np.abs( side_y ) * extra_lateral_space_per_side_um +
		( 1 - np.abs( side_y ) ) * fdtd_region_size_lateral_um ) * 1e-6

	side_block['index'] = device_background_index


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

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

# todo(groberts): should we go fixed step size?  The current method is somewhat adaptive.
step_size_start = fixed_step_size # 10 * 0.001

#
# Run the optimization
#
for epoch in range(start_epoch, num_epochs):
	bayer_filter.update_filters(epoch)

	for iteration in range(0, num_iterations_per_epoch):
		print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

		fdtd_hook.switchtolayout()
		cur_permittivity = bayer_filter.get_permittivity()
		cur_index = permittivity_to_index( cur_permittivity )
		fdtd_hook.select("design_import")
		fdtd_hook.importnk2(cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)


		#
		# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
		#
		for xy_idx in range(0, 2):
			disable_all_sources()
			(forward_sources[xy_idx]).enabled = 1
			fdtd_hook.run()

			forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')

			focal_data[xy_names[xy_idx]] = []
			for adj_src_idx in range(0, num_adjoint_sources):
				focal_data[xy_names[xy_idx]].append(get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E'))


		#
		# Step 2: Compute the figure of merit
		#
		figure_of_merit_per_focal_spot = []
		for focal_idx in range(0, num_focal_spots):
			compute_fom = 0

			polarizations = polarizations_focal_plane_map[focal_idx]

			for polarization_idx in range(0, len(polarizations)):
				get_focal_data = focal_data[polarizations[polarization_idx]]

				max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[focal_idx][0] : spectral_focal_plane_map[focal_idx][1] : 1]
				total_weighting = max_intensity_weighting * weight_focal_plane_map[focal_idx]

				for spectral_idx in range(0, total_weighting.shape[0]):
					compute_fom += np.sum(
						(
							np.abs(get_focal_data[focal_idx][:, spectral_focal_plane_map[focal_idx][0] + spectral_idx, 0, 0, 0])**2 /
							total_weighting[spectral_idx]
						)
					)

			figure_of_merit_per_focal_spot.append(compute_fom)

		figure_of_merit_per_focal_spot = np.array(figure_of_merit_per_focal_spot)

		performance_weighting = (2. / num_focal_spots) - figure_of_merit_per_focal_spot**2 / np.sum(figure_of_merit_per_focal_spot**2)
		performance_weighting -= np.min(performance_weighting)
		performance_weighting /= np.sum(performance_weighting)

		figure_of_merit = np.sum(figure_of_merit_per_focal_spot)
		figure_of_merit_evolution[epoch, iteration] = figure_of_merit

		np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)

		#
		# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
		# gradients for x- and y-polarized forward sources.
		#
		cur_permittivity_shape = cur_permittivity.shape
		reversed_field_shape = [cur_permittivity_shape[2], cur_permittivity_shape[1], cur_permittivity_shape[0]]
		xy_polarized_gradients = [ np.zeros(reversed_field_shape, dtype=np.complex), np.zeros(reversed_field_shape, dtype=np.complex) ]

		for adj_src_idx in range(0, num_adjoint_sources):
			polarizations = polarizations_focal_plane_map[adj_src_idx]
			spectral_indices = spectral_focal_plane_map[adj_src_idx]

			gradient_performance_weight = performance_weighting[adj_src_idx]

			adjoint_e_fields = []
			for xy_idx in range(0, 2):
				disable_all_sources()
				(adjoint_sources[adj_src_idx][xy_idx]).enabled = 1
				fdtd_hook.run()

				adjoint_e_fields.append(
					get_complex_monitor_data(design_efield_monitor['name'], 'E'))

			for pol_idx in range(0, len(polarizations)):
				pol_name = polarizations[pol_idx]
				get_focal_data = focal_data[pol_name]
				pol_name_to_idx = polarization_name_to_idx[pol_name]

				for xy_idx in range(0, 2):
					source_weight = np.conj(
						get_focal_data[adj_src_idx][xy_idx, spectral_indices[0] : spectral_indices[1] : 1, 0, 0, 0])

					max_intensity_weighting = max_intensity_by_wavelength[spectral_indices[0] : spectral_indices[1] : 1]
					total_weighting = max_intensity_weighting * weight_focal_plane_map[focal_idx]

					for spectral_idx in range(0, source_weight.shape[0]):
						xy_polarized_gradients[pol_name_to_idx] += np.sum(
							(source_weight[spectral_idx] * gradient_performance_weight / total_weighting[spectral_idx]) *
							adjoint_e_fields[xy_idx][:, spectral_indices[0] + spectral_idx, :, :, :] *
							forward_e_fields[pol_name][:, spectral_indices[0] + spectral_idx, :, :, :],
							axis=0)

		#
		# Step 4: Step the design variable.
		#
		device_gradient_real = 2 * np.real( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
		device_gradient_imag = 2 * np.imag( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
		# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
		# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
		device_gradient_real = np.swapaxes(device_gradient_real, 0, 2)
		device_gradient_imag = np.swapaxes(device_gradient_imag, 0, 2)

		design_gradient = bayer_filter.backpropagate(device_gradient_real, device_gradient_imag)

		# max_change_design = epoch_start_permittivity_change_max
		# min_change_design = epoch_start_permittivity_change_min

		# if num_iterations_per_epoch > 1:

		# 	max_change_design = (
		# 		epoch_end_permittivity_change_max +
		# 		(num_iterations_per_epoch - 1 - iteration) * (epoch_range_permittivity_change_max / (num_iterations_per_epoch - 1))
		# 	)

		# 	min_change_design = (
		# 		epoch_end_permittivity_change_min +
		# 		(num_iterations_per_epoch - 1 - iteration) * (epoch_range_permittivity_change_min / (num_iterations_per_epoch - 1))
		# 	)


		cur_design_variable = bayer_filter.get_design_variable()

		step_size = step_size_start

		# check_last = False
		# last = 0

		# while True:
		# 	proposed_design_variable = cur_design_variable + step_size * design_gradient
		# 	proposed_design_variable = np.maximum(
		# 								np.minimum(
		# 									proposed_design_variable,
		# 									1.0),
		# 								0.0)

		# 	difference = np.abs(proposed_design_variable - cur_design_variable)
		# 	max_difference = np.max(difference)

		# 	if (max_difference <= max_change_design) and (max_difference >= min_change_design):
		# 		break
		# 	elif (max_difference <= max_change_design):
		# 		step_size *= 2
		# 		if (last ^ 1) and check_last:
		# 			break
		# 		check_last = True
		# 		last = 1
		# 	else:
		# 		step_size /= 2
		# 		if (last ^ 0) and check_last:
		# 			break
		# 		check_last = True
		# 		last = 0

		# step_size_start = step_size

		last_design_variable = cur_design_variable.copy()
		bayer_filter.step(-device_gradient_real, -device_gradient_imag, step_size)

		print( "The max amount the density is changing is around " + str( np.max(np.abs(-design_gradient * step_size))) )
		print( "The mean amount the density is changing is around " + str( np.mean(np.abs(-design_gradient * step_size))) )
		print()

		cur_design_variable = bayer_filter.get_design_variable()

		average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
		max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

		step_size_evolution[epoch][iteration] = step_size
		average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
		max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

		np.save(projects_directory_location + '/device_gradient_real.npy', device_gradient_real)
		np.save(projects_directory_location + '/device_gradient_imag.npy', device_gradient_imag)
		np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
		np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
		np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
		np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
		np.save(projects_directory_location + "/cur_design_variable.npy", cur_design_variable)




