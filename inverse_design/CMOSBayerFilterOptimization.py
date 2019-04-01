import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSBayerFilterParameters import *
import CMOSBayerFilter

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

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

shutil.copy2(python_src_directory + "/CMOSBayerFilterParameters.py", projects_directory_location + "/ArchiveCMOSBayerFilterParameters.py")

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
# Add device region and create device permittivity
#
design_import = fdtd_hook.addimport()
design_import['name'] = 'design_import'
design_import['x span'] = device_size_lateral_um * 1e-6
design_import['y span'] = device_size_lateral_um * 1e-6
design_import['z max'] = device_vertical_maximum_um * 1e-6
design_import['z min'] = device_vertical_minimum_um * 1e-6

bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
bayer_filter = CMOSBayerFilter.CMOSBayerFilter(bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_permittivity_0_1_scale)

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

cur_permittivity = bayer_filter.get_permittivity()
fdtd_hook.select("design_import")
fdtd_hook.importnk2(np.sqrt(cur_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)


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


def structured_to_complex(element):
	return element[0] + np.complex(0, 1) * element[1]

def convert_array(input_array, output_dtype, func):
	input_array_shape = input_array.shape
	num_elements = functools.reduce(lambda x, y: x * y, input_array_shape)
	converted_array = np.zeros(num_elements, dtype=output_dtype)

	flatten_input = input_array.flatten()

	for idx in range(0, num_elements):
		converted_array[idx] = func(flatten_input[idx])

	return converted_array.reshape(input_array_shape)

#
# Consolidate the data transfer functionality for getting data from Lumerical FDTD process to
# python process.  This is much faster than going through Lumerical's interop library
#
def get_monitor_data(monitor_name, monitor_field):
	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	extracted_data_name = lumerical_data_name + "_data"
	data_transfer_filename = "data_transfer_" + monitor_name + "_" + monitor_field

	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command_extract_data = extracted_data_name + " = " + lumerical_data_name + "." + monitor_field + ";"
	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + extracted_data_name + ");"

	print(command_read_monitor)
	print(command_extract_data)
	print(command_save_data_to_file)

	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)
	lumapi.evalScript(fdtd_hook.handle, command_extract_data)

	start_time = time.time()

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat")

	print(load_file)
	print(load_file.keys())

	monitor_data = np.array(load_file[extracted_data_name])

	end_time = time.time()

	print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	return convert_array(
		get_monitor_data(monitor_name, monitor_field),
		np.complex,
		structured_to_complex)

#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}
adjoint_e_fields = []

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

#
# Run the optimization
#
for epoch in range(0, num_epochs):
	bayer_filter.update_filters(epoch)

	for iteration in range(0, num_iterations_per_epoch):

		#
		# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
		#
		for xy_idx in range(0, 2):
			disable_all_sources()
			(forward_sources[xy_idx]).enabled = 1
			fdtd_hook.run()

			forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E'),

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
			spectral_indices = spectral_focal_plane_map[focal_idx]

			for polarization_idx in range(0, len(polarizations)):
				compute_fom += np.sum( np.abs(focal_data[polarizations[polarization_idx]][spectral_focal_plane_map])**2 )

			figure_of_merit_per_focal_spot[focal_idx] = compute_fom

		# When we combine figures of merit, we can either just do a straight average or we can do a weighted average
		# based on performance.  Or we can just apply the weighting to the gradient update of the permittivity.

		figure_of_merit = np.sum(figure_of_merit_per_focal_spot)
		figure_of_merit_evolution[epoch, iteration] = figure_of_merit


		#
		# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources.
		#
		for adj_src_idx in range(0, num_adjoint_sources):
			adjoint_e_fields.append({})
			for xy_idx in range(0, 2):
				disable_all_sources()
				(adjoint_sources[adj_src_idx][xy_idx]).enabled = 1
				fdtd_hook.run()

				adjoint_e_fields[adj_src_idx][xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')


		#
		# Step 4: Compute the gradient by appropriately weighting the fields from all the adjoint sources and combining
		# with the fields from the forward source.
		#
		xy_polarized_gradients = [ np.zeros(cur_permittivity.shape), np.zeros(cur_permittivity.shape) ]
		for adj_src_idx in range(0, num_adjoint_sources):
			polarizations = polarizations_focal_plane_map[focal_idx]
			spectral_indices = spectral_focal_plane_map[focal_idx]

			for get_polarization in polarizations:
				for weight_adjoint_polarization in ['x', 'y']:

					weight_adjoint_fields = (
						np.conj(focal_data[get_polarization][spectral_indices, weight_adjoint_polarization]) *
						adjoint_e_fields[adj_src_idx][weight_adjoint_polarization][:, :, :, spectral_indices, :]
					)
					dot_with_forward_fields = weight_adjoint_fields * forward_e_fields[get_polarization][:, :, :, spectral_indices, :]
					sum_dot_product = np.sum(dot_with_forward_fields)

					# Currently, this weights all gradients equally. I believe there is another scaling with wavelength that needs to be
					# added back in.  Maximum value of intensity by wavelength at focal spot
					xy_polarized_gradients[polarization_name_to_idx[get_polarization]] += sum_dot_product






