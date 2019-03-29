import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSBayerFilterParameters import *
import CMOSBayerFilter

import lumapi

import numpy as np
import scipy.io
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
fdtd['mesh cells x'] = fdtd_region_minimum_vertical_voxels
fdtd['mesh cells y'] = fdtd_region_minimum_vertical_voxels
fdtd['mesh cells z'] = fdtd_region_minimum_lateral_voxels
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
	for xy_idx in range(0, 2):
		(forward_sources[xy_idx]).enabled = 0

	for adj_src_idx in range(0, num_adjoint_sources):
		for xy_idx in range(0, 2):
			(adjoint_sources[adj_src_idx][xy_idx]).enabled = 0


#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}
adjoint_e_fields = {}

def get_monitor_data(monitor_name, monitor_field):
	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	data_transfer_filename = "data_transfer_" + monitor_name + "_" + monitor_field

	lumerical_handle = lumapi.open('fdtd')

	command1 = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command2 = "matlabsave(\'" + data_transfer_filename + "\', " + lumerical_data_name + ");"

	print(command1)
	print(command2)
	lumapi.evalScript(lumerical_handle, command1)

	start_time = time.time()
	lumapi.evalScript(lumerical_handle, command2)
	monitor_data = scipy.io.loadmat(data_transfer_filename)
	end_time = time.time()

	print("\nIt took " + str(end_time - start_time) + " seconds to retrieve the monitor data\n")

	lumapi.close(lumerical_handle)

	return monitor_data

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

			# forward_e_fields[xy_names[xy_idx]] = get_monitor_data(design_efield_monitor['name'], 'E')

			focal_data[xy_names[xy_idx]] = []
			for adj_src_idx in range(0, num_adjoint_sources):
				focal_data[xy_names[xy_idx]].append(get_monitor_data(focal_monitors[adj_src_idx]['name'], 'E'))


		#
		# Step 2: Run all the adjoint optimizations for both x- and y-polarized adjoint sources
		#
		# for adj_src_idx in range(0, num_adjoint_sources):
		# 	for xy_idx in range(0, 2):
		# 		disable_all_sources()
		# 		(adjoint_sources[adj_src_idx][xy_idx]).enabled = 1
		# 		fdtd_hook.run()



lumapi.close(lumerical_handle)

