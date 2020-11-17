import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredMWIRBridgesBayerFilterParameters import *
import LayeredMWIRBridgesBayerFilter
import ip_dip_dispersion


import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
import lumapi

import functools
import h5py
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
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
# projects_directory_location += "/" + project_name

projects_directory_location = "/central/groups/Faraon_Computing/projects" 
projects_directory_location += "/" + project_name# + '_parallel'


if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredMWIRBridgesBayerFilterParameters.py", projects_directory_location + "/ArchiveLayeredMWIRBridgesBayerFilterParameters.py")

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

#
# Add a block of polymer at the top where the device will be adhered to a Silicon substrate
#
# permittivity_layer_substrate = fdtd_hook.addrect()
# permittivity_layer_substrate['name'] = 'polymer_layer_substrate'
# permittivity_layer_substrate['x'] = 0
# permittivity_layer_substrate['x span'] = ( device_size_lateral_um ) * 1e-6
# permittivity_layer_substrate['y'] = 0
# permittivity_layer_substrate['y span'] = ( device_size_lateral_um ) * 1e-6
# permittivity_layer_substrate['z min'] = ( device_vertical_maximum_um ) * 1e-6
# # Send this outside the region FDTD and let the source sit inside of it
# permittivity_layer_substrate['z max'] = ( device_vertical_maximum_um + 1.5 * vertical_gap_size_um ) * 1e-6
# permittivity_layer_substrate['index'] = max_device_index

permittivity_layer_substrate = fdtd_hook.addimport()
permittivity_layer_substrate['name'] = 'permittivity_layer_substrate'
permittivity_layer_substrate['x span'] = device_size_lateral_um * 1e-6
permittivity_layer_substrate['y span'] = device_size_lateral_um * 1e-6
permittivity_layer_substrate['z min'] = device_vertical_maximum_um * 1e-6
permittivity_layer_substrate['z max'] = ( fdtd_region_maximum_vertical_um - silicon_thickness_um ) * 1e-6

platform_x_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, 2 )
platform_y_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, 2 )
platform_z_range = 1e-6 * np.linspace( device_vertical_maximum_um, fdtd_region_maximum_vertical_um - silicon_thickness_um, 2 )

platform_index = np.ones( ( 2, 2, 2 ), dtype=np.complex )


silicon_substrate = fdtd_hook.addrect()
silicon_substrate['name'] = 'silicon_substrate'
silicon_substrate['x'] = 0
silicon_substrate['x span'] = fdtd_region_size_lateral_um * 1e-6
silicon_substrate['y'] = 0
silicon_substrate['y span'] = fdtd_region_size_lateral_um * 1e-6
silicon_substrate['z min'] = ( fdtd_region_maximum_vertical_um - silicon_thickness_um ) * 1e-6
# Send this outside the region FDTD and let the source sit inside of it
silicon_substrate['z max'] = fdtd_region_maximum_vertical_um * 1e-6
silicon_substrate['material'] = 'Si (Silicon) - Palik'


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
bayer_filter = LayeredMWIRBridgesBayerFilter.LayeredMWIRBridgesBayerFilter(
	bayer_filter_size_voxels,
	[ 0.0, 1.0 ],
	# [min_device_permittivity, max_device_permittivity],
	init_permittivity_0_1_scale,
	num_vertical_layers,
	topology_num_free_iterations_between_patches)

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
figure_of_merit_by_wl_evolution = np.zeros((num_epochs, num_iterations_per_epoch, 4))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = 1.

#
# Set up queue for parallel jobs
#
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
			if queue_in.qsize() > 0:
				small_queue.put( queue_in.get() )

		run_jobs_inner( small_queue )

def run_jobs_inner( queue_in ):
	processes = []
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


ip_dip_dispersion_model = ip_dip_dispersion.IPDipDispersion()

cur_design_variable = np.load( projects_directory_location + "/cur_design_variable.npy" )
bayer_filter.w[0] = cur_design_variable
bayer_filter.update_permittivity()

#
# Run the optimization
#
start_epoch = 6
for epoch in range(start_epoch, num_epochs):
	bayer_filter.update_filters(epoch)
	bayer_filter.update_permittivity()

	start_iter = 0
	if epoch == start_epoch:
		start_iter = 20
	for iteration in range(start_iter, num_iterations_per_epoch):
		print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

		job_names = {}

		fdtd_hook.switchtolayout()
		cur_density = bayer_filter.get_permittivity()

		#
		# Step 1: Run the forward optimization for both x- and y-polarized plane waves.  Run a different optimization for each color band so
		# that you can input a different index into each of those cubes.  Further note that the polymer slab needs to also change index when
		# the cube index changes.
		#
		for dispersive_range_idx in range( 0, num_dispersive_ranges ):
			dispersive_max_permittivity = ip_dip_dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
			disperesive_max_index = ip_dip_dispersion.index_from_permittivity( dispersive_max_permittivity )

			fdtd_hook.switchtolayout()

			platform_index[ : ] = disperesive_max_index

			fdtd_hook.select( 'permittivity_layer_substrate' )
			fdtd_hook.importnk2( platform_index, platform_x_range, platform_y_range, platform_z_range )

			cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
			cur_index = ip_dip_dispersion.index_from_permittivity( cur_permittivity )

			fdtd_hook.select( 'design_import' )
			fdtd_hook.importnk2( cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )


			for xy_idx in range(0, 2):
				disable_all_sources()

				fdtd_hook.select( forward_sources[xy_idx]['name'] )
				fdtd_hook.set( 'enabled', 1 )

				job_name = 'forward_job_' + str( xy_idx ) + "_" + str( dispersive_range_idx ) + '.fsp'
				fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
				job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )


			for adj_src_idx in range(0, num_adjoint_sources):
				adjoint_e_fields = []
				for xy_idx in range(0, 2):
					disable_all_sources()

					fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
					fdtd_hook.set( 'enabled', 1 )

					job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '_' + str( dispersive_range_idx ) + '.fsp'
					fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
					job_names[ ( 'adjoint', adj_src_idx, xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )


		run_jobs( jobs_queue )


		#
		# Step 2: Compute the figure of merit
		#
		for xy_idx in range(0, 2):
			focal_data[xy_names[xy_idx]] = [ None for idx in range( 0, num_focal_spots ) ]

		for dispersive_range_idx in range( 0, num_dispersive_ranges ):
			for xy_idx in range(0, 2):
				fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )

				fwd_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

				if ( dispersive_range_idx == 0 ):

					forward_e_fields[xy_names[xy_idx]] = fwd_e_fields
				else:
					get_wavelength_bucket = spectral_focal_plane_map[ dispersive_range_to_adjoint_src_map[ dispersive_range_idx ][ 0 ] ]

					for spectral_idx in range( get_wavelength_bucket[ 0 ], get_wavelength_bucket[ 1 ] ):
						forward_e_fields[xy_names[xy_idx]][ :, spectral_idx, :, :, : ] = fwd_e_fields[ :, spectral_idx, :, :, : ]

				for adj_src_idx in dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]:
					focal_data[xy_names[xy_idx]][ adj_src_idx ] = get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E')



		figure_of_merit_per_focal_spot = []
		for focal_idx in range(0, num_focal_spots):
			compute_fom = 0

			polarizations = polarizations_focal_plane_map[focal_idx]

			for polarization_idx in range(0, len(polarizations)):
				get_focal_data = focal_data[polarizations[polarization_idx]]

				max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[focal_idx][0] : spectral_focal_plane_map[focal_idx][1] : 1]
				total_weighting = max_intensity_weighting / weight_focal_plane_map[focal_idx]

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
		figure_of_merit_by_wl_evolution[epoch, iteration] = figure_of_merit_per_focal_spot

		np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)
		np.save(projects_directory_location + "/figure_of_merit_by_wl.npy", figure_of_merit_by_wl_evolution)

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

			lookup_dispersive_range_idx = adjoint_src_to_dispersive_range_map[ adj_src_idx ]


			adjoint_e_fields = []
			for xy_idx in range(0, 2):
				#
				# This is ok because we are only going to be using the spectral idx corresponding to this range in teh summation below
				#
				fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ] )

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
					total_weighting = max_intensity_weighting / weight_focal_plane_map[focal_idx]

					#
					# We need to properly account here for the current real and imaginary index
					#
					dispersive_max_permittivity = ip_dip_dispersion_model.average_permittivity( dispersive_ranges_um[ lookup_dispersive_range_idx ] )
					delta_real_permittivity = np.real( dispersive_max_permittivity - min_device_permittivity )
					delta_imag_permittivity = np.imag( dispersive_max_permittivity - min_device_permittivity )


					for spectral_idx in range(0, source_weight.shape[0]):
						gradient_component = np.sum(
							(source_weight[spectral_idx] * gradient_performance_weight / total_weighting[spectral_idx]) *
							adjoint_e_fields[xy_idx][:, spectral_indices[0] + spectral_idx, :, :, :] *
							forward_e_fields[pol_name][:, spectral_indices[0] + spectral_idx, :, :, :],
							axis=0)

						real_part_gradient = np.real( gradient_component )
						imag_part_gradient = -np.imag( gradient_component )

						get_grad_density = delta_real_permittivity * real_part_gradient + delta_imag_permittivity * imag_part_gradient

						xy_polarized_gradients[pol_name_to_idx] += get_grad_density

		#
		# Step 4: Step the design variable.
		#
		# device_gradient = 2 * np.real( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
		device_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )

		# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
		# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
		device_gradient = np.swapaxes(device_gradient, 0, 2)

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

		if use_fixed_step_size:
			step_size = fixed_step_size
		else:
			step_size = step_size_start

			check_last = False
			last = 0

			while True:
				proposed_design_variable = cur_design_variable + step_size * design_gradient
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
		bayer_filter.step(-device_gradient, step_size)
		cur_design_variable = bayer_filter.get_design_variable()
		cur_design = bayer_filter.get_permittivity()

		average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
		max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

		step_size_evolution[epoch][iteration] = step_size
		average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
		max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

		np.save(projects_directory_location + '/device_gradient.npy', device_gradient)
		np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
		np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
		np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
		np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
		np.save(projects_directory_location + "/cur_design_variable.npy", cur_design_variable)
		np.save(projects_directory_location + "/cur_design.npy", cur_design)




