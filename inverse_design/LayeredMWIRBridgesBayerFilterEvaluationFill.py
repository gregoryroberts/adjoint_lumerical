import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredMWIRBridgesBayerFilterEvaluationParameters import *
import LayeredMWIRBridgesBayerFilter
import ip_dip_dispersion


run_on_cluster = True

if run_on_cluster:
	import imp
	imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
else:
	import imp
	imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )

import lumapi


import functools
import h5py
import numpy as np
import time

import queue

import subprocess

import platform

import re

import skimage.morphology as skim

import scipy

num_nodes_available = int( sys.argv[ 1 ] )

if run_on_cluster:
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

	num_cpus_per_node = 8
	cluster_hostnames = get_slurm_node_list()
else:
	num_cpus_per_node = 1
	cluster_hostnames = []

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


if run_on_cluster:
	projects_directory_location_base = "/central/groups/Faraon_Computing/projects" 
	projects_directory_location_base += "/" + project_name
	projects_directory_location = projects_directory_location_base + '_filled_dilation'
else:
	projects_directory_location_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
	projects_directory_location_base += "/" + project_name
	projects_directory_location = projects_directory_location_base + '_filled_dilation'


if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredMWIRBridgesBayerFilterEvaluationParameters.py", projects_directory_location + "/LayeredMWIRBridgesBayerFilterEvaluationParameters.py")
shutil.copy2(python_src_directory + "/LayeredMWIRBridgesBayerFilterEvaluationFill.py", projects_directory_location + "/LayeredMWIRBridgesBayerFilterEvaluationFill.py")

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
fdtd['mesh cells x'] = fdtd_region_minimum_lateral_voxels * 2
fdtd['mesh cells y'] = fdtd_region_minimum_lateral_voxels * 2
fdtd['mesh cells z'] = fdtd_region_minimum_vertical_voxels * 2
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
	forward_src['z max'] = (src_maximum_vertical_um + 1) * 1e-6
	forward_src['z min'] = src_minimum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_max_um * 1e-6

	forward_sources.append(forward_src)


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


transmission_monitors = []

for adj_src in range(0, num_adjoint_sources):
	transmission_monitor = fdtd_hook.addpower()
	transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
	transmission_monitor['monitor type'] = '2D Z-normal'
	transmission_monitor['x span'] = 0.5 * device_size_lateral_um * 1e-6
	transmission_monitor['y span'] = 0.5 * device_size_lateral_um * 1e-6
	transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
	transmission_monitor['z'] = adjoint_vertical_um * 1e-6
	transmission_monitor['override global monitor settings'] = 1
	transmission_monitor['use wavelength spacing'] = 1
	transmission_monitor['use source limits'] = 1
	transmission_monitor['frequency points'] = num_design_frequency_points

	transmission_monitors.append(transmission_monitor)


focal_transmission_monitor = fdtd_hook.addpower()
focal_transmission_monitor['name'] = 'focal_transmission_monitor'
focal_transmission_monitor['monitor type'] = '2D Z-normal'
focal_transmission_monitor['x span'] = device_size_lateral_um * 1e-6
focal_transmission_monitor['y span'] = device_size_lateral_um * 1e-6
focal_transmission_monitor['x'] = 0 * 1e-6
focal_transmission_monitor['y'] = 0 * 1e-6
focal_transmission_monitor['z'] = adjoint_vertical_um * 1e-6
focal_transmission_monitor['override global monitor settings'] = 1
focal_transmission_monitor['use wavelength spacing'] = 1
focal_transmission_monitor['use source limits'] = 1
focal_transmission_monitor['frequency points'] = num_design_frequency_points

transmission_monitors.append( focal_transmission_monitor )

#
# Add a block of polymer at the top where the device will be adhered to a Silicon substrate
#

permittivity_layer_substrate = fdtd_hook.addimport()
permittivity_layer_substrate['name'] = 'permittivity_layer_substrate'
permittivity_layer_substrate['x span'] = device_size_lateral_um * 1e-6
permittivity_layer_substrate['y span'] = device_size_lateral_um * 1e-6
permittivity_layer_substrate['z min'] = device_vertical_maximum_um * 1e-6
permittivity_layer_substrate['z max'] = ( device_vertical_maximum_um + pedestal_thickness_um ) * 1e-6

platform_x_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, 2 )
platform_y_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, 2 )
platform_z_range = 1e-6 * np.linspace( device_vertical_maximum_um, device_vertical_maximum_um + pedestal_thickness_um, 2 )

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

dilation_erosion_test = True
dilation_amount = 1

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

if dilation_erosion_test:
	bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, 2 * device_voxels_lateral)
	bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, 2 * device_voxels_lateral)
	bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, 2 * device_voxels_vertical)


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	for xy_idx in range(0, 2):
		fdtd_hook.select( forward_sources[xy_idx]['name'] )
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

cur_design_variable = np.load( projects_directory_location_base + "/cur_design_variable.npy" )
bayer_filter.w[0] = cur_design_variable
bayer_filter.update_filters(num_epochs - 1)
bayer_filter.update_permittivity()

cur_fabrication_target = ( bayer_filter.get_permittivity() > 0.5 )

pad_cur_fabrication_target = np.pad(
	cur_fabrication_target,
	( ( 1, 1 ), ( 1, 1 ), ( 1, 1 ) ),
	mode='constant'
)
pad_cur_fabrication_target[ :, :, pad_cur_fabrication_target.shape[ 2 ] - 1 ] = 1

[solid_labels, num_solid_labels] = skim.label( pad_cur_fabrication_target, connectivity=1, return_num=True )
[void_labels, num_void_labels] = skim.label( 1 - pad_cur_fabrication_target, connectivity=1, return_num=True )

layer_step = int( cur_fabrication_target.shape[ 2 ] / num_vertical_layers )

new_device = np.zeros( cur_fabrication_target.shape )

for layer_idx in range( 0, num_vertical_layers ):
	pull_data = void_labels[ 1 : void_labels.shape[ 0 ] - 1, 1 : void_labels.shape[ 1 ] - 1, layer_idx * layer_step + 1 ]

	for internal_layer in range( 0, layer_step ):
		new_device[ :, :, layer_idx * layer_step + internal_layer ] = 1.0 * np.greater( cur_fabrication_target[ :, :, layer_idx * layer_step + 1 ], 0.5 )
		new_device[ :, :, layer_idx * layer_step + internal_layer ] += 1.0 * np.greater( pull_data, 1 )
new_device[ :, :, new_device.shape[ 2 ] - 1 ] = new_device[ :, :, new_device.shape[ 2 ] - 2 ]


num_eval_points = 50
if ( num_eval_points % num_nodes_available ) > 0:
	num_eval_points += ( num_nodes_available - ( num_eval_points % num_nodes_available ) )

assert ( num_eval_points % num_nodes_available ) == 0, "Expected the number of nodes of available to evenly divide the number of eval points!"


num_outer_loops = int( num_eval_points / num_nodes_available )

job_names = {}

fdtd_hook.switchtolayout()
cur_density = new_device.copy()
# cur_density = bayer_filter.get_permittivity()
cur_density = 1.0 * np.greater_equal( cur_density, 0.5 )

reinterpolate_density = np.zeros( [ 2 * cur_density.shape[ idx ] for idx in range( 0, len( cur_density.shape ) ) ] )
for z_idx in range( 0, reinterpolate_density.shape[ 2 ] ):
	for x_idx in range( 0, reinterpolate_density.shape[ 0 ] ):
		for y_idx in range( 0, reinterpolate_density.shape[ 1 ] ):
			down_x = int( 0.5 * x_idx )
			down_y = int( 0.5 * y_idx )
			down_z = int( 0.5 * z_idx )

			reinterpolate_density[ x_idx, y_idx, z_idx ] = cur_density[ down_x, down_y, down_z ]

	if dilation_erosion_test:
		if dilation_amount > 0:
			reinterpolate_density[ :, :, z_idx ] = scipy.ndimage.binary_dilation( reinterpolate_density[ :, :, z_idx ], iterations=np.abs( dilation_amount ) )
		else:
			reinterpolate_density[ :, :, z_idx ] = scipy.ndimage.binary_erosion( reinterpolate_density[ :, :, z_idx ], iterations=np.abs( dilation_amount ) )

# import matplotlib.pyplot as plt
# for layer in range( 0, 5 ):
# 	layer_cur = cur_density[ :, :, 10 + 20 * layer ]
# 	layer_up = reinterpolate_density[ :, :, 20 + 40 * layer ]
# 	plt.subplot( 1, 2, 1 )
# 	plt.imshow( layer_cur )
# 	plt.colorbar()
# 	plt.subplot( 1, 2, 2 )
# 	plt.imshow( layer_up )
# 	plt.colorbar()
# 	plt.show()

# for layer in range( 0, 5 ):
# 	layer_up = reinterpolate_density[ :, :, 20 + 40 * layer ]
# 	layer_up_dilated = scipy.ndimage.binary_dilation( layer_up, iterations=1 )
# 	layer_up_eroded = scipy.ndimage.binary_erosion( layer_up, iterations=1 )
# 	plt.subplot( 1, 3, 1 )
# 	plt.imshow( layer_up )
# 	plt.colorbar()
# 	plt.subplot( 1, 3, 2 )
# 	plt.imshow( layer_up_dilated )
# 	plt.colorbar()
# 	plt.subplot( 1, 3, 3 )
# 	plt.imshow( layer_up_eroded )
# 	plt.colorbar()
# 	plt.show()

# sys.exit(0)	

eval_lambda_min_um = lambda_min_um - 0.5
eval_lambda_max_um = lambda_max_um + 0.5

eval_lambda_um = np.linspace( eval_lambda_min_um, eval_lambda_max_um, num_eval_points )

transmission_data = np.zeros( ( 2, num_adjoint_sources + 1, num_eval_points ) )

for outer_loop in range( 0, num_outer_loops ):
	eval_point_start = int( outer_loop * num_nodes_available )

	for eval_point_idx in range( eval_point_start, eval_point_start + num_nodes_available ):
		dispersive_max_permittivity = ip_dip_dispersion_model.average_permittivity( [ eval_lambda_um[ eval_point_idx ], eval_lambda_um[ eval_point_idx ] ] )
		# disperesive_max_index = np.real( ip_dip_dispersion.index_from_permittivity( dispersive_max_permittivity ) )
		disperesive_max_index = ip_dip_dispersion.index_from_permittivity( dispersive_max_permittivity )

		fdtd_hook.switchtolayout()

		platform_index[ : ] = disperesive_max_index

		fdtd_hook.select( 'permittivity_layer_substrate' )
		fdtd_hook.importnk2( platform_index, platform_x_range, platform_y_range, platform_z_range )

		cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
		# cur_index = np.real( ip_dip_dispersion.index_from_permittivity( cur_permittivity ) )
		cur_index = ip_dip_dispersion.index_from_permittivity( cur_permittivity )

		fdtd_hook.select( 'design_import' )
		fdtd_hook.importnk2( cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )

		for focal_idx in range( 0, len( transmission_monitors ) ):
			transmission_monitors[ focal_idx ]['use source limits'] = 0
			transmission_monitors[ focal_idx ]['frequency points'] = 1
			transmission_monitors[ focal_idx ]['wavelength center'] = eval_lambda_um[ eval_point_idx ] * 1e-6

		for pol_idx in range(0, 2):
			disable_all_sources()

			fdtd_hook.select( forward_sources[pol_idx]['name'] )
			fdtd_hook.set( 'enabled', 1 )

			job_name = 'forward_job_' + str( pol_idx ) + "_" + str( eval_point_idx % num_nodes_available ) + '.fsp'
			fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
			job_names[ ( 'forward', pol_idx, eval_point_idx % num_nodes_available ) ] = add_job( job_name, jobs_queue )


	run_jobs( jobs_queue )

	for eval_point_idx in range( eval_point_start, eval_point_start + num_nodes_available ):
		for pol_idx in range( 0, 2 ):

			fdtd_hook.load( job_names[ ( 'forward', pol_idx, eval_point_idx % num_nodes_available ) ] )

			for focal_idx in range( 0, num_adjoint_sources ):

				T = fdtd_hook.getresult( transmission_monitors[ focal_idx ][ 'name' ], 'T' )
				transmission_data[ pol_idx, focal_idx, eval_point_idx ] = T[ 'T' ][ 0 ]

			T = fdtd_hook.getresult( transmission_monitors[ num_adjoint_sources ][ 'name' ], 'T' )
			transmission_data[ pol_idx, num_adjoint_sources, eval_point_idx ] = T[ 'T' ][ 0 ]


	np.save( projects_directory_location + "/filled_dispersive_transmission_data_full.npy", transmission_data )

np.save( projects_directory_location + "/filled_dispersive_transmission_data_full.npy", transmission_data )

