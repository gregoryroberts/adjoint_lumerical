import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyAMParameters import *
import LayeredLithographyAMBayerFilter

import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.ndimage import gaussian_filter

import queue

import subprocess

import platform

import re

import reinterpolate


#
# Code from Conner // START
#

def configure_resources_for_cluster( fdtd_hook, node_hostnames, N_resources=2, N_threads_per_resource=8 ):
	'''
	Take in a list of hostnames (different nodes on the cluster), and configure
	them to have N_threads_per_resource.
	'''
	if len(node_hostnames) != N_resources:
		raise ValueError('Length of node_hostnames should be N_resources')

	# Use different MPIs depending on platform.
	if platform.system() == 'Windows':
		mpi_type = 'Remote: Intel MPI'
	else:
		mpi_type = 'Remote: MPICH2'
	# Delete all resources. Lumerical doesn't let us delete the last resource, so
	# we stop when it throws an Exception.
	while True:
		try:
			fdtd_hook.deleteresource("FDTD", 1)
		except lumapi.LumApiError:
			break
	# Change the one resource we have to have the proper number of threads.
	fdtd_hook.setresource("FDTD", 1, "processes", N_threads_per_resource)
	fdtd_hook.setresource('FDTD', 1, 'Job launching preset', mpi_type)
	fdtd_hook.setresource('FDTD', 1, 'hostname', node_hostnames[0])
	# Now add and configure the rest.
	for i in np.arange(1, N_resources):
		try:
			fdtd_hook.addresource("FDTD")
		except:
			pass
		finally:
			fdtd_hook.setresource("FDTD", i+1, "processes", N_threads_per_resource)
			fdtd_hook.setresource('FDTD', i+1, 'Job launching preset', mpi_type)
			fdtd_hook.setresource('FDTD', i+1, 'hostname', node_hostnames[i])


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


#
# Code from Conner // END
#

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD( hide=True )

num_nodes_available = int( sys.argv[ 1 ] )
num_cpus_per_node = 8
cluster_hostnames = get_slurm_node_list()
# configure_resources_for_cluster( fdtd_hook, slurm_list, N_resources=num_nodes_to_use, N_threads_per_resource=num_cpus_per_node )

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

projects_directory_location = "/central/groups/Faraon_Computing/projects" 
projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredLithographyAMParameters.py", projects_directory_location + "/LayeredLithographyAMParameters.py")
shutil.copy2(python_src_directory + "/LayeredLithographyAMBayerFilter.py", projects_directory_location + "/LayeredLithographyAMBayerFilter.py")
shutil.copy2(python_src_directory + "/LayeredLithographyAMOptimization.py", projects_directory_location + "/LayeredLithographyAMOptimization.py")

#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['z max'] = fdtd_region_maximum_vertical_um * 1e-6
fdtd['z min'] = fdtd_region_minimum_vertical_um * 1e-6
# fdtd['mesh type'] = 'uniform'
# fdtd['define x mesh by'] = 'number of mesh cells'
# fdtd['define y mesh by'] = 'number of mesh cells'
# fdtd['define z mesh by'] = 'number of mesh cells'
# fdtd['mesh cells x'] = fdtd_region_minimum_lateral_voxels
# fdtd['mesh cells y'] = fdtd_region_minimum_lateral_voxels
# fdtd['mesh cells z'] = fdtd_region_minimum_vertical_voxels
fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
fdtd['background index'] = background_index

design_mesh = fdtd_hook.addmesh()
design_mesh['name'] = 'design_override_mesh'
design_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
design_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
design_mesh['z max'] = device_vertical_maximum_um * 1e-6
design_mesh['z min'] = device_vertical_minimum_um * 1e-6
design_mesh['dx'] = mesh_spacing_um * 1e-6
design_mesh['dy'] = mesh_spacing_um * 1e-6
design_mesh['dz'] = mesh_spacing_um * 1e-6


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
substrate = fdtd_hook.addrect()
substrate['name'] = 'substrate'
substrate['x span'] = fdtd_region_size_lateral_um * 1e-6
substrate['y span'] = fdtd_region_size_lateral_um * 1e-6
substrate['z min'] = device_vertical_maximum_um * 1e-6
substrate['z max'] = fdtd_region_maximum_vertical_um * 1e-6
substrate['index'] = index_substrate

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
design_import['x span'] = ( fdtd_region_size_lateral_um - mesh_spacing_um ) * 1e-6
design_import['y span'] = ( fdtd_region_size_lateral_um - mesh_spacing_um ) * 1e-6
design_import['z max'] = ( device_vertical_maximum_um - 0.5 * mesh_spacing_um ) * 1e-6
design_import['z min'] = ( device_vertical_minimum_um + 0.5 * mesh_spacing_um ) * 1e-6

bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
bayer_filter = LayeredLithographyAMBayerFilter.LayeredLithographyAMBayerFilter(
	bayer_filter_size_voxels,
	[min_device_permittivity, max_device_permittivity],
	init_permittivity_0_1_scale,
	num_vertical_layers,
	spacer_size_voxels,
	last_layer_permittivities )

np.random.seed( random_seed )
num_random = device_voxels_lateral * device_voxels_lateral * device_voxels_vertical
random_device = np.random.normal( init_permittivity_0_1_scale, 0.25, num_random )
random_device = np.minimum( np.maximum( random_device, 0.0 ), 1.0 )

reshape_device = np.reshape( random_device, [ device_voxels_lateral, device_voxels_lateral, device_voxels_vertical ] )
blur_random_device = np.zeros( reshape_device.shape )

for z_idx in range( 0, reshape_device.shape[ 2 ] ):
	blur_random_device[ :, :, z_idx ] = gaussian_filter( reshape_device[ :, :, z_idx ], sigma=2 )

bayer_filter.set_design_variable( blur_random_device )

# bayer_filter.set_design_variable( np.random.random( bayer_filter.get_design_variable().shape ) )
# bayer_filter.step(
# 	np.random.random( bayer_filter.get_design_variable().shape ),
# 	0.01,
# 	True,
# 	projects_directory_location
# )
# sys.exit(0)

# bayer_filter.set_design_variable( np.load(projects_directory_location + "/cur_design_variable.npy") )

# bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, import_region_voxels_lateral)
# bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, import_region_voxels_lateral)
# bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

# bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, import_region_voxels_lateral)
# bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, import_region_voxels_lateral)
# bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um + 0.5 * mesh_spacing_um, device_vertical_maximum_um - 0.5 * mesh_spacing_um, simulated_device_voxels_vertical)

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um + 0.5 * mesh_spacing_um, device_vertical_maximum_um - 0.5 * mesh_spacing_um, simulated_device_voxels_vertical)


# def border_and_replicate_device( device_index, border_size_voxels, border_index ):
# 	z_size = device_index.shape[ 2 ]
# 	replicated_index = np.zeros( [ import_region_voxels_lateral, import_region_voxels_lateral, simulated_device_voxels_vertical ] )

# 	total_size_remaining = import_region_voxels_lateral - device_voxels_lateral
# 	pad_right = int( 0.5 * total_size_remaining )
# 	pad_left = total_size_remaining - pad_right

# 	for z_idx in range( 0, z_size ):
# 		extract_layer = np.squeeze( device_index[ :, :, z_idx ] )

# 		bordered_layer = np.pad(
# 			extract_layer,
# 			( ( pad_left, pad_right ), ( pad_left, pad_right ) ),
# 			mode='constant',
# 			constant_values=(( border_index, border_index ), ( border_index, border_index )) )

# 		# replicated_index[ :, :, z_idx ] = np.pad( bordered_layer, ( ( pad_left, pad_right ), ( pad_left, pad_right ) ), mode='wrap' )

# 	return replicated_index


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	# fdtd_hook.switchtolayout()
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	for xy_idx in range(0, 2):
		fdtd_hook.select( forward_sources[xy_idx]['name'] )
		fdtd_hook.set( 'enabled', 0 )
		# (forward_sources[xy_idx]).enabled = 0

	for adj_src_idx in range(0, num_adjoint_sources):
		for xy_idx in range(0, 2):
			fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
			fdtd_hook.set( 'enabled', 0 )
			# (adjoint_sources[adj_src_idx][xy_idx]).enabled = 0


def get_efield_interpolated( monitor_name, spatial_limits_um, new_size ):
	field_polarizations = [ 'Ex', 'Ey', 'Ez' ]
	spatial_components = [ 'x', 'y', 'z' ]
	data_xfer_size_MB = 0

	start = time.time()

	for coord_idx in range( 0, len( spatial_limits_um ) ):
		command_setup_old_coord = "old_coord_space_" + str( coord_idx ) + " = getdata(\'" + monitor_name + "\', \'" + spatial_components[ coord_idx ] + "\');"
		command_setup_new_coord = "new_coord_space_" + str( coord_idx ) + " = 1e-6 * linspace( " + str( spatial_limits_um[ coord_idx ][ 0 ] ) + ", " + str( spatial_limits_um[ coord_idx ][ 1 ] ) + ", " + str( new_size[ coord_idx ] ) + " );"
		lumapi.evalScript(fdtd_hook.handle, command_setup_old_coord)
		lumapi.evalScript(fdtd_hook.handle, command_setup_new_coord)


	total_efield = np.zeros( [ len (field_polarizations ) ] + list( new_size ) + [ num_design_frequency_points ], dtype=np.complex )

	for pol_idx in range( 0, len( field_polarizations ) ):
		lumerical_data_name = "monitor_data_" + monitor_name + "_E"
		command_make_interpolated_array = "interpolated_data = zeros( "
		for coord_idx in range( 0, len( spatial_limits_um ) ):
			command_make_interpolated_array += str( new_size[ coord_idx ] ) + ", "
		command_make_interpolated_array += str( num_design_frequency_points ) + " );"

		lumapi.evalScript(fdtd_hook.handle, command_make_interpolated_array)

		command_read_monitor = lumerical_data_name + " = getdata(\'" + monitor_name + "\', \'" + field_polarizations[ pol_idx ] + "\');"
		lumapi.evalScript(fdtd_hook.handle, command_read_monitor)

		for wl_idx in range( 0, num_design_frequency_points ):
			command_data_by_wavelength = "wl_data = pinch( " + lumerical_data_name + "( :, :, :, " + str( wl_idx + 1 ) + " ) );"
			command_reassemble_by_wavelength = "interpolated_data( :, :, :, " + str( wl_idx + 1 ) + " ) = interpolated;"

			command_interpolate = "interpolated = interp( wl_data, "

			for coord_idx in range( 0, len( spatial_limits_um ) ):
				command_interpolate += "old_coord_space_" + str( coord_idx ) + ", "

			for coord_idx in range( 0, len( spatial_limits_um ) ):
				command_interpolate += "new_coord_space_" + str( coord_idx )

				if coord_idx < ( len( spatial_limits_um ) - 1 ):
					command_interpolate += ", "

			command_interpolate += " );"

			lumapi.evalScript(fdtd_hook.handle, command_data_by_wavelength)
			lumapi.evalScript(fdtd_hook.handle, command_interpolate)
			lumapi.evalScript(fdtd_hook.handle, command_reassemble_by_wavelength)

		Epol = fdtd_hook.getv( "interpolated_data" )
		data_xfer_size_MB += Epol.nbytes / ( 1024. * 1024. )

		total_efield[ pol_idx ] = Epol


	# Epol_0 = fdtd_hook.getdata( monitor_name, field_polarizations[ 0 ] )
	# data_xfer_size_MB += Epol_0.nbytes / ( 1024. * 1024. )

	# total_efield = np.zeros( [ len (field_polarizations ) ] + list( Epol_0.shape ), dtype=np.complex )
	# total_efield[ 0 ] = Epol_0

	# for pol_idx in range( 1, len( field_polarizations ) ):
	# 	Epol = fdtd_hook.getdata( monitor_name, field_polarizations[ pol_idx ] )
	# 	data_xfer_size_MB += Epol.nbytes / ( 1024. * 1024. )

	# 	total_efield[ pol_idx ] = Epol

	elapsed = time.time() - start

	date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
	log_file.write( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )
	log_file.close()

	return total_efield


def get_efield( monitor_name ):
	field_polarizations = [ 'Ex', 'Ey', 'Ez' ]
	data_xfer_size_MB = 0

	start = time.time()

	Epol_0 = fdtd_hook.getdata( monitor_name, field_polarizations[ 0 ] )
	data_xfer_size_MB += Epol_0.nbytes / ( 1024. * 1024. )

	total_efield = np.zeros( [ len (field_polarizations ) ] + list( Epol_0.shape ), dtype=np.complex )
	total_efield[ 0 ] = Epol_0

	for pol_idx in range( 1, len( field_polarizations ) ):
		Epol = fdtd_hook.getdata( monitor_name, field_polarizations[ pol_idx ] )
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
# forward_e_fields = {}
# focal_data = {}

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = 0.001

if start_epoch > 0:
	design_variable_reload = np.load( projects_directory_location + '/cur_design_variable_' + str( start_epoch - 1 ) + '.npy' )
	bayer_filter.set_design_variable( design_variable_reload )
	figure_of_merit_evolution_load = np.load( projects_directory_location + "/figure_of_merit.npy" )

	num_epochs_reload = figure_of_merit_evolution_load.shape[ 0 ]
	figure_of_merit_evolution[ 0 : num_epochs_reload ] = figure_of_merit_evolution_load


fdtd_hook.save( projects_directory_location + "/optimization.fsp" )


jobs_queue = queue.Queue()

def add_job( job_name, queue ):
	full_name = projects_directory_location + "/" + job_name
	fdtd_hook.save( full_name )
	queue.put( full_name )

	return full_name

def run_jobs( queue ):
	processes = []
	# for job_idx in range( 0, len( queue ) ):
	# really should protect against number of available engines here
	job_idx = 0
	while not queue.empty():
		get_job_path = queue.get()

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



spatial_limits_device_um = [
	[ -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um ],
	[ -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um ],
	[ device_vertical_minimum_um, device_vertical_maximum_um ]
]

interpolated_size = [ device_voxels_lateral, device_voxels_lateral, device_voxels_vertical ]



#
# Run the optimization
#
for epoch in range(start_epoch, num_epochs):
	bayer_filter.update_filters(epoch)

	for iteration in range(0, num_iterations_per_epoch):
		iteration_start_time = time.time()
		print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

		fdtd_hook.load( projects_directory_location + "/optimization.fsp" )

		job_names = {}

		# fdtd_hook.switchtolayout()
		lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

		cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )
		cur_index = np.sqrt( cur_permittivity )

		# bordered_replicated_index = border_and_replicate_device( cur_index, device_border_voxels, min_device_index )

		fdtd_hook.select( design_import[ 'name' ] )
		fdtd_hook.importnk2(cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)


		num_fdtd_jobs = 0

		forward_e_fields_job_queued = {}
		#
		# Step 1: Get all the simulations we need queued up and run in parallel and then we will
		# put all the data together later.
		#
		for xy_idx in range(0, 2):
			get_symmetry_fields = forward_e_fields_job_queued.get( forward_symmetry[ xy_idx ], None )
			if get_symmetry_fields is None:
				disable_all_sources()
				fdtd_hook.select( forward_sources[xy_idx]['name'] )
				fdtd_hook.set( 'enabled', 1 )

				job_name = 'forward_job_' + str( xy_idx ) + '.fsp'
				# job_name_review = 'forward_job_' + str( xy_idx ) + '_review.fsp'
				# job_names[ ( 'forward', xy_idx ) ] = job_name

				fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
				# fdtd_hook.save( projects_directory_location + "/" + job_name )
				# fdtd_hook.save( projects_directory_location + "/" + job_name_review )

				# fdtd_hook.addjob( job_name )
				job_names[ ( 'forward', xy_idx ) ] = add_job( job_name, jobs_queue )
				num_fdtd_jobs += 1
				forward_e_fields_job_queued[xy_names[xy_idx]] = 1


		adjoint_e_fields_job_queued = [ {} for i in range( 0, num_adjoint_sources ) ]

		for adj_src_idx in range(0, num_adjoint_sources):
			adjoint_symmetry_loc = adjoint_symmetry_location[ adj_src_idx ]

			for xy_idx in range(0, 2):
				get_adj_symmetry_fields = adjoint_e_fields_job_queued[ adjoint_symmetry_loc ].get( adjoint_symmetry_pol[ xy_idx ], None )

				if get_adj_symmetry_fields is None:
					disable_all_sources()
					fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
					fdtd_hook.set( 'enabled', 1 )

					job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '.fsp'
					# job_name_review = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '_review.fsp'
					# job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] = job_name

					fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
					# fdtd_hook.save( projects_directory_location + "/" + job_name )
					# fdtd_hook.save( projects_directory_location + "/" + job_name_review )
					# fdtd_hook.addjob( job_name )
					job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] = add_job( job_name, jobs_queue )
					num_fdtd_jobs += 1

					adjoint_e_fields_job_queued[ adj_src_idx ][ xy_names[ xy_idx ] ] = 1




		#
		# Step 2: Now that all the jobs are queued up, let's get them all run!
		#
		start = time.time()
		# fdtd_hook.runjobs()
		run_jobs( jobs_queue )
		elapsed = time.time() - start

		log_file = open( projects_directory_location + "/log.txt", 'a' )
		log_file.write( "To run all " + str( num_fdtd_jobs ) + " jobs in parallel took " + str( elapsed ) + " seconds\n\n" )
		log_file.close()

		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )



















		forward_e_fields = {}
		focal_data = [ {} for i in range( 0, num_adjoint_sources ) ]
		#
		# Step 3: Get all the forward data from the simulations
		#
		for xy_idx in range(0, 2):
			get_symmetry_fields = forward_e_fields.get( forward_symmetry[ xy_idx ], None )
			if get_symmetry_fields is not None:
				# fields are organized as [ pol, wavelength, z, y, x ]
				# fields are organized as [ pol, x, y, z, wavelength ]
				# get_symmetry_fields = np.swapaxes( get_symmetry_fields, 3, 4 )
				get_symmetry_fields = np.swapaxes( get_symmetry_fields, 1, 2 )
				get_symmetry_fields_ypol = ( get_symmetry_fields[ 1 ] ).copy()
				get_symmetry_fields[ 1 ] = get_symmetry_fields[ 0 ]
				get_symmetry_fields[ 0 ] = get_symmetry_fields_ypol

				forward_e_fields[ xy_names[ xy_idx ] ] = get_symmetry_fields

				for adj_src_idx in range( 0, num_adjoint_sources ):
					adjoint_symmetry_loc = adjoint_symmetry_location[ adj_src_idx ]
					get_symmetry_focal = focal_data[ adjoint_symmetry_loc ][ forward_symmetry[ xy_idx ] ]
					get_symmetry_focal_ypol = ( get_symmetry_focal[ 1 ] ).copy()
					get_symmetry_focal[ 1 ] = get_symmetry_focal[ 0 ]
					get_symmetry_focal[ 0 ] = get_symmetry_focal_ypol

					focal_data[ adj_src_idx ][ xy_names[ xy_idx ] ] = get_symmetry_focal

			else:
				fdtd_hook.load( job_names[ ( 'forward', xy_idx ) ] )

				# forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')
				# forward_e_fields[xy_names[xy_idx]] = get_efield( design_efield_monitor['name' ] )
				forward_e_fields[xy_names[xy_idx]] = get_efield_interpolated(
					design_efield_monitor['name' ],
					spatial_limits_device_um,
					interpolated_size
				)


				for adj_src_idx in range(0, num_adjoint_sources):
					# pull_focal_data = get_complex_monitor_data( focal_monitors[ adj_src_idx ][ 'name' ], 'E' )
					pull_focal_data = get_efield( focal_monitors[ adj_src_idx ][ 'name' ] )
					# pull_focal_data = pull_focal_data[ :, :, 0, 0, 0 ]
					pull_focal_data = pull_focal_data[ :, 0, 0, 0, : ]
					focal_data[ adj_src_idx ][ xy_names[ xy_idx ] ] = pull_focal_data


		#
		# Step 4: Compute the figure of merit
		#
		figure_of_merit_per_focal_spot = []
		figure_of_merit_by_focal_spot_by_wavelength = np.zeros( ( num_focal_spots, num_points_per_band ) )
		for focal_idx in range(0, num_focal_spots):
			compute_fom = 0

			polarizations = polarizations_focal_plane_map[focal_idx]

			for polarization_idx in range(0, len(polarizations)):
				get_focal_data = focal_data[ focal_idx ][ polarizations[ polarization_idx ] ]

				max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[focal_idx][0] : spectral_focal_plane_map[focal_idx][1] : 1]
				total_weighting = weight_focal_plane_map[focal_idx] / max_intensity_weighting

				for spectral_idx in range(0, total_weighting.shape[0]):
					get_fom_by_wl = np.sum(
						np.abs(get_focal_data[:, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 *
							total_weighting[spectral_idx]
					)

					figure_of_merit_by_focal_spot_by_wavelength[ focal_idx, spectral_idx ] += get_fom_by_wl

					compute_fom += get_fom_by_wl

			figure_of_merit_per_focal_spot.append(compute_fom)

		figure_of_merit_per_focal_spot = np.array(figure_of_merit_per_focal_spot)

		flatten_fom_by_focal_wl = figure_of_merit_by_focal_spot_by_wavelength.flatten()
		performance_weighting_with_wl = ( 2. / ( num_focal_spots * num_points_per_band ) ) - flatten_fom_by_focal_wl**2 / np.sum( flatten_fom_by_focal_wl**2 )
		performance_weighting_with_wl = np.maximum( performance_weighting_with_wl, 0 )
		performance_weighting_with_wl /= np.sum( performance_weighting_with_wl )
		performance_weighting_with_wl = np.reshape( performance_weighting_with_wl, figure_of_merit_by_focal_spot_by_wavelength.shape )

		# We are currenty only doing weighting per focal spot and not doing anything that is wavelength dependent in each focal spot.. just a note
		# performance_weighting = (2. / num_focal_spots) - figure_of_merit_per_focal_spot**2 / np.sum(figure_of_merit_per_focal_spot**2)
		# performance_weighting -= np.min(performance_weighting)
		# performance_weighting = np.maximum( performance_weighting, 0 )
		# performance_weighting /= np.sum( performance_weighting )

		figure_of_merit = np.sum(figure_of_merit_per_focal_spot)
		figure_of_merit_evolution[epoch, iteration] = figure_of_merit

		np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)

		#
		# Step 5: Get all the adjoint optimization data for both x- and y-polarized adjoint sources and use the results to compute the
		# gradients for x- and y-polarized forward sources.
		#
		cur_permittivity_shape = cur_permittivity.shape
		xy_polarized_gradients = [
			np.zeros((device_voxels_lateral, device_voxels_lateral, device_voxels_vertical), dtype=np.complex),
			np.zeros((device_voxels_lateral, device_voxels_lateral, device_voxels_vertical), dtype=np.complex) ]

		adjoint_e_fields = [ {} for i in range( 0, num_adjoint_sources ) ]

		for adj_src_idx in range(0, num_adjoint_sources):
			polarizations = polarizations_focal_plane_map[adj_src_idx]
			spectral_indices = spectral_focal_plane_map[adj_src_idx]

			adjoint_symmetry_loc = adjoint_symmetry_location[ adj_src_idx ]

			# gradient_performance_weight = performance_weighting[adj_src_idx]
			gradient_performance_weight = performance_weighting_with_wl[adj_src_idx]

			for xy_idx in range(0, 2):
				get_adj_symmetry_fields = adjoint_e_fields[ adjoint_symmetry_loc ].get( adjoint_symmetry_pol[ xy_idx ], None )

				if get_adj_symmetry_fields is not None:
					# fields are organized as [ pol, wavelength, z, y, x ]
					# fields are organized as [ pol, x, y, z, wavelength ]
					# get_adj_symmetry_fields = np.swapaxes( get_adj_symmetry_fields, 3, 4 )
					get_adj_symmetry_fields = np.swapaxes( get_adj_symmetry_fields, 1, 2 )
					get_adj_symmetry_fields_ypol = ( get_adj_symmetry_fields[ 1 ] ).copy()
					get_adj_symmetry_fields[ 1 ] = get_adj_symmetry_fields[ 0 ]
					get_adj_symmetry_fields[ 0 ] = get_adj_symmetry_fields_ypol

					adjoint_e_fields[ adj_src_idx ][ xy_names[ xy_idx ] ] = get_adj_symmetry_fields

				else:
					fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] )

					# adjoint_e_fields[ adj_src_idx ][ xy_names[ xy_idx ] ] = get_complex_monitor_data( design_efield_monitor['name'] ,'E' )
					# adjoint_e_fields[ adj_src_idx ][ xy_names[ xy_idx ] ] = get_efield( design_efield_monitor['name'] )
					adjoint_e_fields[ adj_src_idx ][ xy_names[ xy_idx ] ] = get_efield_interpolated(
						design_efield_monitor['name'],
						spatial_limits_device_um,
						interpolated_size )

			for pol_idx in range(0, len(polarizations)):
				pol_name = polarizations[pol_idx]
				# get_focal_data = focal_data[pol_name]
				get_focal_data = focal_data[ adj_src_idx ][ polarizations[ pol_idx ] ]

				pol_name_to_idx = polarization_name_to_idx[pol_name]

				for xy_idx in range(0, 2):
					source_weight = np.conj(
						get_focal_data[xy_idx, spectral_indices[0] : spectral_indices[1] : 1])

					max_intensity_weighting = max_intensity_by_wavelength[spectral_indices[0] : spectral_indices[1] : 1]
					total_weighting = weight_focal_plane_map[focal_idx] / max_intensity_weighting

					for spectral_idx in range(0, source_weight.shape[0]):
						# xy_polarized_gradients[pol_name_to_idx] += np.sum(
						# 	(source_weight[spectral_idx] * gradient_performance_weight * total_weighting[spectral_idx]) *
						# 	adjoint_e_fields[ adj_src_idx ][ xy_names[ xy_idx ] ][:, spectral_indices[0] + spectral_idx, :, :, :] *
						# 	forward_e_fields[ pol_name ][ :, spectral_indices[0] + spectral_idx, :, :, : ],
						# 	axis=0)

						xy_polarized_gradients[pol_name_to_idx] += np.sum(
							(source_weight[spectral_idx] * gradient_performance_weight[ spectral_idx ] * total_weighting[spectral_idx]) *
							adjoint_e_fields[ adj_src_idx ][ xy_names[ xy_idx ] ][ :, :, :, :, spectral_indices[0] + spectral_idx ] *
							forward_e_fields[ pol_name ][ :, :, :, :, spectral_indices[0] + spectral_idx ],
							axis=0 )


		# fdtd_hook.switchtolayout()
		lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
		shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/optimization_start_epoch_" + str( epoch ) + ".fsp" )

		#
		# Step 4: Step the design variable.
		#
		# device_gradient_simulation_mesh = 2 * np.real( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
		device_gradient = 2 * np.real( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
		# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
		# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
		# device_gradient = np.swapaxes(device_gradient, 0, 2)

		# device_gradient = reinterpolate.reinterpolate( device_gradient_simulation_mesh, cur_permittivity.shape )

		design_gradient = bayer_filter.backpropagate(device_gradient)

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

		# step_size = step_size_start

		step_size_density = design_change_start_epoch + ( iteration / ( num_iterations_per_epoch - 1 ) ) * ( design_change_end_epoch - design_change_start_epoch )
		step_size_density /= np.max( np.abs( design_gradient ) )

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
		#
		# todo: fix this in other files! the step already does the backpropagation so you shouldn't
		# pass it an already backpropagated gradient!  Sloppy, these files need some TLC and cleanup!
		#
		# enforce_binarization = False
		# if epoch >= binarization_start_epoch:
		# 	enforce_binarization = True
		device_gradient = np.flip( device_gradient, axis=2 )
		bayer_filter.step(-device_gradient, step_size_density)#, enforce_binarization, projects_directory_location)
		cur_design_variable = bayer_filter.get_design_variable()

		average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
		max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

		step_size_evolution[epoch][iteration] = step_size_density
		average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
		max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

		np.save(projects_directory_location + '/device_gradient.npy', device_gradient)
		np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
		np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
		np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
		np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
		np.save(projects_directory_location + "/cur_design_variable.npy", cur_design_variable)
		np.save(projects_directory_location + "/cur_design_variable_" + str( epoch ) + ".npy", cur_design_variable)

		iteration_elapsed_time = time.time() - iteration_start_time
		log_file = open( projects_directory_location + "/log.txt", 'a' )
		log_file.write( "To do one iteration took " + str( iteration_elapsed_time ) + " seconds = " + str( iteration_elapsed_time / 60. ) + " minutes\n\n" )
		log_file.close()


	fdtd_hook.switchtolayout()
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
	fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
	shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/optimization_end_epoch_" + str( epoch ) + ".fsp" )


