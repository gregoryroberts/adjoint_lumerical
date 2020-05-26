import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from SingleLayerIRCircularSplitterParameters import *
import SingleLayerIRCircularSplitterDevice
from SingleLayerLSF import *

import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )

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
# num_nodes_to_use = 3
# num_cpus_per_node = 8
# slurm_list = get_slurm_node_list()
# configure_resources_for_cluster( fdtd_hook, slurm_list, N_resources=num_nodes_to_use, N_threads_per_resource=num_cpus_per_node )

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
# projects_directory_location += "/" + project_name

projects_directory_location = "/central/groups/Faraon_Computing/projects" 
projects_init_design_directory = projects_directory_location + "/" + project_name + '_parallel'

projects_directory_location += "/" + project_name + '_parallel_lsf_fd_v2'

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/SingleLayerIRCircularSplitterParameters.py", projects_directory_location + "/SingleLayerIRCircularSplitterParameters.py")

#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['x span'] = fdtd_region_width_um * 1e-6
fdtd['y span'] = fdtd_region_height_um * 1e-6
fdtd['z max'] = fdtd_region_maximum_vertical_um * 1e-6
fdtd['z min'] = fdtd_region_minimum_vertical_um * 1e-6
fdtd['mesh type'] = 'uniform'
fdtd['define x mesh by'] = 'number of mesh cells'
fdtd['define y mesh by'] = 'number of mesh cells'
fdtd['define z mesh by'] = 'number of mesh cells'
fdtd['mesh cells x'] = fdtd_region_minimum_width_voxels
fdtd['mesh cells y'] = fdtd_region_minimum_height_voxels
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
	forward_src['x span'] = lateral_aperture_width_um * 1e-6
	forward_src['y span'] = lateral_aperture_height_um * 1e-6
	forward_src['z max'] = src_maximum_vertical_um * 1e-6
	forward_src['z min'] = src_minimum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_src_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_src_max_um * 1e-6

	forward_sources.append( forward_src )

#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = []

for adj_src_idx in range( 0, num_adjoint_sources ):
	adjoint_sources.append( [ ] )
	for xy_idx in range(0, 2):
		adj_src = fdtd_hook.adddipole()
		adj_src['name'] = 'adj_src_' + str(adj_src_idx) + xy_names[xy_idx]
		adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
		adj_src['y'] = adjoint_y_positions_um[adj_src_idx] * 1e-6
		adj_src['z'] = adjoint_vertical_um * 1e-6
		adj_src['theta'] = 90
		adj_src['phi'] = xy_phi_rotations[xy_idx]
		adj_src['wavelength start'] = lambda_src_min_um * 1e-6
		adj_src['wavelength stop'] = lambda_src_max_um * 1e-6

		adjoint_sources[ adj_src_idx ].append( adj_src )

#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
# todo: We may want to be doing the other type of field monitor to get interpolated fields instead of
# exact positions
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['monitor type'] = '3D'
design_efield_monitor['x span'] = device_width_um * 1e-6
design_efield_monitor['y span'] = device_height_um * 1e-6
design_efield_monitor['z max'] = device_vertical_maximum_um * 1e-6
design_efield_monitor['z min'] = device_vertical_minimum_um * 1e-6
design_efield_monitor['override global monitor settings'] = 1
design_efield_monitor['use wavelength spacing'] = 1
design_efield_monitor['use source limits'] = 0
design_efield_monitor['minimum wavelength'] = lambda_min_um * 1e-6
design_efield_monitor['maximum wavelength'] = lambda_max_um * 1e-6
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
	focal_monitor['use source limits'] = 0
	focal_monitor['minimum wavelength'] = lambda_min_um * 1e-6
	focal_monitor['maximum wavelength'] = lambda_max_um * 1e-6
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)

transmission_focal_monitors = []

for adj_src in range(0, num_adjoint_sources):
	transmission_focal_monitor = fdtd_hook.addpower()
	transmission_focal_monitor['name'] = 'transmission_focal_monitor_' + str(adj_src)
	transmission_focal_monitor['monitor type'] = '2D Z-Normal'
	transmission_focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_focal_monitor['x span'] = 0.5 * device_width_um * 1e-6
	transmission_focal_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
	transmission_focal_monitor['y span'] = device_height_um * 1e-6
	transmission_focal_monitor['z'] = adjoint_vertical_um * 1e-6
	transmission_focal_monitor['override global monitor settings'] = 1
	transmission_focal_monitor['use wavelength spacing'] = 1
	transmission_focal_monitor['use source limits'] = 0
	transmission_focal_monitor['minimum wavelength'] = lambda_min_um * 1e-6
	transmission_focal_monitor['maximum wavelength'] = lambda_max_um * 1e-6
	transmission_focal_monitor['frequency points'] = num_eval_frequency_points
	transmission_focal_monitor.enabled = 0

	transmission_focal_monitors.append(transmission_focal_monitor)

transmission_focal = fdtd_hook.addpower()
transmission_focal['name'] = 'transmission_focal'
transmission_focal['monitor type'] = '2D Z-Normal'
transmission_focal['x'] = 0 * 1e-6
transmission_focal['x span'] = device_width_um * 1e-6
transmission_focal['y'] = 0 * 1e-6
transmission_focal['y span'] = device_height_um * 1e-6
transmission_focal['z'] = adjoint_vertical_um * 1e-6
transmission_focal['override global monitor settings'] = 1
transmission_focal['use wavelength spacing'] = 1
transmission_focal['use source limits'] = 0
transmission_focal['minimum wavelength'] = lambda_min_um * 1e-6
transmission_focal['maximum wavelength'] = lambda_max_um * 1e-6
transmission_focal['frequency points'] = num_eval_frequency_points
transmission_focal.enabled = 0


#
# Add SiO2 at the top
#
sio2_top = fdtd_hook.addrect()
sio2_top['name'] = 'sio2_top'
sio2_top['x span'] = fdtd_region_width_um * 1e-6
sio2_top['y span'] = fdtd_region_height_um * 1e-6
sio2_top['z min'] = device_vertical_maximum_um * 1e-6
sio2_top['z max'] = fdtd_region_maximum_vertical_um * 1e-6
sio2_top['index'] = index_sio2


#
# Add device region and create device permittivity
#
design_import = fdtd_hook.addimport()
design_import['name'] = 'design_import'
design_import['x span'] = device_width_um * 1e-6
design_import['y span'] = device_height_um * 1e-6
design_import['z max'] = device_vertical_maximum_um * 1e-6
design_import['z min'] = device_vertical_minimum_um * 1e-6


bayer_filter_size_voxels = np.array([device_width_voxels, device_height_voxels, device_voxels_vertical])
bayer_filter = SingleLayerIRCircularSplitterDevice.SingleLayerIRCircularSplitterDevice(
	bayer_filter_size_voxels,
	[min_device_permittivity, max_device_permittivity],
	init_permittivity_0_1_scale,
	max_binarize_movement,
	desired_binarize_change)


bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_width_um, 0.5 * device_width_um, device_width_voxels)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_height_um, 0.5 * device_height_um, device_height_voxels)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

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



#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_by_focal_spot_by_wavelength_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_focal_spots, 3, num_design_frequency_points))

step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = 0.001

bayer_filter.update_filters(start_epoch - 1)
bayer_filter.set_design_variable( np.load( projects_init_design_directory + '/cur_design_variable_' + str( start_epoch - 1 ) + '.npy' ) )
bayer_filter_permittivity = bayer_filter.get_permittivity()

# Let's ensure the design has the given symmetry
bayer_filter_permittivity = 0.5 * ( bayer_filter_permittivity + np.flip( bayer_filter_permittivity, axis=0 ) )

permittivity_to_density = ( bayer_filter_permittivity - min_device_permittivity ) / ( max_device_permittivity - min_device_permittivity )

design_variable_reload = np.real( np.flip( permittivity_to_density, axis=2 ) )

####
rbf_sigma = 1
rbf_eval_cutoff = 5

level_set_alpha = read_density_into_alpha( design_variable_reload )
level_set_alpha = level_set_alpha[ :, :, int( 0.5 * level_set_alpha.shape[ 2 ] ) ]
level_set_function = compute_lsf( level_set_alpha, rbf_sigma, rbf_eval_cutoff )

np.save(projects_directory_location + "/init_alpha.npy", level_set_alpha)
np.save(projects_directory_location + "/init_level_set_function.npy", level_set_function)

binary_design = read_lsf_into_density( level_set_function )
np.save(projects_directory_location + "/init_binary_design.npy", binary_design)
device_permittivity = np.ones( ( device_width_voxels, device_height_voxels, device_voxels_vertical ) )
for voxel_vertical in range( 0, device_voxels_vertical ):
	device_permittivity[ :, :, voxel_vertical ] = min_device_permittivity + ( max_device_permittivity - min_device_permittivity ) * binary_design
####

fdtd_hook.save( projects_directory_location + "/optimization.fsp" )


#
# Figure of merit selection
#
choose_focal_spot = 0
choose_wl = int( 0.5 * num_design_frequency_points )


#
# Get starting figure of merit
#

def compute_figure_of_merit( binary_structure, level_set_alpha, level_set_function, compute_gradient=False ):
	job_names = {}

	fdtd_hook.load( projects_directory_location + "/optimization.fsp" )

	fdtd_hook.switchtolayout()
	for voxel_vertical in range( 0, device_voxels_vertical ):
		device_permittivity[ :, :, voxel_vertical ] = min_device_permittivity + ( max_device_permittivity - min_device_permittivity ) * binary_structure
	fdtd_hook.select("design_import")
	fdtd_hook.importnk2(np.sqrt(device_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)

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

	run_jobs( jobs_queue )

	fdtd_hook.save( projects_directory_location + "/optimization.fsp" )

	for xy_idx in range(0, 2):
		fdtd_hook.load( job_names[ ( 'forward', xy_idx ) ] )

		forward_e_fields[xy_names[xy_idx]] = get_efield(design_efield_monitor['name'])

		focal_data[xy_names[xy_idx]] = []
		for focal_idx in range( 0, num_focal_spots ):
			focal_monitor_data = get_efield( focal_monitors[ focal_idx ][ 'name' ])

			if xy_idx == 0:
				Qxx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]
				Qxy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
			else:
				Qyy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
				Qyx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]


	analyzer_vector = jones_polarizations[ choose_focal_spot ]

	create_forward_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ choose_focal_spot, choose_wl ] + analyzer_vector[ 1 ] * Qyx[ choose_focal_spot, choose_wl ]
	create_forward_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ choose_focal_spot, choose_wl ] + analyzer_vector[ 1 ] * Qyy[ choose_focal_spot, choose_wl ]

	create_reflected_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ 1 - choose_focal_spot, choose_wl ] + analyzer_vector[ 1 ] * Qyx[ 1 - choose_focal_spot, choose_wl ]
	create_reflected_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ 1 - choose_focal_spot, choose_wl ] + analyzer_vector[ 1 ] * Qyy[ 1 - choose_focal_spot, choose_wl ]

	parallel_intensity = np.abs( create_forward_parallel_response_x )**2 + np.abs( create_forward_parallel_response_y )**2
	parallel_fom = parallel_intensity / max_intensity_by_wavelength[ choose_wl ]

	figure_of_merit = parallel_fom


	alpha_grad = None

	if compute_figure_of_merit:

		for xy_idx in range(0, 2):
			disable_all_sources()
			fdtd_hook.select( adjoint_sources[choose_focal_spot][xy_idx]['name'] )
			fdtd_hook.set( 'enabled', 1 )

			job_name = 'adjoint_job_' + str( choose_focal_spot ) + '_' + str( xy_idx ) + '.fsp'

			fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
			job_names[ ( 'adjoint', choose_focal_spot, xy_idx ) ] = add_job( job_name, jobs_queue )

		run_jobs( jobs_queue )


		adjoint_ex_fields = []
		adjoint_ey_fields = []
		for xy_idx in range(0, 2):
			job_name = 'adjoint_job_' + str( choose_focal_spot ) + '_' + str( xy_idx ) + '.fsp'
			fdtd_hook.load( job_name )

			if xy_idx == 0:
				adjoint_ex_fields.append(
					get_efield(design_efield_monitor['name']))
			else:
				adjoint_ey_fields.append(
					get_efield(design_efield_monitor['name']))

		maximization_gradient = np.zeros( forward_e_fields[ 'x' ][ 0, :, :, :, 0 ].shape )
		analyzer_vector = jones_polarizations[ choose_focal_spot ]

		create_forward_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ choose_focal_spot, : ] + analyzer_vector[ 1 ] * Qyx[ choose_focal_spot, : ]
		create_forward_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ choose_focal_spot, : ] + analyzer_vector[ 1 ] * Qyy[ choose_focal_spot, : ]

		parallel_intensity = np.abs( create_forward_parallel_response_x )**2 + np.abs( create_forward_parallel_response_y )**2
		reflected_parallel_intensity = np.abs( create_reflected_parallel_response_x )**2 + np.abs( create_reflected_parallel_response_y )**2

		parallel_fom_by_wavelength = parallel_intensity / max_intensity_by_wavelength
		reflected_fom_by_wavelength = np.maximum( 0, 1 - ( reflected_parallel_intensity / max_intensity_by_wavelength ) )

		create_forward_e_fields = analyzer_vector[ 0 ] * forward_e_fields[ 'x' ] + analyzer_vector[ 1 ] * forward_e_fields[ 'y' ]

		alpha_grad = np.zeros( level_set_alpha.shape )

		alpha_grad += 2 * alpha_perturbations(
			create_forward_e_fields[ :, :, :, :, choose_wl ],
			np.conj( create_forward_parallel_response_x[ choose_wl ] ) * adjoint_ex_fields[ choose_focal_spot ][ :, :, :, :, choose_wl ],
			level_set_function,
			level_set_alpha,
			rbf_sigma,
			rbf_eval_cutoff,
			max_device_permittivity,
			min_device_permittivity
		)

		alpha_grad += 2 * alpha_perturbations(
			create_forward_e_fields[ :, :, :, :, choose_wl ],
			np.conj( create_forward_parallel_response_y[ choose_wl ] ) * adjoint_ey_fields[ choose_focal_spot ][ :, :, :, :, choose_wl ],
			level_set_function,
			level_set_alpha,
			rbf_sigma,
			rbf_eval_cutoff,
			max_device_permittivity,
			min_device_permittivity
		)

		np.save( projects_directory_location + "/fwd_e_fields.npy", create_forward_e_fields[ :, :, :, :, choose_wl ] )
		np.save( projects_directory_location + "/conj_weighting.npy", np.conj( create_forward_parallel_response_y[ choose_wl ] ) )
		np.save( projects_directory_location + "/adj_ex_fields.npy", adjoint_ex_fields[ choose_focal_spot ][ :, :, :, :, choose_wl ] )
		np.save( projects_directory_location + "/adj_ey_fields.npy", adjoint_ey_fields[ choose_focal_spot ][ :, :, :, :, choose_wl ] )
		np.save( projects_directory_location + "/grad_level_set_function.npy", level_set_function )
		np.save( projects_directory_location + "/grad_level_set_alpha.npy", level_set_alpha )
		np.save( projects_directory_location + "/rbf_sigma.npy", np.array( rbf_sigma ) )
		np.save( projects_directory_location + "/rbf_eval_cutoff.npy", np.array( rbf_eval_cutoff ) )
		np.save( projects_directory_location + "/min_perm.npy", np.array( min_device_permittivity ) )
		np.save( projects_directory_location + "/max_perm.npy", np.array( max_device_permittivity ) )

	return figure_of_merit, alpha_grad



init_fom, init_grad = compute_figure_of_merit( binary_design, level_set_alpha, level_set_function, compute_gradient=True )

np.save( projects_directory_location + '/adjoint_grad_alpha.npy', init_grad )


fd_x = int( 0.5 * device_width_voxels )
alpha_spread = 3
num_fd_pts = 20
fd_y = np.arange( alpha_spread, alpha_spread + num_fd_pts )
finite_diff_alpha = np.zeros( len( fd_y ) )
fd_delta = 0.05

for fd_y_idx in range( 0, len( fd_y ) ):

	y_idx = fd_y[ fd_y_idx ]

	fd_alpha = level_set_alpha.copy()
	fd_alpha[ ( fd_x - alpha_spread ) : ( fd_x + alpha_spread + 1 ), ( y_idx - alpha_spread ) : ( y_idx + alpha_spread + 1 ) ] += fd_delta


	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Working on fd idx = " + str( fd_y_idx ) + " out of " + str( num_fd_pts ) + "\n" )
	log_file.close()


	level_set_function_step = compute_lsf( fd_alpha, rbf_sigma, rbf_eval_cutoff )
	binary_design_step = read_lsf_into_density( level_set_function_step )

	np.save( projects_directory_location + "/binary_step_" + str( fd_y_idx ) + ".npy", binary_design_step )

	fom_up, none_grad = compute_figure_of_merit( binary_design_step, fd_alpha, level_set_function_step, compute_gradient=False )

	finite_diff_alpha[ fd_y_idx ] = ( fom_up - init_fom ) / fd_delta

	np.save( projects_directory_location + "/finite_diff_alpha.npy", finite_diff_alpha )
