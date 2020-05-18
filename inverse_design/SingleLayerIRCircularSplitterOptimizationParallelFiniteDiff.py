import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from SingleLayerIRCircularSplitterParameters import *
import SingleLayerIRCircularSplitterDevice

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
projects_directory_location += "/" + project_name + '_parallel_finite_difference'

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


fdtd_hook.save( projects_directory_location + "/optimization.fsp" )

job_names = {}

fdtd_hook.load( projects_directory_location + "/optimization.fsp" )

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
	# job_name_review = 'forward_job_' + str( xy_idx ) + '_review.fsp'
	# job_names[ ( 'forward', xy_idx ) ] = job_name

	fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
	# fdtd_hook.save( projects_directory_location + "/" + job_name )
	# fdtd_hook.save( projects_directory_location + "/" + job_name_review )
	job_names[ ( 'forward', xy_idx ) ] = add_job( job_name, jobs_queue )

	# fdtd_hook.addjob( job_name )


for adj_src_idx in range(0, num_adjoint_sources):

	for xy_idx in range(0, 2):
		disable_all_sources()
		fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
		fdtd_hook.set( 'enabled', 1 )

		job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '.fsp'
		# job_name_review = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '_review.fsp'
		# job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] = job_name

		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
		# fdtd_hook.save( projects_directory_location + "/" + job_name )
		# fdtd_hook.save( projects_directory_location + "/" + job_name_review )
		job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] = add_job( job_name, jobs_queue )
		
		# fdtd_hook.addjob( job_name )


run_jobs( jobs_queue )
# fdtd_hook.runjobs()

fdtd_hook.save( projects_directory_location + "/optimization.fsp" )


for xy_idx in range(0, 2):
	fdtd_hook.load( job_names[ ( 'forward', xy_idx ) ] )

	# forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')
	forward_e_fields[xy_names[xy_idx]] = get_efield(design_efield_monitor['name'])

	focal_data[xy_names[xy_idx]] = []
	for focal_idx in range( 0, num_focal_spots ):
		# focal_monitor_data = get_complex_monitor_data( focal_monitors[ focal_idx ][ 'name' ], 'E' )
		focal_monitor_data = get_efield( focal_monitors[ focal_idx ][ 'name' ])

		if xy_idx == 0:
			Qxx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]
			Qxy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
		else:
			Qyy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
			Qyx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]


fom_by_focal_spot_by_wavelength = np.zeros( ( num_focal_spots, num_design_frequency_points ) )
for focal_idx in range( 0, num_focal_spots ):
	analyzer_vector = jones_polarizations[ focal_idx ]

	create_forward_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ focal_idx, : ]
	create_forward_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ focal_idx, : ]

	create_reflected_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ 1 - focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ 1 - focal_idx, : ]
	create_reflected_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ 1 - focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ 1 - focal_idx, : ]

	parallel_intensity = np.abs( create_forward_parallel_response_x )**2 + np.abs( create_forward_parallel_response_y )**2
	reflected_parallel_intensity = np.abs( create_reflected_parallel_response_x )**2 + np.abs( create_reflected_parallel_response_y )**2

	parallel_fom_by_wavelength = parallel_intensity / max_intensity_by_wavelength
	reflected_intensity_by_wavelength = reflected_parallel_intensity / max_intensity_by_wavelength
	reflected_fom_by_wavelength = 1 - ( reflected_parallel_intensity / max_intensity_by_wavelength )

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( str( parallel_fom_by_wavelength ) + "\n" )
	log_file.write( str( reflected_fom_by_wavelength ) + "\n\n" )
	log_file.write( str( reflected_intensity_by_wavelength ) + "\n\n" )
	log_file.close()

	reflected_fom_by_wavelength = np.maximum( 0, reflected_fom_by_wavelength )

	fom_by_wavelength = parallel_fom_by_wavelength * reflected_fom_by_wavelength

	fom_by_focal_spot_by_wavelength[ focal_idx, : ] = fom_by_wavelength

all_fom = fom_by_focal_spot_by_wavelength.flatten()
fom_weightings = np.ones( all_fom.shape )
fom_weightings /= np.sum( fom_weightings )
fom_start = np.mean( fom_weightings * all_fom )
fom_weightings = np.reshape( fom_weightings, fom_by_focal_spot_by_wavelength.shape )

#
# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
# gradients for x- and y-polarized forward sources.
#
adjoint_ex_fields = []
adjoint_ey_fields = []
for adj_src_idx in range(0, num_adjoint_sources):
	for xy_idx in range(0, 2):
		job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '.fsp'
		fdtd_hook.load( job_name )

		if xy_idx == 0:
			adjoint_ex_fields.append(
				get_efield(design_efield_monitor['name']))
		else:
			adjoint_ey_fields.append(
				get_efield(design_efield_monitor['name']))

maximization_gradient = np.zeros( forward_e_fields[ 'x' ][ 0, :, :, :, 0 ].shape )
for focal_idx in range( 0, num_focal_spots ):
	analyzer_vector = jones_polarizations[ focal_idx ]

	create_forward_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ focal_idx, : ]
	create_forward_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ focal_idx, : ]

	create_reflected_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ 1 - focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ 1 - focal_idx, : ]
	create_reflected_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ 1 - focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ 1 - focal_idx, : ]

	parallel_intensity = np.abs( create_forward_parallel_response_x )**2 + np.abs( create_forward_parallel_response_y )**2
	reflected_parallel_intensity = np.abs( create_reflected_parallel_response_x )**2 + np.abs( create_reflected_parallel_response_y )**2

	parallel_fom_by_wavelength = parallel_intensity / max_intensity_by_wavelength
	reflected_fom_by_wavelength = np.maximum( 0, 1 - ( reflected_parallel_intensity / max_intensity_by_wavelength ) )

	create_forward_e_fields = analyzer_vector[ 0 ] * forward_e_fields[ 'x' ] + analyzer_vector[ 1 ] * forward_e_fields[ 'y' ]

	for wl_idx in range( 0, num_design_frequency_points ):
		maximization_gradient += 2 * np.sum(
			np.real(
				reflected_fom_by_wavelength[ wl_idx ] *
				fom_weightings[ focal_idx, wl_idx ] *
				np.conj( create_forward_parallel_response_x[ wl_idx ] ) *
				create_forward_e_fields[ :, :, :, :, wl_idx ] *
				adjoint_ex_fields[ focal_idx ][ :, :, :, :, wl_idx ]
			),
		axis=0 )

		maximization_gradient -= 2 * np.sum(
			np.real(
				parallel_fom_by_wavelength[ wl_idx ] *
				fom_weightings[ focal_idx, wl_idx ] *
				np.conj( create_reflected_parallel_response_x[ wl_idx ] ) *
				create_forward_e_fields[ :, :, :, :, wl_idx ] *
				adjoint_ex_fields[ 1 - focal_idx ][ :, :, :, :, wl_idx ]
			),
		axis=0 )

		maximization_gradient += 2 * np.sum(
			np.real(
				reflected_fom_by_wavelength[ wl_idx ] *
				fom_weightings[ focal_idx, wl_idx ] *
				np.conj( create_forward_parallel_response_y[ wl_idx ] ) *
				create_forward_e_fields[ :, :, :, :, wl_idx ] *
				adjoint_ey_fields[ focal_idx ][ :, :, :, :, wl_idx ]
			),
		axis=0 )

		maximization_gradient -= 2 * np.sum(
			np.real(
				parallel_fom_by_wavelength[ wl_idx ] *
				fom_weightings[ focal_idx, wl_idx ] *
				np.conj( create_reflected_parallel_response_y[ wl_idx ] ) *
				create_forward_e_fields[ :, :, :, :, wl_idx ] *
				adjoint_ey_fields[ 1 - focal_idx ][ :, :, :, :, wl_idx ]
			),
		axis=0 )

#
# Step 4: Step the design variable.
#
device_gradient = -maximization_gradient
device_gradient = np.flip( device_gradient, axis=2 )

np.save( projects_directory_location + '/device_gradient.npy', device_gradient )


fd_z = int( 0.5 * device_voxels_vertical )
fd_y_mid = int( 0.5 * device_height_voxels )
fd_x_mid = int( 0.5 * device_width_voxels )

num_fd_points = 20

fd_y_low = fd_y_mid - int( 0.5 * num_fd_points )
fd_x_low = fd_x_mid - num_fd_points

fd_y_range = np.linspace( fd_y_low, fd_y_low + num_fd_points, num_fd_points )
fd_x_range = np.linspace( fd_x_low, fd_x_low + 2 * num_fd_points, num_fd_points )

fd_points = np.zeros( ( num_fd_points, 3 ) )
for fd_pt in range( 0, num_fd_points ):
	fd_points[ fd_pt ] = [ int( fd_x_range[ fd_pt ] ), int( fd_y_range[ fd_pt ] ), fd_z ]

np.save( projects_directory_location + "/fd_points.npy", fd_points )

h = 1e-3

cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )

finite_difference = np.zeros( num_fd_points )

for fd_pt in range( 0, num_fd_points ):
	get_pt = fd_points[ fd_pt ]
	x_pt = int( get_pt[ 0 ] )
	y_pt = int( get_pt[ 1 ] )
	z_pt = int( get_pt[ 2 ] )

	next_permittivity = cur_permittivity.copy()
	next_permittivity[ x_pt, y_pt, z_pt ] += h

	fdtd_hook.load( projects_directory_location + "/optimization.fsp" )
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
	fdtd_hook.select("design_import")
	fdtd_hook.importnk2(np.sqrt(next_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)

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
		# job_name_review = 'forward_job_' + str( xy_idx ) + '_review.fsp'
		# job_names[ ( 'forward', xy_idx ) ] = job_name

		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
		# fdtd_hook.save( projects_directory_location + "/" + job_name )
		# fdtd_hook.save( projects_directory_location + "/" + job_name_review )
		job_names[ ( 'forward', xy_idx ) ] = add_job( job_name, jobs_queue )

		# fdtd_hook.addjob( job_name )

	run_jobs( jobs_queue )
	# fdtd_hook.runjobs()

	fdtd_hook.save( projects_directory_location + "/optimization.fsp" )


	for xy_idx in range(0, 2):
		fdtd_hook.load( job_names[ ( 'forward', xy_idx ) ] )

		# forward_e_fields[xy_names[xy_idx]] = get_complex_monitor_data(design_efield_monitor['name'], 'E')
		forward_e_fields[xy_names[xy_idx]] = get_efield(design_efield_monitor['name'])

		focal_data[xy_names[xy_idx]] = []
		for focal_idx in range( 0, num_focal_spots ):
			# focal_monitor_data = get_complex_monitor_data( focal_monitors[ focal_idx ][ 'name' ], 'E' )
			focal_monitor_data = get_efield( focal_monitors[ focal_idx ][ 'name' ])

			if xy_idx == 0:
				Qxx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]
				Qxy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
			else:
				Qyy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
				Qyx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]


	fom_by_focal_spot_by_wavelength = np.zeros( ( num_focal_spots, num_design_frequency_points ) )
	for focal_idx in range( 0, num_focal_spots ):
		analyzer_vector = jones_polarizations[ focal_idx ]

		create_forward_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ focal_idx, : ]
		create_forward_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ focal_idx, : ]

		create_reflected_parallel_response_x = analyzer_vector[ 0 ] * Qxx[ 1 - focal_idx, : ] + analyzer_vector[ 1 ] * Qyx[ 1 - focal_idx, : ]
		create_reflected_parallel_response_y = analyzer_vector[ 0 ] * Qxy[ 1 - focal_idx, : ] + analyzer_vector[ 1 ] * Qyy[ 1 - focal_idx, : ]

		parallel_intensity = np.abs( create_forward_parallel_response_x )**2 + np.abs( create_forward_parallel_response_y )**2
		reflected_parallel_intensity = np.abs( create_reflected_parallel_response_x )**2 + np.abs( create_reflected_parallel_response_y )**2

		parallel_fom_by_wavelength = parallel_intensity / max_intensity_by_wavelength
		reflected_intensity_by_wavelength = reflected_parallel_intensity / max_intensity_by_wavelength
		reflected_fom_by_wavelength = 1 - ( reflected_parallel_intensity / max_intensity_by_wavelength )

		reflected_fom_by_wavelength = np.maximum( 0, reflected_fom_by_wavelength )

		fom_by_wavelength = parallel_fom_by_wavelength * reflected_fom_by_wavelength

		fom_by_focal_spot_by_wavelength[ focal_idx, : ] = fom_by_wavelength

	all_fom = fom_by_focal_spot_by_wavelength.flatten()
	fom_weightings = np.ones( all_fom.shape )
	fom_weightings /= np.sum( fom_weightings )
	fom_fd = np.mean( fom_weightings * all_fom )

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( 'Cur fd for idx = ' + str( fd_pt ) + ' is ' + str( ( fom_fd - fom_start ) / h ) + "\n\n" )
	log_file.close()

	finite_difference[ fd_pt ] = ( fom_fd - fom_start ) / h

	np.save( projects_directory_location + '/finite_difference.npy', finite_difference )


