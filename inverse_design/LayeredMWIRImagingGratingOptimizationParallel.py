import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredMWIRImagingGratingParameters import *
import LayeredMWIRPolarizationBayerFilter


import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )


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
# projects_directory_location += "/" + project_name + "_parallel"

projects_directory_location = "/central/groups/Faraon_Computing/projects" 
projects_directory_location += "/" + project_name + '_parallel'

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredMWIRImagingGratingParameters.py", projects_directory_location + "/LayeredMWIRImagingGratingParameters.py")

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

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

fdtd['x min bc'] = 'Bloch'
fdtd['x max bc'] = 'Bloch'
fdtd['y min bc'] = 'Bloch'
fdtd['y max bc'] = 'Bloch'

fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
fdtd['background index'] = background_index

#
# General polarized source information
#
xy_phi_rotations = [0, 90]
xy_names = ['x', 'y']


# Add a TFSF plane wave forward source at normal incidence
#
forward_sources = []

log_file = open( projects_directory_location + "/log.txt", 'a' )
log_file.write( "Pre adding fwd source\n" )
log_file.close()


for fwd_src_idx in range( 0, num_forward_sources ):
	forward_src = fdtd_hook.addtfsf()
	forward_src['name'] = 'forward_src_' + str( fwd_src_idx )
	forward_src['polarization angle'] = 0
	forward_src['angle phi'] = forward_sources_phi_angles_degrees[ fwd_src_idx ]
	forward_src['angle theta'] = forward_sources_theta_angle_degrees
	forward_src['direction'] = 'Backward'
	forward_src['x span'] = lateral_aperture_um * 1e-6
	forward_src['y span'] = lateral_aperture_um * 1e-6
	forward_src['z'] = src_maximum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_max_um * 1e-6

	forward_sources.append( forward_src )

log_file = open( projects_directory_location + "/log.txt", 'a' )
log_file.write( "Post adding fwd source\n" )
log_file.close()


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

log_file = open( projects_directory_location + "/log.txt", 'a' )
log_file.write( "Post adding adj source\n" )
log_file.close()


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
	transmission_focal_monitor['frequency points'] = num_design_frequency_points
	transmission_focal_monitor['output Hx'] = 1
	transmission_focal_monitor['output Hy'] = 1
	transmission_focal_monitor['output Hz'] = 1
	transmission_focal_monitor.enabled = 1

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
transmission_focal['frequency points'] = num_design_frequency_points
transmission_focal['output Hx'] = 1
transmission_focal['output Hy'] = 1
transmission_focal['output Hz'] = 1
transmission_focal.enabled = 1


#
# Add IP-Dip at the bottom
#
ip_dip_bottom = fdtd_hook.addrect()
ip_dip_bottom['name'] = 'ip_dip_bottom'
ip_dip_bottom['x span'] = fdtd_region_size_lateral_um * 1e-6
ip_dip_bottom['y span'] = fdtd_region_size_lateral_um * 1e-6
ip_dip_bottom['z min'] = fdtd_region_minimum_vertical_um * 1e-6
ip_dip_bottom['z max'] = device_vertical_minimum_um * 1e-6
ip_dip_bottom['index'] = max_device_index

log_file = open( projects_directory_location + "/log.txt", 'a' )
log_file.write( "Post adding ip dip bottom\n" )
log_file.close()


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
bayer_filter = LayeredMWIRPolarizationBayerFilter.LayeredMWIRPolarizationBayerFilter(
	bayer_filter_size_voxels,
	[min_device_permittivity, max_device_permittivity],
	init_permittivity_0_1_scale,
	num_vertical_layers,
	topology_num_free_iterations_between_patches)

log_file = open( projects_directory_location + "/log.txt", 'a' )
log_file.write( "Post creating filter\n" )
log_file.close()


bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	for fwd_src_idx in range(0, num_forward_sources):
		fdtd_hook.select( forward_sources[fwd_src_idx]['name'] )
		fdtd_hook.set( 'enabled', 0 )

	for adj_src_idx in range(0, num_adjoint_sources):
		for xy_idx in range(0, 2):
			fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
			fdtd_hook.set( 'enabled', 0 )


def get_afield( monitor_name, field_indicator ):
	field_polariations = [ field_indicator + 'x', field_indicator + 'y', field_indicator + 'z' ]
	data_xfer_size_MB = 0

	start = time.time()

	field_pol_0 = fdtd_hook.getdata( monitor_name, field_polariations[ 0 ] )
	data_xfer_size_MB += field_pol_0.nbytes / ( 1024. * 1024. )

	total_field = np.zeros( [ len (field_polariations ) ] + list( field_pol_0.shape ), dtype=np.complex )
	total_field[ 0 ] = field_pol_0

	for pol_idx in range( 1, len( field_polariations ) ):
		field_pol = fdtd_hook.getdata( monitor_name, field_polariations[ pol_idx ] )
		data_xfer_size_MB += field_pol.nbytes / ( 1024. * 1024. )

		total_field[ pol_idx ] = field_pol

	elapsed = time.time() - start

	date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
	log_file.write( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )
	log_file.close()

	return total_field

def get_hfield( monitor_name ):
	return get_afield( monitor_name, 'H' )

def get_efield( monitor_name ):
	return get_afield( monitor_name, 'E' )

def compute_transmission( E_field_focal, H_field_focal, power_normalization_by_wl, width_spatial_units, height_spatial_units ):
	xpol = 0
	ypol = 1

	E_field_focal = np.squeeze( E_field_focal )
	H_field_focal = np.squeeze( H_field_focal )

	assert len( E_field_focal.shape ) == 4, "We expected a differently shaped E field for the transmission computation"
	assert len( E_field_focal.shape ) == 4, "We expected a differently shaped E field for the transmission computation"

	power_z = np.squeeze(
		np.real( E_field_focal[ ypol ] * np.conj( H_field_focal[ xpol ] ) - E_field_focal[ xpol ] * np.conj( H_field_focal[ ypol ] ) )
	)

	assert len( power_z.shape ) == 3, "We expected a differently shaped power matrix for the transmission computation"

	voxel_size_normalization = width_spatial_units * height_spatial_units / ( power_z.shape[ 0 ] * power_z.shape[ 1 ] )

	num_wl = E_field_focal.shape[ 3 ]
	transmission_by_wl = np.zeros( num_wl )

	for wl_idx in range( 0, num_wl ):
		transmission_by_wl[ wl_idx ] = np.sum( power_z[ :, :, wl_idx ] ) / power_normalization_by_wl[ wl_idx ]

	transmission_by_wl *= voxel_size_normalization

	return transmission_by_wl


#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_by_focal_spot_by_wavelength_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_focal_spots, num_design_frequency_points))

step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = 0.001

if start_epoch > 0:
	design_variable_reload = np.load( projects_directory_location + '/cur_design_variable_' + str( start_epoch - 1 ) + '.npy' )
	bayer_filter.set_design_variable( design_variable_reload )
	figure_of_merit_evolution = np.load( projects_directory_location + "/figure_of_merit.npy" )
	figure_of_merit_by_focal_spot_by_wavelength_evolution = np.load( projects_directory_location + "/figure_of_merit_by_focal_spot_by_type_by_wavelength.npy" )

jobs_queue = queue.Queue()

def add_job( job_name, queue_in ):
	full_name = projects_directory_location + "/" + job_name
	fdtd_hook.save( full_name )
	queue_in.put( full_name )

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Adding job " + str( job_name ) + "\n" )
	log_file.close()

	return full_name

def run_jobs( queue_in ):
	small_queue = queue.Queue()

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Running jobs\n" )
	log_file.close()

	while not queue_in.empty():
		for node_idx in range( 0, num_nodes_available ):
			if queue_in.qsize() > 0:
				small_queue.put( queue_in.get() )

		run_jobs_inner( small_queue )

def run_jobs_inner( queue_in ):
	processes = []
	# for job_idx in range( 0, len( queue_in ) ):
	# really should protect against number of available engines here
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


fdtd_hook.save( projects_directory_location + "/optimization.fsp" )


#
# Run the optimization
#
for epoch in range(start_epoch, num_epochs):
	bayer_filter.update_filters(epoch)

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Starting first epoch\n" )
	log_file.close()


	for iteration in range(0, num_iterations_per_epoch):
		print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

		job_names = {}

		fdtd_hook.switchtolayout()
		# cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )
		cur_permittivity = bayer_filter.get_permittivity()
		fdtd_hook.select("design_import")
		fdtd_hook.importnk2(np.sqrt(cur_permittivity), bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)

		log_file = open( projects_directory_location + "/log.txt", 'a' )
		log_file.write( "Imported filter\n" )
		log_file.close()

		#
		# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
		#
		Qx = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )
		Qy = np.zeros( ( num_focal_spots, num_design_frequency_points ), dtype=np.complex )

		for fwd_src_idx in range( 0, num_forward_sources ):
			log_file = open( projects_directory_location + "/log.txt", 'a' )
			log_file.write( "Pre source disable\n" )
			log_file.close()

			disable_all_sources()

			log_file = open( projects_directory_location + "/log.txt", 'a' )
			log_file.write( "Post source disable\n" )
			log_file.close()


			fdtd_hook.select( forward_sources[ fwd_src_idx ][ 'name' ] )
			fdtd_hook.set( 'enabled', 1 )

			log_file = open( projects_directory_location + "/log.txt", 'a' )
			log_file.write( "post source enable\n" )
			log_file.close()


			job_name = 'forward_job_' + str( fwd_src_idx ) + '.fsp'
			fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
			log_file = open( projects_directory_location + "/log.txt", 'a' )
			log_file.write( "Pre file save\n" )
			log_file.close()

			job_names[ ( 'forward', fwd_src_idx ) ] = add_job( job_name, jobs_queue )


			log_file = open( projects_directory_location + "/log.txt", 'a' )
			log_file.write( "Post add job\n" )
			log_file.close()



		for adj_src_idx in range(0, num_adjoint_sources):
			for xy_idx in range(0, 2):
				disable_all_sources()
				fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
				fdtd_hook.set( 'enabled', 1 )

				job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '.fsp'

				fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
				job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] = add_job( job_name, jobs_queue )



		run_jobs( jobs_queue )


####
		E_focal_transmission = None
		H_focal_transmission = None
		for fwd_src_idx in range( 0, num_forward_sources ):
			fdtd_hook.load( job_names[ ( 'forward', fwd_src_idx ) ] )

			forward_e_fields[ fwd_src_idx ] = get_efield( design_efield_monitor['name'] )

			focal_data[xy_names[fwd_src_idx]] = []
			for focal_idx in range( 0, num_focal_spots ):
				focal_monitor_data = get_efield( focal_monitors[ focal_idx ][ 'name' ] )

				Qx[ focal_idx, : ] = focal_monitor_data[ 0, 0, 0, 0, : ]
				Qy[ focal_idx, : ] = focal_monitor_data[ 1, 0, 0, 0, : ]
####
		
		focal_intensity = np.zoers( ( num_focal_spots, num_design_frequency_points ) )
		for focal_idx in range( 0, num_focal_spots ):
			focal_intensity[ focal_idx, : ] = ( np.abs( Qx[ focal_idx, : ] )**2 + np.abs( Qy[ focal_idx, : ] )**2 ) / max_intensity_by_wavelength

		figure_of_merit_by_focal_spot_by_wavelength = focal_intensity
		figure_of_merit = np.mean( focal_intensity )

		figure_of_merit_by_focal_spot_by_wavelength_evolution[ epoch, iteration ] = figure_of_merit_by_focal_spot_by_wavelength
		figure_of_merit_evolution[ epoch, iteration ] = figure_of_merit

		print( 'The current figure of merit = ' + str( figure_of_merit ) )

		np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution )
		np.save(projects_directory_location + "/figure_of_merit_by_focal_spot_by_wavelength.npy", figure_of_merit_by_focal_spot_by_wavelength_evolution )

		#
		# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
		# gradients for x- and y-polarized forward sources.
		#
		adjoint_ex_fields = []
		adjoint_ey_fields = []
		for adj_src_idx in range(0, num_adjoint_sources):
			for xy_idx in range(0, 2):
				fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx ) ] )

				if xy_idx == 0:
					adjoint_ex_fields.append(
						get_efield( design_efield_monitor['name'] ) )
				else:
					adjoint_ey_fields.append(
						get_efield( design_efield_monitor['name'] ) )


		flatten_individual_fom = figure_of_merit_by_focal_spot_by_wavelength.flatten()
		performance_weights = ( 2. / len( flatten_individual_fom ) ) - flatten_individual_fom**2 / np.sum( flatten_individual_fom**2 )
		performance_weights = np.maximum( performance_weights, 0 )


		device_gradient = np.zeros( forward_e_fields[ 0 ][ 0, :, :, :, 0 ].shape )
		for fwd_src_idx in range( 0, num_forward_sources ):
			get_fwd_fields = forward_e_fields[ fwd_src_idx ]
			get_adj_x_fields = adjoint_ex_fields[ fwd_src_idx ]
			get_adj_y_fields = adjoint_ey_fields[ fwd_src_idx ]

			x_gradient_component =  np.conj( Qx[ focal_idx, : ] ) * get_fwd_fields * get_adj_x_fields
			y_gradient_component =  np.conj( Qy[ focal_idx, : ] ) * get_fwd_fields * get_adj_y_fields

			combined_gradient_component = np.sum( x_gradient_component + y_gradient_component, axis=0 )

			for wl_idx in range( 0, num_design_frequency_points ):
				performance_weight = performance_weights[ fwd_src_idx * num_design_frequency_points + wl_idx ]
				net_weight = performance_weight / max_intensity_by_wavelength[ wl_idx ]

				device_gradient += net_weight * combined_gradient_component[ :, :, :, wl_idx ]

		device_gradient = 2 * np.real( device_gradient )
		# Enforce x=y symmetry
		device_gradient = 0.5 * ( device_gradient + np.swapaxes( device_gradient, 0, 1 ) )

		#
		# Step 4: Step the design variable.
		#
		# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
		# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
		# device_gradient = np.swapaxes(device_gradient, 0, 2)

		design_gradient = bayer_filter.backpropagate(device_gradient)


		cur_design_variable = bayer_filter.get_design_variable()


		step_size_density = design_change_start_epoch + ( iteration / ( num_iterations_per_epoch - 1 ) ) * ( design_change_end_epoch - design_change_start_epoch )
		step_size_density /= np.max( np.abs( design_gradient ) )


		last_design_variable = cur_design_variable.copy()
		#
		# todo: fix this in other files! the step already does the backpropagation so you shouldn't
		# pass it an already backpropagated gradient!  Sloppy, these files need some TLC and cleanup!
		#
		# enforce_binarization = False
		# if epoch >= binarization_start_epoch:
		# 	enforce_binarization = True
		# device_gradient = np.flip( device_gradient, axis=2 )
		bayer_filter.step( -device_gradient, step_size_density, require_xy_symmetry )
		#(, enforce_binarization, projects_directory_location)
		cur_design_variable = bayer_filter.get_design_variable()

		average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
		max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

		step_size_evolution[epoch][iteration] = step_size_density
		average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
		max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

		fdtd_hook.switchtolayout()
		fdtd_hook.save()
		shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/optimization_start_epoch_" + str( epoch ) + ".fsp" )
		np.save(projects_directory_location + '/device_gradient.npy', device_gradient)
		np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
		np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
		np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
		np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
		np.save(projects_directory_location + "/cur_design_variable.npy", cur_design_variable)
		np.save(projects_directory_location + "/cur_design_variable_" + str( epoch ) + ".npy", cur_design_variable)

	fdtd_hook.switchtolayout()
	fdtd_hook.save()
	shutil.copy( projects_directory_location + "/optimization.fsp", projects_directory_location + "/optimization_end_epoch_" + str( epoch ) + ".fsp" )


