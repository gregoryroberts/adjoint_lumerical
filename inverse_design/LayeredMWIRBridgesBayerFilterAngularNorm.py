import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredMWIRBridgesBayerFilterAngularNormParameters import *
import LayeredMWIRBridgesBayerFilter
import ip_dip_dispersion


run_on_cluster = True

if run_on_cluster:
	import imp
	imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
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
	projects_directory_location = projects_directory_location_base + '_angular_bfast_32_snell_large_focal_norm_xpol_v3'
else:
	projects_directory_location_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
	projects_directory_location_base += "/" + project_name
	projects_directory_location = projects_directory_location_base + '_angular_bfast_32_snell_large_focal_norm_xpol_v3'


if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredMWIRBridgesBayerFilterAngularNormParameters.py", projects_directory_location + "/LayeredMWIRBridgesBayerFilterAngularNormParameters.py")
shutil.copy2(python_src_directory + "/LayeredMWIRBridgesBayerFilterAngularNorm.py", projects_directory_location + "/LayeredMWIRBridgesBayerFilterAngularNorm.py")

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
fdtd['pml profile'] = 3#'steep angle'
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
	# forward_src = fdtd_hook.addtfsf()
	# forward_src['name'] = 'forward_src_' + xy_names[xy_idx]
	# forward_src['angle phi'] = xy_phi_rotations[xy_idx]
	# forward_src['direction'] = 'Backward'
	# forward_src['x span'] = lateral_aperture_um * 1e-6
	# forward_src['y span'] = lateral_aperture_um * 1e-6
	# forward_src['z max'] = src_maximum_vertical_um * 1e-6
	# forward_src['z min'] = src_minimum_vertical_um * 1e-6
	# forward_src['wavelength start'] = lambda_min_um * 1e-6
	# forward_src['wavelength stop'] = lambda_max_um * 1e-6

	# forward_sources.append(forward_src)


	forward_src = fdtd_hook.addplane()
	forward_src['name'] = 'forward_src_' + xy_names[xy_idx]
	forward_src['plane wave type'] = 'Diffracting'
	forward_src['angle phi'] = 0#xy_phi_rotations[xy_idx]
	forward_src['polarization angle'] = xy_phi_rotations[xy_idx]
	forward_src['direction'] = 'Backward'
	# forward_src['x span'] = lateral_aperture_um * 1e-6
	forward_src['x span'] = fdtd_region_size_lateral_um * 1e-6
	forward_src['y span'] = fdtd_region_size_lateral_um * 1e-6
	forward_src['z'] = src_maximum_vertical_um * 1e-6
	# forward_src['z max'] = src_maximum_vertical_um * 1e-6
	# forward_src['z min'] = src_minimum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_max_um * 1e-6
	forward_src['frequency dependent profile'] = 1
	forward_src['number of field profile samples'] = 10

	forward_sources.append(forward_src)




device_input_monitor = fdtd_hook.addpower()
device_input_monitor['name'] = 'device_input_monitor'
device_input_monitor['monitor type'] = '2D Z-normal'
# device_input_monitor['x span'] = device_size_lateral_um * 1e-6
# device_input_monitor['y span'] = device_size_lateral_um * 1e-6
device_input_monitor['x span'] = fdtd_region_size_lateral_um * 1e-6
device_input_monitor['y span'] = fdtd_region_size_lateral_um * 1e-6
device_input_monitor['x'] = 0 * 1e-6
device_input_monitor['y'] = 0 * 1e-6
device_input_monitor['z'] = ( device_size_verical_um + pedestal_thickness_um ) * 1e-6
device_input_monitor['override global monitor settings'] = 1
device_input_monitor['use wavelength spacing'] = 1
device_input_monitor['use source limits'] = 1
device_input_monitor['frequency points'] = num_design_frequency_points


silicon_substrate = fdtd_hook.addrect()
silicon_substrate['name'] = 'silicon_substrate'
silicon_substrate['x'] = 0
silicon_substrate['x span'] = fdtd_region_size_lateral_um * 1e-6
silicon_substrate['y'] = 0
silicon_substrate['y span'] = fdtd_region_size_lateral_um * 1e-6
silicon_substrate['z min'] = fdtd_region_minimum_vertical_um * 1e-6
# Send this outside the region FDTD and let the source sit inside of it
silicon_substrate['z max'] = fdtd_region_maximum_vertical_um * 1e-6
silicon_substrate['material'] = 'Si (Silicon) - Palik'


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



eval_lambda_min_um = lambda_min_um - 0.5
eval_lambda_max_um = lambda_max_um + 0.5

eval_lambda_um = np.linspace( eval_lambda_min_um, eval_lambda_max_um, num_eval_points )

device_input_efields = np.zeros( ( num_phi, num_theta, 3, 1 + fdtd_region_minimum_lateral_voxels, 1 + fdtd_region_minimum_lateral_voxels, num_design_frequency_points ), dtype=np.complex )
device_input_hfields = np.zeros( ( num_phi, num_theta, 3, 1 + fdtd_region_minimum_lateral_voxels, 1 + fdtd_region_minimum_lateral_voxels, num_design_frequency_points ), dtype=np.complex )

job_names = {}


for phi_idx in range( 0, num_phi ):
	for theta_idx in range( 0, num_theta ):
		fdtd_hook.switchtolayout()

		forward_sources[ eval_pol_idx ][ 'angle theta' ] = eval_theta_degrees[ theta_idx ]
		forward_sources[ eval_pol_idx ][ 'angle phi' ] = eval_phi_degrees[ phi_idx ]
		forward_sources[ eval_pol_idx ][ 'polarization angle' ] = xy_phi_rotations[ eval_pol_idx ] - eval_phi_degrees[ phi_idx ]

		disable_all_sources()

		fdtd_hook.select( forward_sources[eval_pol_idx]['name'] )
		fdtd_hook.set( 'enabled', 1 )

		job_name = 'forward_job_' + str( eval_pol_idx ) + "_" + str( theta_idx ) + '.fsp'
		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
		job_names[ ( 'forward', eval_pol_idx, theta_idx ) ] = add_job( job_name, jobs_queue )

	run_jobs( jobs_queue )

	for theta_idx in range( 0, num_theta ):

		fdtd_hook.load( job_names[ ( 'forward', eval_pol_idx, theta_idx ) ] )

		focal_E = get_efield( device_input_monitor[ 'name' ] )
		device_input_efields[ phi_idx, theta_idx, :, :, :, : ] = np.squeeze( focal_E )

		focal_H = get_hfield( device_input_monitor[ 'name' ] )
		device_input_hfields[ phi_idx, theta_idx, :, :, :, : ] = np.squeeze( focal_H )


np.save( projects_directory_location + "/device_input_efields.npy", device_input_efields )
np.save( projects_directory_location + "/device_input_hfields.npy", device_input_hfields )

