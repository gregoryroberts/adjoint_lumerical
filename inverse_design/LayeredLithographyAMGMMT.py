import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyAMCtrlPtsParameters import *
import LayeredLithographyAMBayerFilterCtrlPts

import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
import lumapi

import functools
import h5py
# import matplotlib.pyplot as plt
import numpy as np
import time

import miepy

from scipy.ndimage import gaussian_filter

import queue

import subprocess

import platform

import re

import reinterpolate

import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.graphical_output as go
import smuthi.postprocessing.scattered_field as sf
import smuthi.postprocessing.internal_field as intf
import smuthi.postprocessing.far_field as ff


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
projects_directory_location += "/" + project_name + "_gmmt_v3_test"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredLithographyAMCtrlPtsParameters.py", projects_directory_location + "/LayeredLithographyAMCtrlPtsParameters.py")
shutil.copy2(python_src_directory + "/LayeredLithographyAMGMMT.py", projects_directory_location + "/LayeredLithographyAMGMMT.py")
shutil.copy2(python_src_directory + "/LayeredLithographyAMCtrlPtsOptimization.py", projects_directory_location + "/LayeredLithographyAMCtrlPtsOptimization.py")

#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['z max'] = fdtd_region_maximum_vertical_um * 1e-6
fdtd['z min'] = fdtd_region_minimum_vertical_um * 1e-6
fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
fdtd['background index'] = background_index

design_mesh = fdtd_hook.addmesh()
design_mesh['name'] = 'design_override_mesh'
design_mesh['x span'] = device_size_lateral_um * 1e-6
design_mesh['y span'] = device_size_lateral_um * 1e-6
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
# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
# gradient.
#
focal_monitors = []

probe_wavelength_nm = 500

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
	focal_monitor['frequency points'] = 1
	focal_monitor['wavelength center'] = probe_wavelength_nm * 1e-9

	focal_monitors.append(focal_monitor)

focal_plane_intensity = fdtd_hook.addpower()
focal_plane_intensity['name'] = 'focal_plane_intensity'
focal_plane_intensity['monitor type'] = '2D Z-Normal'
focal_plane_intensity['x'] = 0 * 1e-6
focal_plane_intensity['x span'] = device_size_lateral_um * 1e-6
focal_plane_intensity['y'] = 0 * 1e-6
focal_plane_intensity['y span'] = device_size_lateral_um * 1e-6
focal_plane_intensity['z'] = adjoint_vertical_um * 1e-6
focal_plane_intensity['override global monitor settings'] = 1
focal_plane_intensity['use wavelength spacing'] = 1
focal_plane_intensity['use source limits'] = 0
focal_plane_intensity['frequency points'] = 1
focal_plane_intensity['wavelength center'] = probe_wavelength_nm * 1e-9


#
# Add SiO2 at the top
#
# substrate = fdtd_hook.addrect()
# substrate['name'] = 'substrate'
# substrate['x span'] = fdtd_region_size_lateral_um * 1e-6
# substrate['y span'] = fdtd_region_size_lateral_um * 1e-6
# substrate['z min'] = device_vertical_maximum_um * 1e-6
# substrate['z max'] = fdtd_region_maximum_vertical_um * 1e-6
# substrate['index'] = index_substrate

# air_bottom = fdtd_hook.addrect()
# air_bottom['name'] = 'air_bottom'
# air_bottom['x span'] = fdtd_region_size_lateral_um * 1e-6
# air_bottom['y span'] = fdtd_region_size_lateral_um * 1e-6
# air_bottom['z min'] = fdtd_region_minimum_vertical_um * 1e-6
# air_bottom['z max'] = device_vertical_minimum_um * 1e-6
# air_bottom['index'] = index_air


#
# Add device region and create device permittivity
#
# design_import = fdtd_hook.addimport()
# design_import['name'] = 'design_import'
# design_import['x span'] = ( fdtd_region_size_lateral_um - mesh_spacing_um ) * 1e-6
# design_import['y span'] = ( fdtd_region_size_lateral_um - mesh_spacing_um ) * 1e-6
# design_import['z max'] = ( device_vertical_maximum_um - 0.5 * mesh_spacing_um ) * 1e-6
# design_import['z min'] = ( device_vertical_minimum_um + 0.5 * mesh_spacing_um ) * 1e-6

# bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
# bayer_filter = LayeredLithographyAMBayerFilterCtrlPts.LayeredLithographyAMBayerFilterCtrlPts(
# 	bayer_filter_size_voxels,
# 	[ lateral_box_sampling, lateral_box_sampling ],
# 	gaussian_blur_filter_sigma,
# 	[ min_device_permittivity, max_device_permittivity ],
# 	init_permittivity_0_1_scale,
# 	num_vertical_layers,
# 	spacer_size_voxels )

np.random.seed( random_seed )
# num_random = device_voxels_lateral * device_voxels_lateral * device_voxels_vertical
# random_device = np.random.normal( init_permittivity_0_1_scale, 0.25, num_random )
# random_device = np.minimum( np.maximum( random_device, 0.0 ), 1.0 )

# bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
# bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
# bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um + 0.5 * mesh_spacing_um, device_vertical_maximum_um - 0.5 * mesh_spacing_um, simulated_device_voxels_vertical)


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	for xy_idx in range(0, 2):
		fdtd_hook.select( forward_sources[xy_idx]['name'] )
		fdtd_hook.set( 'enabled', 0 )


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



spatial_limits_device_um = [
	[ -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um ],
	[ -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um ],
	[ device_vertical_minimum_um, device_vertical_maximum_um ]
]

interpolated_size = [ device_voxels_lateral, device_voxels_lateral, device_voxels_vertical ]



num_comparisons = 10#200
just_mie = True
record_focal_plane = True
focal_x_points = 50
focal_y_points = 50

gmmt_data = np.zeros( ( num_comparisons, num_focal_spots ) )
gmmt_focal_intensity = np.zeros( ( num_comparisons, focal_x_points, focal_y_points ) )
fdtd_sphere_data = np.zeros( ( num_comparisons, num_focal_spots ) )
fdtd_cylinder_data = np.zeros( ( num_comparisons, num_focal_spots ) )
sphere_focal_intensity = np.zeros( ( num_comparisons, 101, 101 ) )
cylinder_focal_intensity = np.zeros( ( num_comparisons, 101, 101 ) )

nm = 1e-9

layer_thickness_nm = 400
layer_spacing_nm = 800
num_layers = 5
device_height_nm = num_layers * layer_spacing_nm
focal_length_nm = 1500

sphere_z_global_offset_nm = 1000

x_bounds_nm = [ -1000, 1000 ]
y_bounds_nm = [ -1000, 1000 ]

device_size_x_nm = x_bounds_nm[ 1 ] - x_bounds_nm[ 0 ]
device_size_y_nm = y_bounds_nm[ 1 ] - y_bounds_nm[ 0 ]

sphere_radius_nm = 50
sphere_spacing_nm = 200

sphere_gen_probability = 0.1

sphere_index = 2.4

sphere_dielectric = miepy.constant_material( sphere_index**2 )
background_dielectric = miepy.constant_material( 1.46**2 )
interface_dielectric = miepy.materials.vacuum()

two_layers = smuthi.layers.LayerSystem( thicknesses=[0, 0], refractive_indices=[ 1.0, 1.46 ] )

smuthi_plane_wave = smuthi.initial_field.PlaneWave(
											vacuum_wavelength=probe_wavelength_nm,
											polar_angle=np.pi,#np.pi,#4*np.pi/5, # from top
											azimuthal_angle=0,
											polarization=1 )         # 0=TE 1=TM


plane_wave = miepy.sources.plane_wave( [ 1, 0 ] )
air_interface = miepy.interface( interface_dielectric, z=( ( device_height_nm + sphere_z_global_offset_nm ) * nm ) )
lmax = 2#3

focal_x_nm = np.linspace( x_bounds_nm[ 0 ], x_bounds_nm[ 1 ], focal_x_points )
focal_y_nm = np.linspace( y_bounds_nm[ 0 ], y_bounds_nm[ 1 ], focal_y_points )
focal_z_nm = sphere_z_global_offset_nm + device_height_nm + focal_length_nm


def gen_random_cluster( sphere_probability ):

	x_start_nm = x_bounds_nm[ 0 ] + sphere_radius_nm
	y_start_nm = y_bounds_nm[ 0 ] + sphere_radius_nm

	x_end_nm = x_bounds_nm[ 1 ] - sphere_radius_nm
	y_end_nm = y_bounds_nm[ 1 ] - sphere_radius_nm

	sphere_positions_m = []
	sphere_radii_m = []
	sphere_materials = []
	sphere_indices = []

	for layer_idx in range( 0, num_layers ):

		layer_z_nm = layer_idx * layer_spacing_nm + 0.5 * layer_thickness_nm + sphere_z_global_offset_nm

		cur_x_nm = x_start_nm

		while ( cur_x_nm <= x_end_nm ):

			cur_y_nm = y_start_nm

			while ( cur_y_nm <= y_end_nm ):

				gen_random = np.random.random( 1 )[ 0 ]
				if gen_random < sphere_probability:

					sphere_positions_m.append( [ cur_x_nm * nm, cur_y_nm * nm, layer_z_nm * nm ] )
					sphere_radii_m.append( sphere_radius_nm * nm )
					sphere_materials.append( sphere_dielectric )
					sphere_indices.append( sphere_index )

				cur_y_nm += sphere_spacing_nm

			cur_x_nm += sphere_spacing_nm

	return np.array( sphere_positions_m ), np.array( sphere_radii_m ), sphere_materials, sphere_indices

lumerical_sphere_objects = []
lumerical_cylinder_objects = []


focal_lateral_search_mesh_x_nm = np.zeros( ( focal_x_points, focal_y_points ) )
focal_lateral_search_mesh_y_nm = np.zeros( ( focal_x_points, focal_y_points ) )
focal_lateral_search_mesh_z_nm = np.zeros( ( focal_x_points, focal_y_points ) )
for x_idx in range( 0, focal_x_points ):
	for y_idx in range( 0, focal_y_points ):
		focal_lateral_search_mesh_x_nm[ x_idx, y_idx ] = focal_x_nm[ x_idx ] * nm
		focal_lateral_search_mesh_y_nm[ x_idx, y_idx ] = focal_y_nm[ y_idx ] * nm
		focal_lateral_search_mesh_z_nm[ x_idx, y_idx ] = focal_z_nm * nm


comparison = 0
while comparison < num_comparisons:
	num_jobs = int( np.minimum( num_nodes_available, num_comparisons - comparison ) )
	job_names = {}

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Working on comparison " + str( comparison ) + " out of " + str( num_comparisons - 1 ) + " total\n" )
	log_file.close()


	for job_idx in range( 0, num_jobs ):

		random_centers, random_radii, random_materials, random_indices = gen_random_cluster( sphere_gen_probability )

		smuthi_spheres = []
		for sphere_idx in range( 0, len( random_centers ) ):
			get_center = random_centers[ sphere_idx ] / nm
			# get_center[ 2 ] -= sphere_z_global_offset_nm
			# get_center[ 2 ] += device_height_nm

			flip_z = get_center[ 2 ] - sphere_z_global_offset_nm
			flip_z = device_height_nm - flip_z - ( layer_spacing_nm - layer_thickness_nm )

			get_center[ 2 ] = flip_z

			smuthi_spheres.append( 
				smuthi.particles.Sphere(
					position=list( get_center ),
					refractive_index=random_indices[ sphere_idx ],
					radius=( random_radii[ sphere_idx ] / nm ),
					l_max=lmax ) )

		mie_start = time.time()

		simulation = smuthi.simulation.Simulation(
												layer_system=two_layers,
												particle_list=smuthi_spheres,
												initial_field=smuthi_plane_wave,
												solver_type='gmres',
												store_coupling_matrix=False,
												# coupling_matrix_interpolator_kind='linear',
												coupling_matrix_lookup_resolution=5,
												solver_tolerance=1e-4,
												length_unit='nm' )
		prep_time, solution_time, post_time = simulation.run()

		log_file = open( projects_directory_location + "/log.txt", 'a' )
		log_file.write( "Prep, Solution, Post times are: " + str( prep_time ) + ", " + str( solution_time ) + ", " + str( post_time ) + "\n" )
		log_file.close()


		vacuum_wavelength = simulation.initial_field.vacuum_wavelength
		dim1vec = np.arange(-1000, 1000 + 25/2, 25)
		dim2vec = np.arange(-1000, 1000 + 25/2, 25)
		xarr, yarr = np.meshgrid(dim1vec, dim2vec)
		zarr = xarr - xarr - focal_length_nm

		scat_fld_exp = sf.scattered_field_piecewise_expansion(vacuum_wavelength,
																simulation.particle_list, simulation.layer_system,
																'default', 'default', None)

		e_x_scat, e_y_scat, e_z_scat = scat_fld_exp.electric_field(xarr, yarr, zarr)
		e_x_init, e_y_init, e_z_init = simulation.initial_field.electric_field(xarr, yarr, zarr, simulation.layer_system)

		e_x, e_y, e_z = e_x_scat + e_x_init, e_y_scat + e_y_init, e_z_scat + e_z_init

		intensity = abs(e_x)**2 + abs(e_y)**2 + abs(e_z)**2
		intensity = np.swapaxes( intensity, 0, 1 )

		np.save( projects_directory_location + "/sumith_intensity_" + str( comparison + job_idx ) + ".npy", intensity )

		# go.show_near_field(
		# 			quantities_to_plot=['norm(E)'],
		# 			show_plots=False,
		# 			show_opts=[{'label':'raw_data'},
		# 						{'interpolation':'quadric'},
		# 						{'interpolation':'quadric'}],
		# 			save_plots=True,
		# 			save_opts=[{'format':'png'},
		# 						{'format':'raw'}],
		# 			outputdir=(projects_directory_location + "/smuthi_output_" + str( comparison + job_idx ) + "/"),
		# 			xmin=-1000,
		# 			xmax=1000,
		# 			ymin=-1000,
		# 			ymax=1000,
		# 			zmin=focal_length_nm,
		# 			zmax=focal_length_nm,
		# 			resolution_step=50,
		# 			simulation=simulation,
		# 			show_internal_field=False)


		# mie_cluster = miepy.sphere_cluster(
		# 	medium=background_dielectric,
		# 	position=random_centers, radius=random_radii, material=random_materials,
		# 	source=plane_wave, wavelength=(probe_wavelength_nm*nm), lmax=lmax, interface=air_interface )

		# gmmt_data[ comparison + job_idx, 0 ] = np.sum(
		# 	np.abs( mie_cluster.E_field( 0.25 * device_size_x_nm * nm, 0.25 * device_size_y_nm * nm, focal_z_nm * nm ) )**2 )
		# gmmt_data[ comparison + job_idx, 1 ] = np.sum(
		# 	np.abs( mie_cluster.E_field( -0.25 * device_size_x_nm * nm, 0.25 * device_size_y_nm * nm, focal_z_nm * nm ) )**2 )
		# gmmt_data[ comparison + job_idx, 2 ] = np.sum(
		# 	np.abs( mie_cluster.E_field( -0.25 * device_size_x_nm * nm, -0.25 * device_size_y_nm * nm, focal_z_nm * nm ) )**2 )
		# gmmt_data[ comparison + job_idx, 3 ] = np.sum(
		# 	np.abs( mie_cluster.E_field( 0.25 * device_size_x_nm * nm, -0.25 * device_size_y_nm * nm, focal_z_nm * nm ) )**2 )

		# gmmt_focal_intensity[ comparison + job_idx ] = np.sum(
		# 	np.abs( mie_cluster.E_field( focal_lateral_search_mesh_x_nm, focal_lateral_search_mesh_y_nm, focal_lateral_search_mesh_z_nm ) )**2,
		# 	axis=0 )


		# for x_idx in range( 0, focal_x_points ):
		# 	for y_idx in range( 0, focal_y_points ):
		# 		gmmt_focal_intensity[ comparison + job_idx, x_idx, y_idx ] = np.sum(
		# 			np.abs( mie_cluster.E_field( focal_x_nm[ x_idx ] * nm, focal_y_nm[ y_idx ] * nm, focal_z_nm * nm ) )**2 )

		mie_time = time.time() - mie_start

		log_file = open( projects_directory_location + "/log.txt", 'a' )
		log_file.write( "Mie time for single wavelength took " + str( mie_time ) + " seconds\n" )
		log_file.close()

		for sphere_object in lumerical_sphere_objects:
			sphere_object['enabled'] = 0
		for cylinder_object in lumerical_cylinder_objects:
			cylinder_object['enabled'] = 0

		for sphere_idx in range( 0, len( random_centers ) ):
			if sphere_idx >= len( lumerical_sphere_objects ):
				make_sphere = fdtd_hook.addsphere()
				make_sphere['name'] = 'sphere_' + str( sphere_idx )
			else:
				make_sphere = lumerical_sphere_objects[ sphere_idx ]

			flip_z = random_centers[ sphere_idx ][ 2 ] - sphere_z_global_offset_nm * nm
			flip_z = device_height_nm * nm - flip_z - ( layer_spacing_nm - layer_thickness_nm ) * nm

			make_sphere['index'] = random_indices[ sphere_idx ]
			make_sphere['radius'] = random_radii[ sphere_idx ]
			make_sphere['x'] = random_centers[ sphere_idx ][ 0 ]
			make_sphere['y'] = random_centers[ sphere_idx ][ 1 ]
			make_sphere['z'] = flip_z
			make_sphere['enabled'] = 1

			if sphere_idx >= len( lumerical_sphere_objects ):
				lumerical_sphere_objects.append( make_sphere )

		#
		# Add FDTD job
		#		
		disable_all_sources()
		fdtd_hook.select( forward_sources[ 0 ][ 'name' ] )
		fdtd_hook.set( 'enabled', 1 )

		job_name = 'spheres_' + str( job_idx ) + '.fsp'
		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )

		job_names[ ( 'spheres', job_idx ) ] = add_job( job_name, jobs_queue )

		for sphere_object in lumerical_sphere_objects:
			sphere_object['enabled'] = 0
		for cylinder_object in lumerical_cylinder_objects:
			cylinder_object['enabled'] = 0

		for cylinder_idx in range( 0, len( random_centers ) ):
			if cylinder_idx >= len( lumerical_cylinder_objects ):
				make_cylinder = fdtd_hook.addcircle()
				make_cylinder['name'] = 'cylinder_' + str( cylinder_idx )
			else:
				make_cylinder = lumerical_cylinder_objects[ cylinder_idx ]

			flip_z = random_centers[ cylinder_idx ][ 2 ] - sphere_z_global_offset_nm * nm
			flip_z = device_height_nm * nm - flip_z - ( layer_spacing_nm - layer_thickness_nm ) * nm

			make_cylinder['index'] = random_indices[ cylinder_idx ]
			make_cylinder['radius'] = random_radii[ cylinder_idx ]
			make_cylinder['x'] = random_centers[ cylinder_idx ][ 0 ]
			make_cylinder['y'] = random_centers[ cylinder_idx ][ 1 ]
			make_cylinder['z'] = flip_z
			make_cylinder['z span'] = device_height_per_layer_um * 1e-6
			make_cylinder['enabled'] = 1

			if cylinder_idx >= len( lumerical_cylinder_objects ):
				lumerical_cylinder_objects.append( make_cylinder )

		#
		# Add FDTD job
		#		
		disable_all_sources()
		fdtd_hook.select( forward_sources[ 0 ][ 'name' ] )
		fdtd_hook.set( 'enabled', 1 )

		job_name = 'cylinders_' + str( job_idx ) + '.fsp'
		fdtd_hook.save( projects_directory_location + "/optimization.fsp" )

		job_names[ ( 'cylinders', job_idx ) ] = add_job( job_name, jobs_queue )


	if just_mie:
		comparison += num_nodes_available
		np.save( projects_directory_location + "/gmmt_data.npy", gmmt_data )
		np.save( projects_directory_location + "/gmmt_focal_intensity.npy", gmmt_focal_intensity )
		continue

	run_jobs( jobs_queue )

	for job_idx in range( 0, num_jobs ):
		fdtd_hook.load( job_names[ ( 'spheres', job_idx ) ] )

		focal_plane_data = np.squeeze( get_efield( 'focal_plane_intensity' ) )
		sphere_focal_intensity[ comparison + job_idx ] = np.sum( np.abs( focal_plane_data )**2, axis=0 )

		for focal_idx in range(0, num_adjoint_sources):
			pull_focal_data = get_efield( focal_monitors[ focal_idx ][ 'name' ] )
			pull_focal_data = pull_focal_data[ :, 0, 0, 0, 0 ]
			fdtd_sphere_data[ comparison + job_idx ][ focal_idx ] = np.sum( np.abs( pull_focal_data )**2 )

		lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	for job_idx in range( 0, num_jobs ):
		fdtd_hook.load( job_names[ ( 'cylinders', job_idx ) ] )

		focal_plane_data = np.squeeze( get_efield( 'focal_plane_intensity' ) )
		cylinder_focal_intensity[ comparison + job_idx ] = np.sum( np.abs( focal_plane_data )**2, axis=0 )

		for focal_idx in range(0, num_adjoint_sources):
			pull_focal_data = get_efield( focal_monitors[ focal_idx ][ 'name' ] )
			pull_focal_data = pull_focal_data[ :, 0, 0, 0, 0 ]
			fdtd_cylinder_data[ comparison + job_idx ][ focal_idx ] = np.sum( np.abs( pull_focal_data )**2 )

		lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	comparison += num_nodes_available

	np.save( projects_directory_location + "/gmmt_data.npy", gmmt_data )
	np.save( projects_directory_location + "/gmmt_focal_intensity.npy", gmmt_focal_intensity )
	np.save( projects_directory_location + "/fdtd_sphere_data.npy", fdtd_sphere_data )
	np.save( projects_directory_location + "/sphere_focal_intensity.npy", sphere_focal_intensity )
	np.save( projects_directory_location + "/fdtd_cylinder_data.npy", fdtd_cylinder_data )
	np.save( projects_directory_location + "/cylinder_focal_intensity.npy", cylinder_focal_intensity )
