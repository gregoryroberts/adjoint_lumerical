import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyAMCtrlPtsParameters import *
import LayeredLithographyAMBayerFilterCtrlPts

# import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
# import lumapi

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
import smuthi.utility.cuda as cu



#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# projects_directory_location = "/central/groups/Faraon_Computing/projects" 
# projects_directory_location += "/" + project_name + "_gmmt_v3_test"
projects_directory_location = "/home/ec2-user/gmmt_test/"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

shutil.copy2(python_src_directory + "/LayeredLithographyAMCtrlPtsParameters.py", projects_directory_location + "/LayeredLithographyAMCtrlPtsParameters.py")
shutil.copy2(python_src_directory + "/GMMTTest.py", projects_directory_location + "/GMMTTest.py")
shutil.copy2(python_src_directory + "/LayeredLithographyAMCtrlPtsOptimization.py", projects_directory_location + "/LayeredLithographyAMCtrlPtsOptimization.py")


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

probe_wavelength_nm = 500

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
use_gpu = True
while comparison < num_comparisons:
	num_jobs = int( np.minimum( 1, num_comparisons - comparison ) )
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

		cu.enable_gpu(use_gpu)
		if use_gpu and not cu.use_gpu:
			print("Failed to load pycuda, skipping simulation")
			sys.exit( 1 )


		print( "number of particles = " + str( len( smuthi_spheres ) ) )
		simulation = smuthi.simulation.Simulation(
												layer_system=two_layers,
												particle_list=smuthi_spheres,
												initial_field=smuthi_plane_wave,
												solver_type='gmres',
												store_coupling_matrix=False,
												# coupling_matrix_interpolator_kind='linear',
												# coupling_matrix_lookup_resolution=5,
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

