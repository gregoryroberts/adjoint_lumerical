
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


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

	return total_efield


project_name = 'e_field_convergence'
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

lambda_min_um = 1.05
lambda_max_um = 1.55

num_observed_frequency_points = 10

num_index_values = 5
block_indices = np.linspace( 1.5, 3.5, num_index_values )

R_min_um = 0.25 * lambda_min_um
R_max_um = 1.0 * lambda_max_um
block_width_um = 0.5 * lambda_min_um

lateral_gap_um = 0.5 * lambda_max_um
tfsf_lateral_buffer_um = 0.25 * lambda_max_um
tfsf_width_um = 0.5 * block_width_um + R_max_um + tfsf_lateral_buffer_um

block_height_um = 2 * lambda_min_um

vertical_gap_um = lambda_max_um
tfsf_top_gap_um = 1.25 * lambda_max_um
tfsf_bottom_gap_um = 0.25 * lambda_max_um

simulation_lateral_size_um = 2 * lateral_gap_um + 2 * tfsf_width_um
simulation_vertical_size_um = 2 * vertical_gap_um + block_height_um + tfsf_top_gap_um + tfsf_bottom_gap_um

simulation_vertical_min_um = -( vertical_gap_um + tfsf_bottom_gap_um )
simulation_vertical_max_um = ( vertical_gap_um + tfsf_top_gap_um + block_height_um )

mesh_fractions = np.array( [ 1. / ( 4 * ( 6 - k ) ) for k in range( 0, 5 ) ] )
num_mesh_sizes = len( mesh_fractions )

num_R_values = 5
R_values_um = np.linspace( R_min_um, R_max_um, num_R_values )

fdtd_simulation_time_fs = 4 * 700

fdtd = fdtd_hook.addfdtd()
fdtd['x span'] = simulation_lateral_size_um * 1e-6
fdtd['y span'] = simulation_lateral_size_um * 1e-6
fdtd['z max'] = simulation_vertical_max_um * 1e-6
fdtd['z min'] = simulation_vertical_min_um * 1e-6
fdtd['mesh type'] = 'uniform'
fdtd['define x mesh by'] = 'number of mesh cells'
fdtd['define y mesh by'] = 'number of mesh cells'
fdtd['define z mesh by'] = 'number of mesh cells'
index_air = 1.0
fdtd['background index'] = index_air
fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15

excitation = fdtd_hook.addtfsf()
excitation['name'] = 'excitation'
excitation['angle phi'] = 0
excitation['direction'] = 'Backward'
excitation['x span'] = ( 2 * tfsf_width_um ) * 1e-6
excitation['y span'] = ( 2 * tfsf_width_um ) * 1e-6
excitation['z max'] = ( tfsf_top_gap_um + block_height_um ) * 1e-6
excitation['z min'] = -( tfsf_bottom_gap_um ) * 1e-6
excitation['wavelength start'] = lambda_min_um * 1e-6
excitation['wavelength stop'] = lambda_max_um * 1e-6


E_monitor_x = fdtd_hook.addpower()
E_monitor_x['name'] = 'focal_monitor_x'
E_monitor_x['monitor type'] = 'point'
E_monitor_x['spatial interpolation'] = 'specified position'
E_monitor_x['x'] = 0 * 1e-6
E_monitor_x['y'] = 0 * 1e-6
E_monitor_x['z'] = block_height_um * 1e-6
E_monitor_x['override global monitor settings'] = 1
E_monitor_x['use wavelength spacing'] = 1
E_monitor_x['use source limits'] = 1
E_monitor_x['frequency points'] = num_observed_frequency_points



phase_correction = np.zeros( ( num_index_values, num_mesh_sizes, 3, num_observed_frequency_points ), dtype=np.complex )

for index_idx in range( 0, num_index_values ):

	print( 'Currently working on index ' + str( index_idx ) + ' out of ' + str( num_index_values - 1 ) )

	block_index = block_indices[ index_idx ]

	mesh_sizes_um = ( 1. / block_index ) * mesh_fractions * lambda_min_um

	for mesh_idx in range( 0, num_mesh_sizes ):
		print( 'Currently mesh ' + str( mesh_idx ) + ' out of ' + str( num_mesh_sizes - 1 ) )

		mesh_size_um = mesh_sizes_um[ mesh_idx ]

		#
		# Adjust mesh
		#
		fdtd_region_minimum_vertical_voxels = int( np.ceil( simulation_vertical_size_um / mesh_size_um ) )
		fdtd_region_minimum_lateral_voxels = int( np.ceil( simulation_lateral_size_um / mesh_size_um ) )

		fdtd_hook.switchtolayout()

		fdtd['mesh cells x'] = fdtd_region_minimum_lateral_voxels
		fdtd['mesh cells y'] = fdtd_region_minimum_lateral_voxels
		fdtd['mesh cells z'] = fdtd_region_minimum_vertical_voxels

		fdtd_hook.run()

		E_data_phase_correction = get_efield( E_monitor_x[ 'name' ] )
		phase_correction[ index_idx, mesh_idx, :, : ] = E_data_phase_correction[ :, 0, 0, 0, : ]

np.save( projects_directory_location + "/phase_correction.npy", phase_correction )
