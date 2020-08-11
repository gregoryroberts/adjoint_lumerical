import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyAMPostprocessParameters import *
# import LayeredLithographyAMBayerFilter
import LayeredLithographyAMPostprocessBayerFilter

# import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )
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
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD( hide=False )

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

projects_directory_location = base_directory + "/projects/cluster/nir/"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/postprocess")

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
design_mesh['x span'] = device_size_lateral_um * 1e-6
design_mesh['y span'] = device_size_lateral_um * 1e-6
design_mesh['z max'] = device_vertical_maximum_um * 1e-6
design_mesh['z min'] = device_vertical_minimum_um * 1e-6
design_mesh['dx'] = mesh_spacing_um * 1e-6
design_mesh['dy'] = mesh_spacing_um * 1e-6
design_mesh['dz'] = mesh_spacing_um * 1e-6

transmission_min_um = lambda_min_um
transmission_max_um = lambda_max_um

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
design_efield_monitor.enabled = 0

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
	transmission_focal_monitor['use source limits'] = 0
	transmission_focal_monitor['frequency points'] = num_eval_frequency_points
	transmission_focal_monitor['minimum wavelength'] = transmission_min_um * 1e-6
	transmission_focal_monitor['maximum wavelength'] = transmission_max_um * 1e-6
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
transmission_focal['use source limits'] = 0
transmission_focal['frequency points'] = num_eval_frequency_points
transmission_focal['minimum wavelength'] = transmission_min_um * 1e-6
transmission_focal['maximum wavelength'] = transmission_max_um * 1e-6
transmission_focal.enabled = 1

transmission_device = fdtd_hook.addpower()
transmission_device['name'] = 'transmission_device'
transmission_device['monitor type'] = '2D Z-Normal'
transmission_device['x'] = 0 * 1e-6
transmission_device['x span'] = device_size_lateral_um * 1e-6
transmission_device['y'] = 0 * 1e-6
transmission_device['y span'] = device_size_lateral_um * 1e-6
transmission_device['z'] = device_vertical_maximum_um * 1e-6
transmission_device['override global monitor settings'] = 1
transmission_device['use wavelength spacing'] = 1
transmission_device['use source limits'] = 0
transmission_device['frequency points'] = num_eval_frequency_points
transmission_device['minimum wavelength'] = transmission_min_um * 1e-6
transmission_device['maximum wavelength'] = transmission_max_um * 1e-6
transmission_device.enabled = 1


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
bayer_filter = LayeredLithographyAMPostprocessBayerFilter.LayeredLithographyAMBayerFilter(
	bayer_filter_size_voxels,
	[min_device_permittivity, max_device_permittivity],
	init_permittivity_0_1_scale,
	num_vertical_layers,
	spacer_size_voxels,
	last_layer_permittivities )

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um + 0.5 * mesh_spacing_um, device_vertical_maximum_um - 0.5 * mesh_spacing_um, simulated_device_voxels_vertical)

final_design_variable = np.load( projects_directory_location + "cur_design_variable.npy" )
bayer_filter.update_filters( num_epochs - 1 )
bayer_filter.set_design_variable( final_design_variable )

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	# fdtd_hook.switchtolayout()
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	fdtd_hook.select( angled_source['name'] )
	fdtd_hook.set( 'enabled', 0 )

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

spatial_limits_device_um = [
	[ -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um ],
	[ -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um ],
	[ device_vertical_minimum_um, device_vertical_maximum_um ]
]

interpolated_size = [ device_voxels_lateral, device_voxels_lateral, device_voxels_vertical ]

eval_forward_idx = 0

#
# Run the optimization
#
# fdtd_hook.switchtolayout()

def blur_size_to_sigma( blur_size_um ):
	return ( ( ( blur_size_um / design_spacing_um ) - 1 ) / 2 )

# blur_sizes_um = [ 0, 0.075, 0.1, 0.125 ]
blur_sizes_um = [ 0, 0.1 ]
blur_sigmas = [ blur_size_to_sigma( x ) for x in blur_sizes_um ]
colors = [ 'b', 'g', 'r', 'g' ]
# linestyles = [ '-', '-', '-', '--' ]
linestyles = [ '--', '-.', '-', ':', ]
# linestyles = [ '--', '-' ]

# plt.clf()

num_angles = 4
angle_phi = 45
min_angle_degrees = 0
max_angle_degrees = 12
angles_degrees = np.linspace( min_angle_degrees, max_angle_degrees, num_angles )

angled_source = fdtd_hook.addplane()
angled_source['name'] = 'angled_source'
angled_source['angle phi'] = xy_phi_rotations[0] + angle_phi
angled_source['angle theta'] = angles_degrees[ 0 ]
angled_source['direction'] = 'Backward'
angled_source['x span'] = 1.25 * device_size_lateral_um * 1e-6
angled_source['y span'] = 1.25 * device_size_lateral_um * 1e-6
angled_source['z'] = src_maximum_vertical_um * 1e-6
# angled_source['z min'] = src_minimum_vertical_um * 1e-6
angled_source['wavelength start'] = lambda_min_um * 1e-6
angled_source['wavelength stop'] = lambda_max_um * 1e-6

for angle_idx in range( 0, num_angles ):


	disable_all_sources()
	design_import.enabled = 0
	air_bottom.enabled = 0
	substrate['index'] = background_index

	angled_source['angle theta'] = angles_degrees[ angle_idx ]
	fdtd_hook.select( angled_source['name'] )
	fdtd_hook.set( 'enabled', 1 )
	fdtd_hook.run()

	transmission_device_normalization = fdtd_hook.getresult( transmission_device['name'], 'T' )


	bayer_filter = LayeredLithographyAMPostprocessBayerFilter.LayeredLithographyAMBayerFilter(
		bayer_filter_size_voxels,
		[min_device_permittivity, max_device_permittivity],
		init_permittivity_0_1_scale,
		num_vertical_layers,
		spacer_size_voxels,
		last_layer_permittivities,
		blur_sigmas[ 1 ] )

	bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
	bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
	bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um + 0.5 * mesh_spacing_um, device_vertical_maximum_um - 0.5 * mesh_spacing_um, simulated_device_voxels_vertical)

	bayer_filter.update_filters( num_epochs - 1 )
	bayer_filter.set_design_variable( final_design_variable )

	# bayer_filter.plot_layers( 2, 3, 'Greens', projects_directory_location + "/layers_sigma_" + str( blur_sigma_idx ) + ".png" )

	cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )
	cur_index = np.sqrt( cur_permittivity )

	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
	design_import.enabled = 1
	air_bottom.enabled = 1
	substrate['index'] = index_substrate
	fdtd_hook.select( design_import[ 'name' ] )
	fdtd_hook.importnk2(cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)
	fdtd_hook.save( projects_directory_location + "/postprocess.fsp" )


	#
	# Step 1: Get all the simulations we need queued up and run in parallel and then we will
	# put all the data together later.
	#
	disable_all_sources()
	fdtd_hook.select( angled_source['name'] )
	fdtd_hook.set( 'enabled', 1 )

	fdtd_hook.run()

	#
	# Step 4: Compute the figure of merit
	#

	figure_of_merit_per_focal_spot = []
	figure_of_merit_by_focal_spot_by_wavelength = np.zeros( ( num_focal_spots, num_points_per_band ) )
	for focal_idx in range(0, num_focal_spots):
		compute_fom = 0

		# polarizations = polarizations_focal_plane_map[focal_idx]

		transmission_data = fdtd_hook.getresult( transmission_focal_monitors[focal_idx]['name'], 'T' )
		plt.plot(
			np.linspace( lambda_min_um, lambda_max_um, num_eval_frequency_points ), transmission_data[ 'T' ] / transmission_device_normalization[ 'T' ],
			color=colors[ focal_idx ],
			linestyle=linestyles[ angle_idx ],
			linewidth=2 )

plt.ylabel( 'Transmission', fontsize=16 )
plt.xlabel( 'Wavelength (um)', fontsize=16 )
plt.savefig( projects_directory_location + "/angled_test.png" )
sys.exit(0)




for blur_sigma_idx in range( 0, len( blur_sigmas ) ):
	bayer_filter = LayeredLithographyAMPostprocessBayerFilter.LayeredLithographyAMBayerFilter(
		bayer_filter_size_voxels,
		[min_device_permittivity, max_device_permittivity],
		init_permittivity_0_1_scale,
		num_vertical_layers,
		spacer_size_voxels,
		last_layer_permittivities,
		blur_sigmas[ blur_sigma_idx ] )

	bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
	bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um + 0.5 * mesh_spacing_um, 0.5 * device_size_lateral_um - 0.5 * mesh_spacing_um, simulated_device_voxels_lateral)
	bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um + 0.5 * mesh_spacing_um, device_vertical_maximum_um - 0.5 * mesh_spacing_um, simulated_device_voxels_vertical)

	bayer_filter.update_filters( num_epochs - 1 )
	bayer_filter.set_design_variable( final_design_variable )

	# bayer_filter.plot_layers( 2, 3, 'Greens', projects_directory_location + "/layers_sigma_" + str( blur_sigma_idx ) + ".png" )

	cur_permittivity = np.flip( bayer_filter.get_permittivity(), axis=2 )
	cur_index = np.sqrt( cur_permittivity )

	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
	fdtd_hook.select( design_import[ 'name' ] )
	fdtd_hook.importnk2(cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z)
	fdtd_hook.save( projects_directory_location + "/postprocess.fsp" )


	#
	# Step 1: Get all the simulations we need queued up and run in parallel and then we will
	# put all the data together later.
	#
	disable_all_sources()
	fdtd_hook.select( forward_sources[eval_forward_idx]['name'] )
	fdtd_hook.set( 'enabled', 1 )

	fdtd_hook.run()

	forward_e_fields = {}
	focal_data = [ {} for i in range( 0, num_adjoint_sources ) ]
	#
	# Step 3: Get all the forward data from the simulations
	#
	# forward_e_fields[xy_names[eval_forward_idx]] = get_efield_interpolated(
	# 	design_efield_monitor['name' ],
	# 	spatial_limits_device_um,
	# 	interpolated_size
	# )

	for adj_src_idx in range(0, num_adjoint_sources):
		# pull_focal_data = get_complex_monitor_data( focal_monitors[ adj_src_idx ][ 'name' ], 'E' )
		pull_focal_data = get_efield( focal_monitors[ adj_src_idx ][ 'name' ] )
		# pull_focal_data = pull_focal_data[ :, :, 0, 0, 0 ]
		pull_focal_data = pull_focal_data[ :, 0, 0, 0, : ]
		focal_data[ adj_src_idx ][ xy_names[ eval_forward_idx ] ] = pull_focal_data

	#
	# Step 4: Compute the figure of merit
	#

	figure_of_merit_per_focal_spot = []
	figure_of_merit_by_focal_spot_by_wavelength = np.zeros( ( num_focal_spots, num_points_per_band ) )
	for focal_idx in range(0, num_focal_spots):
		compute_fom = 0

		# polarizations = polarizations_focal_plane_map[focal_idx]

		transmission_data = fdtd_hook.getresult( transmission_focal_monitors[focal_idx]['name'], 'T' )
		plt.plot(
			np.linspace( lambda_min_um, lambda_max_um, num_eval_frequency_points ), -transmission_data[ 'T' ],
			color=colors[ focal_idx ],
			linestyle=linestyles[ blur_sigma_idx ],
			linewidth=2 )

		# for polarization_idx in range(0, len(polarizations)):
		# 	get_focal_data = focal_data[ focal_idx ][ polarizations[ polarization_idx ] ]

		# 	max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[focal_idx][0] : spectral_focal_plane_map[focal_idx][1] : 1]
		# 	total_weighting = weight_focal_plane_map[focal_idx] / max_intensity_weighting

		# 	for spectral_idx in range(0, total_weighting.shape[0]):
		# 		get_fom_by_wl = np.sum(
		# 			np.abs(get_focal_data[:, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 *
		# 				total_weighting[spectral_idx]
		# 		)

		# 		figure_of_merit_by_focal_spot_by_wavelength[ focal_idx, spectral_idx ] += get_fom_by_wl

		# 		compute_fom += get_fom_by_wl

		# figure_of_merit_per_focal_spot.append(compute_fom)

legend = [
	'No Blur Q1', 'No Blur Q2', 'No Blur Q3', 'No Blur Q4',
	'50nm Blur Q1', '50nm Blur Q2', '50nm Blur Q3', '50nm Blur Q4',
	'100nm Blur Q1', '100nm Blur Q2', '100nm Blur Q3', '100nm Blur Q4',
	'150nm Blur Q1', '150nm Blur Q2', '150nm Blur Q3', '150nm Blur Q4',
]
# plt.legend( legend )
plt.ylabel( 'Transmission', fontsize=16 )
plt.xlabel( 'Wavelength (um)', fontsize=16 )
# plt.show()
plt.savefig( projects_directory_location + "/blur_test_less_bin.png" )

