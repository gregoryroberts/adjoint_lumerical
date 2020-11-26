import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LayeredLithographyPostsParameters import *

from skimage import measure

# import imp
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

import csv

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LayeredLithographyPostsParameters.py", projects_directory_location + "/LayeredLithographyPostsParameters.py")

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


transmission_monitors = []

for adj_src in range(0, num_adjoint_sources):
	transmission_monitor = fdtd_hook.addpower()
	transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
	transmission_monitor['monitor type'] = '2D Z-normal'
	transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
	transmission_monitor['x span'] = 0.5 * device_size_lateral_um * 1e-6
	transmission_monitor['y span'] = 0.5 * device_size_lateral_um * 1e-6
	transmission_monitor['z'] = adjoint_vertical_um * 1e-6
	transmission_monitor['override global monitor settings'] = 1
	transmission_monitor['use wavelength spacing'] = 1
	transmission_monitor['use source limits'] = 1
	transmission_monitor['frequency points'] = num_design_frequency_points

	transmission_monitors.append(transmission_monitor)


#
# Add device region and create device permittivity
#
# design_import = fdtd_hook.addimport()
# design_import['name'] = 'design_import'
# design_import['x span'] = device_size_lateral_um * 1e-6
# design_import['y span'] = device_size_lateral_um * 1e-6
# design_import['z max'] = device_vertical_maximum_um * 1e-6
# design_import['z min'] = device_vertical_minimum_um * 1e-6


# bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
# bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
# bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)



filepath = projects_directory_location + "/"

f = h5py.File( filepath + 'tio2_sio2_binary_index.mat', 'r' )
get_index = f[ 'bin_index' ]
get_index = np.swapaxes( get_index, 0, 2 )

low_index = np.min( get_index )
high_index = np.max( get_index )

mid_index = 0.5 * ( low_index + high_index )

bin_index = 1.0 * np.greater_equal( get_index, mid_index )

start = 10
step = 20
num = 5

um_per_layer = 0.4

# post_filename = projects_directory_location + "/color_splitter.csv"

# with open(post_filename, 'w', newline='') as csvfile:
# 	post_writer = csv.writer(csvfile, delimiter=',',
# 							quotechar='|', quoting=csv.QUOTE_MINIMAL)
# 	post_writer.writerow(['X center (um)', 'Y center (um)', 'Z minimum (um)', 'Z maximum (um)', 'Radius (um)'])

for layer in range( 0, num ):
	data = np.squeeze( bin_index[ :, :, start + step * layer ] )

	contours = measure.find_contours(data, 0.5)


	post_data = []

	post_idx = 0
	for contour in contours:
		M = measure.moments_coords(contour, order=1)
		area = M[ 0, 0 ]
		radius = np.sqrt( area / np.pi )
		center_x = ( M[ 1, 0 ] / M[ 0, 0 ] ) * device_size_lateral_um / device_voxels_lateral
		center_y = ( M[ 0, 1 ] / M[ 0, 0 ] ) * device_size_lateral_um / device_voxels_lateral
		draw_circle = plt.Circle((center_x, center_y), radius, fill=False, edgecolor='r', linewidth=2)

		z_start_um = layer * um_per_layer
		z_end_um = z_start_um + um_per_layer

		adjusted_area = area * um_per_layer / ( um_per_layer - ( lambda_max_um / 20. ) )
		radius_um = np.sqrt( adjusted_area / np.pi ) * device_size_lateral_um / device_voxels_lateral

		post_data.append( [ radius_um, center_x, center_y, z_start_um, z_end_um ] )


	for post_idx in range( 0, len( post_data ) ):
		get_post = post_data[ post_idx ]
		sorted_posts = sorted( post_data, key=lambda other_post: np.sqrt( ( other_post[ 1 ] - get_post[ 1 ] )**2 + ( other_post[ 2 ] - get_post[ 2 ] )**2 ) - other_post[ 0 ] )

		limiting_post = sorted_posts[ 1 ]
		radius_reduction = -( lambda_max_um / 20. ) + np.sqrt( ( limiting_post[ 1 ] - get_post[ 1 ] )**2 + ( limiting_post[ 2 ] - get_post[ 2 ] )**2 ) - limiting_post[ 0 ] - get_post[ 0 ]

		if radius_reduction >= 0:
			continue
		else:
			print( radius_reduction )
			post_data[ post_idx ][ 0 ] += radius_reduction

	filtered_posts = []

	for post_idx in range( 0, len( post_data ) ):
		if post_data[ post_idx ][ 0 ] >= ( lambda_max_um / 20. ):
			filtered_posts.append( post_data[ post_idx ] )

	for post_idx in range( 0, len( filtered_posts ) ):
		get_filtered_post = filtered_posts[ post_idx ]

		radius_um = get_filtered_post[ 0 ]
		center_x_um = get_filtered_post[ 1 ]
		center_y_um = get_filtered_post[ 2 ]
		z_start_um = get_filtered_post[ 3 ]
		z_end_um = get_filtered_post[ 4 ]

		# post_writer.writerow(
		# 	[ -0.5 * device_size_lateral_um + center_x_um, -0.5 * device_size_lateral_um + center_y_um,
		# 	z_start_um + ( 0.5 * lambda_max_um / 20. ), z_end_um - ( 0.5 * lambda_max_um / 20. ),
		# 	radius_um ] )

		post = fdtd_hook.addcircle()
		post[ 'name' ] = 'layer_' + str( layer ) + '_post_' + str( post_idx )
		post[ 'radius' ] = radius_um * 1e-6
		post[ 'x' ] = ( -0.5 * device_size_lateral_um + center_x_um  ) * 1e-6
		post[ 'y' ] = ( -0.5 * device_size_lateral_um + center_y_um  ) * 1e-6
		post[ 'z min' ] = ( z_start_um + ( 0.5 * lambda_max_um / 20. ) ) * 1e-6
		post[ 'z max' ] = ( z_end_um - ( 0.5 * lambda_max_um / 20. ) ) * 1e-6
		post[ 'index' ] = max_device_index


fdtd_hook.save( projects_directory_location + '/optimization.fsp' )

