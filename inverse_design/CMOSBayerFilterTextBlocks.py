import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSBayerFilterParametersCopper import *
import CMOSBayerFilter

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

shutil.copy2(python_src_directory + "/CMOSBayerFilterParameters.py", projects_directory_location + "/ArchiveCMOSBayerFilterParameters.py")

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
design_efield_monitor['use linear wavelength spacing'] = 1
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
	focal_monitor['use linear wavelength spacing'] = 1
	focal_monitor['use source limits'] = 1
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)



#
# Add device region and create device permittivity
#
# design_import = fdtd_hook.addimport()
# design_import['name'] = 'design_import'
# design_import['x span'] = device_size_lateral_um * 1e-6
# design_import['y span'] = device_size_lateral_um * 1e-6
# design_import['z max'] = device_vertical_maximum_um * 1e-6
# design_import['z min'] = device_vertical_minimum_um * 1e-6

bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
bayer_filter = CMOSBayerFilter.CMOSBayerFilter(bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_permittivity_0_1_scale, num_vertical_layers)

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

fdtd_hook.redrawoff()

device_group = fdtd_hook.addstructuregroup()
device_group['name'] = 'device_volume'

x_line_group = fdtd_hook.addstructuregroup()
x_line_group_name = 'device_x_line_' + str(0)
x_line_group['name'] = x_line_group_name

next_rect = fdtd_hook.addrect()
voxel_name = 'voxel_' + str(0)
next_rect['name'] = voxel_name
next_rect['x min'] = ( (-device_size_lateral_um / 2.) + 0 * mesh_spacing_um ) * 1e-6
next_rect['x max'] = ( (-device_size_lateral_um / 2.) + ( 0 + 1 ) * mesh_spacing_um ) * 1e-6
next_rect['y min'] = ( (-device_size_lateral_um / 2.) + 0 * mesh_spacing_um ) * 1e-6
next_rect['y max'] = ( (-device_size_lateral_um / 2.) + ( 0 + 1 ) * mesh_spacing_um ) * 1e-6
next_rect['z min'] = ( device_vertical_minimum_um + 0 * mesh_spacing_um ) * 1e-6
next_rect['z max'] = ( device_vertical_minimum_um + ( 0 + 1 ) * mesh_spacing_um ) * 1e-6
fdtd_hook.select(voxel_name)
fdtd_hook.addtogroup(x_line_group_name)

for x in range(1, device_voxels_lateral):
	fdtd_hook.select(voxel_name)
	fdtd_hook.copy(mesh_spacing_um * 1e-6, 0, 0)
	fdtd_hook.set('name', 'voxel_' + str(x))
	# fdtd_hook.addtogroup(x_line_group_name)

	# next_rect = fdtd_hook.addrect()
	# voxel_name = 'voxel_' + str(x)
	# next_rect['name'] = voxel_name
	# next_rect['x min'] = ( (-device_size_lateral_um / 2.) + x * mesh_spacing_um ) * 1e-6
	# next_rect['x max'] = ( (-device_size_lateral_um / 2.) + ( x + 1 ) * mesh_spacing_um ) * 1e-6
	# next_rect['y min'] = ( (-device_size_lateral_um / 2.) + 0 * mesh_spacing_um ) * 1e-6
	# next_rect['y max'] = ( (-device_size_lateral_um / 2.) + ( 0 + 1 ) * mesh_spacing_um ) * 1e-6
	# next_rect['z min'] = ( device_vertical_minimum_um + 0 * mesh_spacing_um ) * 1e-6
	# next_rect['z max'] = ( device_vertical_minimum_um + ( 0 + 1 ) * mesh_spacing_um ) * 1e-6
	# fdtd_hook.select(voxel_name)
	# fdtd_hook.addtogroup(x_line_group_name)

xy_plane_group = fdtd_hook.addstructuregroup()
xy_plane_group_name = 'device_xy_plane_' + str(0)
xy_plane_group['name'] = xy_plane_group_name

for y in range(1, device_voxels_lateral):
	fdtd_hook.select(x_line_group_name)
	fdtd_hook.copy(0, mesh_spacing_um * 1e-6, 0)
	fdtd_hook.addtogroup(xy_plane_group_name)

fdtd_hook.select(xy_plane_group_name)
fdtd_hook.addtogroup('device_volume')




# for x in range(0, 1):#device_voxels_lateral):
# 	for y in range(0, 1):#device_voxels_lateral):

# 		next_rect = fdtd_hook.addrect()
# 		next_rect['name'] = 'voxel_' + str(x) + '_' + str(y)
# 		next_rect['x min'] = ( (-device_size_lateral_um / 2.) + x * mesh_spacing_um ) * 1e-6
# 		next_rect['x max'] = ( (-device_size_lateral_um / 2.) + ( x + 1 ) * mesh_spacing_um ) * 1e-6
# 		next_rect['y min'] = ( (-device_size_lateral_um / 2.) + y * mesh_spacing_um ) * 1e-6
# 		next_rect['y max'] = ( (-device_size_lateral_um / 2.) + ( y + 1 ) * mesh_spacing_um ) * 1e-6
# 		next_rect['z min'] = ( device_vertical_minimum_um + 0 * mesh_spacing_um ) * 1e-6
# 		next_rect['z max'] = ( device_vertical_minimum_um + ( 0 + 1 ) * mesh_spacing_um ) * 1e-6
# 		# next_rect['index'] = str( np.sqrt( get_permittivity ) )
# 		fdtd_hook.addtogroup('device_group')

# 		for z in range(1, device_voxels_vertical):
# 			fdtd_hook.copy(0, 0, mesh_spacing_um * 1e-6)
# 			fdtd_hook.set('name', 'voxel_' + str(x) + '_' + str(y) + '_' + str(z))

# 			#get_permittivity = 1 + (max_device_permittivity - 1) * (x**2 + y**2 + z**2) / (device_voxels_lateral**2 + device_voxels_lateral**2 + device_voxels_vertical**2)



fdtd_hook.redrawon()
fdtd_hook.redraw()
