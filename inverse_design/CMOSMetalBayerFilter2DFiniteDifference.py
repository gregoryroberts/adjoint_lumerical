import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSMetalBayerFilter2DParameters import *
import CMOSMetalBayerFilter2D

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


#
# Create project folder and save out the parameter file for documentation for this optimization
#
project_subfolder = ""
if len(sys.argv) > 1:
	project_subfolder = "/" + sys.argv[1] + "/"

use_random_design_seed = False
if len(sys.argv) > 3:
	random_seed = int( sys.argv[2] )
	np.random.seed( random_seed )
	use_random_design_seed = True
	step_size_multiplier = float( sys.argv[3] )
	adaptive_step_size *= step_size_multiplier

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/')) + project_subfolder

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/CMOSMetalBayerFilter2DParameters.py", projects_directory_location + "/ArchiveCMOSMetalBayerFilter.py")

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
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[extracted_data_name])

	# end_time = time.time()

	# print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	data = get_monitor_data(monitor_name, monitor_field)
	return (data['real'] + np.complex(0, 1) * data['imag'])

#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['dimension'] = '2D'
fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
fdtd['y max'] = fdtd_region_maximum_vertical_um * 1e-6
fdtd['y min'] = fdtd_region_minimum_vertical_um * 1e-6
fdtd['mesh type'] = 'uniform'
fdtd['define x mesh by'] = 'number of mesh cells'
fdtd['define y mesh by'] = 'number of mesh cells'
fdtd['mesh cells x'] = fdtd_region_minimum_lateral_voxels
fdtd['mesh cells y'] = fdtd_region_minimum_vertical_voxels
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

forward_src = fdtd_hook.addtfsf()
forward_src['name'] = 'forward_src'
forward_src['polarization angle'] = 90
forward_src['direction'] = 'Backward'
forward_src['x span'] = lateral_aperture_um * 1e-6
forward_src['y max'] = src_maximum_vertical_um * 1e-6
forward_src['y min'] = src_minimum_vertical_um * 1e-6
forward_src['wavelength start'] = lambda_min_um * 1e-6
forward_src['wavelength stop'] = lambda_max_um * 1e-6

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	fdtd_hook.switchtolayout()

	forward_src.enabled = 0

	for adj_src_idx in range(0, num_adjoint_sources):
		(adjoint_sources[adj_src_idx]).enabled = 0



#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = []

for adj_src_idx in range(0, num_adjoint_sources):
	adj_src = fdtd_hook.adddipole()
	adj_src['name'] = 'adj_src_' + str(adj_src_idx)
	adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
	adj_src['y'] = adjoint_vertical_um * 1e-6
	adj_src['theta'] = 0
	adj_src['phi'] = 0
	adj_src['wavelength start'] = lambda_min_um * 1e-6
	adj_src['wavelength stop'] = lambda_max_um * 1e-6

	adjoint_sources.append(adj_src)

#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
# design_efield_monitor['monitor type'] = '2D'
design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
design_efield_monitor['y min'] = designable_device_vertical_minimum_um * 1e-6
design_efield_monitor['y max'] = designable_device_vertical_maximum_um * 1e-6
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
	focal_monitor['y'] = adjoint_vertical_um * 1e-6
	focal_monitor['override global monitor settings'] = 1
	focal_monitor['use linear wavelength spacing'] = 1
	focal_monitor['use source limits'] = 1
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)

for adj_src in range(0, num_adjoint_sources):
	transmission_monitor = fdtd_hook.addpower()
	transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
	transmission_monitor['monitor type'] = 'Linear X'
	transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_monitor['x span'] = ( 1 / 3. ) * device_size_lateral_um * 1e-6
	transmission_monitor['y'] = adjoint_vertical_um * 1e-6
	transmission_monitor['override global monitor settings'] = 1
	transmission_monitor['use linear wavelength spacing'] = 1
	transmission_monitor['use source limits'] = 1
	transmission_monitor['frequency points'] = num_eval_frequency_points


#
# Build the dielectric stack on top of the metal
#
num_dielectric_layers = len( top_dielectric_layer_thickness_um )
for dielectric_layer_idx in range( 0, num_dielectric_layers ):
	dielectric_layer = fdtd_hook.addrect()
	dielectric_layer['name'] = 'dielectric_layer_' + str( dielectric_layer_idx )
	dielectric_layer['x span'] = fdtd_region_size_lateral_um * 1e-6

	dielectric_layer_z_min_um = dielectric_stack_start_um + np.sum( top_dielectric_layer_thickness_um[ 0 : dielectric_layer_idx ] )
	dielectric_layer_z_max_um = dielectric_layer_z_min_um + top_dielectric_layer_thickness_um[ dielectric_layer_idx ]

	dielectric_layer['y min'] = dielectric_layer_z_min_um * 1e-6
	dielectric_layer['y max'] = dielectric_layer_z_max_um * 1e-6

	dielectric_layer['index'] = top_dielectric_layer_refractice_index[ dielectric_layer_idx ]


#
# Make bottom layer reflective for a reflective device.  We will do this with an nk2 material that we import the maximum
# real and imaginary permittivity parts in for that we are using for the design.  Thus, it will reflect and account for
# metallic loss.
#
# metal_reflector_import = fdtd_hook.addimport()
# metal_reflector_import['name'] = 'bottom_metal_reflector'
# metal_reflector_import['x span'] = fdtd_region_size_lateral_um * 1e-6
# metal_reflector_import['y min'] = bottom_metal_reflector_start_um * 1e-6
# metal_reflector_import['y max'] = bottom_metal_reflector_end_um * 1e-6

# Note - why does it look like it follows one path after initial optimization.  The first move basically shows you the structure.
# is there something physical here? how true is this? can you quantify it? PCA (kind of like Phil mentioned that one time, I think
# it was an interesting point)

# metal_reflector_permittivity = (
# 		# ( max_real_permittivity + 1j * max_imag_permittivity ) *
# 		( reflector_real_permittivity + 1j * reflector_imag_permittivity ) *
# 		np.ones( ( fdtd_region_size_lateral_voxels, bottom_metal_reflector_size_vertical_voxels, 2 ) )
# 	)
# metal_reflector_index = permittivity_to_index( metal_reflector_permittivity )

# metal_reflector_region_x = 1e-6 * np.linspace(-0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, fdtd_region_size_lateral_voxels)
# metal_reflector_region_y = 1e-6 * np.linspace(bottom_metal_reflector_start_um, bottom_metal_reflector_end_um, bottom_metal_reflector_size_vertical_voxels)
# metal_reflector_region_z = 1e-6 * np.linspace(-0.51, 0.51, 2)

# fdtd_hook.select('bottom_metal_reflector')
# fdtd_hook.importnk2(metal_reflector_index, metal_reflector_region_x, metal_reflector_region_y, metal_reflector_region_z)


#
# Add device region and create device permittivity
#

min_device_permittivity = min_real_permittivity + 1j * min_imag_permittivity
max_device_permittivity = max_real_permittivity + 1j * max_imag_permittivity

#
# Here, many devices will actually be added, one for each actually designable region.  When the region is not
# designable, we will just add a block of material there.  This applies for things like the via and capping layers
#
number_device_layers = len( layer_thicknesses_um )

bayer_filters = []
bayer_filter_regions_y = []
design_imports = []

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(-0.51, 0.51, 2)

for device_layer_idx in range( 0, number_device_layers ):
	layer_vertical_maximum_um = designable_device_vertical_maximum_um - np.sum( layer_thicknesses_um[ 0 : device_layer_idx ] )
	layer_vertical_minimum_um = layer_vertical_maximum_um - layer_thicknesses_um[ device_layer_idx ]

	if is_layer_designable[ device_layer_idx ]:

		one_vertical_layer = 1
		layer_bayer_filter_size_voxels = np.array([device_voxels_lateral, layer_thicknesses_voxels[device_layer_idx], 1])
		layer_bayer_filter = CMOSMetalBayerFilter2D.CMOSMetalBayerFilter2D(
			layer_bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_permittivity_0_1_scale, one_vertical_layer)

		if use_random_design_seed:
			random_design_seed = 0.1 * np.random.random( layer_bayer_filter_size_voxels )
			layer_bayer_filter.set_design_variable( random_design_seed )

		bayer_filter_region_y = 1e-6 * np.linspace(layer_vertical_minimum_um, layer_vertical_maximum_um, layer_thicknesses_voxels[device_layer_idx])

		# bayer_filter.set_design_variable( np.load( projects_directory_location + "/cur_design_variable.npy" ) )

		layer_import = fdtd_hook.addimport()
		layer_import['name'] = 'layer_import_' + str( device_layer_idx )
		layer_import['x span'] = device_size_lateral_um * 1e-6
		layer_import['y min'] = layer_vertical_minimum_um * 1e-6
		layer_import['y max'] = layer_vertical_maximum_um * 1e-6
		layer_import['z min'] = -0.51 * 1e-6
		layer_import['z max'] = 0.51 * 1e-6

		print("For layer " + str( device_layer_idx ), " the min um spot is " + str( layer_vertical_minimum_um ) + " and max um spot is " + str( layer_vertical_maximum_um ) )

		bayer_filters.append( layer_bayer_filter )
		bayer_filter_regions_y.append( bayer_filter_region_y )
		design_imports.append( layer_import )


	else:
		blank_layer = fdtd_hook.addrect()
		blank_layer['name'] = 'device_layer_' + str(device_layer_idx)
		blank_layer['x span'] = device_size_lateral_um * 1e-6
		blank_layer['y min'] = layer_vertical_minimum_um * 1e-6
		blank_layer['y max'] = layer_vertical_maximum_um * 1e-6
		blank_layer['index'] = layer_background_index[ device_layer_idx ]



#
# Add blocks of dielectric on the side of the designable region because we will be leaving those as blank, unpatterned dielectric material
# Would it be better to have this be a part of the bayer filter material inmport and jus tnot modify it (i.e. - mask out any changes to it).  I'm
# thinking of how the mesh behaves here between these interacting objects.  For now, we will add the blocks around the side because this will make
# it simpler at first and then we can move towards making a more sophisticated device class or subclass.
# Further, in reality, this may be a stack of material in general.  However, it will be mostly the low-K dielctric background material so we will assume
# this is not a stratified stack and is actaully just a single piece of low index material background.
#
extra_lateral_space_per_side_um = 0.5 * ( fdtd_region_size_lateral_um - device_size_lateral_um )
extra_lateral_space_offset_um = 0.5 * ( device_size_lateral_um + extra_lateral_space_per_side_um )

def side_to_string( side_number ):
	side_integer = int( side_number )
	if side_integer < 0:
		return ( "n" + str( np.abs( side_integer ) ) )
	else:
		return str( side_integer )

device_background_side_x = [ -1, 1 ]#, 0, 0 ]
# device_background_side_y = [ 0, 0, -1, 1 ]

for device_background_side_idx in range( 0, 2 ):
	side_x = device_background_side_x[ device_background_side_idx ]

	side_block = fdtd_hook.addrect()

	side_block['name'] = 'device_background_' + side_to_string( side_x )
	side_block['y min'] = designable_device_vertical_minimum_um * 1e-6
	side_block['y max'] = designable_device_vertical_maximum_um * 1e-6
	side_block['x'] = side_x * extra_lateral_space_offset_um * 1e-6
	side_block['x span'] = (
		np.abs( side_x ) * extra_lateral_space_per_side_um +
		( 1 - np.abs( side_x ) ) * fdtd_region_size_lateral_um ) * 1e-6

	side_block['index'] = device_background_index



#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = fixed_step_size

def import_previous_seed( filename_prefix ):
	fdtd_hook.switchtolayout()
	print( "Importing from a previous seed point with filename prefix " + filename_prefix )

	for device_layer_idx in range( 0, len( bayer_filters ) ):
		bayer_filter = bayer_filters[ device_layer_idx ]

		import_mirrored_data = np.load( projects_directory_location + '/' + filename_prefix + str( device_layer_idx ) + '.npy' )
		bayer_filter.set_design_variable( import_mirrored_data )

		cur_permittivity = bayer_filter.get_permittivity()
		cur_index = permittivity_to_index( cur_permittivity )

		design_import = design_imports[ device_layer_idx ]

		fdtd_hook.select( design_import[ "name" ] )
		fdtd_hook.importnk2( cur_index, bayer_filter_region_x, bayer_filter_regions_y[ device_layer_idx ], bayer_filter_region_z )

def import_bayer_filters():
	fdtd_hook.switchtolayout()

	for device_layer_idx in range( 0, len( bayer_filters ) ):
		bayer_filter = bayer_filters[ device_layer_idx ]

		cur_permittivity = bayer_filter.get_permittivity()
		cur_index = permittivity_to_index( cur_permittivity )

		import_index = np.zeros( ( cur_index.shape[ 0 ], cur_index.shape[ 1 ], 2 ), dtype=np.complex )
		import_index[ :, :, 0 ] = cur_index[ :, :, 0 ]
		import_index[ :, :, 1 ] = cur_index[ :, :, 0 ]

		design_import = design_imports[ device_layer_idx ]

		fdtd_hook.select( design_import[ "name" ] )
		fdtd_hook.importnk2( import_index, bayer_filter_region_x, bayer_filter_regions_y[ device_layer_idx ], bayer_filter_region_z )

def update_bayer_filters( device_step_real, device_step_imag, step_size ):
	max_max_design_variable_change = 0
	min_max_design_variable_change = 1.0
	
	log_file = open(projects_directory_location + "/log.txt", 'a+')

	for device_layer_idx in range( 0, len( bayer_filters ) ):
		bayer_filter = bayer_filters[ device_layer_idx ]

		cur_design_variable = bayer_filter.get_design_variable()
		last_design_variable = cur_design_variable.copy()

		layer_vertical_minimum_um = 1e6 * bayer_filter_regions_y[ device_layer_idx ][ 0 ]
		layer_vertical_maximum_um = 1e6 * bayer_filter_regions_y[ device_layer_idx ][ -1 ]

		layer_vertical_minimum_voxels = int( ( layer_vertical_minimum_um - designable_device_vertical_minimum_um ) / mesh_spacing_um )
		layer_vertical_maximum_voxels = layer_vertical_minimum_voxels + len( bayer_filter_regions_y[ device_layer_idx ] )

		bayer_filter.step(
			device_step_real[ :, layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, : ],
			device_step_imag[ :, layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, : ],
			step_size )

		cur_design_variable = bayer_filter.get_design_variable()

		average_design_variable_change = np.mean( np.abs( last_design_variable - cur_design_variable ) )
		max_design_variable_change = np.max( np.abs( last_design_variable - cur_design_variable ) )

		max_max_design_variable_change = np.maximum( max_max_design_variable_change, max_design_variable_change )
		min_max_design_variable_change = np.maximum( min_max_design_variable_change, max_design_variable_change )

		log_file.write("The layer number is " + str( device_layer_idx ) + " and thickness is " + str( layer_vertical_maximum_um - layer_vertical_minimum_um )  + "\n")
		log_file.write( "This bayer filter is expecting something of size " + str( bayer_filter.size ) + " and it has been fed something of size " +
			str( device_step_real[ :, layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, : ].shape )  + "\n")
		log_file.write( "The gradient information is being taken between " + str( layer_vertical_minimum_voxels ) + " and " + str( layer_vertical_maximum_voxels )
			+ "\nout of total number height of " + str( designable_device_voxels_vertical ) + " (voxels)\n")
		log_file.write( "The max amount the density is changing for layer " + str( device_layer_idx ) + " is " + str( max_design_variable_change ) + "\n")
		log_file.write( "The mean amount the density is changing for layer " + str( device_layer_idx ) + " is " + str( average_design_variable_change ) + "\n")
		log_file.write("\n")

		np.save(projects_directory_location + "/cur_design_variable_" + str( device_layer_idx ) + ".npy", cur_design_variable)

	log_file.write("\n\n")
	log_file.close()

	if max_max_design_variable_change > desired_max_max_design_change:
		return 0.5
	elif min_max_design_variable_change < desired_min_max_design_change:
		return 2

	return 1


fd_layer = 2
fd_bayer_filter = bayer_filters[ fd_layer ]

fd_layer_vertical_minimum_um = 1e6 * bayer_filter_regions_y[ fd_layer ][ 0 ]
fd_layer_vertical_maximum_um = 1e6 * bayer_filter_regions_y[ fd_layer ][ -1 ]
fd_layer_vertical_minimum_voxels = int( ( fd_layer_vertical_minimum_um - designable_device_vertical_minimum_um ) / mesh_spacing_um )
fd_layer_vertical_maximum_voxels = fd_layer_vertical_minimum_voxels + len( bayer_filter_regions_y[ fd_layer ] )

fd_scan_x = np.arange(0, device_voxels_lateral)

fd_values_lambda_low_spot_left = np.zeros( device_voxels_lateral )
fd_values_lambda_high_spot_center = np.zeros( device_voxels_lateral )

disable_all_sources()
forward_src.enabled = 1
fdtd_hook.run()

forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

focal_data = []
for adj_src_idx in range(0, num_adjoint_sources):
    focal_data.append(
        get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )

#
# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
# gradients for x- and y-polarized forward sources.
#
reversed_field_shape = [1, designable_device_voxels_vertical, device_voxels_lateral]
xy_polarized_gradients_lambda_low_spot_left = np.zeros(reversed_field_shape, dtype=np.complex)
xy_polarized_gradients_lambda_high_spot_center = np.zeros(reversed_field_shape, dtype=np.complex)

adj_values_lambda_low_spot_left = np.zeros( device_voxels_lateral )
adj_values_lambda_high_spot_center = np.zeros( device_voxels_lateral )

adj_src_idxs = [ 0, 1 ]
lambda_idxs = [ 0, num_design_frequency_points - 1 ]
adj_calc_gradients = [ xy_polarized_gradients_lambda_low_spot_left, xy_polarized_gradients_lambda_high_spot_center ]
adj_values = [ adj_values_lambda_low_spot_left, adj_values_lambda_high_spot_center ]

fd_grad_baseline = [ 0, 0 ]

for adj_calc_idx in range( 0, len( adj_src_idxs ) ):
    adj_src_idx = adj_src_idxs[ adj_calc_idx ]
    lambda_idx = lambda_idxs[ adj_calc_idx ]

    max_intensity_weighting = max_intensity_by_wavelength[lambda_idx]
    total_weighting = max_intensity_weighting * weight_focal_plane_map[adj_src_idx]

    fd_grad_baseline[ adj_calc_idx ] = np.sum( np.abs(focal_data[adj_src_idx][:, lambda_idx, 0, 0, 0])**2 / total_weighting )

    disable_all_sources()
    (adjoint_sources[adj_src_idx]).enabled = 1
    fdtd_hook.run()

    adjoint_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

    source_weight = np.conj(
        focal_data[adj_src_idx][2, lambda_idx, 0, 0, 0])

    adj_calc_gradients[ adj_calc_idx ] = np.sum(
        (source_weight * total_weighting) *
        adjoint_e_fields[:, lambda_idx, :, :, :] *
        forward_e_fields[:, lambda_idx, :, :, :],
        axis=0)

    #
    # Step 4: Step the design variable.
    #
    device_gradient_real = 2 * np.real( adj_calc_gradients[ adj_calc_idx ] )
    device_gradient_imag = 2 * np.imag( adj_calc_gradients[ adj_calc_idx ] )
    # Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
    # [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
    device_gradient_real = np.swapaxes(device_gradient_real, 0, 2)
    device_gradient_imag = np.swapaxes(device_gradient_imag, 0, 2)

    density_gradient = fd_bayer_filter.backpropagate(
        device_gradient_real[ :, fd_layer_vertical_minimum_voxels : fd_layer_vertical_maximum_voxels, : ],
        device_gradient_imag[ :, fd_layer_vertical_minimum_voxels : fd_layer_vertical_maximum_voxels, : ],
    )

    adj_values[ adj_calc_idx ] = density_gradient[ :, 0 ]




h = 0.005
for fd_x in range( 0, len( fd_scan_x ) ):
    print("Working on fd_x = " + str( fd_x ) + " out of " + str( len( fd_scan_x ) ) )
    cur_design_variable = fd_bayer_filter.get_design_variable()

    cur_design_variable[ fd_x, : ] += h
    fd_bayer_filter.set_design_variable( cur_design_variable )
    import_bayer_filters()

    disable_all_sources()
    forward_src.enabled = 1
    fdtd_hook.run()

    focal_data = []
    for adj_src_idx in range(0, num_adjoint_sources):
        focal_data.append(
            get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )

    fd_perturb = [ 0, 0 ]

    for adj_calc_idx in range( 0, len( adj_src_idxs ) ):
        adj_src_idx = adj_src_idxs[ adj_calc_idx ]
        lambda_idx = lambda_idxs[ adj_calc_idx ]

        max_intensity_weighting = max_intensity_by_wavelength[lambda_idx]
        total_weighting = max_intensity_weighting * weight_focal_plane_map[adj_src_idx]

        fd_perturb[ adj_calc_idx ] = np.sum( np.abs(focal_data[adj_src_idx][:, lambda_idx, 0, 0, 0])**2 / total_weighting )

    fd_values_lambda_low_spot_left[ fd_x ] = ( fd_perturb[ 0 ] - fd_grad_baseline[ 0 ] ) / h
    fd_values_lambda_high_spot_center[ fd_x ] = ( fd_perturb[ 1 ] - fd_grad_baseline[ 1 ] ) / h


    cur_design_variable[ fd_x, : ] -= h
    fd_bayer_filter.set_design_variable( cur_design_variable )
    import_bayer_filters()

np.save( "fd_values_lambda_high_spot_center_dielectric.npy", fd_values_lambda_high_spot_center )
np.save( "fd_values_lambda_low_spot_left_dielectric.npy", fd_values_lambda_low_spot_left )
np.save( "adj_values_lambda_high_spot_center_dielectric.npy", adj_values[ 0 ] )
np.save( "adj_values_lambda_low_spot_left_dielectric.npy", adj_values[ 1 ] )

# adj_values = [ None, None ]

# fd_values_lambda_high_spot_center = np.load( "fd_values_lambda_high_spot_center_dielectric.npy" )
# fd_values_lambda_low_spot_left = np.load( "fd_values_lambda_low_spot_left_dielectric.npy" )
# adj_values[ 0 ] = np.load( "adj_values_lambda_high_spot_center_dielectric.npy" )
# adj_values[ 1 ] = np.load( "adj_values_lambda_low_spot_left_dielectric.npy" )

print(np.mean( fd_values_lambda_high_spot_center / ( 1e-4 + adj_values[0] ) ))
print("\n\n")
print(np.mean( fd_values_lambda_low_spot_left / ( 1e-4 + adj_values[1] ) ))
print("\n\n")
print(fd_values_lambda_high_spot_center)
print(adj_values[0])

plt.plot( adj_values[ 0 ] / np.max( np.abs( adj_values[ 0 ] ) ), color='g', linewidth=2 )
plt.plot( fd_values_lambda_low_spot_left / np.max( np.abs( fd_values_lambda_low_spot_left ) ), color='r', linewidth=2, linestyle='--' )
plt.show()

plt.plot( adj_values[ 1 ] / np.max( np.abs( adj_values[ 1 ] ) ), color='g', linewidth=2 )
plt.plot( fd_values_lambda_high_spot_center / np.max( np.abs( fd_values_lambda_high_spot_center ) ), color='r', linewidth=2, linestyle='--' )
plt.show()


