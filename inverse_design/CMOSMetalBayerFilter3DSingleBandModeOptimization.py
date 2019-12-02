import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSMetalBayerFilter3DSingleBandModeParameters import *
import CMOSMetalBayerFilter3D

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

shutil.copy2(python_src_directory + "/CMOSMetalBayerFilter3DSingleBandModeParameters.py", projects_directory_location + "/ArchiveCMOSMetalBayerFilter.py")

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
fdtd['dimension'] = '3D'
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
# fdtd['dt stability factor'] = fdtd_dt_stability_factor

#
# General polarized source information
#
xy_phi_rotations = { 'x' : 0, 'y' : 90 }
xy_index_idx = { 'x' : 0, 'y' : 1 }
xy_names = ['x', 'y']


#
# Add a TFSF plane wave forward source at normal incidence
#
plane_wave_sources = {}

forward_src_xpol = fdtd_hook.addtfsf()
forward_src_xpol['name'] = 'forward_src_xpol'
forward_src_xpol['angle phi'] = xy_phi_rotations['x']
# forward_src_xpol['direction'] = 'Backward'
forward_src_xpol['direction'] = 'Forward'
forward_src_xpol['x span'] = 1.3 * device_size_lateral_um * 1e-6
forward_src_xpol['y span'] = 1.3 * device_size_lateral_um * 1e-6
# forward_src_xpol['z min'] = src_minimum_vertical_um * 1e-6
# forward_src_xpol['z max'] = src_maximum_vertical_um * 1e-6
forward_src_xpol['z min'] = src_maximum_vertical_um * 1e-6
forward_src_xpol['z max'] = fdtd_region_maximum_vertical_um * 1e-6
forward_src_xpol['wavelength start'] = src_lambda_min_um * 1e-6
forward_src_xpol['wavelength stop'] = src_lambda_max_um * 1e-6

forward_src_ypol = fdtd_hook.addtfsf()
forward_src_ypol['name'] = 'forward_src_ypol'
forward_src_ypol['angle phi'] = xy_phi_rotations['y']
# forward_src_ypol['direction'] = 'Backward'
forward_src_ypol['direction'] = 'Forward'
forward_src_ypol['x span'] = 1.3 * device_size_lateral_um * 1e-6
forward_src_ypol['y span'] = 1.3 * device_size_lateral_um * 1e-6
# forward_src_ypol['z min'] = src_minimum_vertical_um * 1e-6
# forward_src_ypol['z max'] = src_maximum_vertical_um * 1e-6
forward_src_ypol['z min'] = src_maximum_vertical_um * 1e-6
forward_src_ypol['z max'] = fdtd_region_maximum_vertical_um * 1e-6
forward_src_ypol['wavelength start'] = src_lambda_min_um * 1e-6
forward_src_ypol['wavelength stop'] = src_lambda_max_um * 1e-6

plane_wave_sources['x'] = forward_src_xpol
plane_wave_sources['y'] = forward_src_ypol

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
    fdtd_hook.switchtolayout()

    plane_wave_sources['x'].enabled = 0
    plane_wave_sources['y'].enabled = 0


#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['monitor type'] = '3D'
design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
design_efield_monitor['y span'] = device_size_lateral_um * 1e-6
design_efield_monitor['z min'] = designable_device_vertical_minimum_um * 1e-6
design_efield_monitor['z max'] = designable_device_vertical_maximum_um * 1e-6
design_efield_monitor['override global monitor settings'] = 1
design_efield_monitor['use linear wavelength spacing'] = 1
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
mode_reflection_monitor_delta_um = 0.25 * vertical_gap_size_top_um
mode_reflection_monitor = fdtd_hook.addpower()
mode_reflection_monitor['name'] = 'mode_reflection_monitor'
mode_reflection_monitor['monitor type'] = '2D Z-normal'
mode_reflection_monitor['x span'] = plane_wave_sources['x']['x span']
mode_reflection_monitor['y span'] = plane_wave_sources['x']['y span']
mode_reflection_monitor['z'] = ( src_maximum_vertical_um + mode_reflection_monitor_delta_um ) * 1e-6
mode_reflection_monitor['override global monitor settings'] = 1
mode_reflection_monitor['use linear wavelength spacing'] = 1
mode_reflection_monitor['use source limits'] = 0
mode_reflection_monitor['minimum wavelength'] = lambda_min_um * 1e-6
mode_reflection_monitor['maximum wavelength'] = lambda_max_um * 1e-6
mode_reflection_monitor['frequency points'] = num_design_frequency_points

#
# Run a normalization run for the adjoint problem
#
mode_e_fields = {}
mode_h_fields = {}

phase_corrections_reflection = {}
phase_corrections_reflection['x'] = []
phase_corrections_reflection['y'] = []

disable_all_sources()
plane_wave_sources['x'].enabled = 1
fdtd_hook.run()

mode_e_fields['x'] = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
mode_h_fields['x'] = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

mode_midpt_x = int( mode_e_fields['x'].shape[ 4 ] / 2 )
mode_midpt_y = int( mode_e_fields['x'].shape[ 3 ] / 2 )

for wl_idx in range( 0, num_design_frequency_points ):
    wavelength_um = lambda_values_um[ wl_idx ]
    phase_shift = 2 * np.pi * mode_reflection_monitor_delta_um / wavelength_um
    mode_phase = -np.angle( mode_e_fields['x'][ 2, wl_idx, 0, mode_midpt_y, mode_midpt_x ] )
    phase_corrections_reflection['x'].append( 2 * ( phase_shift + mode_phase ) )

fdtd_hook.switchtolayout()

plane_wave_sources['x']['direction'] = 'Backward'
plane_wave_sources['x']['z min'] = src_minimum_vertical_um * 1e-6
plane_wave_sources['x']['z max'] = src_maximum_vertical_um * 1e-6


disable_all_sources()
plane_wave_sources['y'].enabled = 1
fdtd_hook.run()

mode_e_fields['y'] = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
mode_h_fields['y'] = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

phase_corrections_transmission = []

for wl_idx in range( 0, num_design_frequency_points ):
    wavelength_um = lambda_values_um[ wl_idx ]
    phase_shift = 2 * np.pi * mode_reflection_monitor_delta_um / wavelength_um
    mode_phase = -np.angle( mode_e_fields['y'][ 2, wl_idx, 0, mode_midpt_y, mode_midpt_x ] )
    phase_corrections_reflection['y'].append( 2 * ( phase_shift + mode_phase ) )

fdtd_hook.switchtolayout()

plane_wave_sources['y']['direction'] = 'Backward'
plane_wave_sources['y']['z min'] = src_minimum_vertical_um * 1e-6
plane_wave_sources['y']['z max'] = src_maximum_vertical_um * 1e-6




# Add Si absorbing layer
silicon_absorbing_layer = fdtd_hook.addrect()
silicon_absorbing_layer['name'] = 'bottom_metal_absorber'
silicon_absorbing_layer['x span'] = fdtd_region_size_lateral_um * 1e-6
silicon_absorbing_layer['y span'] = fdtd_region_size_lateral_um * 1e-6
silicon_absorbing_layer['z min'] = bottom_metal_absorber_start_um * 1e-6
silicon_absorbing_layer['z max'] = bottom_metal_absorber_end_um * 1e-6
silicon_absorbing_layer['material'] = 'Si (Silicon) - Palik'

# metal_absorber_index = (
#         ( silicon_absober_index_real + 1j * silicon_absober_index_imag ) *
# 		# ( 4.32 + 1j * 0.073 ) *
# 		np.ones( ( fdtd_region_size_lateral_voxels, bottom_metal_absorber_size_vertical_voxels, 2 ) )
# 	)
# metal_reflector_index = permittivity_to_index( metal_reflector_permittivity )

# metal_absorber_region_x = 1e-6 * np.linspace(-0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, fdtd_region_size_lateral_voxels)
# metal_absorber_region_y = 1e-6 * np.linspace(bottom_metal_absorber_start_um, bottom_metal_absorber_end_um, bottom_metal_absorber_size_vertical_voxels)
# metal_absorber_region_z = 1e-6 * np.linspace(-0.51, 0.51, 2)

# fdtd_hook.select('bottom_metal_absorber')
# fdtd_hook.importnk2(metal_absorber_index, metal_absorber_region_x, metal_absorber_region_y, metal_absorber_region_z)

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
bayer_filter_regions_z = []
design_imports = []
lock_design_to_reflective = []

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
# bayer_filter_region_z = 1e-6 * np.linspace(-0.51, 0.51, 2)

for device_layer_idx in range( 0, number_device_layers ):
    layer_vertical_maximum_um = designable_device_vertical_maximum_um - np.sum( layer_thicknesses_um[ 0 : device_layer_idx ] )
    layer_vertical_minimum_um = layer_vertical_maximum_um - layer_thicknesses_um[ device_layer_idx ]

    if is_layer_designable[ device_layer_idx ]:

        one_vertical_layer = 1
        layer_bayer_filter_size_voxels = np.array([device_voxels_lateral, device_voxels_lateral, layer_thicknesses_voxels[device_layer_idx]])

        init_design = init_permittivity_0_1_scale
        layer_bayer_filter = CMOSMetalBayerFilter3D.CMOSMetalBayerFilter3D(
            layer_bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_design, one_vertical_layer)

        if not restrict_layered_device:
            layer_bayer_filter = CMOSMetalBayerFilter3D.CMOSMetalBayerFilter3D(
                layer_bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_design, layer_thicknesses_voxels[device_layer_idx])

        if fix_layer_permittivity_to_reflective[ device_layer_idx ]:
            init_design = 1.0
            layer_bayer_filter = CMOSMetalBayerFilter3D.CMOSMetalBayerFilter3D(
                layer_bayer_filter_size_voxels, [1+0j, -10-3j], init_design, one_vertical_layer)

        elif init_from_random:
            random_design_seed = init_max_random_0_1_scale * np.random.random( layer_bayer_filter_size_voxels )
            layer_bayer_filter.set_design_variable( random_design_seed )

        bayer_filter_region_z = 1e-6 * np.linspace(layer_vertical_minimum_um, layer_vertical_maximum_um, layer_thicknesses_voxels[device_layer_idx])

        # bayer_filter.set_design_variable( np.load( projects_directory_location + "/cur_design_variable.npy" ) )

        layer_import = fdtd_hook.addimport()
        layer_import['name'] = 'layer_import_' + str( device_layer_idx )
        layer_import['x span'] = device_size_lateral_um * 1e-6
        layer_import['y span'] = device_size_lateral_um * 1e-6
        layer_import['z min'] = layer_vertical_minimum_um * 1e-6
        layer_import['z max'] = layer_vertical_maximum_um * 1e-6

        # print("For layer " + str( device_layer_idx ), " the min um spot is " + str( layer_vertical_minimum_um ) + " and max um spot is " + str( layer_vertical_maximum_um ) )

        bayer_filters.append( layer_bayer_filter )
        bayer_filter_regions_z.append( bayer_filter_region_z )
        design_imports.append( layer_import )
        lock_design_to_reflective.append( fix_layer_permittivity_to_reflective[ device_layer_idx ] )


    else:
        blank_layer = fdtd_hook.addrect()
        blank_layer['name'] = 'device_layer_' + str(device_layer_idx)
        blank_layer['x span'] = device_size_lateral_um * 1e-6
        blank_layer['y span'] = device_size_lateral_um * 1e-6
        blank_layer['z min'] = layer_vertical_minimum_um * 1e-6
        blank_layer['z max'] = layer_vertical_maximum_um * 1e-6
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

device_background_side_x = [ -1, 1, 0, 0 ]
device_background_side_y = [ 0, 0, -1, 1 ]

for device_background_side_idx in range( 0, 4 ):
    side_x = device_background_side_x[ device_background_side_idx ]
    side_y = device_background_side_y[ device_background_side_idx ]

    side_block = fdtd_hook.addrect()

    side_block['name'] = 'device_background_' + side_to_string( side_x ) + "_" + side_to_string( side_y )

    side_block['z min'] = designable_device_vertical_minimum_um * 1e-6
    side_block['z max'] = designable_device_vertical_maximum_um * 1e-6

    side_block['x'] = side_x * extra_lateral_space_offset_um * 1e-6
    side_block['x span'] = (
        np.abs( side_x ) * extra_lateral_space_per_side_um +
        ( 1 - np.abs( side_x ) ) * fdtd_region_size_lateral_um ) * 1e-6

    side_block['y'] = side_y * extra_lateral_space_offset_um * 1e-6
    side_block['y span'] = (
        np.abs( side_y ) * extra_lateral_space_per_side_um +
        ( 1 - np.abs( side_y ) ) * fdtd_region_size_lateral_um ) * 1e-6


    side_block['index'] = device_background_index



#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}
focal_data = {}

figure_of_merit_evolution = np.zeros((2, num_epochs, num_iterations_per_epoch))
figure_of_merit_evolution_transmission = np.zeros((2, num_epochs, num_iterations_per_epoch))
figure_of_merit_evolution_reflect_low_band = np.zeros((2, num_epochs, num_iterations_per_epoch))
figure_of_merit_evolution_reflect_high_band = np.zeros((2, num_epochs, num_iterations_per_epoch))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

step_size_start = fixed_step_size

def import_bayer_filters():
    fdtd_hook.switchtolayout()

    for device_layer_idx in range( 0, len( bayer_filters ) ):
        bayer_filter = bayer_filters[ device_layer_idx ]

        cur_permittivity = bayer_filter.get_permittivity()
        cur_index = permittivity_to_index( cur_permittivity )

        design_import = design_imports[ device_layer_idx ]

        fdtd_hook.select( design_import[ "name" ] )
        fdtd_hook.importnk2( cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_regions_z[ device_layer_idx ] )

def import_previous():
    for device_layer_idx in range( 0, len( bayer_filters ) ):
        if lock_design_to_reflective[ device_layer_idx ]:
            continue

        bayer_filter = bayer_filters[ device_layer_idx ]

        cur_design_variable = np.load(projects_directory_location + "/cur_design_variable_" + str( device_layer_idx ) + ".npy")
        cur_design_variable = 1.0 * np.greater( cur_design_variable, 0.25 )
        bayer_filter.set_design_variable( cur_design_variable )

    import_bayer_filters()

def update_bayer_filters( device_step_real, device_step_imag, step_size, do_simulated_annealing, current_temperature ):
    max_max_design_variable_change = 0
    min_max_design_variable_change = 1.0
    
    log_file = open(projects_directory_location + "/log.txt", 'a+')

    for device_layer_idx in range( 0, len( bayer_filters ) ):
        bayer_filter = bayer_filters[ device_layer_idx ]

        cur_design_variable = bayer_filter.get_design_variable()

        np.save(projects_directory_location + "/cur_design_variable_" + str( device_layer_idx ) + ".npy", cur_design_variable)

        if lock_design_to_reflective[ device_layer_idx ]:
            continue

        last_design_variable = cur_design_variable.copy()

        layer_vertical_minimum_um = 1e6 * bayer_filter_regions_z[ device_layer_idx ][ 0 ]
        layer_vertical_maximum_um = 1e6 * bayer_filter_regions_z[ device_layer_idx ][ -1 ]

        layer_vertical_minimum_voxels = int( ( layer_vertical_minimum_um - designable_device_vertical_minimum_um ) / mesh_spacing_um )
        layer_vertical_maximum_voxels = layer_vertical_minimum_voxels + len( bayer_filter_regions_z[ device_layer_idx ] )

        bayer_filter.step(
            device_step_real[ layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, :, : ],
            device_step_imag[ layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, :, : ],
            step_size,
            do_simulated_annealing,
            current_temperature)

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


    log_file.write("\n\n")
    log_file.close()

    if max_max_design_variable_change > desired_max_max_design_change:
        return 0.5
    elif min_max_design_variable_change < desired_min_max_design_change:
        return 2

    return 1

if init_from_old:
    import_previous()

def mode_overlap_fom(
    electric_fields_forward, magnetic_fields_forward,
    electric_mode_fields, magnetic_mode_fields, normal_weighting,
    mode_overlap_norm=None ):

    num_wavelengths = electric_fields_forward.shape[ 1 ]
    total_norm = 1.0 / num_wavelengths
    fom_by_wavelength = np.zeros( num_wavelengths )

    for wl_idx in range( 0, num_wavelengths ):
        choose_electric_mode = electric_mode_fields
        choose_magnetic_mode = magnetic_mode_fields

        choose_electric_forward = electric_fields_forward
        choose_magnetic_forward = magnetic_fields_forward

        numerator_term_1 = (
            np.sum( choose_electric_forward[ 0, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 1, wl_idx, 0, :, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 0, wl_idx, 0, :, : ] * choose_magnetic_forward[ 1, wl_idx, 0, :, : ] ) ) )

        numerator_term_2 = -(
            np.sum( choose_electric_forward[ 1, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, :, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 1, wl_idx, 0, :, : ] * choose_magnetic_forward[ 0, wl_idx, 0, :, : ] ) ) )

        numerator = numerator_term_1 + numerator_term_2
        numerator = np.abs( numerator )**2

        denominator = 8.0 * np.real(
            np.sum( choose_electric_mode[ 0, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 1, wl_idx, 0, :, : ] ) ) -
            np.sum( choose_electric_mode[ 1, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, :, : ] ) )
        )

        fom_by_wavelength[ wl_idx ] = ( numerator / denominator )
        if mode_overlap_norm is not None:
            fom_by_wavelength[ wl_idx ] = ( numerator / ( mode_overlap_norm[ wl_idx ] * denominator ) )
    
        fom_by_wavelength[ wl_idx ] *= normal_weighting

    return total_norm * fom_by_wavelength

def mode_overlap_gradient(
    figure_of_merit,
    electric_fields_forward, magnetic_fields_forward,
    electric_mode_fields, magnetic_mode_fields,
    electric_fields_gradient_forward, electric_fields_gradient_adjoint,
    normal_weighting,
    mode_overlap_norm ):

    num_wavelengths = electric_fields_forward.shape[ 1 ]

    fom_weighting = ( 2.0 / num_wavelengths ) - figure_of_merit**2 / np.sum( figure_of_merit**2 )
    fom_weighting = np.maximum( fom_weighting, 0 )
    fom_weighting /= np.sum( fom_weighting )

    gradient = np.zeros( electric_fields_gradient_forward.shape[ 2 : ], dtype=np.complex )

    for wl_idx in range( 0, num_wavelengths ):
        choose_electric_mode = electric_mode_fields
        choose_magnetic_mode = magnetic_mode_fields

        choose_electric_forward = electric_fields_forward
        choose_magnetic_forward = magnetic_fields_forward


        numerator_term_1 = (
            np.sum( choose_electric_forward[ 0, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 1, wl_idx, 0, :, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 0, wl_idx, 0, :, : ] * choose_magnetic_forward[ 1, wl_idx, 0, :, : ] ) ) )

        numerator_term_2 = -(
            np.sum( choose_electric_forward[ 1, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, :, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 1, wl_idx, 0, :, : ] * choose_magnetic_forward[ 0, wl_idx, 0, :, : ] ) ) )

        numerator = np.conj( numerator_term_1 + numerator_term_2 )

        denominator = 4.0 * np.real(
            np.sum( choose_electric_mode[ 0, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 1, wl_idx, 0, :, : ] ) ) -
            np.sum( choose_electric_mode[ 1, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, :, : ] ) )
        )

        adjoint_phase = np.conj( numerator ) / ( denominator * mode_overlap_norm[ wl_idx ] )
        gradient += normal_weighting * ( 
            fom_weighting[ wl_idx ] * adjoint_phase *
            np.sum( electric_fields_gradient_forward[ wl_idx, :, :, :, : ] * electric_fields_gradient_adjoint[ wl_idx, :, :, :, : ], axis=0 ) )

    return -gradient / num_wavelengths


mode_overlap_maxima_r = []

for reflection_band in range( 0, len( reflection_fom_map) ):
    wavelength_range = reflection_fom_map[ reflection_band ]
    num_wavelengths = wavelength_range[ 1 ] - wavelength_range[ 0 ]

    #
    # Just choose the x-polarized input because the overlap normalizations should be the same based
    # on symmetry
    #
    mode_e_field = mode_e_fields['x']
    mode_h_field = mode_h_fields['x']

    mode_e_field_shape = mode_e_field.shape
    mode_h_field_shape = mode_h_field.shape

    band_mode_e_field_shape = np.array( mode_e_field_shape )
    band_mode_h_field_shape = np.array( mode_h_field_shape )

    band_mode_e_field_shape[ 1 ] = num_wavelengths
    band_mode_h_field_shape[ 1 ] = num_wavelengths

    select_mode_e_field_band = np.zeros( band_mode_e_field_shape, dtype=np.complex )
    select_mode_h_field_band = np.zeros( band_mode_h_field_shape, dtype=np.complex )

    for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

        select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
        select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

    mode_overlap_maxima_r.append(  num_wavelengths * mode_overlap_fom( select_mode_e_field_band, select_mode_h_field_band, select_mode_e_field_band, select_mode_h_field_band, 1 ) )


#
# Run the optimization
#
for epoch in range(start_epoch, num_epochs):
    integrated_transmission_evolution = np.zeros( ( num_iterations_per_epoch, num_focal_spots ) )
    fom_focal_spot_evolution = np.zeros( ( num_iterations_per_epoch, num_focal_spots ) )

    for device_layer_idx in range( 0, len( bayer_filters ) ):
        bayer_filters[ device_layer_idx ].update_filters( epoch )

    for iteration in range(0, num_iterations_per_epoch):
        print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

        # Import all the bayer filters
        import_bayer_filters()

        #
        # Step 1: Run the forward optimization for both x- and y-polarized plane waves.  These are also the adjoint simulations! Woo!
        #
        forward_e_fields = {}
        reflected_e_fields = {}
        reflected_h_fields = {}

        for pol in ['x', 'y']:
            disable_all_sources()
            plane_wave_sources[pol].enabled = 1
            fdtd_hook.run()

            forward_e_fields[pol] = get_complex_monitor_data(design_efield_monitor['name'], 'E')

            reflected_e_fields[pol] = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
            reflected_h_fields[pol] = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

        #
        # Step 2: Compute the figure of merit
        #

        fom_by_task = {}
        figure_of_merit = {}
        reflection_performance = {}

        for pol in ['x', 'y']:
            fom_by_task[ pol ] = []
            reflection_performance[ pol ] = []

        # fom_by_task = []
        # reflection_performance = []
        # transmission_performance = []

            mode_e_field = mode_e_fields[ pol ]
            mode_h_field = mode_h_fields[ pol ]

            for reflection_band in range( 0, len( reflection_fom_map ) ):
                wavelength_range = reflection_fom_map[ reflection_band ]
                num_wavelengths = wavelength_range[ 1 ] - wavelength_range[ 0 ]

                mode_e_field_shape = mode_e_field.shape
                mode_h_field_shape = mode_h_field.shape

                band_mode_e_field_shape = np.array( mode_e_field_shape )
                band_mode_h_field_shape = np.array( mode_h_field_shape )

                band_mode_e_field_shape[ 1 ] = num_wavelengths
                band_mode_h_field_shape[ 1 ] = num_wavelengths

                reflected_e_field_band = np.zeros( band_mode_e_field_shape, dtype=np.complex )
                reflected_h_field_band = np.zeros( band_mode_h_field_shape, dtype=np.complex )

                select_mode_e_field_band = np.zeros( band_mode_e_field_shape, dtype=np.complex )
                select_mode_h_field_band = np.zeros( band_mode_h_field_shape, dtype=np.complex )

                mode_overlap_norm = mode_overlap_maxima_r[ reflection_band ]

                for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

                    reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_fields[ pol ][ :, wl_idx, :, :, : ]
                    reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_fields[ pol ][ :, wl_idx, :, :, : ]

                    select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
                    select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

                cur_reflection_overlap = mode_overlap_fom( reflected_e_field_band, reflected_h_field_band, select_mode_e_field_band, select_mode_h_field_band, 1, mode_overlap_norm )

                if reflection_max[ reflection_band ]:
                    reflection_performance[ pol ].append( cur_reflection_overlap )
                else:
                    reflection_performance[ pol ].append( ( 1 / num_points_per_band ) - cur_reflection_overlap )
                fom_by_task[ pol ].append( np.sum( reflection_performance[ pol ][ reflection_band ] ) )

            fom_by_task[ pol ] = np.array( fom_by_task[ pol ] )
            figure_of_merit[ pol ] = np.mean( fom_by_task[ pol ] )

        task_weightings = {}

        for pol in ['x', 'y']:
            task_weightings[ pol ] = ( 2.0 / len( fom_by_task[ pol ] ) ) - fom_by_task[ pol ]**2 / np.sum( fom_by_task[ pol ]**2 )
            task_weightings[ pol ] = np.maximum( task_weightings[ pol ], 0 )

            print( "fom by task = " + str( fom_by_task[ pol ] ) + " for pol " + pol )
            print( "task weightings = " + str( task_weightings[ pol ] ) + " for pol " + pol )
            print()

            # task_weightings[ pol ] = [ 0, 0, 1 ]

            figure_of_merit_evolution_reflect_low_band[ xy_index_idx[ pol ], epoch, iteration ] = fom_by_task[ pol ][ 0 ]
            figure_of_merit_evolution_reflect_high_band[ xy_index_idx[ pol ], epoch, iteration ] = fom_by_task[ pol ][ 1 ]
            figure_of_merit_evolution_transmission[ xy_index_idx[ pol ], epoch, iteration ] = fom_by_task[ pol ][ 2 ]
            figure_of_merit_evolution[ xy_index_idx[ pol ], epoch, iteration ] = figure_of_merit[ pol ]

        np.save(projects_directory_location + "/figure_of_merit_reflect_low.npy", figure_of_merit_evolution_reflect_low_band)
        np.save(projects_directory_location + "/figure_of_merit_reflect_high.npy", figure_of_merit_evolution_reflect_high_band)
        np.save(projects_directory_location + "/figure_of_merit_transmission.npy", figure_of_merit_evolution_transmission)
        np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)

        #
        # Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
        # gradients for x- and y-polarized forward sources.
        #
        reversed_field_shape = [1, designable_device_voxels_vertical, device_voxels_lateral, device_voxels_lateral]
        xy_polarized_gradients = {}
        # np.zeros(reversed_field_shape, dtype=np.complex)

        for pol in ['x', 'y']:

            all_gradients = []

            adjoint_e_fields = forward_e_fields[ pol ]

            mode_e_field = mode_e_fields[ pol ]
            mode_h_field = mode_h_fields[ pol ]

            for reflection_band in range( 0, len( reflection_fom_map) ):
                wavelength_range = reflection_fom_map[ reflection_band ]
                num_wavelengths = wavelength_range[ 1 ] - wavelength_range[ 0 ]

                mode_e_field_shape = mode_e_field.shape
                mode_h_field_shape = mode_h_field.shape

                band_mode_e_field_shape = np.array( mode_e_field_shape )
                band_mode_h_field_shape = np.array( mode_h_field_shape )

                band_mode_e_field_shape[ 1 ] = num_wavelengths
                band_mode_h_field_shape[ 1 ] = num_wavelengths

                reflected_e_field_band = np.zeros( band_mode_e_field_shape, dtype=np.complex )
                reflected_h_field_band = np.zeros( band_mode_h_field_shape, dtype=np.complex )

                select_mode_e_field_band = np.zeros( band_mode_e_field_shape, dtype=np.complex )
                select_mode_h_field_band = np.zeros( band_mode_h_field_shape, dtype=np.complex )

                adjoint_reflection_e_fields = np.zeros(
                    ( 3, num_wavelengths, designable_device_voxels_vertical, device_voxels_lateral, device_voxels_lateral ), dtype=np.complex )

                forward_reflection_e_fields = np.zeros(
                    ( 3, num_wavelengths, designable_device_voxels_vertical, device_voxels_lateral, device_voxels_lateral ), dtype=np.complex )

                for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

                    reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_fields[ pol ][ :, wl_idx, :, :, : ]
                    reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_fields[ pol ][ :, wl_idx, :, :, : ]

                    select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
                    select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

                    adjoint_reflection_e_fields_backup = adjoint_e_fields[ :, wl_idx, :, :, : ].copy() * np.exp( 1j * phase_corrections_reflection[ pol ][ wl_idx ] )

                    adjoint_reflection_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = adjoint_reflection_e_fields_backup
                    forward_reflection_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = forward_e_fields[ :, wl_idx, :, :, : ]

                forward_reflection_e_fields = np.swapaxes( forward_reflection_e_fields, 0, 1 )
                adjoint_reflection_e_fields = np.swapaxes( adjoint_reflection_e_fields, 0, 1 )

                mode_overlap_norm = mode_overlap_maxima_r[ reflection_band ]

                cur_reflection_gradient = mode_overlap_gradient(
                        reflection_performance[ reflection_band ],
                        reflected_e_field_band, reflected_h_field_band,
                        select_mode_e_field_band, select_mode_h_field_band,
                        forward_reflection_e_fields, adjoint_reflection_e_fields,
                        1,
                        mode_overlap_norm
                    ) / 1j

                if reflection_max[ reflection_band ]:
                    all_gradients.append(
                        cur_reflection_gradient
                    )
                else:
                    all_gradients.append(
                        -cur_reflection_gradient
                    )


            for gradient_idx in range( 0, len( task_weightings[ pol ] ) ):
                all_gradients[ gradient_idx ] *= task_weightings[ pol ][ gradient_idx ]
            xy_polarized_gradients[ pol ] = np.sum( all_gradients, axis=0 )
        
        #
        # Currently, no polarization performance weighting!  But this is something we can add in if one
        # polarization is dominating the performance.
        #
        final_gradient = 0.5 * ( xy_polarized_gradients[ 'x' ] + xy_polarized_gradients[ 'y' ] )
        #
        # Step 4: Step the design variable.
        #
        device_gradient_real = 2 * np.real( xy_polarized_gradients )
        device_gradient_imag = 2 * np.imag( xy_polarized_gradients )
        # Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
        # [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
        device_gradient_real = np.swapaxes(device_gradient_real, 0, 2)
        device_gradient_imag = np.swapaxes(device_gradient_imag, 0, 2)

        step_size = step_size_start

        if use_adaptive_step_size:
            step_size = adaptive_step_size

        do_simulated_annealing = use_simulated_annealing and ( iteration < simulated_annealing_cutoff_iteration )
        current_temperature = temperature_scaling / np.log( iteration + 2 )

        adaptive_step_size *= update_bayer_filters( -device_gradient_real, -device_gradient_imag, step_size, do_simulated_annealing, current_temperature )

        #
        # Would be nice to see how much it actually changed because some things will just
        # hit the wall even if they have a high desired change, but for now this is ok.  We
        # will do that for each individual iteration
        #
        # average_design_variable_change = np.mean( np.abs( step_size * design_gradient ) )
        # max_design_variable_change = np.max( np.abs( step_size * design_gradient ) )

        step_size_evolution[epoch][iteration] = step_size
        # average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
        # max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

        np.save(projects_directory_location + '/device_gradient_real.npy', device_gradient_real)
        np.save(projects_directory_location + '/device_gradient_imag.npy', device_gradient_imag)
        # np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
        np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
        # np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
        # np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
    np.save(projects_directory_location + '/fom_focal_spot_evolution.npy', fom_focal_spot_evolution)
    np.save(projects_directory_location + '/integrated_transmission_evolution.npy', integrated_transmission_evolution)



# disable_all_sources()
# plane_wave_sources['x'].enabled = 1
# fdtd_hook.run()

