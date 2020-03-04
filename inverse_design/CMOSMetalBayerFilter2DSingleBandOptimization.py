import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CMOSMetalBayerFilter2DSingleBandParameters import *
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

shutil.copy2(python_src_directory + "/CMOSMetalBayerFilter2DSingleBandParameters.py", projects_directory_location + "/ArchiveCMOSMetalBayerFilter.py")

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
# fdtd['dt stability factor'] = fdtd_dt_stability_factor

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
forward_src['x span'] = 1.3 * device_size_lateral_um * 1e-6
forward_src['y min'] = src_minimum_vertical_um * 1e-6
forward_src['y max'] = src_maximum_vertical_um * 1e-6
forward_src['wavelength start'] = src_lambda_min_um * 1e-6
forward_src['wavelength stop'] = src_lambda_max_um * 1e-6

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
    fdtd_hook.switchtolayout()

    forward_src.enabled = 0

    for adj_src_idx in range(0, num_focus_adjoint_sources):
        (focus_adjoint_sources[adj_src_idx]).enabled = 0

    reflection_adjoint_source.enabled = 0

#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
focus_adjoint_sources = []
reflection_adjoint_sources = []

for focus_adj_src_idx in range(0, num_focus_adjoint_sources):
    adj_src = fdtd_hook.adddipole()
    adj_src['name'] = 'adj_src_' + str(focus_adj_src_idx)
    adj_src['x'] = adjoint_x_positions_um[focus_adj_src_idx] * 1e-6
    adj_src['y'] = focusing_adjoint_vertical_um * 1e-6
    adj_src['theta'] = 0
    adj_src['phi'] = 0
    adj_src['wavelength start'] = src_lambda_min_um * 1e-6
    adj_src['wavelength stop'] = src_lambda_max_um * 1e-6

    focus_adjoint_sources.append(adj_src)

reflection_adjoint_source = fdtd_hook.addtfsf()
reflection_adjoint_source['name'] = 'reflection_adjoint_source'
reflection_adjoint_source['x span'] = forward_src['x span']
reflection_adjoint_source['y min'] = reflection_adjoint_vertical_um * 1e-6
reflection_adjoint_source['y max'] = fdtd_region_maximum_vertical_um * 1e-6
reflection_adjoint_source['polarization angle'] = 90
reflection_adjoint_source['wavelength start'] = src_lambda_min_um * 1e-6
reflection_adjoint_source['wavelength stop'] = src_lambda_max_um * 1e-6
#
# Start it backward so we can get the phase normalization for the source.
# Then, we can flip the direction when we inject the actual adjoint source.
#
reflection_adjoint_source['direction'] = 'Forward'


#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
design_efield_monitor['y min'] = designable_device_vertical_minimum_um * 1e-6
design_efield_monitor['y max'] = designable_device_vertical_maximum_um * 1e-6
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
focal_monitors = []

for adj_src in range(0, num_focus_adjoint_sources):
    focal_monitor = fdtd_hook.addpower()
    focal_monitor['name'] = 'focal_monitor' + str(adj_src)
    focal_monitor['monitor type'] = 'point'
    focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
    focal_monitor['y'] = focusing_adjoint_vertical_um * 1e-6
    focal_monitor['override global monitor settings'] = 1
    focal_monitor['use linear wavelength spacing'] = 1
    focal_monitor['use source limits'] = 0
    focal_monitor['minimum wavelength'] = lambda_min_um * 1e-6
    focal_monitor['maximum wavelength'] = lambda_max_um * 1e-6
    focal_monitor['frequency points'] = num_design_frequency_points

    focal_monitors.append(focal_monitor)


for adj_src in range(0, num_focus_adjoint_sources):
    transmission_monitor = fdtd_hook.addpower()
    transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
    transmission_monitor['monitor type'] = 'Linear X'
    transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
    transmission_monitor['x span'] = transmission_monitor_width_um * 1e-6
    transmission_monitor['y'] = focusing_adjoint_vertical_um * 1e-6
    transmission_monitor['override global monitor settings'] = 1
    transmission_monitor['use linear wavelength spacing'] = 1
    transmission_monitor['use source limits'] = 0
    transmission_monitor['minimum wavelength'] = lambda_min_um * 1e-6
    transmission_monitor['maximum wavelength'] = lambda_max_um * 1e-6
    transmission_monitor['frequency points'] = num_eval_frequency_points

reflection_monitor = fdtd_hook.addpower()
reflection_monitor['name'] = 'reflection_monitor'
reflection_monitor['monitor type'] = 'Linear X'
reflection_monitor['x span'] = reflection_adjoint_source['x span']
reflection_monitor['y'] = reflection_adjoint_vertical_um * 1e-6
reflection_monitor['override global monitor settings'] = 1
reflection_monitor['use linear wavelength spacing'] = 1
reflection_monitor['use source limits'] = 0
reflection_monitor['minimum wavelength'] = lambda_min_um * 1e-6
reflection_monitor['maximum wavelength'] = lambda_max_um * 1e-6
reflection_monitor['frequency points'] = num_design_frequency_points

mode_reflection_monitor_delta_um = 0.2 * vertical_gap_size_um
mode_reflection_monitor = fdtd_hook.addpower()
mode_reflection_monitor['name'] = 'mode_reflection_monitor'
mode_reflection_monitor['monitor type'] = 'Linear X'
mode_reflection_monitor['x span'] = reflection_adjoint_source['x span']
mode_reflection_monitor['y'] = ( reflection_adjoint_vertical_um + mode_reflection_monitor_delta_um ) * 1e-6
mode_reflection_monitor['override global monitor settings'] = 1
mode_reflection_monitor['use linear wavelength spacing'] = 1
mode_reflection_monitor['use source limits'] = 0
mode_reflection_monitor['minimum wavelength'] = lambda_min_um * 1e-6
mode_reflection_monitor['maximum wavelength'] = lambda_max_um * 1e-6
mode_reflection_monitor['frequency points'] = num_design_frequency_points


#
# Run a normalization run for the adjoint problem
#
disable_all_sources()
reflection_adjoint_source.enabled = 1
fdtd_hook.run()

mode_e_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
mode_h_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

print(mode_e_field.shape)
print(mode_h_field.shape)

phase_corrections = []

for wl_idx in range( 0, num_design_frequency_points ):
    wavelength_um = lambda_values_um[ wl_idx ]
    phase_shift = 2 * np.pi * mode_reflection_monitor_delta_um / wavelength_um
    mode_phase = -np.angle( mode_e_field[ 2, wl_idx, 0, 0, int( mode_e_field.shape[ 4 ] / 2 ) ] )
    phase_corrections.append( 2 * ( phase_shift + mode_phase ) )
    # phase_corrections.append( phase_shift + mode_phase )
    # phase_corrections.append( -mode_phase )

    print(phase_shift)
    print(mode_phase)
    print()

fdtd_hook.switchtolayout()

reflection_adjoint_source[ 'direction' ] = 'Backward'
reflection_adjoint_source['y min'] = src_minimum_vertical_um * 1e-6
reflection_adjoint_source['y max'] = reflection_adjoint_vertical_um * 1e-6

focal_plane_monitor = fdtd_hook.addpower()
focal_plane_monitor['name'] = 'focal_plane_monitor'
focal_plane_monitor['monitor type'] = 'Linear X'
focal_plane_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
focal_plane_monitor['x span'] = fdtd_region_size_lateral_um * 1e-6
focal_plane_monitor['y'] = focusing_adjoint_vertical_um * 1e-6
focal_plane_monitor['override global monitor settings'] = 1
focal_plane_monitor['use linear wavelength spacing'] = 1
focal_plane_monitor['use source limits'] = 0
focal_plane_monitor['minimum wavelength'] = lambda_min_um * 1e-6
focal_plane_monitor['maximum wavelength'] = lambda_max_um * 1e-6
focal_plane_monitor['frequency points'] = num_eval_frequency_points

input_aperture_monitor = fdtd_hook.addpower()
input_aperture_monitor['name'] = 'input_aperture_monitor'
input_aperture_monitor['monitor type'] = 'Linear X'
input_aperture_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
input_aperture_monitor['x span'] = fdtd_region_size_lateral_um * 1e-6
input_aperture_monitor['y'] = designable_device_vertical_maximum_um * 1e-6
input_aperture_monitor['override global monitor settings'] = 1
input_aperture_monitor['use linear wavelength spacing'] = 1
input_aperture_monitor['use source limits'] = 0
input_aperture_monitor['minimum wavelength'] = lambda_min_um * 1e-6
input_aperture_monitor['maximum wavelength'] = lambda_max_um * 1e-6
input_aperture_monitor['frequency points'] = num_eval_frequency_points


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
lock_design_to_reflective = []

bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
bayer_filter_region_z = 1e-6 * np.linspace(-0.51, 0.51, 2)

for device_layer_idx in range( 0, number_device_layers ):
    layer_vertical_maximum_um = designable_device_vertical_maximum_um - np.sum( layer_thicknesses_um[ 0 : device_layer_idx ] )
    layer_vertical_minimum_um = layer_vertical_maximum_um - layer_thicknesses_um[ device_layer_idx ]

    if is_layer_designable[ device_layer_idx ]:

        one_vertical_layer = 1
        layer_bayer_filter_size_voxels = np.array([device_voxels_lateral, layer_thicknesses_voxels[device_layer_idx], 1])

        init_design = init_permittivity_0_1_scale
        layer_bayer_filter = CMOSMetalBayerFilter2D.CMOSMetalBayerFilter2D(
            layer_bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_design, one_vertical_layer)

        if not restrict_layered_device:
            layer_bayer_filter = CMOSMetalBayerFilter2D.CMOSMetalBayerFilter2D(
                layer_bayer_filter_size_voxels, [min_device_permittivity, max_device_permittivity], init_design, layer_thicknesses_voxels[device_layer_idx])


        if fix_layer_permittivity_to_reflective[ device_layer_idx ]:
            init_design = 1.0
            layer_bayer_filter = CMOSMetalBayerFilter2D.CMOSMetalBayerFilter2D(
                layer_bayer_filter_size_voxels, [1+0j, -10-3j], init_design, one_vertical_layer)

        elif init_from_random:
            random_design_seed = init_max_random_0_1_scale * np.random.random( layer_bayer_filter_size_voxels )
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

        # print("For layer " + str( device_layer_idx ), " the min um spot is " + str( layer_vertical_minimum_um ) + " and max um spot is " + str( layer_vertical_maximum_um ) )

        bayer_filters.append( layer_bayer_filter )
        bayer_filter_regions_y.append( bayer_filter_region_y )
        design_imports.append( layer_import )
        lock_design_to_reflective.append( fix_layer_permittivity_to_reflective[ device_layer_idx ] )


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

device_background_side_x = [ -1, 1 ]

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
figure_of_merit_evolution_focus = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_evolution_reflect_low_band = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_evolution_reflect_high_band = np.zeros((num_epochs, num_iterations_per_epoch))
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

        import_index = np.zeros( ( cur_index.shape[ 0 ], cur_index.shape[ 1 ], 2 ), dtype=np.complex )
        import_index[ :, :, 0 ] = cur_index[ :, :, 0 ]
        import_index[ :, :, 1 ] = cur_index[ :, :, 0 ]

        design_import = design_imports[ device_layer_idx ]

        fdtd_hook.select( design_import[ "name" ] )
        fdtd_hook.importnk2( import_index, bayer_filter_region_x, bayer_filter_regions_y[ device_layer_idx ], bayer_filter_region_z )

def import_previous():
    for device_layer_idx in range( 0, len( bayer_filters ) ):
        if lock_design_to_reflective[ device_layer_idx ]:
            continue

        bayer_filter = bayer_filters[ device_layer_idx ]

        cur_design_variable = np.load(projects_directory_location + "/cur_design_variable_" + str( device_layer_idx ) + ".npy")
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

        layer_vertical_minimum_um = 1e6 * bayer_filter_regions_y[ device_layer_idx ][ 0 ]
        layer_vertical_maximum_um = 1e6 * bayer_filter_regions_y[ device_layer_idx ][ -1 ]

        layer_vertical_minimum_voxels = int( ( layer_vertical_minimum_um - designable_device_vertical_minimum_um ) / mesh_spacing_um )
        layer_vertical_maximum_voxels = layer_vertical_minimum_voxels + len( bayer_filter_regions_y[ device_layer_idx ] )

        bayer_filter.step(
            device_step_real[ :, layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, : ],
            device_step_imag[ :, layer_vertical_minimum_voxels : layer_vertical_maximum_voxels, : ],
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
    electric_mode_fields, magnetic_mode_fields,
    mode_overlap_norm=None ):

    num_wavelengths = electric_fields_forward[0].shape[ 1 ]
    total_norm = 1.0 / num_wavelengths
    fom_by_wavelength = np.zeros( num_wavelengths )

    for wl_idx in range( 0, num_wavelengths ):
        choose_electric_mode = electric_mode_fields
        choose_magnetic_mode = magnetic_mode_fields

        choose_electric_forward = electric_fields_forward
        choose_magnetic_forward = magnetic_fields_forward

        numerator = (
            np.sum( choose_electric_forward[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 2, wl_idx, 0, 0, : ] ) * choose_magnetic_forward[ 0, wl_idx, 0, 0, : ] ) )
        numerator = np.abs( numerator )**2
        denominator = 8.0 * np.real( np.sum( choose_electric_mode[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) )

        fom_by_wavelength[ wl_idx ] = ( numerator / denominator )
        if mode_overlap_norm is not None:
            # print('mode overlap norm = ' + str(mode_overlap_norm))
            fom_by_wavelength[ wl_idx ] = ( numerator / ( mode_overlap_norm[ wl_idx ] * denominator ) )
    
    return total_norm * fom_by_wavelength

def mode_overlap_gradient(
    figure_of_merit,
    electric_fields_forward, magnetic_fields_forward,
    electric_mode_fields, magnetic_mode_fields,
    electric_fields_gradient_forward, electric_fields_gradient_adjoint,
    mode_overlap_norm ):

    num_fom = len( figure_of_merit )
    fom_weighting = ( 2.0 / num_fom ) - figure_of_merit**2 / np.sum( figure_of_merit**2 )
    fom_weighting = np.maximum( fom_weighting, 0 )
    fom_weighting /= np.sum( fom_weighting )

    num_wavelengths = electric_fields_forward[0].shape[ 1 ]
    fom_by_wavelength = np.zeros( num_wavelengths )

    gradient = np.zeros( electric_fields_gradient_forward.shape[ 2 : ], dtype=np.complex )

    for wl_idx in range( 0, num_wavelengths ):
        choose_electric_mode = electric_mode_fields
        choose_magnetic_mode = magnetic_mode_fields

        choose_electric_forward = electric_fields_forward
        choose_magnetic_forward = magnetic_fields_forward

        numerator = (
            np.sum( choose_electric_forward[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 2, wl_idx, 0, 0, : ] ) * choose_magnetic_forward[ 0, wl_idx, 0, 0, : ] ) )
        denominator = 4.0 * np.real( np.sum( choose_electric_mode[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) )

        adjoint_phase = np.conj( numerator ) / ( denominator * mode_overlap_norm[ wl_idx ] )
        gradient += ( 
            fom_weighting[ wl_idx ] * adjoint_phase *
            np.sum( electric_fields_gradient_forward[ wl_idx, :, :, :, : ] * electric_fields_gradient_adjoint[ wl_idx, :, :, :, : ], axis=0 ) )

    return -gradient / num_wavelengths

def focusing_fom(
    normalized_intensity_focal_point_wavelength ):

    num_points = normalized_intensity_focal_point_wavelength.shape[ 0 ]
    total_norm = 1.0 / ( num_points )
    fom = np.zeros( num_points )
    for wl_idx in range( 0, num_points ):
        fom[ wl_idx ] = normalized_intensity_focal_point_wavelength[ wl_idx ]

    return total_norm * fom

def focusing_gradient(
    cur_fom,
    conjugate_weighting_focal_point_wavelength,
    forward_e_fields_wavelength,
    adjoint_e_fields_focal_point_wavelength ):

    num_fom = len( cur_fom )
    fom_weighting = ( 2.0 / num_fom ) - cur_fom**2 / np.sum( cur_fom**2 )
    fom_weighting = np.maximum( fom_weighting, 0 )
    fom_weighting /= np.sum( fom_weighting )

    num_total_points = num_points_per_band
    total_norm = 1.0 / num_total_points

    fields_shape = forward_e_fields_wavelength.shape
    gradient = np.zeros( fields_shape[ 2 : ], dtype=np.complex )

    for wl_idx in range( 0, num_points_per_band ):
        gradient += fom_weighting[ wl_idx ] * np.sum(
            ( conjugate_weighting_focal_point_wavelength[ wl_idx ] * adjoint_e_fields_focal_point_wavelength[ wl_idx ] * forward_e_fields_wavelength[ wl_idx ] ),
            axis=0 )

    return ( total_norm * gradient )


mode_overlap_maxima = []

for reflection_band in range( 0, len( reflection_fom_map) ):
    wavelength_range = reflection_fom_map[ reflection_band ]
    num_wavelengths = wavelength_range[ 1 ] - wavelength_range[ 0 ]

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

    mode_overlap_maxima.append(  mode_overlap_fom( select_mode_e_field_band, select_mode_h_field_band, select_mode_e_field_band, select_mode_h_field_band ) )


if run_finite_difference_check:
    choose_bayer_filter_idx = 0

    bayer_filter = bayer_filters[ choose_bayer_filter_idx ]
    cur_design_variable = bayer_filter.get_design_variable()

    layer_vertical_minimum_um = 1e6 * bayer_filter_regions_y[ choose_bayer_filter_idx ][ 0 ]
    layer_vertical_maximum_um = 1e6 * bayer_filter_regions_y[ choose_bayer_filter_idx ][ -1 ]

    layer_vertical_minimum_voxels = int( ( layer_vertical_minimum_um - designable_device_vertical_minimum_um ) / mesh_spacing_um )
    layer_vertical_maximum_voxels = layer_vertical_minimum_voxels + len( bayer_filter_regions_y[ choose_bayer_filter_idx ] )


    adj_y = int( 0.5 * ( layer_vertical_minimum_voxels + layer_vertical_maximum_voxels ) )
    fd_y = int( 0.5 * ( layer_vertical_maximum_voxels - layer_vertical_minimum_voxels ) )


    # Import all the bayer filters
    import_bayer_filters()

    #
    # Step 1: Run the forward optimization for both x- and y-polarized plane waves.
    #
    disable_all_sources()
    forward_src.enabled = 1
    fdtd_hook.run()

    forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

    reflected_e_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
    reflected_h_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

    reflection_performance = []

    for reflection_band in range( 0, 1 ):
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

        mode_overlap_norm = mode_overlap_maxima[ reflection_band ]

        for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

            reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_field[ :, wl_idx, :, :, : ]
            reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_field[ :, wl_idx, :, :, : ]

            select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
            select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

        reflection_performance.append(  mode_overlap_fom( reflected_e_field_band, reflected_h_field_band, select_mode_e_field_band, select_mode_h_field_band, mode_overlap_norm ) )



    all_gradients = []

    disable_all_sources()
    reflection_adjoint_source.enabled = 1
    fdtd_hook.run()

    adjoint_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )

    for reflection_band in range( 0, 1 ):
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
            ( 3, num_wavelengths, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

        forward_reflection_e_fields = np.zeros(
            ( 3, num_wavelengths, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

        for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

            reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_field[ :, wl_idx, :, :, : ]
            reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_field[ :, wl_idx, :, :, : ]

            select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
            select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

            # for wl_idx in range( 0, num_design_frequency_points ):
            # wavelength_um = lambda_values_um[ wl_idx ]
            # phase_shift = 2 * np.pi * mode_reflection_monitor_delta_um / wavelength_um

            adjoint_reflection_e_fields_backup = adjoint_e_fields[ :, wl_idx, :, :, : ].copy() * np.exp( 1j * phase_corrections[ wl_idx ] )
            # adjoint_reflection_e_fields_backup *= np.exp( -1j * source_phases[ wl_idx ] )
            # print( np.exp( -2 * 1j * source_phases[ wl_idx]))

            adjoint_reflection_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = adjoint_reflection_e_fields_backup
            #adjoint_e_fields[ :, wl_idx, :, :, : ] * np.exp( -2 * 1j * mode_phases[ wl_idx ] )
            forward_reflection_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = forward_e_fields[ :, wl_idx, :, :, : ]

        forward_reflection_e_fields = np.swapaxes( forward_reflection_e_fields, 0, 1 )
        adjoint_reflection_e_fields = np.swapaxes( adjoint_reflection_e_fields, 0, 1 )

        mode_overlap_norm = mode_overlap_maxima[ 0 ]

        # all_gradients.append(
        #     (max_real_permittivity - min_real_permittivity) * np.real( mode_overlap_gradient(
        #         reflection_performance[ 0 ],
        #         reflected_e_field_band, reflected_h_field_band,
        #         select_mode_e_field_band, select_mode_h_field_band,
        #         forward_reflection_e_fields, adjoint_reflection_e_fields,
        #         mode_overlap_norm )
        #     )
        # )

        all_gradients.append(
            (max_real_permittivity - min_real_permittivity) * ( mode_overlap_gradient(
                reflection_performance[ 0 ],
                reflected_e_field_band, reflected_h_field_band,
                select_mode_e_field_band, select_mode_h_field_band,
                forward_reflection_e_fields, adjoint_reflection_e_fields,
                mode_overlap_norm )
            )
        )

    finite_diff = np.zeros( device_voxels_lateral )
    
    h = 5e-3
    for fd_x in range( 0, device_voxels_lateral ):
        break
        print("Working on fd_x = " + str(fd_x))
        cur_design_variable[ fd_x, fd_y ] += h
        bayer_filter.set_design_variable( cur_design_variable )

        # Import all the bayer filters
        import_bayer_filters()

        #
        # Step 1: Run the forward optimization for both x- and y-polarized plane waves.
        #
        disable_all_sources()
        forward_src.enabled = 1
        fdtd_hook.run()

        forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

        reflected_e_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
        reflected_h_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

        reflection_performance2 = []

        for reflection_band in range( 0, 1 ):
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

            mode_overlap_norm = mode_overlap_maxima[ reflection_band ]

            for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

                reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_field[ :, wl_idx, :, :, : ]
                reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_field[ :, wl_idx, :, :, : ]

                select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
                select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

            reflection_performance2.append(  mode_overlap_fom( reflected_e_field_band, reflected_h_field_band, select_mode_e_field_band, select_mode_h_field_band, mode_overlap_norm ) )


        finite_diff[ fd_x ] = ( reflection_performance2[ 0 ] - reflection_performance[ 0 ] ) / (h)
        cur_design_variable[ fd_x, fd_y ] -= h

        print('finite diff = ' + str(finite_diff[ fd_x ]))


    test_phases = np.linspace(50 * np.pi / 32, 51 * np.pi / 32, 3)

    for phase_idx in range(0, 3):
        test_phase = test_phases[ phase_idx ]
        test_grad = all_gradients[0][ 0, adj_y, : ] * np.exp( 1j * test_phase )
        test_grad = np.real(test_grad)
        plt.plot( test_grad / np.max( np.abs( test_grad ) ), linestyle='--' )

    print(test_phases)
    print(2 * np.pi - test_phases)

    # np.save('finite_diff_reflection_take2.npy', finite_diff)
    np.save('adjoint_reflectionp2.npy', all_gradients[0][ 0, adj_y, : ] )
    finite_diff = np.load('finite_diff_reflection_take2.npy')

    zeroth = np.real( all_gradients[0][ 0, adj_y, : ] / 1j )

    plt.plot( finite_diff / np.max(np.abs(finite_diff)), color='g', linewidth=2 )
    plt.plot( zeroth / np.max(np.abs(zeroth)), color='b', linewidth=2)
    plt.show()
    sys.exit(1)

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
        # Step 1: Run the forward optimization for both x- and y-polarized plane waves.
        #
        disable_all_sources()
        forward_src.enabled = 1
        fdtd_hook.run()

        forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

        focal_data = []
        for adj_src_idx in range(0, num_focus_adjoint_sources):
            focal_data.append(
                get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )


        reflected_e_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
        reflected_h_field = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

        #
        # Step 2: Compute the figure of merit
        #

        normalized_intensity_focal_bands_wavelength = []
        conjugate_weighting_focal_bands_wavelength = []

        for focal_band in range( 0, len( focus_fom_map ) ):
            wavelength_range = focus_fom_map[ focal_band ]
            num_wavelengths = wavelength_range[ 1 ] - wavelength_range[ 0 ]

            normalized_intensity_focal_point_wavelength = np.zeros( ( num_wavelengths ) )
            conjugate_weighting_focal_point_wavelength = np.zeros( ( num_wavelengths ), dtype=np.complex )

            for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):
                normalized_intensity_focal_point_wavelength[ wl_idx - wavelength_range[ 0 ] ] = (
                    np.sum( np.abs( focal_data[ 0 ][ :, wl_idx, 0, 0, 0 ])**2 ) / max_intensity_by_wavelength[ wl_idx ] )

                conjugate_weighting_focal_point_wavelength[ wl_idx - wavelength_range[ 0 ] ] = np.conj(
                    focal_data[ 0 ][ 2, wl_idx, 0, 0, 0] / max_intensity_by_wavelength[ wl_idx ]
                )

            normalized_intensity_focal_bands_wavelength.append( normalized_intensity_focal_point_wavelength )
            conjugate_weighting_focal_bands_wavelength.append( conjugate_weighting_focal_point_wavelength )

        fom_by_task = []
        focus_performance = []
        reflection_performance = []

        for focal_band in range( 0, len( focus_fom_map ) ):
            focus_performance.append( focusing_fom( normalized_intensity_focal_bands_wavelength[ focal_band ] ) )
            fom_by_task.append( np.sum( np.array( focus_performance[ focal_band ] ) ) )

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

            mode_overlap_norm = mode_overlap_maxima[ reflection_band ]

            for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

                reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_field[ :, wl_idx, :, :, : ]
                reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_field[ :, wl_idx, :, :, : ]

                select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
                select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

            reflection_performance.append(  mode_overlap_fom( select_mode_e_field_band, select_mode_h_field_band, select_mode_e_field_band, select_mode_h_field_band ) )
            fom_by_task.append( np.sum( reflection_performance[ reflection_band ] ) )

        fom_by_task = np.array( fom_by_task )
        figure_of_merit = np.mean( fom_by_task )

        task_weightings = ( 2.0 / len( fom_by_task ) ) - fom_by_task**2 / np.sum( fom_by_task**2 )
        task_weightings = np.maximum( task_weightings, 0 )

        print( "fom by task = " + str( fom_by_task ) )
        print( "task weightings = " + str( task_weightings ) )
        print()

        # task_weightings = [ 1, 0, 0 ]

        # figure_of_merit_per_wavelength = splitting_fom(
        #     normalized_intensity_focal_point_wavelength )
        # figure_of_merit = np.sum( figure_of_merit_per_wavelength )

        # figure_of_merit_per_wavelength = splitting_contrast_fom( normalized_intensity_focal_point_wavelength )
        # figure_of_merit = np.sum( figure_of_merit_per_wavelength )


        # figure_of_merit = contrast_figure_of_merit( normalized_intensity_focal_point_wavelength )
        figure_of_merit_evolution_focus[epoch, iteration] = fom_by_task[ 0 ]
        figure_of_merit_evolution_reflect_low_band[epoch, iteration] = fom_by_task[ 1 ]
        figure_of_merit_evolution_reflect_high_band[epoch, iteration] = fom_by_task[ 2 ]
        figure_of_merit_evolution[epoch, iteration] = figure_of_merit

        np.save(projects_directory_location + "/figure_of_merit_focus.npy", figure_of_merit_evolution_focus)
        np.save(projects_directory_location + "/figure_of_merit_reflect_low.npy", figure_of_merit_evolution_reflect_low_band)
        np.save(projects_directory_location + "/figure_of_merit_reflect_high.npy", figure_of_merit_evolution_reflect_high_band)

        np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)

        #
        # Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
        # gradients for x- and y-polarized forward sources.
        #
        reversed_field_shape = [1, designable_device_voxels_vertical, device_voxels_lateral]
        xy_polarized_gradients = np.zeros(reversed_field_shape, dtype=np.complex)
        np.zeros( ( num_focus_adjoint_sources, 3, num_design_frequency_points, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

        all_gradients = []

        for focal_band in range( 0, len( focus_fom_map ) ):
            wavelength_range = focus_fom_map[ focal_band ]

            num_wavelengths = wavelength_range[ 1 ] - wavelength_range[ 0 ]

            adjoint_focus_e_fields = np.zeros(
                ( 3, num_wavelengths, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

            forward_focus_e_fields = np.zeros(
                ( 3, num_wavelengths, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

            disable_all_sources()
            (focus_adjoint_sources[0]).enabled = 1
            fdtd_hook.run()

            adjoint_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )

            for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):
                adjoint_focus_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = adjoint_e_fields[ :, wl_idx, :, :, : ]
                forward_focus_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = forward_e_fields[ :, wl_idx, :, :, : ]

            forward_focus_e_fields = np.swapaxes( forward_focus_e_fields, 0, 1 )
            adjoint_focus_e_fields = np.swapaxes( adjoint_focus_e_fields, 0, 1 )

            all_gradients.append(
                focusing_gradient(
                    focus_performance[ focal_band ],
                    conjugate_weighting_focal_bands_wavelength[ focal_band ],
                    forward_focus_e_fields, adjoint_focus_e_fields
                )
            )

        disable_all_sources()
        reflection_adjoint_source.enabled = 1
        fdtd_hook.run()

        adjoint_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )

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
                ( 3, num_wavelengths, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

            forward_reflection_e_fields = np.zeros(
                ( 3, num_wavelengths, 1, designable_device_voxels_vertical, device_voxels_lateral ), dtype=np.complex )

            for wl_idx in range( wavelength_range[ 0 ], wavelength_range[ 1 ] ):

                reflected_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_e_field[ :, wl_idx, :, :, : ]
                reflected_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = reflected_h_field[ :, wl_idx, :, :, : ]

                select_mode_e_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_e_field[ :, wl_idx, :, :, : ]
                select_mode_h_field_band[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = mode_h_field[ :, wl_idx, :, :, : ]

                # for wl_idx in range( 0, num_design_frequency_points ):

                adjoint_reflection_e_fields_backup = adjoint_e_fields[ :, wl_idx, :, :, : ].copy() * np.exp( 1j * phase_corrections[ wl_idx ] )

                adjoint_reflection_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = adjoint_reflection_e_fields_backup
                #adjoint_e_fields[ :, wl_idx, :, :, : ] * np.exp( -2 * 1j * mode_phases[ wl_idx ] )
                forward_reflection_e_fields[ :, wl_idx - wavelength_range[ 0 ], :, :, : ] = forward_e_fields[ :, wl_idx, :, :, : ]

            forward_reflection_e_fields = np.swapaxes( forward_reflection_e_fields, 0, 1 )
            adjoint_reflection_e_fields = np.swapaxes( adjoint_reflection_e_fields, 0, 1 )

            mode_overlap_norm = mode_overlap_maxima[ reflection_band ]

            all_gradients.append(
                mode_overlap_gradient(
                    transmission_performance[ transmission_band ],
                    transmitted_e_field_band, transmitted_h_field_band,
                    select_mode_e_field_band, select_mode_h_field_band,
                    forward_transmission_e_fields, adjoint_transmission_e_fields,
                    1,
                    mode_overlap_norm
                ) / 1j
            )


        # gradient_weighting = np.array( [ 0, 1, 0 ] )
        for gradient_idx in range( 0, len( task_weightings ) ):
            all_gradients[ gradient_idx ] *= task_weightings[ gradient_idx ]
        xy_polarized_gradients = np.sum( all_gradients, axis=0 )
        
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
# forward_src.enabled = 1
# fdtd_hook.run()

# transmisison_low = -get_monitor_data( 'transmission_monitor_0', 'T' )
# transmisison_high = -get_monitor_data( 'transmission_monitor_1', 'T' )
# # transmisison_red = -get_monitor_data( 'transmission_monitor_2', 'T' )

# np.save(projects_directory_location + '/transmission_low.npy', transmisison_low)
# np.save(projects_directory_location + '/transmission_high.npy', transmisison_high)
# # np.save(projects_directory_location + '/transmission_red.npy', transmisison_red)


