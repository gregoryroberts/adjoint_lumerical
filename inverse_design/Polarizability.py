import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# from CMOSMetalBayerFilter3DSingleBandModeParameters import *
# import CMOSMetalBayerFilter3D

# import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
import lumapi

import functools
import h5py
import numpy as np
import time
from scipy.ndimage import gaussian_filter

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))

if not os.path.isdir(projects_directory_location):
    os.mkdir(projects_directory_location)

projects_directory_location += "/polarizability_test_staircase_v9/"

if not os.path.isdir(projects_directory_location):
    os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

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


mesh_spacing_um = 0.025

focal_length_um = 1.5

device_thickness_um = 0.5
device_thickness_voxels = 2 + int( device_thickness_um / mesh_spacing_um )

device_size_lateral_um = 1.0
device_size_lateral_voxels = 2 + int( device_size_lateral_um / mesh_spacing_um )

permittivity_max = 2.0**2
permittivity_min = 1.0**2

num_design_frequency_points = 30

lambda_min_um = 0.4
lambda_max_um = 0.7

wavelengths_um = np.linspace( lambda_min_um, lambda_max_um, num_design_frequency_points )
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * wavelengths_um**2)

top_gap_um = 1.6
lateral_gap_um = 0.75
bottom_gap_um = 0.6

fdtd_lateral_size_um = device_size_lateral_um + 2 * lateral_gap_um
fdtd_lateral_size_voxels = 1 + int( fdtd_lateral_size_um / mesh_spacing_um )
fdtd_vertical_size_um = top_gap_um + bottom_gap_um + focal_length_um + device_thickness_um
fdtd_vertical_size_voxels = 1 + int( fdtd_vertical_size_um / mesh_spacing_um )

fdtd_simulation_time_fs = 5000

#
# Set up the FDTD region and mesh
#
fdtd = fdtd_hook.addfdtd()
fdtd['dimension'] = '3D'
fdtd['x span'] = fdtd_lateral_size_um * 1e-6
fdtd['y span'] = fdtd_lateral_size_um * 1e-6
fdtd['z max'] = ( device_thickness_um + top_gap_um ) * 1e-6
fdtd['z min'] = -( focal_length_um + bottom_gap_um ) * 1e-6
fdtd['mesh type'] = 'uniform'
fdtd['mesh refinement'] = 'staircase'
fdtd['define x mesh by'] = 'number of mesh cells'
fdtd['define y mesh by'] = 'number of mesh cells'
fdtd['define z mesh by'] = 'number of mesh cells'
fdtd['mesh cells x'] = fdtd_lateral_size_voxels
fdtd['mesh cells y'] = fdtd_lateral_size_voxels
fdtd['mesh cells z'] = fdtd_vertical_size_voxels
fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
fdtd['background index'] = 1.0

#
# General polarized source information
#
xy_phi_rotations = { 'x' : 0, 'y' : 90 }
xy_index_idx = { 'x' : 0, 'y' : 1 }
xy_names = ['x', 'y']

forward_src = fdtd_hook.addtfsf()
forward_src['name'] = 'forward_src'
forward_src['angle phi'] = xy_phi_rotations[ 'x' ]
forward_src['direction'] = 'Backward'
forward_src['x span'] = 1.1 * device_size_lateral_um * 1e-6
forward_src['y span'] = 1.1 * device_size_lateral_um * 1e-6
forward_src['z max'] = ( device_thickness_um + 0.5 * top_gap_um ) * 1e-6
forward_src['z min'] = -( focal_length_um + 0.5 * bottom_gap_um ) * 1e-6
forward_src['wavelength start'] = lambda_min_um * 1e-6
forward_src['wavelength stop'] = lambda_max_um * 1e-6


adjoint_src = fdtd_hook.adddipole()
adjoint_src['name'] = 'adjoint_src'
adjoint_src['theta'] = 90
adjoint_src['phi'] = 90
adjoint_src['x'] = 0.25 * device_size_lateral_um * 1e-6
adjoint_src['y'] = -0.3 * device_size_lateral_um * 1e-6
adjoint_src['z'] = -focal_length_um * 1e-6
adjoint_src['wavelength start'] = lambda_min_um * 1e-6
adjoint_src['wavelength stop'] = lambda_max_um * 1e-6


device_import = fdtd_hook.addimport()
device_import['name'] = 'device_import'
device_import['x span'] = device_size_lateral_um * 1e-6
device_import['y span'] = device_size_lateral_um * 1e-6
device_import['z min'] = 0 * 1e-6
device_import['z max'] = device_thickness_um * 1e-6

device_x_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_size_lateral_voxels )
device_y_range = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_size_lateral_voxels )
device_z_range = 1e-6 * np.linspace( 0, device_thickness_um, device_thickness_voxels )


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
    fdtd_hook.switchtolayout()

    forward_src.enabled = 0
    adjoint_src.enabled = 0


#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#
design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['monitor type'] = '3D'
design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
design_efield_monitor['y span'] = device_size_lateral_um * 1e-6
design_efield_monitor['z min'] = 0 * 1e-6
design_efield_monitor['z max'] = device_thickness_um * 1e-6
design_efield_monitor['override global monitor settings'] = 1
design_efield_monitor['use wavelength spacing'] = 1
design_efield_monitor['use source limits'] = 0
design_efield_monitor['minimum wavelength'] = lambda_min_um * 1e-6
design_efield_monitor['maximum wavelength'] = lambda_max_um * 1e-6
design_efield_monitor['frequency points'] = num_design_frequency_points
design_efield_monitor['output Hx'] = 0
design_efield_monitor['output Hy'] = 0
design_efield_monitor['output Hz'] = 0


focal_monitor = fdtd_hook.addpower()
focal_monitor['name'] = 'focal_monitor'
focal_monitor['monitor type'] = 'point'
focal_monitor['x'] = adjoint_src['x']
focal_monitor['y'] = adjoint_src['y']
focal_monitor['z'] = -focal_length_um * 1e-6
focal_monitor['override global monitor settings'] = 1
focal_monitor['use wavelength spacing'] = 1
focal_monitor['use source limits'] = 1
focal_monitor['frequency points'] = num_design_frequency_points

def compute_fom_by_wavelength( input_permittivity ):
    fdtd_hook.switchtolayout()
    fdtd_hook.select( device_import[ 'name' ] )
    fdtd_hook.importnk2( np.sqrt( input_permittivity ), device_x_range, device_y_range, device_z_range )

    disable_all_sources()
    forward_src.enabled = 1
    fdtd_hook.run()

    focal_e = get_complex_monitor_data( focal_monitor[ 'name' ], 'E' )

    intensity_y = np.zeros( num_design_frequency_points )
    for wl_idx in range( 0, num_design_frequency_points ):
        intensity_y[ wl_idx ] = np.sum( np.abs( focal_e[ 0, wl_idx, 0, 0, 0 ] )**2 / max_intensity_by_wavelength[ wl_idx ] )

    return intensity_y


# random_generator_seed = 6245234
# np.random.seed( random_generator_seed )

# np.save( projects_directory_location + "/random_generator_seed.npy", np.array( [ random_generator_seed ] ) )

device_permittivity = 0.5 * ( permittivity_max + permittivity_min ) * np.ones( ( device_size_lateral_voxels, device_size_lateral_voxels, device_thickness_voxels ) )

# device_permittivity = np.zeros( ( device_size_lateral_voxels, device_size_lateral_voxels, device_thickness_voxels ) )
# random_design_seed = 0.25 * np.random.random( device_permittivity.shape )
# random_design_seed = gaussian_filter( random_design_seed, sigma=3 )
# device_permittivity = permittivity_min + ( permittivity_max - permittivity_min ) * random_design_seed

region_start_x = int( 0.35 * device_size_lateral_voxels )
region_start_y = int( 0.68 * device_size_lateral_voxels )
region_start_z = int( 0.57 * device_thickness_voxels )

region_dim = 5

region_end_x = region_start_x + region_dim
region_end_y = region_start_y + region_dim
region_end_z = region_start_z + region_dim

stencil = 0.5 * ( permittivity_max + permittivity_min ) * np.ones( ( region_dim, region_dim, region_dim ) )

device_permittivity[ region_start_x : region_end_x, region_start_y : region_end_y, region_start_z : region_end_z ] = stencil

# initial_fom_by_wavelength = compute_fom_by_wavelength( device_permittivity )

# deps_values = [ 2**(-8), 2**(-9), 2**(-10), 2**(-11), 2**(-12) ]
# deps_values = [ 0.5 * ( 2**(-8) + 2**(-9) ), 0.5 * ( 2**(-9) + 2**(-10) ) ]
# deps_values = [ 2**(-13), 2**(-14) ]
deps_values = [ 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5 ]

# deps_values = [ 2**(-8) ]
# deps_values = [ 0.5 * ( 2**(-8) + 2**(-9) ) ]
# deps_values = [ 0.5 * ( 2**(-9) + 2**(-10) ) ]

for deps_idx in range( 0, len( deps_values ) ):
    deps = deps_values[ deps_idx ]

    fd_grad_in_stencil = np.zeros( list( stencil.shape ) + [ num_design_frequency_points ] )

    for stencil_x in range( 0, region_dim ):
        print( "Working on stencil x = " + str( stencil_x ) )
        for stencil_y in range( 0, region_dim ):
            for stencil_z in range( 0, region_dim ):

                permittivity_step = device_permittivity.copy()
                permittivity_step[ stencil_x + region_start_x, stencil_y + region_start_y, stencil_z + region_start_z ] += deps

                fom_up_by_wavelength = compute_fom_by_wavelength( permittivity_step )

                permittivity_step = device_permittivity.copy()
                permittivity_step[ stencil_x + region_start_x, stencil_y + region_start_y, stencil_z + region_start_z ] -= deps

                fom_down_by_wavelength = compute_fom_by_wavelength( permittivity_step )

                fd_grad_in_stencil[ stencil_x, stencil_y, stencil_z ] = ( fom_up_by_wavelength - fom_down_by_wavelength ) / ( 2 * deps )

    np.save( projects_directory_location + '/fd_grad_in_stencil_' + str( deps_idx ) + '.npy', fd_grad_in_stencil )

    user_shapes = [
        np.ones( stencil.shape ) ]

    three_box = np.zeros( stencil.shape )
    three_box[ 1:4, 1:4, 1:4 ] = 1
    user_shapes.append( three_box )

    three_box_right = np.zeros( stencil.shape )
    three_box_right[ 2:5, 1:4, 1:4 ] = 1
    user_shapes.append( three_box_right )

    three_box_up_left = np.zeros( stencil.shape )
    three_box_up_left[ 0:3, 0:3, 2:5 ] = 1
    user_shapes.append( three_box_up_left )

    two_box = np.zeros( stencil.shape )
    two_box[ 2:4, 1:3, 3:5 ] = 1
    user_shapes.append( two_box )

    four_box = np.zeros( stencil.shape )
    four_box[ 1:5, 1:5, 1:5 ] = 1
    user_shapes.append( four_box )

    four_box2 = np.zeros( stencil.shape )
    four_box2[ 0:4, 1:5, 0:4 ] = 1
    user_shapes.append( four_box2 )

    for i in range( 0, region_dim ):
        slender_x = np.zeros( stencil.shape )
        slender_x[ :, i, i ] = 1

        user_shapes.append( slender_x )

    tall_slender_x = np.zeros( stencil.shape )
    tall_slender_x[ :, 0, : ] = 1
    user_shapes.append( tall_slender_x )

    for i in range( 0, region_dim ):
        slender_y = np.zeros( stencil.shape )
        slender_y[ i, :, i ] = 1

        user_shapes.append( slender_y )

    tall_slender_y = np.zeros( stencil.shape )
    tall_slender_y[ 3, :, : ] = 1
    user_shapes.append( tall_slender_y )

    tall_slender_45 = np.zeros( stencil.shape )
    for i in range( 0, region_dim ):
        lower = np.maximum( i - 1, 0 )
        upper = np.minimum( i + 2, region_dim )
        tall_slender_45[ lower : upper, lower : upper, : ] = 1

    user_shapes.append( tall_slender_45 )

    for j in range( 0, region_dim ):
        slender_45 = np.zeros( stencil.shape )
        for i in range( 0, region_dim ):
            lower = np.maximum( i - 1, 0 )
            upper = np.minimum( i + 2, region_dim )
            slender_45[ lower : upper, lower : upper, j ] = 1

        user_shapes.append( slender_45 )

    fd_grad_by_user_shape = np.zeros( [ len( user_shapes ), num_design_frequency_points ] )
    shapes_by_user = np.zeros( [ len( user_shapes ) ] + list( stencil.shape ) )

    for shape in range( 0, len( user_shapes ) ):
        print( "Working on user shape " + str( shape ) + " out of " + str( len( user_shapes ) - 1 ) )
        permittivity_step = device_permittivity.copy()
        permittivity_step[ region_start_x : region_end_x, region_start_y : region_end_y, region_start_z : region_end_z ] += deps * user_shapes[ shape ]

        shapes_by_user[ shape ] = user_shapes[ shape ]

        fom_up_by_wavelength = compute_fom_by_wavelength( permittivity_step )

        permittivity_step = device_permittivity.copy()
        permittivity_step[ region_start_x : region_end_x, region_start_y : region_end_y, region_start_z : region_end_z ] -= deps * user_shapes[ shape ]

        fom_down_by_wavelength = compute_fom_by_wavelength( permittivity_step )

        fd_grad_by_user_shape[ shape ] = ( fom_up_by_wavelength - fom_down_by_wavelength ) / ( 2 * deps )

    np.save( projects_directory_location + '/fd_grad_by_user_shape_' + str( deps_idx ) + '.npy', fd_grad_by_user_shape )
    np.save( projects_directory_location + '/shapes_by_user_' + str( deps_idx ) + '.npy', shapes_by_user )

