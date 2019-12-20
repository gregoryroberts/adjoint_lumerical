import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from GenerateColorSplitterDataParameters import *

import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
import lumapi

import functools
import h5py
import numpy as np
import time

from scipy.ndimage import gaussian_filter

def permittivity_to_index( permittivity ):
    eps_real = np.real( permittivity )
    eps_imag = np.imag( permittivity )

    eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

    n = np.sqrt( ( eps_mag + eps_real ) / 2. )
    kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

    return ( n + 1j * kappa )

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
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


#
# Create project folder and save out the parameter file for documentation for this optimization
#
if not( len(sys.argv) == 3 ):
    print( "Error: Expected the following command line.." )
    print( "python " + str( sys.argv[ 0 ] ) + " { seed for pseudorandom generator } { number of designs to generate }" )
    sys.exit( 1 )

pseudorandom_seed = int( sys.argv[ 1 ] )
number_designs = int( sys.argv[ 2 ] )

np.random.seed( pseudorandom_seed )

for design_idx in range( 0, number_designs ):

    #
    # There are a lot of ways we can make a bad design.  We will submit the bad design with a given weight as the random seed we start with
    # and then we will put its gradient-optimized version in the good queue
    #

    projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/generated_data/' + data_prefix + str( pseudorandom_seed ) + "_id_" + str( design_idx ) + "/"))

    if not os.path.isdir(projects_directory_location):
        os.mkdir(projects_directory_location)

    log_file = open(projects_directory_location + "/log.txt", 'w')
    log_file.write("Log\n")
    log_file.close()

    fdtd_hook.newproject()
    fdtd_hook.save(projects_directory_location + "/optimization")

    shutil.copy2(python_src_directory + "/GenerateColorSplitterDataParameters.py", projects_directory_location + "/ArchiveGenerateColorSplitterDataParamters.py")

    index_low = index_low_bounds[ 0 ] + ( index_low_bounds[ 1 ] - index_low_bounds[ 0 ] ) * np.random.random( 1 )[ 0 ]
    index_contrast = index_contrast_ratio_bounds[ 0 ] + ( index_contrast_ratio_bounds[ 1 ] - index_contrast_ratio_bounds[ 0 ] ) * np.random.random( 1 )[ 0 ]
    index_high = index_low * ( 1 + index_contrast )

    lambda_minus_um = lambda_minus_bounds_um[ 0 ] + ( lambda_minus_bounds_um[ 1 ] - lambda_minus_bounds_um[ 0 ] ) * np.random.random( 1 )[ 0 ]
    lambda_low_um = lambda_plus_um  - lambda_minus_um
    lambda_high_um = lambda_plus_um + lambda_minus_um

    lambda_range_um = 2 * lambda_minus_um
    src_lambda_min_um = lambda_low_um - 0.25 * lambda_range_um
    src_lambda_max_um = lambda_high_um + 0.25 * lambda_range_um

    aperture_size_lambda_plus = aperture_size_bounds_lambda_plus_units[ 0 ] + ( aperture_size_bounds_lambda_plus_units[ 1 ] - aperture_size_bounds_lambda_plus_units[ 0 ] ) * np.random.random( 1 )[ 0 ]
    device_depth_lambda_plus = device_depth_bounds_lambda_plus_units[ 0 ] + ( device_depth_bounds_lambda_plus_units[ 1 ] - device_depth_bounds_lambda_plus_units[ 0 ] ) * np.random.random( 1 )[ 0 ]
    focal_length_lambda_plus = focal_length_bounds_lambda_plus_units[ 0 ] + ( focal_length_bounds_lambda_plus_units[ 1 ] - focal_length_bounds_lambda_plus_units[ 0 ] ) * np.random.random( 1 )[ 0 ]

    aperture_size_um = lambda_plus_um * aperture_size_lambda_plus
    device_depth_um = lambda_plus_um * device_depth_lambda_plus
    focal_length_um = lambda_plus_um * focal_length_lambda_plus


    fdtd_region_size_lateral_um = aperture_size_um + 2 * lateral_gap_um
    fdtd_region_size_vertical_um = device_depth_um + focal_length_um + 2 * vertical_gap_um
    fdtd_region_maximum_vertical_um = device_depth_um + vertical_gap_um
    fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_um

    fdtd_region_size_lateral_voxels = 1 + int( fdtd_region_size_lateral_um / mesh_spacing_um )
    fdtd_region_size_vertical_voxels = 1 + int( fdtd_region_size_vertical_um / mesh_spacing_um )

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
    fdtd['mesh cells x'] = fdtd_region_size_lateral_voxels
    fdtd['mesh cells y'] = fdtd_region_size_vertical_voxels
    fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
    fdtd['background index'] = background_index


    src_maximum_vertical_um = device_depth_um + 0.5 * vertical_gap_um
    src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_um
    src_lateral_aperture_um = 1.1 * aperture_size_um

    #
    # Add a TFSF plane wave forward source at normal incidence
    #
    forward_src = fdtd_hook.addtfsf()
    forward_src['name'] = 'forward_src'
    forward_src['polarization angle'] = 90
    forward_src['direction'] = 'Backward'
    forward_src['x span'] = src_lateral_aperture_um * 1e-6
    forward_src['y max'] = src_maximum_vertical_um * 1e-6
    forward_src['y min'] = src_minimum_vertical_um * 1e-6
    forward_src['wavelength start'] = src_lambda_min_um * 1e-6
    forward_src['wavelength stop'] = src_lambda_max_um * 1e-6

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
    adjoint_x_positions_um = []
    for adj_src_idx in range(0, num_adjoint_sources):
        adj_src = fdtd_hook.adddipole()
        adj_src['name'] = 'adj_src_' + str(adj_src_idx)

        adjoint_x_position_um = ( -0.5 + ( 2 * ( adj_src_idx + 1 ) - 1 ) / ( num_adjoint_sources + 2 ) ) * aperture_size_um
        adjoint_x_positions_um.append( adjoint_x_position_um )

        adj_src['x'] = adjoint_x_position_um * 1e-6
        adj_src['y'] = -focal_length_um * 1e-6
        adj_src['theta'] = 0
        adj_src['phi'] = 0
        adj_src['wavelength start'] = src_lambda_min_um * 1e-6
        adj_src['wavelength stop'] = src_lambda_max_um * 1e-6

        adjoint_sources.append(adj_src)

    #
    # Set up the volumetric electric field monitor inside the design region.  We will need this compute
    # the adjoint gradient
    #
    design_efield_monitor = fdtd_hook.addprofile()
    design_efield_monitor['name'] = 'design_efield_monitor'
    design_efield_monitor['x span'] = aperture_size_um * 1e-6
    design_efield_monitor['y min'] = 0 * 1e-6
    design_efield_monitor['y max'] = device_depth_um * 1e-6
    design_efield_monitor['override global monitor settings'] = 1
    design_efield_monitor['use wavelength spacing'] = 1
    design_efield_monitor['use source limits'] = 0
    design_efield_monitor['minimum wavelength'] = lambda_low_um * 1e-6
    design_efield_monitor['maximum wavelength'] = lambda_high_um * 1e-6
    design_efield_monitor['frequency points'] = 2
    design_efield_monitor['output Hx'] = 0
    design_efield_monitor['output Hy'] = 0
    design_efield_monitor['output Hz'] = 0

    #
    # Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
    # compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
    # gradient.
    #
    focal_monitors = []

    for adj_src_idx in range(0, num_adjoint_sources):
        focal_monitor = fdtd_hook.addpower()
        focal_monitor['name'] = 'focal_monitor' + str(adj_src_idx)
        focal_monitor['monitor type'] = 'point'
        focal_monitor['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
        focal_monitor['y'] = -focal_length_um * 1e-6
        focal_monitor['override global monitor settings'] = 1
        focal_monitor['use wavelength spacing'] = 1
        focal_monitor['use source limits'] = 0
        focal_monitor['minimum wavelength'] = lambda_low_um * 1e-6
        focal_monitor['maximum wavelength'] = lambda_high_um * 1e-6
        focal_monitor['frequency points'] = 2

        focal_monitors.append(focal_monitor)


    disable_all_sources()
    fdtd_hook.run()
    efield_data = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )
    design_shape_voxels = np.flip( efield_data.shape[ 2 : ] )
    device_voxels_lateral = design_shape_voxels[ 0 ]
    device_voxels_vertical = design_shape_voxels[ 1 ]
    fdtd_hook.switchtolayout()

    min_device_permittivity = index_low**2
    max_device_permittivity = index_high**2

    device_density = np.random.random( design_shape_voxels )
    device_density = gaussian_filter( device_density, sigma=2 )
    device_density -= np.min( device_density )
    device_density /= np.max( device_density )

    device_permittivity = min_device_permittivity + max_device_permittivity * device_density
    device_index = np.sqrt( device_permittivity )
    device_index_replicate = np.zeros( ( device_index.shape[ 0 ], device_index.shape[ 1 ], 2 ) )
    device_index_replicate[ :, :, 0 ] = device_index[ :, :, 0 ]
    device_index_replicate[ :, :, 1 ] = device_index[ :, :, 0 ]

    device_import = fdtd_hook.addimport()
    device_import['name'] = 'device_import'
    device_import['x span'] = aperture_size_um * 1e-6
    device_import['y min'] = 0 * 1e-6
    device_import['y max'] = device_depth_um * 1e-6
    device_import['z min'] = -0.51 * 1e-6
    device_import['z max'] = 0.51 * 1e-6

    bayer_filter_region_z = 1e-6 * np.linspace(-0.51, 0.51, 2)
    bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * aperture_size_um, 0.5 * aperture_size_um, device_voxels_lateral)
    bayer_filter_region_y = 1e-6 * np.linspace(0, device_depth_um, device_voxels_vertical)

    fdtd_hook.switchtolayout()
    fdtd_hook.select( device_import[ 'name' ] )
    fdtd_hook.importnk2( device_index_replicate, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )

    figure_of_merit = np.zeros( num_iterations )

    lambda_values_um = np.array( [ lambda_low_um, lambda_high_um ] )
    max_intensity_by_wavelength = (aperture_size_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)

    for iteration in range( 0, num_iterations ):
        iter_start_time = time.time()

        log_file = open(projects_directory_location + "/log.txt", 'a+')

       	device_permittivity = min_device_permittivity + max_device_permittivity * device_density
        device_index = np.sqrt( device_permittivity )
        fdtd_hook.switchtolayout()
        fdtd_hook.select( device_import[ 'name' ] )
        device_index_replicate[ :, :, 0 ] = device_index[ :, :, 0 ]
        device_index_replicate[ :, :, 1 ] = device_index[ :, :, 0 ]
        fdtd_hook.importnk2( device_index_replicate, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )

        post_import_time = time.time()
        log_file.write("To do the device import took " + str( post_import_time - iter_start_time) + " seconds.\n")

        #
        # Step 1: Run the forward optimization for both x- and y-polarized plane waves.
        #
        disable_all_sources()
        forward_src.enabled = 1
        fdtd_hook.run()

        run_fdtd_time = time.time()
        log_file.write("To run FDTD took " + str( run_fdtd_time - post_import_time) + " seconds.\n")

        forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

        collect_fields_time = time.time()

        log_file.write("To collect fields took " + str( collect_fields_time - run_fdtd_time) + " seconds.\n")

        focal_data = []
        for adj_src_idx in range(0, num_adjoint_sources):
            focal_data.append(
                get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )

        #
        # Step 2: Compute the figure of merit
        #
        normalized_intensity_focal_point_wavelength = np.zeros( num_adjoint_sources )
        conjugate_weighting_focal_point_wavelength = np.zeros( num_adjoint_sources, dtype=np.complex )

        for focal_idx in range(0, num_adjoint_sources):
            normalized_intensity_focal_point_wavelength[ focal_idx ] = (
                np.sum( np.abs( focal_data[ focal_idx ][ 2, focal_idx, 0, 0, 0 ])**2 )
            ) / max_intensity_by_wavelength[ focal_idx ]

            conjugate_weighting_focal_point_wavelength[ focal_idx ] = np.conj(
                focal_data[ focal_idx ][ 2, focal_idx, 0, 0, 0 ] / max_intensity_by_wavelength[ focal_idx ] )

        figure_of_merit[ iteration ] = np.mean( normalized_intensity_focal_point_wavelength )
        fom_weighting = ( 2.0 / num_adjoint_sources ) - normalized_intensity_focal_point_wavelength**2 / np.sum( normalized_intensity_focal_point_wavelength**2 )
        fom_weighting = np.maximum( fom_weighting, 0 )
        fom_weighting /= np.sum( fom_weighting )

        reversed_field_shape = [1, device_voxels_vertical, device_voxels_lateral]
        gradient = np.zeros(reversed_field_shape, dtype=np.complex)

        for adj_src_idx in range(0, num_adjoint_sources):
            pre_adj_src_time = time.time()
            disable_all_sources()
            (adjoint_sources[adj_src_idx]).enabled = 1
            fdtd_hook.run()
            adj_src_runtime = time.time() - pre_adj_src_time
            log_file.write("To do an adjoint source took " + str( adj_src_runtime ) + " seconds.\n") 
            adjoint_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

            source_weight = np.conj(
                conjugate_weighting_focal_point_wavelength[adj_src_idx])

            gradient += np.sum(
                ( source_weight * fom_weighting[ adj_src_idx ] ) *
                adjoint_e_fields[ :, adj_src_idx, :, :, : ] * forward_e_fields[ :, adj_src_idx, :, :, : ],
                axis=0
            )

        gradient = 2 * np.real( np.swapaxes( gradient, 0, 2 ) )
        density_gradient = ( max_device_permittivity - min_device_permittivity ) * gradient

        current_step_density_max = start_design_change_max + iteration * ( end_design_change_max - start_design_change_max ) / ( num_iterations - 1 )
        step_size = current_step_density_max / np.max( np.abs( density_gradient ) )

        device_density_save = device_density.copy()

        device_density -= step_size * density_gradient
        device_density = np.maximum( np.minimum( device_density, 1 ), 0 )

        rest_of_time = time.time()
        log_file.write("To do one iteration took " + str( rest_of_time - iter_start_time) + " seconds.\n")
        log_file.close()
 

    device_permittivity = min_device_permittivity + max_device_permittivity * device_density
    device_index = np.sqrt( device_permittivity )
    np.save( projects_directory_location + "/figure_of_merit.npy", figure_of_merit )
    np.save( projects_directory_location + "/density.npy", device_density )
    np.save( projects_directory_location + "/permittivity.npy", device_permittivity )
    np.save( projects_directory_location + "/index.npy", device_index )
    np.save( projects_directory_location + "/aperture_size_um.npy", aperture_size_um )
    np.save( projects_directory_location + "/device_depth_um.npy", device_depth_um )
    np.save( projects_directory_location + "/lambda_low_um.npy", lambda_low_um )
    np.save( projects_directory_location + "/lambda_high_um.npy", lambda_high_um )
    np.save( projects_directory_location + "/index_low.npy", index_low )
    np.save( projects_directory_location + "/index_high.npy", index_high )
    np.save( projects_directory_location + "/focal_length_um.npy", focal_length_um )

    disable_all_sources()
    design_efield_monitor['output Hx'] = 1
    design_efield_monitor['output Hy'] = 1
    design_efield_monitor['output Hz'] = 1
    forward_src.enabled = 1
    fdtd_hook.run()

    forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')
    forward_h_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

    np.save( projects_directory_location + "/forward_e_fields.npy", forward_e_fields )
    np.save( projects_directory_location + "/forward_h_fields.npy", forward_h_fields )

