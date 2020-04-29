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

projects_directory_location += "/convert_polarization_focusing_v1/"

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

device_thickness_um = 1.5
device_thickness_voxels = 2 + int( device_thickness_um / mesh_spacing_um )

device_size_lateral_um = 2.0
device_size_lateral_voxels = 2 + int( device_size_lateral_um / mesh_spacing_um )

permittivity_max = 2.0**2
permittivity_min = 1.0**2

num_design_frequency_points = 10

lambda_min_um = 0.5
lambda_max_um = 0.6

wavelengths_um = np.linspace( lambda_min_um, lambda_max_um, num_design_frequency_points )
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * wavelengths_um**2)


top_gap_um = 2.0
lateral_gap_um = 1.0
focal_length_um = 1.5
bottom_gap_um = 1.0

fdtd_lateral_size_um = device_size_lateral_um + 2 * lateral_gap_um
fdtd_lateral_size_voxels = 1 + int( fdtd_lateral_size_um / mesh_spacing_um )
fdtd_vertical_size_um = top_gap_um + bottom_gap_um + focal_length_um + device_thickness_um
fdtd_vertical_size_voxels = 1 + int( fdtd_vertical_size_um / mesh_spacing_um )

fdtd_simulation_time_fs = 5000 * 2

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


forward_src = {}
adjoint_src = {}

forward_src = fdtd_hook.addtfsf()
forward_src['name'] = 'forward_src'
forward_src['angle phi'] = xy_phi_rotations[ 'x' ]
forward_src['direction'] = 'Backward'
forward_src['plane wave type'] = 'Bloch/Periodic'
forward_src['x span'] = 1.1 * fdtd_lateral_size_um * 1e-6
forward_src['y span'] = 1.1 * fdtd_lateral_size_um * 1e-6
forward_src['z max'] = ( device_thickness_um + 0.5 * top_gap_um ) * 1e-6
forward_src['z min'] = -( focal_length_um + 0.5 * bottom_gap_um ) * 1e-6
forward_src['wavelength start'] = lambda_min_um * 1e-6
forward_src['wavelength stop'] = lambda_max_um * 1e-6


adjoint_src = fdtd_hook.adddipole()
adjoint_src['name'] = 'adjoint_src'
adjoint_src['theta'] = 90
adjoint_src['phi'] = 90
adjoint_src['x'] = 0 * 1e-6
adjoint_src['y'] = 0 * 1e-6
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

#
# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
# gradient.
#
mode_transmission_monitor = fdtd_hook.addpower()
mode_transmission_monitor['name'] = 'mode_transmission_monitor'
mode_transmission_monitor['monitor type'] = '2D Z-normal'
mode_transmission_monitor['x span'] = unit_cell_size_um * 1e-6
mode_transmission_monitor['y span'] = unit_cell_size_um * 1e-6
mode_transmission_monitor['z'] = -0.5 * bottom_gap_um * 1e-6
mode_transmission_monitor['override global monitor settings'] = 1
mode_transmission_monitor['use wavelength spacing'] = 1
mode_transmission_monitor['use source limits'] = 0
mode_transmission_monitor['minimum wavelength'] = lambda_min_um * 1e-6
mode_transmission_monitor['maximum wavelength'] = lambda_max_um * 1e-6
mode_transmission_monitor['frequency points'] = num_design_frequency_points

focal_monitor = fdtd_hook.addpower()
focal_monitor['name'] = 'focal_monitor'
focal_monitor['monitor type'] = 'point'
focal_monitor['x'] = 0 * 1e-6
focal_monitor['y'] = 0 * 1e-6
focal_monitor['z'] = -focal_length_um * 1e-6
focal_monitor['override global monitor settings'] = 1
focal_monitor['use wavelength spacing'] = 1
focal_monitor['use source limits'] = 1
focal_monitor['frequency points'] = num_design_frequency_points


np.random.seed( 123123 )

device_permittivity = np.zeros( ( device_size_lateral_voxels, device_size_lateral_voxels, device_thickness_voxels ) )
random_design_seed = 0.25 * np.random.random( device_permittivity.shape )
random_design_seed = gaussian_filter( random_design_seed, sigma=3 )
device_permittivity = permittivity_min + ( permittivity_max - permittivity_min ) * random_design_seed


num_iterations = 100
figure_of_merit_evolution = np.zeros( num_iterations )

#
# Run the optimization
#
fdtd_hook.save(projects_directory_location + "/optimization")
for iteration in range(0, num_iterations):
    print( "Working on iteration " + str( iteration ) )

    fdtd_hook.switchtolayout()
    fdtd_hook.select( device_import[ 'name' ] )
    fdtd_hook.importnk2( np.sqrt( device_permittivity ), device_x_range, device_y_range, device_z_range )

    disable_all_sources()
    forward_src.enabled = 1
    fdtd_hook.run()

    focal_e = get_complex_monitor_data( focal_monitor[ 'name' ], 'E' )
    intensity_y = np.zeros( num_design_frequency_points )
    conj_Ey = np.zeros( num_design_frequency_points, dtype=np.complex )
    for wl_idx in range( 0, num_design_frequency_points ):
        intensity_y = np.sum( np.abs( focal_e[ 1, wl_idx, 0, 0, 0 ] )**2 / max_intensity_by_wavelength[ wl_idx ] )
        conj_Ey = np.squeeze( np.conj( focal_e[ 1, wl_idx, 0, 0, 0 ] ) )

    forward_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )

    print( intensity_y )
    average_intensity_y = np.mean( intensity_y )
    print( "Current average y intensity = " + str( average_intensity_y ) )
    figure_of_merit_evolution[ iteration ] = average_intensity_y

    disable_all_sources()
    adjoint_src.enabled = 1
    fdtd_hook.run()

    adjoint_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )

    fom_weighting = ( 2.0 / num_wavelengths ) - intensity_y**2 / np.sum( intensity_y**2 )
    fom_weighting = np.maximum( fom_weighting, 0 )
    fom_weighting /= np.sum( fom_weighting )

    net_gradient = np.zeros( forward_e_fields[ 0, 0 ].shape )

    for wl_idx in range ( 0, num_design_frequency_points ):
        wl_gradient = ( 1. / max_intensity_by_wavelength[ wl_idx ] ) * np.sum(
            np.real(
                conj_Ey[ wl_idx ] *
                forward_e_fields[ :, wl_idx, :, :, : ] *
                adjoint_e_fields[ :, wl_idx, :, :, : ] ),
            axis=0 )

        net_gradient += fom_weighting * wl_gradient

    net_gradient = np.swapaxes( net_gradient, 0, 2 )

    step_size = 0.025 * ( permittivity_max - permittivity_min ) / np.max( np.abs( net_gradient ) )
    device_permittivity += step_size * net_gradient
    device_permittivity = np.maximum( permittivity_min, np.minimum( device_permittivity, permittivity_max ) )

    np.save( projects_directory_location + '/device_gradient.npy', net_gradient )
    np.save( projects_directory_location + '/figure_of_merit.npy', figure_of_merit_evolution )



