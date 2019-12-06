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

# == 377.1
mu_nought_c = ( 1.257 * 1e-6 ) * ( 3.0 * 1e8 )

monitor_lateral_voxels = 2 + int( 1e6 * mode_reflection_monitor[ 'x span' ] / mesh_spacing_um )
# Organize these as freq, pol, z, y, x

mode_e_field_xpol = np.zeros( ( 3, num_design_frequency_points, 1, monitor_lateral_voxels, monitor_lateral_voxels ), dtype=np.complex )
mode_h_field_xpol = np.zeros( ( 3, num_design_frequency_points, 1, monitor_lateral_voxels, monitor_lateral_voxels ), dtype=np.complex )

mode_e_field_xpol[ 0, :, :, :, : ] = 1
mode_h_field_xpol[ 1, :, :, :, : ] = ( 1. / mu_nought_c )


mode_e_field_ypol = np.zeros( ( 3, num_design_frequency_points, 1, monitor_lateral_voxels, monitor_lateral_voxels ), dtype=np.complex )
mode_h_field_ypol = np.zeros( ( 3, num_design_frequency_points, 1, monitor_lateral_voxels, monitor_lateral_voxels ), dtype=np.complex )

mode_e_field_ypol[ 1, :, :, :, : ] = 1
mode_h_field_ypol[ 0, :, :, :, : ] = -( 1. / mu_nought_c )

mode_e_fields[ 'x' ] = mode_e_field_xpol
mode_h_fields[ 'x' ] = mode_h_field_xpol

mode_e_fields[ 'y' ] = mode_e_field_ypol
mode_h_fields[ 'y' ] = mode_h_field_ypol


phase_corrections_reflection = np.zeros( num_design_frequency_points, dtype=np.complex )

for wl_idx in range( 0, num_design_frequency_points ):
    wavelength_um = lambda_values_um[ wl_idx ]
    phase_shift = 2 * np.pi * mode_reflection_monitor_delta_um / wavelength_um
    phase_corrections_reflection[ wl_idx ] = np.exp( 1j * phase_shift )

plane_wave_sources['x']['direction'] = 'Backward'
plane_wave_sources['x']['z min'] = src_minimum_vertical_um * 1e-6
plane_wave_sources['x']['z max'] = src_maximum_vertical_um * 1e-6

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

#
# Add device region and create device permittivity
#

min_device_permittivity = min_real_permittivity + 1j * min_imag_permittivity
max_device_permittivity = max_real_permittivity + 1j * max_imag_permittivity

#
# Here, many devices will actually be added, one for each actually designable region.  When the region is not
# designable, we will just add a block of material there.  This applies for things like the via and capping layers
#
filter_import = fdtd_hook.addimport()
filter_import['name'] = 'filter_import'
filter_import['x span'] = device_size_lateral_um * 1e-6
filter_import['y span'] = device_size_lateral_um * 1e-6
filter_import['z min'] = designable_device_vertical_minimum_um * 1e-6
filter_import['z max'] = designable_device_vertical_maximum_um * 1e-6

filter_permittivity = 1.5 * np.ones( ( device_voxels_lateral, device_voxels_lateral, designable_device_voxels_vertical ))
filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
filter_region_z = 1e-6 * np.linspace(designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, designable_device_voxels_vertical)

fdtd_hook.select("filter_import")
fdtd_hook.importnk2( filter_region_x, filter_region_y, filter_region_z )



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
            np.sum( np.conj( choose_electric_mode[ 0, wl_idx, 0, :, : ] ) * choose_magnetic_forward[ 1, wl_idx, 0, :, : ] ) )

        numerator_term_2 = -(
            np.sum( choose_electric_forward[ 1, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, :, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 1, wl_idx, 0, :, : ] ) * choose_magnetic_forward[ 0, wl_idx, 0, :, : ] ) )

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
            np.sum( np.conj( choose_electric_mode[ 0, wl_idx, 0, :, : ] ) * choose_magnetic_forward[ 1, wl_idx, 0, :, : ] ) )

        numerator_term_2 = -(
            np.sum( choose_electric_forward[ 1, wl_idx, 0, :, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, :, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 1, wl_idx, 0, :, : ] ) * choose_magnetic_forward[ 0, wl_idx, 0, :, : ] ) )

        numerator = numerator_term_1 + numerator_term_2

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
# Run forward source
#
pol = 'x'
disable_all_sources()
plane_wave_sources[pol].enabled = 1

forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

reflected_e_fields = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
reflected_h_fields = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

fom_0_by_wavelength = np.zeros( num_design_frequency_points )
mode_e_field = mode_e_fields[ pol ]
mode_h_field = mode_h_fields[ pol ]

mode_overlap_maxima_r = 1

for wl_idx in range( 0, num_design_frequency_points ):
    fom_0_by_wavelength[ wl_idx ] = mode_overlap_fom(
        reflected_e_fields[ :, wl_idx, :, :, :], reflected_h_fields[ :, wl_idx, :, :, : ],
        mode_e_field[ :, wl_idx, :, :, : ], mode_h_field[ :, wl_idx, :, :, : ],
        1, mode_overlap_maxima_r
    )

np.save( projects_directory_location + "/forward_e_fields.npy", forward_e_fields )

fd_y = int( device_voxels_lateral / 2.0 )
fd_z = int( designable_device_voxels_vertical / 2.0 )
h = 0.01

fd_by_wavelength = np.zeros( ( device_voxels_lateral, num_design_frequency_points ) )

for fd_x in range( 0, device_voxels_lateral ):
    filter_permittivity[ fd_x, fd_y, fd_z ] += h
    fdtd_hook.select("filter_import")
    fdtd_hook.importnk2( filter_region_x, filter_region_y, filter_region_z )

    fom_1_by_wavelength = np.zeros( num_design_frequency_points )

    forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

    reflected_e_fields = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'E' )
    reflected_h_fields = get_complex_monitor_data( mode_reflection_monitor[ 'name' ], 'H' )

    for wl_idx in range( 0, num_design_frequency_points ):
        fom_1_by_wavelength[ wl_idx ] = mode_overlap_fom(
            reflected_e_fields[ :, wl_idx, :, :, :], reflected_h_fields[ :, wl_idx, :, :, : ],
            mode_e_field[ :, wl_idx, :, :, : ], mode_h_field[ :, wl_idx, :, :, : ],
            1, mode_overlap_maxima_r
        )
    
    fd_by_wavelength[ fd_x, : ] = ( fom_1_by_wavelength - fom_0_by_wavelength ) / h

    filter_permittivity[ fd_x, fd_y, fd_z ] -= h

np.save( projects_directory_location + "/fd_by_wavelength.npy", fd_by_wavelength )


