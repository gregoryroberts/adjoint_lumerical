import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from SwitchableIRReflectorIsolatedParameters import *

# import imp
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )
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


def get_non_struct_data(monitor_name, monitor_field):
	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	data_transfer_filename = projects_directory_location + "/data_transfer_" + monitor_name + "_" + monitor_field

	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + lumerical_data_name + ");"

	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[lumerical_data_name])

	return monitor_data['real']

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

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat", 'r')

	monitor_data = np.array(load_file[extracted_data_name])

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	data = get_monitor_data(monitor_name, monitor_field)
	return (data['real'] + np.complex(0, 1) * data['imag'])

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

projects_directory_location += "/" + project_name + '_convergence'

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/SwitchableIRReflectorParameters.py", projects_directory_location + "/SwitchableIRReflectorParameters.py")



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
# Setting the x min bc to Bloch will automatically set the x max bc to Bloch and lock it
fdtd['x min bc'] = 'PML'
fdtd['x max bc'] = 'PML'
fdtd['y min bc'] = 'PML'
fdtd['y max bc'] = 'PML'
# fdtd['dt stability factor'] = 0.25
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
mode_sources = []
source_polarization_angles = [ 0, 90 ]
affected_coords_by_polarization = [ [ 0, 1 ], [ 2 ] ]

for pol_idx in range( 0, num_polarizations ):
	mode_src = fdtd_hook.addtfsf()
	mode_src['name'] = 'mode_src_' + str( pol_idx )
	mode_src['polarization angle'] = source_polarization_angles[ pol_idx ]
	mode_src['direction'] = 'Backward'
	mode_src['x span'] = lateral_aperture_um * 1e-6
	mode_src['y max'] = ( designable_device_vertical_minimum_um - 0.25 * vertical_gap_size_um ) * 1e-6
	mode_src['y min'] = fdtd_region_minimum_vertical_um * 1e-6
	mode_src['wavelength start'] = lambda_min_um * 1e-6
	mode_src['wavelength stop'] = lambda_max_um * 1e-6

	mode_sources.append( mode_src )


mode_transmission_monitor_delta_um = 0.25 * vertical_gap_size_um
mode_transmission_monitor = fdtd_hook.addpower()
mode_transmission_monitor['name'] = 'mode_transmission_monitor'
mode_transmission_monitor['monitor type'] = 'Linear X'
# mode_transmission_monitor['x span'] = ( fdtd_region_size_lateral_um - 0.5 * lateral_gap_size_um ) * 1e-6
mode_transmission_monitor['x span'] = lateral_aperture_um * 1e-6
mode_transmission_monitor['y'] = ( mode_sources[ 0 ]['y max'] - 1e-6 * mode_transmission_monitor_delta_um )
mode_transmission_monitor['override global monitor settings'] = 1
if is_lumerical_version_2020a:
	mode_transmission_monitor['use wavelength spacing'] = 1
else:
	mode_transmission_monitor['use linear wavelength spacing'] = 1

mode_transmission_monitor['use source limits'] = 0
mode_transmission_monitor['minimum wavelength'] = lambda_min_um * 1e-6
mode_transmission_monitor['maximum wavelength'] = lambda_max_um * 1e-6
mode_transmission_monitor['frequency points'] = num_design_frequency_points


focal_monitor = fdtd_hook.addpower()
focal_monitor['name'] = 'focal_monitor'
focal_monitor['monitor type'] = 'point'
focal_monitor['x'] = 0
focal_monitor['y'] = ( designable_device_vertical_maximum_um + gsst_thickness_um + arc_thickness_um + reflected_focal_length_um ) * 1e-6
focal_monitor['override global monitor settings'] = 1
if is_lumerical_version_2020a:
	focal_monitor['use wavelength spacing'] = 1
else:
	focal_monitor['use linear wavelength spacing'] = 1
focal_monitor['use source limits'] = 0
focal_monitor['minimum wavelength'] = lambda_min_um * 1e-6
focal_monitor['maximum wavelength'] = lambda_max_um * 1e-6
focal_monitor['frequency points'] = num_design_frequency_points

focal_transmission_monitor = fdtd_hook.addpower()
focal_transmission_monitor['name'] = 'focal_transmission_monitor'
focal_transmission_monitor['monitor type'] = 'Linear X'
focal_transmission_monitor['x'] = 0
focal_transmission_monitor['x span'] = device_size_lateral_um * 1e-6
focal_transmission_monitor['y'] = focal_monitor['y']
focal_transmission_monitor['override global monitor settings'] = 1
if is_lumerical_version_2020a:
	focal_transmission_monitor['use wavelength spacing'] = 1
else:
	focal_transmission_monitor['use linear wavelength spacing'] = 1
focal_transmission_monitor['use source limits'] = 0
focal_transmission_monitor['minimum wavelength'] = lambda_min_um * 1e-6
focal_transmission_monitor['maximum wavelength'] = lambda_max_um * 1e-6
focal_transmission_monitor['frequency points'] = num_design_frequency_points



transmission_adjoint_sources = []
for pol_idx in range( 0, num_polarizations ):
	transmission_adjoint_src = fdtd_hook.addtfsf()
	transmission_adjoint_src['name'] = 'transmission_adjoint_src_' + str( pol_idx )
	transmission_adjoint_src['polarization angle'] = source_polarization_angles[ pol_idx ]
	transmission_adjoint_src['direction'] = 'Forward'
	transmission_adjoint_src['x span'] = lateral_aperture_um * 1e-6
	transmission_adjoint_src['y min'] = mode_transmission_monitor['y']
	transmission_adjoint_src['y max'] = fdtd_region_maximum_vertical_um * 1e-6
	transmission_adjoint_src['wavelength start'] = lambda_min_um * 1e-6
	transmission_adjoint_src['wavelength stop'] = lambda_max_um * 1e-6

	transmission_adjoint_sources.append( transmission_adjoint_src )


focusing_adjoint_sources = []
coord_to_phi = [ 0, 90, 0 ]
coord_to_theta = [ 90, 90, 0 ]

for coord_idx in range( 0, 3 ):
	focusing_adjoint_src = fdtd_hook.adddipole()
	focusing_adjoint_src['name'] = 'focusing_adjoint_src_' + str( coord_idx )
	focusing_adjoint_src['x'] = 0
	focusing_adjoint_src['y'] = focal_monitor['y']
	focusing_adjoint_src['theta'] = coord_to_theta[ coord_idx ]
	focusing_adjoint_src['phi'] = coord_to_phi[ coord_idx ]
	focusing_adjoint_src['wavelength start'] = lambda_min_um * 1e-6
	focusing_adjoint_src['wavelength stop'] = lambda_max_um * 1e-6

	focusing_adjoint_sources.append( focusing_adjoint_src )


forward_sources = [ None for i in range( 0, num_optimization_angles * num_polarizations ) ]
for angle_idx in range( 0, num_optimization_angles ):
	for pol_idx in range( 0, num_polarizations ):
		forward_src = fdtd_hook.addtfsf()
		forward_src['name'] = 'forward_src_' + str( angle_idx ) + '_' + str( pol_idx )
		forward_src['polarization angle'] = source_polarization_angles[ pol_idx ]
		forward_src['direction'] = 'Backward'
		forward_src['angle theta'] = optimization_angles_mid_frequency_degrees[ angle_idx ]
		forward_src['x span'] = lateral_aperture_um * 1e-6
		forward_src['y max'] = src_maximum_vertical_um * 1e-6
		forward_src['y min'] = fdtd_region_minimum_vertical_um * 1e-6
		forward_src['wavelength start'] = lambda_min_um * 1e-6
		forward_src['wavelength stop'] = lambda_max_um * 1e-6

		forward_sources[ angle_idx * num_polarizations + pol_idx ] = forward_src


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	fdtd_hook.switchtolayout()

	for pol_idx in range( 0, num_polarizations ):
		mode_sources[ pol_idx ].enabled = 0
		transmission_adjoint_sources[ pol_idx ].enabled = 0

		for angle_idx in range( 0, num_optimization_angles ):
			forward_sources[ angle_idx * num_polarizations + pol_idx ].enabled = 0

	for coord_idx in range( 0, 3 ):
		(focusing_adjoint_sources[ coord_idx ]).enabled = 0



mode_E = []
mode_H = []
for pol_idx in range( 0, num_polarizations ):
	disable_all_sources()
	mode_sources[ pol_idx ].enabled = 1
	# mode_sources[ pol_idx ][ 'direction' ] = 'Forward'
	# mode_sources[ pol_idx ][ 'angle theta' ] = angle_degrees_middle_lambda_first_order
	fdtd_hook.run()

	get_E_mode = get_complex_monitor_data( mode_transmission_monitor['name'], 'E' )
	get_H_mode = get_complex_monitor_data( mode_transmission_monitor['name'], 'H' )

	num_wls = get_E_mode.shape[ 1 ]
	half_x = int( get_E_mode.shape[ 4 ] / 2. )

	for wl_idx in range( 0, num_wls ):
		phase_correction = np.exp( 1j * np.angle( get_E_mode[ 2, wl_idx, 0, 0, half_x ] ) )

		if pol_idx == 0:
			phase_correction = np.exp( 1j * np.angle( get_E_mode[ 0, wl_idx, 0, 0, half_x ] ) )

		get_E_mode[ :, wl_idx, :, :, : ] /= phase_correction
		get_H_mode[ :, wl_idx, :, :, : ] /= phase_correction

		# plt.plot( np.real( get_E_mode[ 2, wl_idx, 0, 0, : ] ), color='g' )
		# plt.plot( np.imag( get_E_mode[ 2, wl_idx, 0, 0, : ] ), color='b', linestyle='--' )
	plt.show()

	mode_E.append( get_E_mode )
	mode_H.append( get_H_mode )

	disable_all_sources()
	# mode_sources[ pol_idx ][ 'direction' ] = 'Backward'
	# mode_sources[ pol_idx ][ 'angle theta' ] = 0


silicon_bottom = fdtd_hook.addrect()
silicon_bottom['name'] = 'silicon_bottom'
silicon_bottom['x span'] = fdtd_region_size_lateral_um * 1e-6
silicon_bottom['y max'] = designable_device_vertical_minimum_um * 1e-6
silicon_bottom['y min'] = fdtd_region_minimum_vertical_um * 1e-6
silicon_bottom['z span'] = 2.0 * 0.51 * 1e-6
silicon_bottom['material'] = 'Si (Silicon) - Palik'

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
if is_lumerical_version_2020a:
	design_efield_monitor['use wavelength spacing'] = 1
else:
	design_efield_monitor['use linear wavelength spacing'] = 1
design_efield_monitor['use source limits'] = 0
design_efield_monitor['minimum wavelength'] = lambda_min_um * 1e-6
design_efield_monitor['maximum wavelength'] = lambda_max_um * 1e-6
design_efield_monitor['frequency points'] = num_design_frequency_points
design_efield_monitor['output Hx'] = 0
design_efield_monitor['output Hy'] = 0
design_efield_monitor['output Hz'] = 0

design_index_monitor = fdtd_hook.addindex()
design_index_monitor['name'] = 'design_index_monitor'
design_index_monitor['x span'] = device_size_lateral_um * 1e-6
design_index_monitor['y min'] = designable_device_vertical_minimum_um * 1e-6
design_index_monitor['y max'] = designable_device_vertical_maximum_um * 1e-6

#
# Add device region and create device permittivity
#

device_import = fdtd_hook.addimport()
device_import['name'] = 'device_import'
device_import['x span'] = device_size_lateral_um * 1e-6
device_import['y min'] = designable_device_vertical_minimum_um * 1e-6
device_import['y max'] = designable_device_vertical_maximum_um * 1e-6
device_import['z min'] = -0.51 * 1e-6
device_import['z max'] = 0.51 * 1e-6

gsst_import = fdtd_hook.addimport()
gsst_import['name'] = 'gsst'
gsst_import['x span'] = fdtd_region_size_lateral_um * 1e-6
gsst_import['y min'] = gsst_min_y_um * 1e-6
gsst_import['y max'] = gsst_max_y_um * 1e-6
gsst_import['z min'] = -0.51 * 1e-6
gsst_import['z max'] = 0.51 * 1e-6

arc_rect = fdtd_hook.addrect()
arc_rect['name'] = 'antireflection layer'
arc_rect['index'] = arc_index_mgf2
arc_rect['x span'] = fdtd_region_size_lateral_um * 1e-6
arc_rect['y min'] = gsst_max_y_um * 1e-6
arc_rect['y max'] = ( gsst_max_y_um + arc_thickness_um ) * 1e-6
arc_rect['z min'] = -0.51 * 1e-6
arc_rect['z max'] = 0.51 * 1e-6


gsst_state_import_data = []

gsst_x_range = 1e-6 * np.linspace( -0.5 * fdtd_region_size_lateral_um, 0.5 * fdtd_region_size_lateral_um, 2 )
gsst_y_range = 1e-6 * np.linspace( gsst_min_y_um, gsst_max_y_um, 2 )
gsst_z_range = 1e-6 * np.linspace( -0.51, 0.51, 2 )

for gsst_state_idx in range( 0, gsst_num_states ):
	gsst_index = ( gsst_n_states[ gsst_state_idx ] + 1j * gsst_k_states[ gsst_state_idx ] ) * np.ones( ( 2, 2, 2 ) )
	gsst_state_import_data.append( gsst_index )

#
# Add blocks of dielectric on the side of the designable region because we will be leaving those as blank, unpatterned dielectric material
# Would it be better to have this be a part of the bayer filter material inmport and jus tnot modify it (i.e. - mask out any changes to it).  I'm
# thinking of how the mesh behaves here between these interacting objects.  For now, we will add the blocks around the side because this will make
# it simpler at first and then we can move towards making a more sophisticated device class or subclass.
# Further, in reality, this may be a stack of material in general.  However, it will be mostly the low-K dielctric background material so we will assume
# this is not a stratified stack and is actaully just a single piece of low index material background.  In fact, actually, we may not have this at all.
# These types of more exact parameters are somehwat undetermined at this point.
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
	side_block['z span'] = 1.02 * 1e-6

	side_block['index'] = device_background_index


gaussian_normalization = np.zeros( num_points_per_band )
middle_point = num_points_per_band / 2.
spacing = 1. / ( num_points_per_band - 1 )
half_bandwidth = 0.4 * num_points_per_band

for wl_idx in range( 0, num_points_per_band ):
	gaussian_normalization[ wl_idx ] =  ( 1. / half_bandwidth ) * np.sqrt( 1 / ( 2 * np.pi ) ) * np.exp( -0.5 * ( wl_idx - middle_point )**2 / ( half_bandwidth**2 ) )
	
gaussian_normalization /= np.sum( gaussian_normalization )
gaussian_normalization_all = np.array( [ gaussian_normalization for i in range( 0, num_bands ) ] ).flatten()

no_normalization_all = np.mean( gaussian_normalization_all ) * np.ones( gaussian_normalization_all.shape )

# plt.plot( np.linspace( lambda_min_um, lambda_max_um, 2 * num_points_per_band ), gaussian_normalization_all )
# plt.show()

reversed_field_shape = [1, designable_device_voxels_vertical, device_voxels_lateral]
reversed_field_shape_with_pol = [num_polarizations, 1, designable_device_voxels_vertical, device_voxels_lateral]
reversed_field_shape_with_angle_and_pol = [num_optimization_angles * num_polarizations, 1, designable_device_voxels_vertical, device_voxels_lateral]




def mode_overlap_fom_ez(
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

        numerator = (
            np.sum( choose_electric_forward[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) +
            np.sum( np.conj( choose_electric_mode[ 2, wl_idx, 0, 0, : ] ) * choose_magnetic_forward[ 0, wl_idx, 0, 0, : ] ) )
        numerator = np.abs( numerator )**2
        denominator = -( -8.0 * np.real( np.sum( choose_electric_mode[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) ) )

        fom_by_wavelength[ wl_idx ] = ( numerator / denominator )
        if mode_overlap_norm is not None:
            fom_by_wavelength[ wl_idx ] = ( numerator / ( mode_overlap_norm[ wl_idx ] * denominator ) )

        fom_by_wavelength[ wl_idx ] *= normal_weighting

    return total_norm * fom_by_wavelength


def mode_overlap_fom_hz(
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

        numerator = (
            -np.sum( choose_electric_forward[ 0, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 2, wl_idx, 0, 0, : ] ) ) -
            np.sum( np.conj( choose_electric_mode[ 0, wl_idx, 0, 0, : ] ) * choose_magnetic_forward[ 2, wl_idx, 0, 0, : ] ) )
        numerator = np.abs( numerator )**2
        denominator = -( -8.0 * np.real( np.sum( choose_electric_mode[ 0, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 2, wl_idx, 0, 0, : ] ) ) ) )

        fom_by_wavelength[ wl_idx ] = ( numerator / denominator )
        if mode_overlap_norm is not None:
            fom_by_wavelength[ wl_idx ] = ( numerator / ( mode_overlap_norm[ wl_idx ] * denominator ) )
    
        fom_by_wavelength[ wl_idx ] *= normal_weighting

    return total_norm * fom_by_wavelength

def mode_overlap_gradient_ez(
    figure_of_merit, fom_weighting,
    electric_fields_forward, magnetic_fields_forward,
    electric_mode_fields, magnetic_mode_fields,
    electric_fields_gradient_forward, electric_fields_gradient_adjoint,
    normal_weighting,
    mode_overlap_norm ):

    num_wavelengths = electric_fields_forward.shape[ 1 ]

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
        gradient += normal_weighting * ( 
            fom_weighting[ wl_idx ] * adjoint_phase *
            np.sum( electric_fields_gradient_forward[ :, wl_idx, :, :, : ] * electric_fields_gradient_adjoint[ :, wl_idx, :, :, : ], axis=0 ) )

    return -gradient / num_wavelengths

def mode_overlap_gradient_hz(
    figure_of_merit, fom_weighting,
    electric_fields_forward, magnetic_fields_forward,
    electric_mode_fields, magnetic_mode_fields,
    electric_fields_gradient_forward, electric_fields_gradient_adjoint,
    normal_weighting,
    mode_overlap_norm ):

    num_wavelengths = electric_fields_forward.shape[ 1 ]

    gradient = np.zeros( electric_fields_gradient_forward.shape[ 2 : ], dtype=np.complex )

    for wl_idx in range( 0, num_wavelengths ):
        choose_electric_mode = electric_mode_fields
        choose_magnetic_mode = magnetic_mode_fields

        choose_electric_forward = electric_fields_forward
        choose_magnetic_forward = magnetic_fields_forward

        numerator = (
            -np.sum( choose_electric_forward[ 0, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 2, wl_idx, 0, 0, : ] ) ) -
            np.sum( np.conj( choose_electric_mode[ 0, wl_idx, 0, 0, : ] ) * choose_magnetic_forward[ 2, wl_idx, 0, 0, : ] ) )
        denominator = 4.0 * np.real( np.sum( choose_electric_mode[ 0, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 2, wl_idx, 0, 0, : ] ) ) )

        adjoint_phase = np.conj( numerator ) / ( denominator * mode_overlap_norm[ wl_idx ] )
        gradient += normal_weighting * ( 
            fom_weighting[ wl_idx ] * adjoint_phase *
            np.sum( electric_fields_gradient_forward[ :, wl_idx, :, :, : ] * electric_fields_gradient_adjoint[ :, wl_idx, :, :, : ], axis=0 ) )

    return -gradient / num_wavelengths


mode_overlap_fom_by_pol = [ mode_overlap_fom_hz, mode_overlap_fom_ez ]
mode_overlap_gradient_by_pol = [ mode_overlap_gradient_hz, mode_overlap_gradient_ez ]
mode_overlap_norm_by_pol = []

normalize_for_device_width = mode_transmission_monitor[ 'x span' ] / ( device_size_lateral_um * 1e-6 )
print("Normalization for device width = " + str( normalize_for_device_width ) )

for pol_idx in range( 0, num_polarizations ):
	mode_overlap_norm_by_pol.append(
		mode_overlap_fom_by_pol[ pol_idx ]( mode_E[ pol_idx ], mode_H[ pol_idx ], mode_E[ pol_idx ], mode_H[ pol_idx ], 1.0 ) / normalize_for_device_width
	)


my_optimization_state = optimization_stages[ 0 ]

get_index = my_optimization_state.assemble_index( 0 )
get_permittivity = get_index**2
device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, get_index.shape[ 0 ] )
device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, get_index.shape[ 1 ] )
device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )


fd_x_cut = int( 0.5 * get_index.shape[ 0 ] )
fd_y_size = get_index.shape[ 1 ]
fd_y_cut = int( 0.5 * get_index.shape[ 1 ] )

# delta = 1e-3

fdtd_hook.switchtolayout()
fdtd_hook.select( gsst_import['name'] )
fdtd_hook.importnk2( gsst_state_import_data[ 0 ], gsst_x_range, gsst_y_range, gsst_z_range )


num_delta_values = 20
start = 1
delta_values = 5**( -np.linspace( start, start + num_delta_values - 1, num_delta_values ) )
print( delta_values )

fd_convergence = np.zeros( num_delta_values )

# for fd_y_idx in range( 0, fd_y_size ):
for delta_idx in range( 0, num_delta_values ):
	delta = delta_values[ delta_idx ]

	fd_permittivity = get_permittivity.copy()
	# fd_permittivity[ fd_x_cut, fd_y_idx ] += delta
	fd_permittivity[ fd_x_cut, fd_y_cut ] += delta

	inflate_index = np.zeros( ( get_index.shape[ 0 ], get_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = np.sqrt( fd_permittivity )
	inflate_index[ :, :, 1 ] = np.sqrt( fd_permittivity )
	fdtd_hook.select( device_import[ 'name' ] )
	fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

	disable_all_sources()
	forward_sources[ 0 ].enabled = 1
	fdtd_hook.run()

	reflected_E = get_complex_monitor_data( focal_monitor[ 'name' ], 'E' )

	fom_up = np.zeros( num_design_frequency_points )

	for wl_idx in range( 0, num_design_frequency_points ):
		fom_up[ wl_idx ] = (
			np.sum( np.abs( reflected_E[ :, wl_idx, 0, 0, 0 ] )**2, axis=0 ) / max_intensity_by_wavelength[ wl_idx ] )

###################################################################################################################

	fd_permittivity = get_permittivity.copy()
	# fd_permittivity[ fd_x_cut, fd_y_idx ] -= delta
	fd_permittivity[ fd_x_cut, fd_y_cut ] -= delta

	inflate_index = np.zeros( ( get_index.shape[ 0 ], get_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = np.sqrt( fd_permittivity )
	inflate_index[ :, :, 1 ] = np.sqrt( fd_permittivity )
	fdtd_hook.select( device_import[ 'name' ] )
	fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

	disable_all_sources()
	forward_sources[ 0 ].enabled = 1
	fdtd_hook.run()

	reflected_E = get_complex_monitor_data( focal_monitor[ 'name' ], 'E' )

	fom_down = np.zeros( num_design_frequency_points )

	for wl_idx in range( 0, num_design_frequency_points ):
		fom_down[ wl_idx ] = (
			np.sum( np.abs( reflected_E[ :, wl_idx, 0, 0, 0 ] )**2, axis=0 ) / max_intensity_by_wavelength[ wl_idx ] )

###################################################################################################################

	fd_grad = ( np.sum( fom_up ) - np.sum( fom_down ) ) / ( 2 * delta )
	fd_convergence[ delta_idx ] = fd_grad

	np.save( projects_directory_location + '/fd_convergence.npy', fd_convergence )
	np.save( projects_directory_location + '/fd_delta.npy', delta_values )

plt.plot( delta_values, fd_convergence, linewidth=2, color='b' )
plt.show()


