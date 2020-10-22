import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ReflectiveColor2DParameters import *

run_on_cluster = True

if run_on_cluster:
	import imp
	imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
else:
	import imp
	imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

import continuous_cmos

from scipy import ndimage, misc

def softplus( x_in ):
	return np.log( 1 + np.exp( x_in ) )

def softplus_prime( x_in ):
	return ( 1. / ( 1 + np.exp( -x_in ) ) )


# def softplus( x_in ):
# 	k = 10
# 	return ( np.log( 1 + np.exp( k * x_in ) ) / k )

# def softplus_prime( x_in ):
# 	k = 10
# 	return ( 1. / ( 1 + np.exp( -k * x_in ) ) )



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
# Figure of merit design, transmission for global search part?
# Generate data for Terry
# Try optimized bounds for binarization code in 2D
# Analyze FTIR data and send update to Andrei
# Shrinking the substrate platform on the mid-IR bayer filter reduces the transmission a good amount
# Angular components on the mid-IR bayer filter do not seem like they will be properly imaged by the reflective microscope
# What does that flat phase front effectively look like for the illumination?
# RECEIPTS
# Catch up on emails
# Email back about tutoring
#



#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))

if run_on_cluster:
	projects_directory_location = "/central/groups/Faraon_Computing/projects"


if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

should_reload = False
# projects_directory_reload = projects_directory_location + "/" + project_name + "_continuous_Hz_sio2_no_constrast_v2"
projects_directory_location += "/" + project_name + "_continuous_reflective_red_v3"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/ReflectiveColor2DParameters.py", projects_directory_location + "/ReflectiveColor2DParameters.py")
shutil.copy2(python_src_directory + "/ReflectiveColor2DOptimization.py", projects_directory_location + "/ReflectiveColor2DOptimization.py")
shutil.copy2(python_src_directory + "/continuous_cmos.py", projects_directory_location + "/continuous_cmos.py")



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
forward_sources = []
source_polarization_angles = [ 90, 0 ]
affected_coords_by_polarization = [ [ 2 ], [ 0, 1 ] ]

for pol_idx in range( 0, num_polarizations ):
	# forward_src = fdtd_hook.addtfsf()
	# forward_src = fdtd_hook.addplane()
	# forward_src['plane wave type'] = 'Diffracting'
	forward_src = fdtd_hook.addgaussian()
	forward_src['name'] = 'forward_src_' + str( pol_idx )
	forward_src['polarization angle'] = source_polarization_angles[ pol_idx ]
	forward_src['direction'] = 'Backward'
	forward_src['x span'] = fdtd_region_size_lateral_um * 1e-6
	forward_src['y'] = src_maximum_vertical_um * 1e-6
	forward_src['waist radius w0'] = 0.5 * lateral_aperture_um * 1e-6
	forward_src['distance from waist'] = ( designable_device_vertical_maximum_um - src_maximum_vertical_um ) * 1e-6
	# forward_src['y max'] = src_maximum_vertical_um * 1e-6
	# forward_src['y min'] = src_minimum_vertical_um * 1e-6
	forward_src['wavelength start'] = lambda_min_um * 1e-6
	forward_src['wavelength stop'] = lambda_max_um * 1e-6

	forward_sources.append( forward_src )


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	fdtd_hook.switchtolayout()

	for pol_idx in range( 0, num_polarizations ):
		forward_sources[ pol_idx ].enabled = 0

	for adj_src_idx in range(0, len( adjoint_sources_reflection ) ):
		(adjoint_sources_reflection[ adj_src_idx ]).enabled = 0
	for adj_src_idx in range(0, len( adjoint_sources_transmission ) ):
		(adjoint_sources_transmission[ adj_src_idx ]).enabled = 0


#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = [ [] for i in range( 0, 3 ) ]
coord_to_phi = [ 0, 90, 0 ]
coord_to_theta = [ 90, 90, 0 ]

adjoint_sources_reflection = []
source_polarization_angles = [ 90, 0 ]
affected_coords_by_polarization = [ [ 2 ], [ 0, 1 ] ]

for pol_idx in range( 0, num_polarizations ):
	# adj_src_refl = fdtd_hook.addtfsf()
	# adj_src_refl = fdtd_hook.addplane()
	# adj_src_refl['plane wave type'] = 'Diffracting'
	adj_src_refl = fdtd_hook.addgaussian()
	adj_src_refl['name'] = 'adj_src_refl_' + str( pol_idx )
	adj_src_refl['polarization angle'] = source_polarization_angles[ pol_idx ]
	adj_src_refl['direction'] = 'Backward'
	adj_src_refl['x span'] = fdtd_region_size_lateral_um * 1e-6
	adj_src_refl['y'] = adjoint_reflection_position_y_um * 1e-6
	# adj_src_refl['y max'] = adjoint_reflection_position_y_um * 1e-6
	# adj_src_refl['y min'] = ( -device_to_mode_match_um - 0.5 * vertical_gap_size_um ) * 1e-6
	adj_src_refl['waist radius w0'] = 0.5 * device_size_lateral_um * 1e-6
	adj_src_refl['distance from waist'] = ( adjoint_transmission_position_y_um - adjoint_reflection_position_y_um ) * 1e-6
	adj_src_refl['wavelength start'] = lambda_min_um * 1e-6
	adj_src_refl['wavelength stop'] = lambda_max_um * 1e-6

	adjoint_sources_reflection.append( adj_src_refl )


adjoint_sources_transmission = []

for pol_idx in range( 0, num_polarizations ):
	# adj_src_trans = fdtd_hook.addtfsf()
	# adj_src_trans = fdtd_hook.addplane()
	# adj_src_trans['plane wave type'] = 'Diffracting'
	adj_src_trans = fdtd_hook.addgaussian()
	adj_src_trans['name'] = 'adj_src_trans_' + str( pol_idx )
	adj_src_trans['polarization angle'] = source_polarization_angles[ pol_idx ]
	adj_src_trans['direction'] = 'Forward'
	adj_src_trans['x span'] = fdtd_region_size_lateral_um * 1e-6
	adj_src_trans['y'] = adjoint_transmission_position_y_um * 1e-6
	# adj_src_trans['y max'] = ( device_size_verical_um + device_to_mode_match_um + 0.5 * vertical_gap_size_um ) * 1e-6
	# adj_src_trans['y min'] = adjoint_transmission_position_y_um * 1e-6
	adj_src_trans['waist radius w0'] = 0.5 * device_size_lateral_um * 1e-6
	adj_src_trans['distance from waist'] = ( adjoint_transmission_position_y_um - adjoint_reflection_position_y_um ) * 1e-6
	adj_src_trans['wavelength start'] = lambda_min_um * 1e-6
	adj_src_trans['wavelength stop'] = lambda_max_um * 1e-6

	adjoint_sources_transmission.append( adj_src_trans )


fdtd_hook.save(projects_directory_location + "/optimization")

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
design_efield_monitor['use wavelength spacing'] = 1
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


# substrate = fdtd_hook.addrect()
# substrate['name'] = 'substrate'
# substrate['x span'] = fdtd_region_size_lateral_um * 1e-6
# substrate['y min'] = designable_device_vertical_maximum_um * 1e-6
# substrate['y max'] = fdtd_region_maximum_vertical_um * 1e-6
# substrate['index'] = substrate_index
# substrate['z min'] = -0.51 * 1e-6
# substrate['z max'] = 0.51 * 1e-6


#
# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
# gradient.
#

transmission_monitor = fdtd_hook.addpower()
transmission_monitor['name'] = 'transmission_monitor'
transmission_monitor['monitor type'] = 'Linear X'
transmission_monitor['x span'] = ( adjoint_aperture_um ) * 1e-6
transmission_monitor['y'] = adjoint_transmission_position_y_um * 1e-6
transmission_monitor['override global monitor settings'] = 1
transmission_monitor['use wavelength spacing'] = 1
transmission_monitor['use source limits'] = 0
transmission_monitor['minimum wavelength'] = lambda_min_um * 1e-6
transmission_monitor['maximum wavelength'] = lambda_max_um * 1e-6
transmission_monitor['frequency points'] = num_design_frequency_points

reflection_monitor = fdtd_hook.addpower()
reflection_monitor['name'] = 'reflection_monitor'
reflection_monitor['monitor type'] = 'Linear X'
reflection_monitor['x span'] = ( adjoint_aperture_um ) * 1e-6
reflection_monitor['y'] = adjoint_reflection_position_y_um * 1e-6
reflection_monitor['override global monitor settings'] = 1
reflection_monitor['use wavelength spacing'] = 1
reflection_monitor['use source limits'] = 0
reflection_monitor['minimum wavelength'] = lambda_min_um * 1e-6
reflection_monitor['maximum wavelength'] = lambda_max_um * 1e-6
reflection_monitor['frequency points'] = num_design_frequency_points


#
# Add device and background group so that they can be rotated together.
#
device_and_backgrond_group = fdtd_hook.addstructuregroup()
device_and_backgrond_group['name'] = 'device_and_backgrond'


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
fdtd_hook.addtogroup( device_and_backgrond_group['name'] )


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
elongations_left_um = [ 1, 0 ]
elongations_right_um = [ 0, 1 ]
# device_background_side_y = [ 0, 0, -1, 1 ]
side_blocks = []

for device_background_side_idx in range( 0, 2 ):
	side_x = device_background_side_x[ device_background_side_idx ]

	side_block = fdtd_hook.addrect()

	center_x_um = side_x * extra_lateral_space_offset_um
	span_x_um = ( np.abs( side_x ) * extra_lateral_space_per_side_um +
		( 1 - np.abs( side_x ) ) * fdtd_region_size_lateral_um )
	left_x_um = center_x_um - 0.5 * span_x_um
	right_x_um = center_x_um + 0.5 * span_x_um

	left_x_um -= elongations_left_um[ device_background_side_idx ]
	right_x_um += elongations_right_um[ device_background_side_idx ]

	side_block['name'] = 'device_background_' + side_to_string( side_x )
	side_block['y min'] = designable_device_vertical_minimum_um * 1e-6
	side_block['y max'] = designable_device_vertical_maximum_um * 1e-6
	side_block['x min'] = left_x_um * 1e-6
	side_block['x max'] = right_x_um * 1e-6
	side_block['index'] = device_background_index
	fdtd_hook.addtogroup( device_and_backgrond_group['name'] )

	side_blocks.append( side_block )


gaussian_normalization = np.zeros( num_points_per_band )
middle_point = num_points_per_band / 2.
# spacing = 1. / ( num_points_per_band - 1 )
half_bandwidth = 0.4 * num_points_per_band

for wl_idx in range( 0, num_points_per_band ):
	gaussian_normalization[ wl_idx ] =  ( 1. / half_bandwidth ) * np.sqrt( 1 / ( 2 * np.pi ) ) * np.exp( -0.5 * ( wl_idx - middle_point )**2 / ( half_bandwidth**2 ) )
	
gaussian_normalization /= np.sum( gaussian_normalization )


flat_normalization = np.ones( gaussian_normalization.shape )

normalization_all = np.array( [ flat_normalization for i in range( 0, num_bands ) ] ).flatten()


reversed_field_shape = [1, designable_device_voxels_vertical, device_voxels_lateral]
reversed_field_shape_with_pol = [num_polarizations, 1, designable_device_voxels_vertical, device_voxels_lateral]

#
# todo(gdroberts): You should update the device again once you have changed optimization states and/or epochs.  This is because the gradient information
# is not correct for that first iteration in a new epoch or optimization stage because it is for a different device so it will take the first step
# incorrectly.  Steps are small so this is probably not a big deal, but still this should be fixed.  This is why all of this stuff needs to get
# put under one umbrella.  Because there are so many versions where this needs to be changed, but there is so much code re-use not getting used.
#

# Propagate forward good parents? - added
# Level set optimize best parents? Or level set optimize everything?

# You likely should verify the gradient for when you do level set optimized devices!

# num_iterations = 200
num_iterations = 100

np.random.seed( 923447 )

my_optimization_state = continuous_cmos.ContinuousCMOS(
	[ min_real_permittivity, max_real_permittivity ],
	lsf_mesh_spacing_um,
	designable_device_vertical_minimum_um,
	device_size_lateral_um,
	feature_size_um_by_profiles,
	device_layer_thicknesses_um,
	device_spacer_thicknesses_um,
	num_iterations,
	1,
	"level_set_optimize",
	device_lateral_background_density )

if should_reload:
	old_index = np.load( projects_directory_reload + "/final_device.npy" )
	old_perm = old_index**2

	old_density = ( old_perm - min_real_permittivity ) / ( max_real_permittivity - min_real_permittivity )

	my_optimization_state.init_profiles_with_density( old_density )
else:
	my_optimization_state.uniform_layer_profiles( 0.5 )

get_index = my_optimization_state.assemble_index()
device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, get_index.shape[ 0 ] )
device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, get_index.shape[ 1 ] )
device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )


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
		denominator = 8.0 * np.real( np.sum( choose_electric_mode[ 2, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 0, wl_idx, 0, 0, : ] ) ) )

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
		denominator = -8.0 * np.real( np.sum( choose_electric_mode[ 0, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 2, wl_idx, 0, 0, : ] ) ) )

		fom_by_wavelength[ wl_idx ] = ( numerator / denominator )
		if mode_overlap_norm is not None:
			fom_by_wavelength[ wl_idx ] = ( numerator / ( mode_overlap_norm[ wl_idx ] * denominator ) )
	
		fom_by_wavelength[ wl_idx ] *= normal_weighting

	return total_norm * fom_by_wavelength

def mode_overlap_gradient_ez(
	# figure_of_merit, fom_weighting,
	# fom_reflection, fom_transmission,
	weights_by_wl,
	#transmission_weights_by_wl,
	electric_fields_forward, magnetic_fields_forward,
	electric_mode_fields, magnetic_mode_fields,
	electric_fields_gradient_forward, electric_fields_gradient_adjoint,
	mode_overlap_norm ):

	normal_weighting = 1.0

	# num_wavelengths = electric_fields_forward.shape[ 1 ]

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

		gradient += weights_by_wl[ wl_idx ] * normal_weighting * ( 
			adjoint_phase *
			np.sum( electric_fields_gradient_forward[ :, wl_idx, :, :, : ] * electric_fields_gradient_adjoint[ :, wl_idx, :, :, : ], axis=0 ) )

	return -gradient
	# return -gradient / num_wavelengths

def mode_overlap_gradient_hz(
	# figure_of_merit, fom_weighting,
	# fom_reflection, fom_transmission,
	weights_by_wl,
	#transmission_weights_by_wl,
	electric_fields_forward, magnetic_fields_forward,
	electric_mode_fields, magnetic_mode_fields,
	electric_fields_gradient_forward, electric_fields_gradient_adjoint,
	mode_overlap_norm ):

	normal_weighting = 1.0

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
		denominator = -4.0 * np.real( np.sum( choose_electric_mode[ 0, wl_idx, 0, 0, : ] * np.conj( choose_magnetic_mode[ 2, wl_idx, 0, 0, : ] ) ) )

		adjoint_phase = np.conj( numerator ) / ( denominator * mode_overlap_norm[ wl_idx ] )

		gradient += weights_by_wl[ wl_idx ] * normal_weighting * ( 
			adjoint_phase *
			np.sum( electric_fields_gradient_forward[ :, wl_idx, :, :, : ] * electric_fields_gradient_adjoint[ :, wl_idx, :, :, : ], axis=0 ) )

	return -gradient
	# return -gradient / num_wavelengths


def reflection_transmission_figure_of_merit( reflection_by_wl, transmission_by_wl, reflection_weights_by_wl, transmission_weights_by_wl ):

	# assert np.all( np.greater_equal( transmission_weights_by_wl, 0.0 ) ), "Expected no negative weights!"
	# assert np.all( np.greater_equal( reflection_weights_by_wl, 0.0 ) ), "Expected no negative weights!"
	assert len( reflection_by_wl ) == len( transmission_by_wl ), "Array sizes do not match up!"
	assert len( reflection_by_wl ) == len( reflection_weights_by_wl ), "Array sizes do not match up!"

	return np.sum( reflection_by_wl * reflection_weights_by_wl ), np.sum( transmission_by_wl * transmission_weights_by_wl )

#
# todo(gdroberts): fom design on random configs to see how well this figure of merit is doing!
#
def fom( pol_idx, rotation_angle_radians, reflection_weights_by_wl, transmission_weights_by_wl ):
	rotation_angle_degrees = rotation_angle_radians * 180. / np.pi

	fdtd_hook.switchtolayout()

	device_and_backgrond_group['first axis'] = 'z'
	device_and_backgrond_group['rotation 1'] = rotation_angle_degrees

	adjust_x_span_efield_monitor = design_efield_monitor['y span'] * np.sin( rotation_angle_radians )
	adjust_y_span_efield_monitor = design_efield_monitor['x span'] * np.sin( rotation_angle_radians )

	design_efield_monitor['x span'] += adjust_x_span_efield_monitor
	design_efield_monitor['y span'] += adjust_y_span_efield_monitor

	capture_x_offset_voxels = int( np.round( 0.5 * designable_device_voxels_vertical * np.sin( rotation_angle_radians ) ) )
	capture_y_offset_voxels = int( np.round( 0.5 * device_voxels_lateral * np.sin( rotation_angle_radians ) ) )


	disable_all_sources()
	forward_sources[ pol_idx ].enabled = 1
	fdtd_hook.run()

	forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	get_E_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'E' )
	get_H_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'H' )

	get_E_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'E' )
	get_H_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'H' )

	mode_overlap_reflection = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_reflection, get_H_fwd_reflection,
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		1.0, mode_overlap_norm_reflection[ pol_idx ] )

	mode_overlap_transmission = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_transmission, get_H_fwd_transmission,
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		-1.0, mode_overlap_norm_transmission[ pol_idx ] )

	fom_reflection, fom_transmission = reflection_transmission_figure_of_merit(
		mode_overlap_reflection, mode_overlap_transmission, reflection_weights_by_wl, transmission_weights_by_wl )

	fdtd_hook.switchtolayout()
	device_and_backgrond_group['first axis'] = 'none'
	design_efield_monitor['x span'] -= adjust_x_span_efield_monitor
	design_efield_monitor['y span'] -= adjust_y_span_efield_monitor

	return fom_reflection, fom_transmission

def fom_and_gradient( pol_idx, rotation_angle_radians, reflection_weights_by_wl, transmission_weights_by_wl ):
	rotation_angle_degrees = rotation_angle_radians * 180. / np.pi

	fdtd_hook.switchtolayout()

	device_and_backgrond_group['first axis'] = 'z'
	device_and_backgrond_group['rotation 1'] = rotation_angle_degrees

	adjust_x_span_efield_monitor = design_efield_monitor['y span'] * np.sin( rotation_angle_radians )
	adjust_y_span_efield_monitor = design_efield_monitor['x span'] * np.sin( rotation_angle_radians )

	design_efield_monitor['x span'] += adjust_x_span_efield_monitor
	design_efield_monitor['y span'] += adjust_y_span_efield_monitor

	capture_x_offset_voxels = int( np.round( 0.5 * designable_device_voxels_vertical * np.sin( rotation_angle_radians ) ) )
	capture_y_offset_voxels = int( np.round( 0.5 * device_voxels_lateral * np.sin( rotation_angle_radians ) ) )


	disable_all_sources()
	forward_sources[ pol_idx ].enabled = 1
	fdtd_hook.run()

	forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	get_E_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'E' )
	get_H_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'H' )

	get_E_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'E' )
	get_H_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'H' )

	mode_overlap_reflection = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_reflection, get_H_fwd_reflection,
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		1.0, mode_overlap_norm_reflection[ pol_idx ] )

	mode_overlap_transmission = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_transmission, get_H_fwd_transmission,
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		-1.0, mode_overlap_norm_transmission[ pol_idx ] )


	fom_reflection, fom_transmission = reflection_transmission_figure_of_merit(
		mode_overlap_reflection, mode_overlap_transmission, reflection_weights_by_wl, transmission_weights_by_wl )

	disable_all_sources()
	adjoint_sources_reflection[ pol_idx ].enabled = 1
	fdtd_hook.run()

	adjoint_e_fields_reflection = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	disable_all_sources()
	adjoint_sources_transmission[ pol_idx ].enabled = 1
	fdtd_hook.run()

	adjoint_e_fields_transmission = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	def reinterpolate_fields( input_field ):
		output_field = np.zeros( input_field.shape, dtype=input_field.dtype )

		for pol_idx in range( 0, 3 ):
			for wl_idx in range( 0, num_design_frequency_points ):
				extract_component = np.squeeze( input_field[ pol_idx, wl_idx ] )
				output_field[ pol_idx, wl_idx ] = (
					ndimage.rotate( np.real( extract_component ), rotation_angle_degrees, reshape=False ) +
					1j * ndimage.rotate( np.imag( extract_component ), rotation_angle_degrees, reshape=False ) )

		rotate_output = output_field[
			:, :, :,
			capture_y_offset_voxels : ( capture_y_offset_voxels + designable_device_voxels_vertical ),
			capture_x_offset_voxels : ( capture_x_offset_voxels + device_voxels_lateral )
		]

		return rotate_output

	forward_e_fields = reinterpolate_fields( forward_e_fields )
	adjoint_e_fields_reflection = reinterpolate_fields( adjoint_e_fields_reflection )
	adjoint_e_fields_transmission = reinterpolate_fields( adjoint_e_fields_transmission )

	adj_grad_reflection = np.real( overlap_gradient_by_pol[ pol_idx ](
		reflection_weights_by_wl,
		get_E_fwd_reflection, get_H_fwd_reflection,
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		forward_e_fields, adjoint_e_fields_reflection,
		mode_overlap_norm_reflection[ pol_idx ] ) / 1j )
	adj_grad_reflection = np.swapaxes( adj_grad_reflection, 0, 2 )

	adj_grad_transmission = np.real( overlap_gradient_by_pol[ pol_idx ](
		transmission_weights_by_wl,
		get_E_fwd_transmission, get_H_fwd_transmission,
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		forward_e_fields, adjoint_e_fields_transmission,
		mode_overlap_norm_transmission[ pol_idx ] ) / 1j )
	adj_grad_transmission = np.swapaxes( adj_grad_transmission, 0, 2 )

	fdtd_hook.switchtolayout()
	device_and_backgrond_group['first axis'] = 'none'
	design_efield_monitor['x span'] -= adjust_x_span_efield_monitor
	design_efield_monitor['y span'] -= adjust_y_span_efield_monitor

	return fom_reflection, fom_transmission, adj_grad_reflection, adj_grad_transmission

def fom_and_gradient_with_rotations( pol_idx ):
	fom_no_rotation_reflection, fom_no_rotation_transmission, grad_no_rotation_reflection, grad_no_rotation_transmission = fom_and_gradient(
		pol_idx, 0.0, normal_reflection_weights, normal_transmission_weights )
	fom_rotation_reflection, fom_rotation_transmission, grad_rotation_reflection, grad_rotation_transmission = fom_and_gradient(
		pol_idx, device_rotation_angle_radians, angled_reflection_weights, angled_transmission_weights )

	# print( fom_no_rotation_reflection )
	# print( fom_rotation_reflection )
	# print( fom_no_rotation_transmission )
	# print( fom_rotation_transmission )
	# print()

	fom_total = softplus( fom_no_rotation_reflection + fom_rotation_reflection ) * softplus( fom_no_rotation_transmission + fom_rotation_transmission )

	# print( softplus( fom_no_rotation_transmission + fom_rotation_transmission ) )
	# print( softplus( fom_no_rotation_reflection + fom_rotation_reflection ) )
	# print()

	grad_reflection = (
		softplus( fom_no_rotation_transmission + fom_rotation_transmission ) * softplus_prime( fom_no_rotation_reflection + fom_rotation_reflection ) *
		( grad_no_rotation_reflection + grad_rotation_reflection )
	)
	grad_transmission = (
		softplus( fom_no_rotation_reflection + fom_rotation_reflection ) * softplus_prime( fom_no_rotation_transmission + fom_rotation_transmission ) *
		( grad_no_rotation_transmission + grad_rotation_transmission )
	)
	grad_total = grad_reflection + grad_transmission

	return fom_total, grad_total

def fom_with_rotations( pol_idx ):
	fom_no_rotation_reflection, fom_no_rotation_transmission = fom( pol_idx, 0.0, normal_reflection_weights, normal_transmission_weights )
	fom_rotation_reflection, fom_rotation_transmission = fom( pol_idx, device_rotation_angle_radians, angled_reflection_weights, angled_transmission_weights )

	fom_total = softplus( fom_no_rotation_reflection + fom_rotation_reflection ) * softplus( fom_no_rotation_transmission + fom_rotation_transmission )

	return fom_total

def compute_gradient( device_index, performance_weight_by_pol=False ):
	inflate_index = np.zeros( ( device_index.shape[ 0 ], device_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = device_index
	inflate_index[ :, :, 1 ] = device_index

	fdtd_hook.switchtolayout()
	fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
	fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

	fom_by_pol = []
	grad_by_pol = []
	for pol_idx in range( 0, num_polarizations ):
		fom, adj_grad = fom_and_gradient_with_rotations( pol_idx )

		fom_by_pol.append( fom )
		grad_by_pol.append( adj_grad )

	fom_by_pol = np.array( fom_by_pol )
	performance_weights = np.ones( len( fom_by_pol ) )
	if performance_weight_by_pol:
		performance_weights = ( 2. / len( fom_by_pol ) ) - ( fom_by_pol**2 / np.sum( fom_by_pol**2 ) )
		performance_weights = np.maximum( performance_weights, 0.0 )
		performance_weights /= np.sum( performance_weights )

	fom_total = 0.0
	grad_total = np.zeros( grad_by_pol[ 0 ].shape )

	for pol_idx in range( 0, num_polarizations ):
		fom_total += pol_weights[ pol_idx ] * fom_by_pol[ pol_idx ]
		grad_total += performance_weights[ pol_idx ] * pol_weights[ pol_idx ] * grad_by_pol[ pol_idx ]

	return fom_total, grad_total

def check_gradient_full( pol_idx ):
	from scipy.ndimage import gaussian_filter

	fd_device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral )
	fd_device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, designable_device_voxels_vertical )
	fd_device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )

	get_index = my_optimization_state.assemble_index( 0 )
	fd_permittivity = 1.0 + 0.5 * np.ones( ( device_voxels_lateral, designable_device_voxels_vertical ) )
	fd_permittivity = 1.2 + 0.5 * np.random.random( ( device_voxels_lateral, designable_device_voxels_vertical ) )
	fd_density = np.random.random( ( device_voxels_lateral, designable_device_voxels_vertical ) )

	fd_density = gaussian_filter(fd_density, sigma=2)
	fd_density -= np.min( fd_density )
	fd_density /= np.max( fd_density )
	fd_permittivity = 1.2 + 0.5 * fd_density

	fd_index = np.sqrt( fd_permittivity )

	inflate_index = np.zeros( ( fd_index.shape[ 0 ], fd_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = fd_index
	inflate_index[ :, :, 1 ] = fd_index

	fdtd_hook.switchtolayout()
	fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
	fdtd_hook.importnk2( inflate_index, fd_device_region_x, fd_device_region_y, fd_device_region_z )

	fom0, adj_grad = fom_and_gradient_with_rotations( pol_idx )

	print( "Before grad bump:" )
	print( fom0 )

	num_fd = 20
	fd_x = int( 0.35 * fd_index.shape[ 0 ] )
	fd_y_start = int( 0.5 * fd_index.shape[ 1 ] )
	fd_y_end = fd_y_start + num_fd

	finite_diff_grad = []

	h = 1e-3

	for fd_idx in range( 0, num_fd ):
		perm_copy = fd_permittivity.copy()
		perm_copy[ fd_x, fd_y_start + fd_idx ] += h

		fd_index = np.sqrt( perm_copy )

		inflate_index = np.zeros( ( fd_index.shape[ 0 ], fd_index.shape[ 1 ], 2 ), dtype=np.complex )
		inflate_index[ :, :, 0 ] = fd_index
		inflate_index[ :, :, 1 ] = fd_index

		fdtd_hook.switchtolayout()
		fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
		fdtd_hook.importnk2( inflate_index, fd_device_region_x, fd_device_region_y, fd_device_region_z )

		fom_fd = fom_with_rotations( pol_idx )
		finite_diff_grad.append( ( fom_fd - fom0 ) / h )


	pull_adj_grad = adj_grad[ fd_x, fd_y_start : fd_y_end ]

	select_fd = []

	for fd_idx in range( 0, len( finite_diff_grad ) ):
		select_fd.append( finite_diff_grad[ fd_idx ] )

	select_fd = np.array( select_fd )

	plt.plot( select_fd / np.max( np.abs( select_fd ) ), color='r', linewidth=2 )
	plt.plot( pull_adj_grad / np.max( np.abs( pull_adj_grad ) ), color='g', linewidth=2, linestyle='--' )
	plt.show()

	perm_copy = fd_permittivity.copy()
	perm_copy += 0.05 * np.squeeze( adj_grad ) / np.max( np.abs( adj_grad ) )

	fd_index = np.sqrt( perm_copy )	

	inflate_index = np.zeros( ( fd_index.shape[ 0 ], fd_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = fd_index
	inflate_index[ :, :, 1 ] = fd_index

	fdtd_hook.switchtolayout()
	fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
	fdtd_hook.importnk2( inflate_index, fd_device_region_x, fd_device_region_y, fd_device_region_z )

	fom1 = fom_with_rotations( pol_idx )

	print( "After grad bump:" )
	print( fom1 )



def check_gradient( pol_idx ):

	rotation_angle_degrees = rotation_angle_radians * 180. / np.pi

	from scipy.ndimage import gaussian_filter

	fd_device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral )
	fd_device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, designable_device_voxels_vertical )
	fd_device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )

	get_index = my_optimization_state.assemble_index( 0 )
	fd_permittivity = 1.0 + 0.5 * np.ones( ( device_voxels_lateral, designable_device_voxels_vertical ) )
	fd_permittivity = 1.2 + 0.5 * np.random.random( ( device_voxels_lateral, designable_device_voxels_vertical ) )
	fd_density = np.random.random( ( device_voxels_lateral, designable_device_voxels_vertical ) )

	fd_density = gaussian_filter(fd_density, sigma=2)
	fd_density -= np.min( fd_density )
	fd_density /= np.max( fd_density )
	fd_permittivity = 1.2 + 0.5 * fd_density

	print( fd_permittivity.shape )
	fd_index = np.sqrt( fd_permittivity )

	inflate_index = np.zeros( ( fd_index.shape[ 0 ], fd_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = fd_index
	inflate_index[ :, :, 1 ] = fd_index


	fdtd_hook.switchtolayout()
	fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
	fdtd_hook.importnk2( inflate_index, fd_device_region_x, fd_device_region_y, fd_device_region_z )

	device_and_backgrond_group['first axis'] = 'z'
	device_and_backgrond_group['rotation 1'] = rotation_angle_degrees

	adjust_x_span_efield_monitor = design_efield_monitor['y span'] * np.sin( rotation_angle_radians )
	adjust_y_span_efield_monitor = design_efield_monitor['x span'] * np.sin( rotation_angle_radians )

	design_efield_monitor['x span'] += adjust_x_span_efield_monitor
	design_efield_monitor['y span'] += adjust_y_span_efield_monitor

	capture_x_offset_voxels = int( np.round( 0.5 * designable_device_voxels_vertical * np.sin( rotation_angle_radians ) ) )
	capture_y_offset_voxels = int( np.round( 0.5 * device_voxels_lateral * np.sin( rotation_angle_radians ) ) )


	disable_all_sources()
	forward_sources[ pol_idx ].enabled = 1
	fdtd_hook.run()

	forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	get_E_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'E' )
	get_H_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'H' )

	get_E_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'E' )
	get_H_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'H' )

	mode_overlap_reflection = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_reflection, get_H_fwd_reflection,
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		1.0, mode_overlap_norm_reflection[ pol_idx ] )

	mode_overlap_transmission = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_transmission, get_H_fwd_transmission,
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		-1.0, mode_overlap_norm_transmission[ pol_idx ] )


	disable_all_sources()
	adjoint_sources_reflection[ pol_idx ].enabled = 1
	fdtd_hook.run()

	adjoint_e_fields_reflection = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	disable_all_sources()
	adjoint_sources_transmission[ pol_idx ].enabled = 1
	fdtd_hook.run()

	adjoint_e_fields_transmission = get_complex_monitor_data(design_efield_monitor['name'], 'E')

	wl_idx = int( 0.5 * num_design_frequency_points )
	weighting_select_wl = np.zeros( num_design_frequency_points )
	weighting_select_wl[ wl_idx ] = 1

	print( "Before grad bump:" )
	print( mode_overlap_reflection[ wl_idx ] )
	print( mode_overlap_transmission[ wl_idx ] )

	def reinterpolate_fields( input_field ):
		output_field = np.zeros( input_field.shape, dtype=input_field.dtype )

		for pol_idx in range( 0, 3 ):
			for wl_idx in range( 0, num_design_frequency_points ):
				extract_component = np.squeeze( input_field[ pol_idx, wl_idx ] )
				output_field[ pol_idx, wl_idx ] = (
					ndimage.rotate( np.real( extract_component ), rotation_angle_degrees, reshape=False ) +
					1j * ndimage.rotate( np.imag( extract_component ), rotation_angle_degrees, reshape=False ) )

		rotate_output = output_field[
			:, :, :,
			capture_y_offset_voxels : ( capture_y_offset_voxels + designable_device_voxels_vertical ),
			capture_x_offset_voxels : ( capture_x_offset_voxels + device_voxels_lateral )
		]

		return rotate_output

	print( forward_e_fields.shape )

	forward_e_fields = reinterpolate_fields( forward_e_fields )
	adjoint_e_fields_reflection = reinterpolate_fields( adjoint_e_fields_reflection )
	adjoint_e_fields_transmission = reinterpolate_fields( adjoint_e_fields_transmission )

	print( forward_e_fields.shape )

	adj_grad_reflection = np.real( overlap_gradient_by_pol[ pol_idx ](
		mode_overlap_reflection, weighting_select_wl,
		get_E_fwd_reflection, get_H_fwd_reflection,
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		forward_e_fields, adjoint_e_fields_reflection,
		mode_overlap_norm_reflection[ pol_idx ] ) / 1j )
	adj_grad_reflection = np.swapaxes( adj_grad_reflection, 0, 2 )

	print( adj_grad_reflection.shape )

	adj_grad_transmission = np.real( overlap_gradient_by_pol[ pol_idx ](
		mode_overlap_transmission, weighting_select_wl,
		get_E_fwd_transmission, get_H_fwd_transmission,
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		forward_e_fields, adjoint_e_fields_transmission,
		mode_overlap_norm_transmission[ pol_idx ] ) / 1j )
	adj_grad_transmission = np.swapaxes( adj_grad_transmission, 0, 2 )


	num_fd = 20
	fd_x = int( 0.35 * fd_index.shape[ 0 ] )
	fd_y_start = int( 0.5 * fd_index.shape[ 1 ] )
	fd_y_end = fd_y_start + num_fd

	fd_reflection = []
	fd_transmission = []

	h = 1e-3

	for fd_idx in range( 0, num_fd ):
		perm_copy = fd_permittivity.copy()
		perm_copy[ fd_x, fd_y_start + fd_idx ] += h

		fd_index = np.sqrt( perm_copy )

		inflate_index = np.zeros( ( fd_index.shape[ 0 ], fd_index.shape[ 1 ], 2 ), dtype=np.complex )
		inflate_index[ :, :, 0 ] = fd_index
		inflate_index[ :, :, 1 ] = fd_index

		fdtd_hook.switchtolayout()
		fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
		fdtd_hook.importnk2( inflate_index, fd_device_region_x, fd_device_region_y, fd_device_region_z )


		disable_all_sources()
		forward_sources[ pol_idx ].enabled = 1
		fdtd_hook.run()


		get_E_fwd_reflection_fd = get_complex_monitor_data( reflection_monitor['name'], 'E' )
		get_H_fwd_reflection_fd = get_complex_monitor_data( reflection_monitor['name'], 'H' )

		get_E_fwd_transmission_fd = get_complex_monitor_data( transmission_monitor['name'], 'E' )
		get_H_fwd_transmission_fd = get_complex_monitor_data( transmission_monitor['name'], 'H' )

		mode_overlap_reflection_fd = overlap_fom_by_pol[ pol_idx ](
			get_E_fwd_reflection_fd, get_H_fwd_reflection_fd,
			mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
			1.0, mode_overlap_norm_reflection[ pol_idx ] )

		mode_overlap_transmission_fd = overlap_fom_by_pol[ pol_idx ](
			get_E_fwd_transmission_fd, get_H_fwd_transmission_fd,
			mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
			-1.0, mode_overlap_norm_transmission[ pol_idx ] )

		fd_reflection.append( ( mode_overlap_reflection_fd - mode_overlap_reflection ) / h )
		fd_transmission.append( ( mode_overlap_transmission_fd - mode_overlap_transmission ) / h )

	pull_grad_reflection = adj_grad_reflection[ fd_x, fd_y_start : fd_y_end ]
	pull_grad_transmission = adj_grad_transmission[ fd_x, fd_y_start : fd_y_end ]

	select_fd_reflection = []
	select_fd_transmission = []

	for fd_idx in range( 0, len( fd_reflection ) ):
		select_fd_reflection.append( fd_reflection[ fd_idx ][ wl_idx ] )
	for fd_idx in range( 0, len( fd_transmission ) ):
		select_fd_transmission.append( fd_transmission[ fd_idx ][ wl_idx ] )

	select_fd_reflection = np.array( select_fd_reflection )
	select_fd_transmission = np.array( select_fd_transmission )

	print( select_fd_reflection )
	print( select_fd_transmission )


	plt.subplot( 1, 2, 1 )
	plt.plot( select_fd_reflection / np.max( np.abs( select_fd_reflection ) ), color='r', linewidth=2 )
	plt.plot( pull_grad_reflection / np.max( np.abs( pull_grad_reflection ) ), color='g', linewidth=2, linestyle='--' )
	plt.subplot( 1, 2, 2 )
	plt.plot( select_fd_transmission / np.max( np.abs( select_fd_transmission ) ), color='r', linewidth=2 )
	plt.plot( pull_grad_transmission / np.max( np.abs( pull_grad_transmission ) ), color='g', linewidth=2, linestyle='--' )
	plt.show()

	perm_copy = fd_permittivity.copy()
	perm_copy += 0.05 * np.squeeze( adj_grad_reflection ) / np.max( np.abs( adj_grad_reflection ) )

	fd_index = np.sqrt( perm_copy )	

	inflate_index = np.zeros( ( fd_index.shape[ 0 ], fd_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = fd_index
	inflate_index[ :, :, 1 ] = fd_index

	fdtd_hook.switchtolayout()
	fdtd_hook.select( device_and_backgrond_group['name'] + "::device_import" )
	fdtd_hook.importnk2( inflate_index, fd_device_region_x, fd_device_region_y, fd_device_region_z )

	fdtd_hook.run()

	get_E_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'E' )
	get_H_fwd_reflection = get_complex_monitor_data( reflection_monitor['name'], 'H' )

	get_E_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'E' )
	get_H_fwd_transmission = get_complex_monitor_data( transmission_monitor['name'], 'H' )

	mode_overlap_reflection = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_reflection, get_H_fwd_reflection,
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		1.0, mode_overlap_norm_reflection[ pol_idx ] )

	mode_overlap_transmission = overlap_fom_by_pol[ pol_idx ](
		get_E_fwd_transmission, get_H_fwd_transmission,
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		-1.0, mode_overlap_norm_transmission[ pol_idx ] )

	print( "After grad bump:" )
	print( mode_overlap_reflection[ wl_idx ] )
	print( mode_overlap_transmission[ wl_idx ] )

	disable_all_sources()
	device_and_backgrond_group['first axis'] = 'none'
	
	design_efield_monitor['x span'] -= adjust_x_span_efield_monitor
	design_efield_monitor['y span'] -= adjust_y_span_efield_monitor


#
# Need to compare to just Hz, big features, level set on 5nm grid
# Start from continuous.
# Maybe that can also give you a good see to then run level set and particle swarm from.
#

def optimize_parent_locally( parent_object, num_iterations ):
	fom_track = []

	field_shape_with_devices = [ parent_object.num_devices ]
	field_shape_with_devices.extend( np.flip( reversed_field_shape ) )
	gradients_real = np.zeros( field_shape_with_devices )
	gradients_imag = np.zeros( field_shape_with_devices )

	gradients_real_lsf = np.zeros( field_shape_with_devices )
	gradients_imag_lsf = np.zeros( field_shape_with_devices )

	for iteration in range( 0, num_iterations ):
		cur_index = parent_object.assemble_index( 0 )

		fom, gradient = compute_gradient( cur_index )

		print( fom )

		fom_track.append( fom )

		#
		# Step 4: Step the design variable.
		#
		device_gradient_real = 2 * gradient
		device_gradient_imag = 0 * gradient

		device_gradient_real_lsf = device_gradient_real
		device_gradient_imag_lsf = device_gradient_imag

		gradients_real[ 0, : ] = device_gradient_real
		gradients_imag[ 0, : ] = device_gradient_imag
		gradients_real_lsf[ 0, : ] = device_gradient_real_lsf
		gradients_imag_lsf[ 0, : ] = device_gradient_imag_lsf

		parent_object.submit_figure_of_merit( fom, iteration, 0 )
		parent_object.update( -gradients_real, -gradients_imag, -gradients_real_lsf, -gradients_imag_lsf, 0, iteration )

		if ( iteration % 10 ) == 0:
			np.save( projects_directory_location + '/device_' + str( int( iteration / 10 ) ) + '.npy', parent_object.assemble_index() )

	return parent_object, fom_track


fdtd_hook.select(device_and_backgrond_group['name'])
fdtd_hook.set('enabled', 0)

mode_E_reflection = []
mode_H_reflection = []

mode_E_transmission = []
mode_H_transmission = []

import matplotlib.pyplot as plt

for pol_idx in range( 0, num_polarizations ):
	disable_all_sources()
	adjoint_sources_transmission[ pol_idx ].enabled = 1
	fdtd_hook.run()

	get_E_mode = get_complex_monitor_data( reflection_monitor['name'], 'E' )
	get_H_mode = get_complex_monitor_data( reflection_monitor['name'], 'H' )

	num_wls = get_E_mode.shape[ 1 ]
	half_x = int( get_E_mode.shape[ 4 ] / 2. )

	for wl_idx in range( 0, num_wls ):
		phase_correction = np.exp( 1j * np.angle( get_E_mode[ 2, wl_idx, 0, 0, half_x ] ) )

		if pol_idx == 1:
			phase_correction = np.exp( 1j * np.angle( get_E_mode[ 0, wl_idx, 0, 0, half_x ] ) )

		get_E_mode[ :, wl_idx, :, :, : ] /= phase_correction
		get_H_mode[ :, wl_idx, :, :, : ] /= phase_correction

	mode_E_reflection.append( get_E_mode )
	mode_H_reflection.append( get_H_mode )

	disable_all_sources()
	adjoint_sources_reflection[ pol_idx ].enabled = 1
	fdtd_hook.run()

	get_E_mode = get_complex_monitor_data( transmission_monitor['name'], 'E' )
	get_H_mode = get_complex_monitor_data( transmission_monitor['name'], 'H' )

	num_wls = get_E_mode.shape[ 1 ]
	half_x = int( get_E_mode.shape[ 4 ] / 2. )

	for wl_idx in range( 0, num_wls ):
		phase_correction = np.exp( 1j * np.angle( get_E_mode[ 2, wl_idx, 0, 0, half_x ] ) )

		if pol_idx == 1:
			phase_correction = np.exp( 1j * np.angle( get_E_mode[ 0, wl_idx, 0, 0, half_x ] ) )

		get_E_mode[ :, wl_idx, :, :, : ] /= phase_correction
		get_H_mode[ :, wl_idx, :, :, : ] /= phase_correction

	mode_E_transmission.append( get_E_mode )
	mode_H_transmission.append( get_H_mode )


mode_overlap_norm_transmission = []
mode_overlap_norm_reflection = []

overlap_fom_by_pol = [ mode_overlap_fom_ez, mode_overlap_fom_hz ]
overlap_gradient_by_pol = [ mode_overlap_gradient_ez, mode_overlap_gradient_hz ]

for pol_idx in range( 0, num_polarizations ):
	mode_overlap_norm_transmission.append( overlap_fom_by_pol[ pol_idx ](
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		mode_E_transmission[ pol_idx ], mode_H_transmission[ pol_idx ],
		-1.0 )
	)

	mode_overlap_norm_reflection.append( overlap_fom_by_pol[ pol_idx ](
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		mode_E_reflection[ pol_idx ], mode_H_reflection[ pol_idx ],
		1.0 )
	)


print( mode_overlap_norm_transmission )
print( mode_overlap_norm_reflection )

fdtd_hook.switchtolayout()

for adj_src_trans in adjoint_sources_transmission:
	adj_src_trans['distance from waist'] = 0 * 1e-6

for adj_src_refl in adjoint_sources_reflection:
	adj_src_refl['distance from waist'] = 0 * 1e-6


fdtd_hook.select(device_and_backgrond_group['name'])
fdtd_hook.set('enabled', 1)

# check_gradient_full( 1 )

# load_index = np.load('/Users/gregory/Downloads/device_9_contrast_3p6_red_v2.npy')
# bin_index = 1.0 + 0.46 * np.greater_equal( load_index, 1.25 )
# compute_gradient( load_index )

# fdtd_hook.run()

# sys.exit( 0 )

figure_of_merit_evolution = []

for epoch_idx in range( 0, 1 ):

	my_optimization_state, local_fom = optimize_parent_locally( my_optimization_state, num_iterations )

	np.save( projects_directory_location + '/final_device.npy', my_optimization_state.assemble_index() )
	np.save( projects_directory_location + '/figure_of_merit.npy', local_fom )
