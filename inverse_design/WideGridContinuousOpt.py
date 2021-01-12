import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LevelSetLocalOptimize2DParameters import *

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


def softplus( x_in ):
	return np.log( 1 + np.exp( x_in ) )

def softplus_prime( x_in ):
	return ( 1. / ( 1 + np.exp( -x_in ) ) )


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

#
# todo: should do a gradient check especially once the figure of merit is settled!
#
should_reload = False#True
# projects_directory_reload = projects_directory_location + "/" + project_name + "_continuous_Hz_sio2_no_constrast_v2"
projects_directory_location += "/" + project_name + "_continuous_Hz_sio2_no_contrast_v3"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LevelSetLocalOptimize2DParameters.py", projects_directory_location + "/LevelSetLocalOptimize2DParameters.py")
shutil.copy2(python_src_directory + "/WideGridContinuousOpt.py", projects_directory_location + "/WideGridContinuousOpt.py")
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
	forward_src = fdtd_hook.addtfsf()
	forward_src['name'] = 'forward_src_' + str( pol_idx )
	forward_src['polarization angle'] = source_polarization_angles[ pol_idx ]
	forward_src['direction'] = 'Backward'
	forward_src['x span'] = lateral_aperture_um * 1e-6
	forward_src['y max'] = src_maximum_vertical_um * 1e-6
	forward_src['y min'] = src_minimum_vertical_um * 1e-6
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

	for coord_idx in range( 0, 3 ):
		for adj_src_idx in range(0, num_adjoint_sources):
			(adjoint_sources[ coord_idx ][ adj_src_idx ]).enabled = 0


#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = [ [] for i in range( 0, 3 ) ]
coord_to_phi = [ 0, 90, 0 ]
coord_to_theta = [ 90, 90, 0 ]

for coord_idx in range( 0, 3 ):
	for adj_src_idx in range(0, num_adjoint_sources):
		adj_src = fdtd_hook.adddipole()
		adj_src['name'] = 'adj_src_' + str(adj_src_idx) + "_" + str( coord_idx )
		adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
		adj_src['y'] = adjoint_vertical_um * 1e-6
		adj_src['theta'] = coord_to_theta[ coord_idx ]
		adj_src['phi'] = coord_to_phi[ coord_idx ]
		adj_src['wavelength start'] = lambda_min_um * 1e-6
		adj_src['wavelength stop'] = lambda_max_um * 1e-6

		adjoint_sources[ coord_idx ].append( adj_src )


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


substrate = fdtd_hook.addrect()
substrate['name'] = 'substrate'
substrate['x span'] = fdtd_region_size_lateral_um * 1e-6
substrate['y min'] = designable_device_vertical_maximum_um * 1e-6
substrate['y max'] = fdtd_region_maximum_vertical_um * 1e-6
substrate['index'] = substrate_index
substrate['z min'] = -0.51 * 1e-6
substrate['z max'] = 0.51 * 1e-6


#
# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
# gradient.
#
focal_monitors = []

for adj_src in range(0, num_adjoint_sources):
	focal_monitor = fdtd_hook.addpower()
	focal_monitor['name'] = 'focal_monitor' + str(adj_src)
	focal_monitor['monitor type'] = 'point'
	focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	focal_monitor['y'] = adjoint_vertical_um * 1e-6
	focal_monitor['override global monitor settings'] = 1
	focal_monitor['use wavelength spacing'] = 1
	focal_monitor['use source limits'] = 0
	focal_monitor['minimum wavelength'] = lambda_min_um * 1e-6
	focal_monitor['maximum wavelength'] = lambda_max_um * 1e-6
	focal_monitor['frequency points'] = num_design_frequency_points

	focal_monitors.append(focal_monitor)


for adj_src in range(0, num_adjoint_sources):
	transmission_monitor = fdtd_hook.addpower()
	transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
	transmission_monitor['monitor type'] = 'Linear X'
	transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
	transmission_monitor['x span'] = ( 1.0 / num_focal_spots ) * device_size_lateral_um * 1e-6
	transmission_monitor['y'] = adjoint_vertical_um * 1e-6
	transmission_monitor['override global monitor settings'] = 1
	transmission_monitor['use wavelength spacing'] = 1
	transmission_monitor['use source limits'] = 1
	transmission_monitor['frequency points'] = num_eval_frequency_points

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

# num_iterations = 150#100
num_iterations = 1000

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

get_index = my_optimization_state.assemble_index(0)
device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, get_index.shape[ 0 ] )
device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, get_index.shape[ 1 ] )
device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )


def evaluate_device( index_profile ):
	fdtd_hook.switchtolayout()
	get_index = index_profile
	inflate_index = np.zeros( ( get_index.shape[ 0 ], get_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = get_index
	inflate_index[ :, :, 1 ] = get_index

	fdtd_hook.select( device_import[ 'name' ] )
	fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

	figure_of_merit_by_pol = np.zeros( num_polarizations )

	figure_of_merit = 0
	for pol_idx in range( 0, num_polarizations ):

		affected_coords = affected_coords_by_polarization[ pol_idx ]

		#
		# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
		#
		disable_all_sources()
		forward_sources[ pol_idx ].enabled = 1
		fdtd_hook.run()

		forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

		focal_data = []
		for adj_src_idx in range(0, num_adjoint_sources):
			focal_data.append(
				get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )

		#
		# Step 2: Compute the figure of merit
		#
		normalized_intensity_focal_point_wavelength = np.zeros( ( num_focal_spots, num_design_frequency_points ) )
		conjugate_weighting_focal_point_wavelength = np.zeros( ( 3, num_focal_spots, num_design_frequency_points ), dtype=np.complex )

		figure_of_merit_total = np.zeros( num_design_frequency_points )
		conjugate_weighting_wavelength = np.zeros( ( num_focal_spots, 3, num_design_frequency_points ), dtype=np.complex )
			
		figure_of_merit_by_band = np.zeros( num_focal_spots )

		correct_focal_by_wl = np.zeros( num_design_frequency_points )
		incorrect_focal_by_wl = np.zeros( num_design_frequency_points )

		for focal_idx in range(0, num_focal_spots):
			spectral_indices = spectral_focal_plane_map[ focal_idx ]
			num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]

			for wl_idx in range( 0, num_design_frequency_points ):
				weighting = 0
				if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
					weighting = 1.0

				for coord_idx in range( 0, len( affected_coords ) ):
					current_coord = affected_coords[ coord_idx ]

					normalize_intensity_k_k = np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] )**2 ) / max_intensity_by_wavelength[ wl_idx ]

					correct_focal_by_wl[ wl_idx ] += band_weights[ focal_idx ] * weighting * normalize_intensity_k_k

					for other_focal_idx in range( 0, num_focal_spots ):
						if other_focal_idx == focal_idx:
							continue

						normalize_intensity_k_j = np.sum( np.abs( focal_data[ other_focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] )**2 ) / max_intensity_by_wavelength[ wl_idx ]

						incorrect_focal_by_wl[ wl_idx ] += band_weights[ focal_idx ] * weighting * normalize_intensity_k_j

					conjugate_weighting_wavelength[ focal_idx, current_coord, wl_idx ] = np.conj(
						focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] / max_intensity_by_wavelength[ wl_idx ] )


		for focal_idx in range(0, num_focal_spots):
			spectral_indices = spectral_focal_plane_map[ focal_idx ]

			for wl_idx in range( 0, num_design_frequency_points ):
				if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
					if band_weights[ focal_idx ] > 0:
						figure_of_merit_by_band[ focal_idx ] += ( ( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] ) - incorrect_focal_by_wl[ wl_idx ] )
						# figure_of_merit_by_band[ focal_idx ] += ( incorrect_focal_by_wl[ wl_idx ] - ( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] ) )

		reselect_fom_by_band = []
		for idx in range( 0, len( figure_of_merit_by_band ) ):
			if band_weights[ idx ] > 0:
				reselect_fom_by_band.append( softplus( figure_of_merit_by_band[ idx ] ) )

		reselect_fom_by_band = np.array( reselect_fom_by_band )
		figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.product( reselect_fom_by_band )
		figure_of_merit += ( 1. / np.sum( pol_weights ) ) * figure_of_merit_by_pol[ pol_idx ]
		
	return figure_of_merit


#
# Need to compare to just Hz, big features, level set on 5nm grid
# Start from continuous.
# Maybe that can also give you a good see to then run level set and particle swarm from.
#

def optimize_parent_locally( parent_object, num_iterations, binarization_terminate ):
	fom_track = []
	binarization_track = []
	for iteration in range( 0, num_iterations ):

		figure_of_merit_by_device = np.zeros( parent_object.num_devices )

		field_shape_with_devices = [ parent_object.num_devices ]
		field_shape_with_devices.extend( np.flip( reversed_field_shape ) )
		gradients_real = np.zeros( field_shape_with_devices )
		gradients_imag = np.zeros( field_shape_with_devices )

		gradients_real_lsf = np.zeros( field_shape_with_devices )
		gradients_imag_lsf = np.zeros( field_shape_with_devices )

		for device in range( 0, parent_object.num_devices ):
			#
			# Start here tomorrow!  Need to do this operation for every device.  Really, the single device operation should
			# be able to fold into here!  You need to get all these code changes under one umbrella.  Including the binarization
			# in the 3D code should be part of a single library.  And add all the postprocessing and evaluation code under
			# the same python library.  Can do things like angled evaluations, ...
			#

			fdtd_hook.switchtolayout()
			get_index = parent_object.assemble_index( iteration )
			inflate_index = np.zeros( ( get_index.shape[ 0 ], get_index.shape[ 1 ], 2 ), dtype=np.complex )
			inflate_index[ :, :, 0 ] = get_index
			inflate_index[ :, :, 1 ] = get_index

			fdtd_hook.select( device_import[ 'name' ] )
			fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

			xy_polarized_gradients_by_pol = np.zeros(reversed_field_shape_with_pol, dtype=np.complex)
			xy_polarized_gradients = np.zeros(reversed_field_shape, dtype=np.complex)
			xy_polarized_gradients_by_pol_lsf = np.zeros(reversed_field_shape_with_pol, dtype=np.complex)
			xy_polarized_gradients_lsf = np.zeros(reversed_field_shape, dtype=np.complex)
			figure_of_merit_by_pol = np.zeros( num_polarizations )

			figure_of_merit = 0
			for pol_idx in range( 0, num_polarizations ):

				affected_coords = affected_coords_by_polarization[ pol_idx ]

				#
				# Step 1: Run the forward optimization for both x- and y-polarized plane waves.
				#
				disable_all_sources()
				forward_sources[ pol_idx ].enabled = 1
				fdtd_hook.run()

				forward_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

				focal_data = []
				for adj_src_idx in range(0, num_adjoint_sources):
					focal_data.append(
						get_complex_monitor_data(focal_monitors[adj_src_idx]['name'], 'E') )

				#
				# Step 2: Compute the figure of merit
				#
				normalized_intensity_focal_point_wavelength = np.zeros( ( num_focal_spots, num_design_frequency_points ) )
				conjugate_weighting_focal_point_wavelength = np.zeros( ( 3, num_focal_spots, num_design_frequency_points ), dtype=np.complex )

				figure_of_merit_total = np.zeros( num_design_frequency_points )
				conjugate_weighting_wavelength = np.zeros( ( num_focal_spots, 3, num_design_frequency_points ), dtype=np.complex )
					
				figure_of_merit_by_band = np.zeros( num_focal_spots )


				correct_focal_by_wl = np.zeros( num_design_frequency_points )
				incorrect_focal_by_wl = np.zeros( num_design_frequency_points )

				for focal_idx in range(0, num_focal_spots):
					spectral_indices = spectral_focal_plane_map[ focal_idx ]
					num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]

					for wl_idx in range( 0, num_design_frequency_points ):
						weighting = 0
						if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
							weighting = 1.0#band_weights[ focal_idx ]

						for coord_idx in range( 0, len( affected_coords ) ):
							current_coord = affected_coords[ coord_idx ]

							normalize_intensity_k_k = np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] )**2 ) / max_intensity_by_wavelength[ wl_idx ]

							correct_focal_by_wl[ wl_idx ] += band_weights[ focal_idx ] * weighting * normalize_intensity_k_k

							for other_focal_idx in range( 0, num_focal_spots ):
								if other_focal_idx == focal_idx:
									continue

								normalize_intensity_k_j = np.sum( np.abs( focal_data[ other_focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] )**2 ) / max_intensity_by_wavelength[ wl_idx ]

								incorrect_focal_by_wl[ wl_idx ] += band_weights[ focal_idx ] * weighting * normalize_intensity_k_j

							conjugate_weighting_wavelength[ focal_idx, current_coord, wl_idx ] = np.conj(
								focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] / max_intensity_by_wavelength[ wl_idx ] )


				for focal_idx in range(0, num_focal_spots):
					spectral_indices = spectral_focal_plane_map[ focal_idx ]

					for wl_idx in range( 0, num_design_frequency_points ):
						if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
							if band_weights[ focal_idx ] > 0:
								# figure_of_merit_by_band[ focal_idx ] += ( np.log( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] / ( fom_quotient_regularization + incorrect_focal_by_wl[ wl_idx ] ) ) )
								# figure_of_merit_by_band[ focal_idx ] += ( ( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] ) - incorrect_focal_by_wl[ wl_idx ] )
								# figure_of_merit_by_band[ focal_idx ] += ( incorrect_focal_by_wl[ wl_idx ] - ( ( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] ) ) )

								figure_of_merit_by_band[ focal_idx ] += correct_focal_by_wl[ wl_idx ]


				# wl_weighting_denominator = num_points_per_band * np.sum( band_weights )

				# figure_of_merit_by_band = np.maximum( figure_of_merit_by_band, 0 )

				reselect_fom_by_band = []
				reselect_fom_by_band_orig = []
				for idx in range( 0, len( figure_of_merit_by_band ) ):
					# reselect_fom_by_band_orig.append( softplus( figure_of_merit_by_band[ idx ] ) )
					reselect_fom_by_band_orig.append( figure_of_merit_by_band[ idx ] )
					if band_weights[ idx ] > 0:
						# reselect_fom_by_band.append( softplus( figure_of_merit_by_band[ idx ] ) )
						reselect_fom_by_band.append( figure_of_merit_by_band[ idx ] )
				reselect_fom_by_band = np.array( reselect_fom_by_band )
				print( reselect_fom_by_band )

				# todo: make sure this figure of merit weighting makes sense the way it is done across wavelengths and focal points
				# figure_of_merit_total = np.maximum( figure_of_merit_total, 0 )#np.min( figure_of_merit_total )
				# fom_weighting = ( 2. / len( figure_of_merit_total ) ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				# fom_weighting = ( 2. / wl_weighting_denominator ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				# fom_weighting = np.maximum( fom_weighting, 0 )

				# fom_weighting /= np.sum( fom_weighting )
				# figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.sum( normalization_all * figure_of_merit_total )
				figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.product( reselect_fom_by_band )
				# figure_of_merit += ( 1. / num_polarizations ) * figure_of_merit_by_pol[ pol_idx ]
				figure_of_merit += ( 1. / np.sum( pol_weights ) ) * figure_of_merit_by_pol[ pol_idx ]
				figure_of_merit_by_device[ device ] = figure_of_merit

				print( figure_of_merit_by_device[ device ] )
				print()

				log_file = open(projects_directory_location + "/log.txt", 'a')
				log_file.write("Figure of merit by device for iteration " + str( iteration ) + ": " + str( figure_of_merit_by_device[ device ] ) + "\n")
				log_file.close()

				#
				# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
				# gradients for x- and y-polarized forward sources.
				#
				polarized_gradient = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)
				polarized_gradient_lsf = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)

				current_index = np.real( get_non_struct_data( design_index_monitor[ 'name' ], 'index_x' ) )
				current_permittivity = np.sqrt( np.squeeze( current_index ) )

				adjoint_e_fields_by_coord_and_focal = {}
				for coord_idx in range( 0, len( affected_coords ) ):
					current_coord = affected_coords[ coord_idx ]

					adjoint_e_fields_by_coord_and_focal[ current_coord ] = {}

				for coord_idx in range( 0, len( affected_coords ) ):
					current_coord = affected_coords[ coord_idx ]
					for adj_src_idx in range( 0, num_adjoint_sources ):
						disable_all_sources()
						(adjoint_sources[current_coord][adj_src_idx]).enabled = 1
						fdtd_hook.run()

						adjoint_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

						adjoint_e_fields_by_coord_and_focal[ current_coord ][ adj_src_idx ] = adjoint_e_fields.copy()


				for coord_idx in range( 0, len( affected_coords ) ):
					current_coord = affected_coords[ coord_idx ]
					for adj_src_idx in range( 0, num_adjoint_sources ):
						adjoint_e_fields = adjoint_e_fields_by_coord_and_focal[ current_coord ][ adj_src_idx ]

						spectral_indices = spectral_focal_plane_map[ adj_src_idx ]
						num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]

						prefactor = 0
						if band_weights[ adj_src_idx ] > 0:
							# prefactor = band_weights[ adj_src_idx ] * softplus_prime( reselect_fom_by_band_orig[ adj_src_idx ] ) * np.product( reselect_fom_by_band ) / reselect_fom_by_band_orig[ adj_src_idx ]
							prefactor = band_weights[ adj_src_idx ] * np.product( reselect_fom_by_band ) / reselect_fom_by_band_orig[ adj_src_idx ]

						# for spectral_idx in range(0, num_points ):
						# for spectral_idx in range(0, num_design_frequency_points ):

						# 	for polarization_idx in range( 0, 3 ):
						# 		# x-coordinate is the perpendicular coorinate to the level set boundary that can be actually changed
						# 		if polarization_idx == 0:
						# 			# todo: the gaussian norm is weird here, the normalization all alredy stacks up all the bands and the spectral indices are counting from 0 to something
						# 			# feels like you want to access at spectral_indices[0] + spectral_idx - effectively I think this is just what is happening
						# 			polarized_gradient_lsf += prefactor * ( ( ( 1. / min_real_permittivity ) - ( 1. / max_real_permittivity ) ) *
						# 				normalization_all[ spectral_idx ] * ( conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx] ) *
						# 				current_permittivity * adjoint_e_fields[polarization_idx, spectral_idx, :, :, :] *
						# 				current_permittivity * forward_e_fields[polarization_idx, spectral_idx, :, :, :]
						# 			)

						# 		else:
						# 			polarized_gradient_lsf += prefactor * ( ( max_real_permittivity - min_real_permittivity ) *
						# 				normalization_all[ spectral_idx ] * (conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx]) *
						# 				current_permittivity * adjoint_e_fields[polarization_idx, spectral_idx, :, :, :] *
						# 				current_permittivity * forward_e_fields[polarization_idx, spectral_idx, :, :, :]
						# 			)

						for spectral_idx in range(0, num_design_frequency_points):

							if ( spectral_idx < spectral_indices[ 1 ] ) and ( spectral_idx >= spectral_indices[ 0 ] ):
								# log_weight = 0

								# if band_weights[ adj_src_idx ] > 0:
								# 	log_weight = ( incorrect_focal_by_wl[ spectral_idx ] + fom_quotient_regularization ) / correct_focal_by_wl[ spectral_idx ]

								log_weight = 1.0

								# on_band_weight = np.log( num_focal_spots - 1. ) / ( incorrect_focal_by_wl[ spectral_idx ] + fom_quotient_regularization )
								on_band_weight = 1.0#( num_focal_spots - 1. )
								# on_band_weight = -( num_focal_spots - 1. )

								polarized_gradient += log_weight * on_band_weight * prefactor * np.sum(
									normalization_all[ spectral_idx ] * (conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx]) *
									adjoint_e_fields[:, spectral_idx, :, :, :] *
									forward_e_fields[:, spectral_idx, :, :, :],
									axis=0)

								for other_focal_idx in range( 0, num_focal_spots ):
									if other_focal_idx == adj_src_idx:
										continue

									# off_band_weight = -( num_focal_spots - 1. ) * correct_focal_by_wl[ spectral_idx ] / ( incorrect_focal_by_wl[ spectral_idx ] + fom_quotient_regularization )**2
									off_band_weight = 0.0#-1.0
									# off_band_weight = 1.0
									off_focal_adjoint_fields = adjoint_e_fields_by_coord_and_focal[ current_coord ][ other_focal_idx ]

									polarized_gradient += log_weight * off_band_weight * prefactor * np.sum(
										normalization_all[ spectral_idx ] * (conjugate_weighting_wavelength[other_focal_idx, current_coord, spectral_idx]) *
										off_focal_adjoint_fields[:, spectral_idx, :, :, :] *
										forward_e_fields[:, spectral_idx, :, :, :],
										axis=0)



				xy_polarized_gradients_by_pol[ pol_idx ] = polarized_gradient
				xy_polarized_gradients_by_pol_lsf[ pol_idx ] = polarized_gradient#polarized_gradient_lsf

			pol_weighting_denominator = np.sum( pol_weights )
			weight_grad_by_pol = ( 2. / pol_weighting_denominator ) - figure_of_merit_by_pol**2 / np.sum( figure_of_merit_by_pol**2 )
			weight_grad_by_pol = np.maximum( weight_grad_by_pol, 0 )

			for idx in range( 0, len( weight_grad_by_pol ) ):
				weight_grad_by_pol[ pol_idx ] *= pol_weights[ pol_idx ]

			weight_grad_by_pol /= np.sum( weight_grad_by_pol )


			for pol_idx in range( 0, num_polarizations ):
				xy_polarized_gradients += weight_grad_by_pol[ pol_idx ] * xy_polarized_gradients_by_pol[ pol_idx ]
				xy_polarized_gradients_lsf += weight_grad_by_pol[ pol_idx ] * xy_polarized_gradients_by_pol_lsf[ pol_idx ]

			#
			# Step 4: Step the design variable.
			#
			device_gradient_real = 2 * np.real( xy_polarized_gradients )
			device_gradient_imag = 2 * np.imag( xy_polarized_gradients )
			# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
			# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
			device_gradient_real = np.swapaxes(device_gradient_real, 0, 2)
			device_gradient_imag = np.swapaxes(device_gradient_imag, 0, 2)

			device_gradient_real_lsf = 2 * np.real( xy_polarized_gradients_lsf )
			device_gradient_imag_lsf = 2 * np.imag( xy_polarized_gradients_lsf )
			# Because of how the data transfer happens between Lumerical and here, the axes are ordered [z, y, x] when we expect them to be
			# [x, y, z].  For this reason, we swap the 0th and 2nd axes to get them into the expected ordering.
			device_gradient_real_lsf = np.swapaxes(device_gradient_real_lsf, 0, 2)
			device_gradient_imag_lsf = np.swapaxes(device_gradient_imag_lsf, 0, 2)

			gradients_real[ device, : ] = device_gradient_real
			gradients_imag[ device, : ] = device_gradient_imag
			gradients_real_lsf[ device, : ] = device_gradient_real_lsf
			gradients_imag_lsf[ device, : ] = device_gradient_imag_lsf


		cur_binarization = parent_object.get_binarization()

		fom_track.append( figure_of_merit_by_device[ 0 ] )
		binarization_track.append( cur_binarization )
		parent_object.submit_figure_of_merit( figure_of_merit_by_device, iteration, 0 )
		parent_object.update( -gradients_real, -gradients_imag, -gradients_real_lsf, -gradients_imag_lsf, 0, iteration )

		if ( iteration % 10 ) == 0:
			np.save( projects_directory_location + '/device_' + str( int( iteration / 10 ) ) + '.npy', parent_object.assemble_index(iteration) )
			np.save( projects_directory_location + '/density_' + str( int( iteration / 10 ) ) + '.npy', parent_object.assemble_index(0) )

		if cur_binarization > binarization_terminate:
			break


	return parent_object, fom_track, binarization_track


post_visualization = False

if post_visualization:

	load_index = np.load('/Users/gregory/Downloads/device_final_splitter_Hz_v3.npy')
	load_perm = load_index**2

	mid_real_permittivity = 0.5 * ( min_real_permittivity + max_real_permittivity )

	bin_profile = np.greater( load_perm, mid_real_permittivity )
	negative_profile = np.less_equal( load_perm, mid_real_permittivity )
	bin_perm = min_real_permittivity + ( max_real_permittivity - min_real_permittivity ) * bin_profile
	bin_index = np.sqrt( bin_perm )

	plt.imshow( bin_index, cmap='winter', extent=[ 0, designable_size_vertical_um, 0, device_size_lateral_um ] )
	plt.xlabel( 'Depth (um)', fontsize=16 )
	plt.ylabel( 'Width (um)', fontsize=16 )
	plt.colorbar()
	plt.show()

	layer_idxs = [ 18, 102, 181, 258, 343, 411, 491 ]

	feature_sizes_by_layer = [ [] for idx in range( 0, len( layer_idxs ) ) ]
	gap_sizes_by_layer = [ [] for idx in range( 0, len( layer_idxs ) ) ]

	def feature_len_by_profile( start_idx, profile ):
		profile_len = len( profile )

		length = 0
		while ( start_idx < len( profile ) ) and profile[ start_idx ]:
			length += 1
			start_idx += 1

		return length


	for layer in range( 0, len( layer_idxs ) ):
		get_layer = bin_profile[ :, layer_idxs[ layer ] ]
		start_idx = 0

		while start_idx < len( get_layer ):

			while ( start_idx < len( get_layer ) ) and ( not get_layer[ start_idx ] ):
				start_idx += 1

			next_length = feature_len_by_profile( start_idx, get_layer )

			if next_length > 0:
				feature_sizes_by_layer[ layer ].append( next_length )

			start_idx += next_length

		get_layer = negative_profile[ :, layer_idxs[ layer ] ]
		start_idx = 0

		while start_idx < len( get_layer ):

			while ( start_idx < len( get_layer ) ) and ( not get_layer[ start_idx ] ):
				start_idx += 1

			next_length = feature_len_by_profile( start_idx, get_layer )

			if next_length > 0:
				gap_sizes_by_layer[ layer ].append( next_length )

			start_idx += next_length

		# print( feature_sizes_by_layer[ layer ] )
		# print( gap_sizes_by_layer[ layer ] )
		# print()
		# plt.subplot( 1, 3, 1 )
		# plt.hist( 1e3 * lsf_mesh_spacing_um * np.array( feature_sizes_by_layer[ layer ] ) )
		# plt.subplot( 1, 3, 2 )
		# plt.hist( 1e3 * lsf_mesh_spacing_um * np.array( gap_sizes_by_layer[ layer ] ) )
		# plt.subplot( 1, 3, 3 )
		# plt.imshow( bin_profile )
		# plt.show()

	features_hist_coarse = []
	gaps_hist_coarse = []
	features_hist_fine = []
	gaps_hist_fine = []
	for layer in range( 0, len( layer_idxs ) - 1 ):
		get_features = feature_sizes_by_layer[ layer ]
		get_gaps = gap_sizes_by_layer[ layer ]

		for f in get_features:
			features_hist_coarse.append( f )
		for g in get_gaps:
			gaps_hist_coarse.append( g )


	get_features = feature_sizes_by_layer[ len( layer_idxs ) - 1 ]
	get_gaps = gap_sizes_by_layer[ len( layer_idxs ) - 1 ]

	for f in get_features:
		features_hist_fine.append( f )
	for g in get_gaps:
		gaps_hist_fine.append( g )

	features_histogram_data = [ 1e3 * lsf_mesh_spacing_um * np.array( features_hist_coarse ), 1e3 * lsf_mesh_spacing_um * np.array( features_hist_fine ) ]
	gaps_histogram_data = [ 1e3 * lsf_mesh_spacing_um * np.array( gaps_hist_coarse ), 1e3 * lsf_mesh_spacing_um * np.array( gaps_hist_fine ) ]

	bin_markings = [ 75 + 20 * idx for idx in range( 0, 30 ) ]

	plt.subplot( 1, 2, 1 )
	plt.hist( features_histogram_data, bins=bin_markings, color=['orange', 'green'] )
	# plt.hist( features_histogram_data, color=['orange', 'green'] )
	plt.title( 'Feature Size Distribution', fontsize=18 )
	plt.xlabel( 'Feature Size (nm)', fontsize=16 )
	plt.ylabel( 'Count', fontsize=16 )
	plt.legend( [ 'Coarse Feature Layers (min 100nm)', 'Fine Feature Layer (min 90nm)' ] )
	plt.subplot( 1, 2, 2 )
	plt.hist( gaps_histogram_data, bins=bin_markings, color=['blue', 'purple'] )
	# plt.hist( gaps_histogram_data, color=['blue', 'purple'] )
	plt.title( 'Gap Size Distribution', fontsize=18 )
	plt.xlabel( 'Feature Size (nm)', fontsize=16 )
	plt.ylabel( 'Count', fontsize=16 )
	plt.legend( [ 'Coarse Gap Layers (min 100nm)', 'Fine Gap Layer (min 90nm)' ] )
	plt.show()

	print( 1e3 * lsf_mesh_spacing_um * np.max( features_hist_coarse ) )
	print( 1e3 * lsf_mesh_spacing_um * np.max( features_hist_fine ) )
	print( 1e3 * lsf_mesh_spacing_um * np.max( gaps_hist_coarse ) )
	print( 1e3 * lsf_mesh_spacing_um * np.max( gaps_hist_fine ) )

	sys.exit( 0 )

	fdtd_hook.switchtolayout()
	inflate_index = np.zeros( ( load_index.shape[ 0 ], load_index.shape[ 1 ], 2 ), dtype=np.complex )
	inflate_index[ :, :, 0 ] = bin_index# load_index
	inflate_index[ :, :, 1 ] = bin_index# load_index
	fdtd_hook.select( device_import[ 'name' ] )
	fdtd_hook.importnk2( inflate_index, device_region_x, device_region_y, device_region_z )

	fdtd_hook.run()

	sys.exit( 0 )


figure_of_merit_evolution = []

for epoch_idx in range( 0, 1 ):

	binarization_terminate_threshold = 0.995

	my_optimization_state, local_fom, local_binarization = optimize_parent_locally( my_optimization_state, num_iterations, binarization_terminate_threshold )

	np.save( projects_directory_location + '/final_device.npy', my_optimization_state.assemble_index( num_iterations - 1 ) )
	np.save( projects_directory_location + '/final_density.npy', my_optimization_state.assemble_index( 0 ) )
	np.save( projects_directory_location + '/figure_of_merit.npy', local_fom )
	np.save( projects_directory_location + '/binarization.npy', local_binarization )
