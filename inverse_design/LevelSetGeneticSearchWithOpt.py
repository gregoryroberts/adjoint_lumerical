import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LevelSetGlobalOptimize2DParameters import *

run_on_cluster = True

if run_on_cluster:
	import imp
	imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
# else:
# 	import imp
# 	imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )

import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

import level_set_cmos


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
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))

if run_on_cluster:
	projects_directory_location = "/central/groups/Faraon_Computing/projects" 


if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

projects_directory_location += "/" + project_name + "_genetic_with_opt_Hz_sio2_v3"

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

shutil.copy2(python_src_directory + "/LevelSetGlobalOptimize2DParameters.py", projects_directory_location + "/LevelSetGlobalOptimize2DParameters.py")
shutil.copy2(python_src_directory + "/LevelSetGeneticSearchWithOpt.py", projects_directory_location + "/LevelSetGeneticSearchWithOpt.py")



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

num_generations = 20#10
num_devices_per_generation = 150#100
num_parents_new_generation = 30#25

num_parents_propagated = 5


num_generations = 20
num_devices_per_generation = 100
num_parents_new_generation = 20

num_parents_propagated = 5




# Probability that on each layer you will just create a random new layer somewhere in a child
mutation_probability_start = 0.25
mutation_probability_end = 0.05
mutation_probability = np.linspace( mutation_probability_start, mutation_probability_end, num_generations - 1 )


def offspring( parent_1, parent_2, mutate_layer_probability ):
	profiles_1 = parent_1.get_layer_profiles()
	profiles_2 = parent_2.get_layer_profiles()

	num_profiles = len( profiles_1 )

	child_profiles = [ None for idx in range( 0, num_profiles ) ]

	crossover_point = np.minimum( int( np.random.uniform( 0, 1 ) * num_profiles ), num_profiles - 1 )

	for fill_idx in range( 0, crossover_point ):
		child_profiles[ fill_idx ] = profiles_1[ fill_idx ].copy()

	for fill_idx in range( crossover_point, num_profiles ):
		child_profiles[ fill_idx ] = profiles_2[ fill_idx ].copy()

	for mutate_idx in range( 0, num_profiles ):
		if np.random.uniform( 0, 1 ) < mutate_layer_probability:
			if mutate_idx < crossover_point:
				random_layer = parent_1.single_random_layer_profile( mutate_idx )
			else:
				random_layer = parent_2.single_random_layer_profile( mutate_idx )
			
			child_profiles[ mutate_idx ] = random_layer.copy()

	child = level_set_cmos.LevelSetCMOS(
		[ min_real_permittivity, max_real_permittivity ],
		lsf_mesh_spacing_um,
		designable_device_vertical_minimum_um,
		device_size_lateral_um,
		feature_size_voxels_by_profiles,
		device_layer_thicknesses_um,
		device_spacer_thicknesses_um,
		num_iterations_per_epoch,
		1,
		"level_set_optimize",
		device_lateral_background_density )

	if np.random.uniform( 0, 1 ) < 0.5:
		child.randomize_layer_profiles( parent_1.feature_gap_width_sigma_voxels, parent_1.feature_probability )
	else:
		child.randomize_layer_profiles( parent_2.feature_gap_width_sigma_voxels, parent_2.feature_probability )

	child.set_layer_profiles( child_profiles )

	return child

# Feels crazy, but you should verify the gradient for when you do level set optimized devices!

individuals_by_generation = [ None for idx in range( 0, num_generations ) ]

generation_0 = []

min_feature_density = 0.15#0.25
max_feature_density = 0.75#0.75

min_size_variability = 0.01#0.01
max_size_variability = 0.15#0.1

np.random.seed( 123123 )

for individual_idx in range( 0, num_devices_per_generation ):
	my_optimization_state = level_set_cmos.LevelSetCMOS(
		[ min_real_permittivity, max_real_permittivity ],
		lsf_mesh_spacing_um,
		designable_device_vertical_minimum_um,
		device_size_lateral_um,
		feature_size_voxels_by_profiles,
		device_layer_thicknesses_um,
		device_spacer_thicknesses_um,
		num_iterations_per_epoch,
		1,
		"level_set_optimize",
		device_lateral_background_density )

	random_feature_density = np.random.uniform( min_feature_density, max_feature_density )
	random_size_variability = np.random.uniform( min_size_variability, max_size_variability )

	my_optimization_state.randomize_layer_profiles( int( random_size_variability / lsf_mesh_spacing_um ), random_feature_density )

	generation_0.append( my_optimization_state )

individuals_by_generation[ 0 ] = generation_0

search_fom = []
search_fom_by_wl = []
search_devices = []


def optimize_parent_locally( parent_object, num_iterations ):
	fom_track = []
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
			get_index = parent_object.assemble_index( device )
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
								figure_of_merit_by_band[ focal_idx ] += ( ( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] ) - incorrect_focal_by_wl[ wl_idx ] )

				# wl_weighting_denominator = num_points_per_band * np.sum( band_weights )

				# figure_of_merit_by_band = np.maximum( figure_of_merit_by_band, 0 )

				reselect_fom_by_band = []
				for idx in range( 0, len( figure_of_merit_by_band ) ):
					if band_weights[ idx ] > 0:
						reselect_fom_by_band.append( softplus( figure_of_merit_by_band[ idx ] ) )
						# reselect_fom_by_band.append( figure_of_merit_by_band[ idx ] )
				reselect_fom_by_band = np.array( reselect_fom_by_band )
				# print( reselect_fom_by_band )

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
							prefactor = band_weights[ adj_src_idx ] * softplus_prime( reselect_fom_by_band[ adj_src_idx ] ) * np.product( reselect_fom_by_band ) / reselect_fom_by_band[ adj_src_idx ]

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
								on_band_weight = ( num_focal_spots - 1. )

								polarized_gradient += log_weight * on_band_weight * prefactor * np.sum(
									normalization_all[ spectral_idx ] * (conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx]) *
									adjoint_e_fields[:, spectral_idx, :, :, :] *
									forward_e_fields[:, spectral_idx, :, :, :],
									axis=0)

								for other_focal_idx in range( 0, num_focal_spots ):
									if other_focal_idx == adj_src_idx:
										continue

									# off_band_weight = -( num_focal_spots - 1. ) * correct_focal_by_wl[ spectral_idx ] / ( incorrect_focal_by_wl[ spectral_idx ] + fom_quotient_regularization )**2
									off_band_weight = -1.0
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

		fom_track.append( figure_of_merit_by_device[ 0 ] )
		parent_object.submit_figure_of_merit( figure_of_merit_by_device, iteration, 0 )
		parent_object.update( -gradients_real, -gradients_imag, -gradients_real_lsf, -gradients_imag_lsf, 0, iteration )

	return parent_object, fom_track

'''
def optimize_parent_locally( parent_object, num_iterations ):
	fom_track = []
	for iteration in range( 0, num_iterations ):

		figure_of_merit_by_device = np.zeros( parent_object.num_devices )

		field_shape_with_devices = [ parent_object.num_devices ]
		field_shape_with_devices.extend( np.flip( reversed_field_shape ) )
		gradients_real = np.zeros( field_shape_with_devices )
		gradients_imag = np.zeros( field_shape_with_devices )

		gradients_real_lsf = np.zeros( field_shape_with_devices )
		gradients_imag_lsf = np.zeros( field_shape_with_devices )

		for device in range( 0, parent_object.num_devices ):
			fdtd_hook.switchtolayout()
			get_index = parent_object.assemble_index( device )
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

				for focal_idx in range(0, num_focal_spots):
					spectral_indices = spectral_focal_plane_map[ focal_idx ]
					num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]
					# normalize = 1. / num_design_frequency_points                    

					for wl_idx in range( 0, num_design_frequency_points ):
						weighting = 0
						if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
							# weighting = 1.0
							weighting = band_weights[ focal_idx ]

						for coord_idx in range( 0, len( affected_coords ) ):
							current_coord = affected_coords[ coord_idx ]

							figure_of_merit_total[ wl_idx ] += weighting * (
								np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ])**2 )
							) / max_intensity_by_wavelength[ wl_idx ]

							figure_of_merit_by_band[ focal_idx ] +=  band_weights[ focal_idx ] * weighting * (
								np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ])**2 )
							) / max_intensity_by_wavelength[ wl_idx ]

							conjugate_weighting_wavelength[ focal_idx, current_coord, wl_idx ] = weighting * np.conj(
								focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] / max_intensity_by_wavelength[ wl_idx ] )

				wl_weighting_denominator = num_points_per_band * np.sum( band_weights )

				figure_of_merit_by_band = np.maximum( figure_of_merit_by_band, 0 )

				reselect_fom_by_band = []
				for idx in range( 0, len( figure_of_merit_by_band ) ):
					if band_weights[ idx ] > 0:
						reselect_fom_by_band.append( figure_of_merit_by_band[ idx ] )
				reselect_fom_by_band = np.array( reselect_fom_by_band )

				# todo: make sure this figure of merit weighting makes sense the way it is done across wavelengths and focal points
				figure_of_merit_total = np.maximum( figure_of_merit_total, 0 )#np.min( figure_of_merit_total )
				# fom_weighting = ( 2. / len( figure_of_merit_total ) ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				fom_weighting = ( 2. / wl_weighting_denominator ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				fom_weighting = np.maximum( fom_weighting, 0 )

				fom_weighting /= np.sum( fom_weighting )

				# figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.sum( normalization_all * figure_of_merit_total )
				figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.product( reselect_fom_by_band )
				# figure_of_merit += ( 1. / num_polarizations ) * figure_of_merit_by_pol[ pol_idx ]
				figure_of_merit += ( 1. / np.sum( pol_weights ) ) * figure_of_merit_by_pol[ pol_idx ]
				figure_of_merit_by_device[ device ] = figure_of_merit

				#
				# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
				# gradients for x- and y-polarized forward sources.
				#
				polarized_gradient = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)
				polarized_gradient_lsf = np.zeros(xy_polarized_gradients.shape, dtype=np.complex)

				current_index = np.real( get_non_struct_data( design_index_monitor[ 'name' ], 'index_x' ) )
				current_permittivity = np.sqrt( np.squeeze( current_index ) )

				for coord_idx in range( 0, len( affected_coords ) ):
					current_coord = affected_coords[ coord_idx ]
					for adj_src_idx in range( 0, num_adjoint_sources ):
						disable_all_sources()
						(adjoint_sources[current_coord][adj_src_idx]).enabled = 1
						fdtd_hook.run()

						adjoint_e_fields = get_complex_monitor_data(design_efield_monitor['name'], 'E')

						spectral_indices = spectral_focal_plane_map[ adj_src_idx ]
						num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]

						prefactor = 0
						if band_weights[ adj_src_idx ] > 0:
							prefactor = np.product( reselect_fom_by_band ) / figure_of_merit_by_band[ adj_src_idx ]

						# for spectral_idx in range(0, num_points ):
						for spectral_idx in range(0, num_design_frequency_points ):

							for polarization_idx in range( 0, 3 ):
								# x-coordinate is the perpendicular coorinate to the level set boundary that can be actually changed
								if polarization_idx == 0:
									# todo: the gaussian norm is weird here, the normalization all alredy stacks up all the bands and the spectral indices are counting from 0 to something
									# feels like you want to access at spectral_indices[0] + spectral_idx - effectively I think this is just what is happening
									polarized_gradient_lsf += prefactor * ( ( ( 1. / min_real_permittivity ) - ( 1. / max_real_permittivity ) ) *
										normalization_all[ spectral_idx ] * ( conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx] ) *
										current_permittivity * adjoint_e_fields[polarization_idx, spectral_idx, :, :, :] *
										current_permittivity * forward_e_fields[polarization_idx, spectral_idx, :, :, :]
									)

								else:
									polarized_gradient_lsf += prefactor * ( ( max_real_permittivity - min_real_permittivity ) *
										normalization_all[ spectral_idx ] * (conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx] ) *
										current_permittivity * adjoint_e_fields[polarization_idx, spectral_idx, :, :, :] *
										current_permittivity * forward_e_fields[polarization_idx, spectral_idx, :, :, :]
									)

						for spectral_idx in range(0, num_design_frequency_points):
							polarized_gradient += prefactor * np.sum(
								normalization_all[ spectral_idx ] * (conjugate_weighting_wavelength[adj_src_idx, current_coord, spectral_idx] ) *
								adjoint_e_fields[:, spectral_idx, :, :, :] *
								forward_e_fields[:, spectral_idx, :, :, :],
								axis=0)


				xy_polarized_gradients_by_pol[ pol_idx ] = polarized_gradient
				xy_polarized_gradients_by_pol_lsf[ pol_idx ] = polarized_gradient_lsf

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

		fom_track.append( figure_of_merit_by_device[ 0 ] )
		parent_object.submit_figure_of_merit( figure_of_merit_by_device, iteration, 0 )
		parent_object.update( -gradients_real, -gradients_imag, -gradients_real_lsf, -gradients_imag_lsf, 0, iteration )

	return parent_object, fom_track
'''

for generation_idx in range( 0, num_generations ):

	generation_fom = [ 0 for idx in range( 0, num_devices_per_generation ) ]
	# generation_fom_by_wl = [ np.zeros( num_design_frequency_points ) for idx in range( 0, num_devices_per_generation ) ]
	generation_devices = [ None for idx in range( 0, num_devices_per_generation ) ]

	for individual_idx in range( 0, num_devices_per_generation ):

		my_optimization_state = individuals_by_generation[ generation_idx ][ individual_idx ]

		get_index = my_optimization_state.assemble_index( 0 )
		device_region_x = 1e-6 * np.linspace( -0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, get_index.shape[ 0 ] )
		device_region_y = 1e-6 * np.linspace( designable_device_vertical_minimum_um, designable_device_vertical_maximum_um, get_index.shape[ 1 ] )
		device_region_z = 1e-6 * np.array( [ -0.51, 0.51 ] )

		figure_of_merit_by_device = np.zeros( my_optimization_state.num_devices )

		field_shape_with_devices = [ my_optimization_state.num_devices ]
		field_shape_with_devices.extend( np.flip( reversed_field_shape ) )
		gradients_real = np.zeros( field_shape_with_devices )
		gradients_imag = np.zeros( field_shape_with_devices )

		gradients_real_lsf = np.zeros( field_shape_with_devices )
		gradients_imag_lsf = np.zeros( field_shape_with_devices )

		for device in range( 0, my_optimization_state.num_devices ):
			#
			# Start here tomorrow!  Need to do this operation for every device.  Really, the single device operation should
			# be able to fold into here!  You need to get all these code changes under one umbrella.  Including the binarization
			# in the 3D code should be part of a single library.  And add all the postprocessing and evaluation code under
			# the same python library.  Can do things like angled evaluations, ...
			#

			fdtd_hook.switchtolayout()
			get_index = my_optimization_state.assemble_index( device )
			inflate_index = np.zeros( ( get_index.shape[ 0 ], get_index.shape[ 1 ], 2 ), dtype=np.complex )
			inflate_index[ :, :, 0 ] = get_index
			inflate_index[ :, :, 1 ] = get_index

			generation_devices[ individual_idx ] = get_index

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

				'''
				for focal_idx in range(0, num_focal_spots):
					spectral_indices = spectral_focal_plane_map[ focal_idx ]
					num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]
					# normalize = 1. / num_design_frequency_points                    

					for wl_idx in range( 0, num_design_frequency_points ):
						weighting = 0
						if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
							weighting = 1.0

						for coord_idx in range( 0, len( affected_coords ) ):
							current_coord = affected_coords[ coord_idx ]

							figure_of_merit_total[ wl_idx ] +=  band_weights[ focal_idx ] * weighting * (
								np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ])**2 )
							) / max_intensity_by_wavelength[ wl_idx ]

							figure_of_merit_by_band[ focal_idx ] +=  band_weights[ focal_idx ] * weighting * (
								np.sum( np.abs( focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ])**2 )
							) / max_intensity_by_wavelength[ wl_idx ]

							conjugate_weighting_wavelength[ focal_idx, current_coord, wl_idx ] = weighting * np.conj(
								focal_data[ focal_idx ][ current_coord, wl_idx, 0, 0, 0 ] / max_intensity_by_wavelength[ wl_idx ] )

				# todo: make sure this figure of merit weighting makes sense the way it is done across wavelengths and focal points
				figure_of_merit_total = np.maximum( figure_of_merit_total, 0 )#np.min( figure_of_merit_total )
				# fom_weighting = ( 2. / len( figure_of_merit_total ) ) - figure_of_merit_total**2 / np.sum( figure_of_merit_total**2 )
				# fom_weighting = np.maximum( fom_weighting, 0 )
				# fom_weighting /= np.sum( fom_weighting )
				figure_of_merit_by_band = np.maximum( figure_of_merit_by_band, 0 )

				reselect_fom_by_band = []
				for idx in range( 0, len( figure_of_merit_by_band ) ):
					if band_weights[ idx ] > 0:
						reselect_fom_by_band.append( figure_of_merit_by_band[ idx ] )
				reselect_fom_by_band = np.array( reselect_fom_by_band )

				figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.product( reselect_fom_by_band )
				# figure_of_merit_by_pol[ pol_idx ] = pol_weights[ pol_idx ] * np.sum( normalization_all * figure_of_merit_total )
				figure_of_merit += ( 1. / np.sum( pol_weights ) ) * figure_of_merit_by_pol[ pol_idx ]
				figure_of_merit_by_device[ device ] = figure_of_merit
				'''

				correct_focal_by_wl = np.zeros( num_design_frequency_points )
				incorrect_focal_by_wl = np.zeros( num_design_frequency_points )

				for focal_idx in range(0, num_focal_spots):
					spectral_indices = spectral_focal_plane_map[ focal_idx ]
					num_points = spectral_indices[ 1 ] - spectral_indices[ 0 ]

					for wl_idx in range( 0, num_design_frequency_points ):
						weighting = 0
						if ( wl_idx < spectral_indices[ 1 ] ) and ( wl_idx >= spectral_indices[ 0 ] ):
							weighting = band_weights[ focal_idx ]

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
								figure_of_merit_by_band[ focal_idx ] += ( ( ( num_focal_spots - 1. ) * correct_focal_by_wl[ wl_idx ] ) - incorrect_focal_by_wl[ wl_idx ] )


				# wl_weighting_denominator = num_points_per_band * np.sum( band_weights )

				# figure_of_merit_by_band = np.maximum( figure_of_merit_by_band, 0 )

				reselect_fom_by_band = []
				for idx in range( 0, len( figure_of_merit_by_band ) ):
					if band_weights[ idx ] > 0:
						reselect_fom_by_band.append( figure_of_merit_by_band[ idx ] )
				reselect_fom_by_band = np.array( reselect_fom_by_band )
				# print( reselect_fom_by_band )

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

				# generation_fom_by_wl[ individual_idx ] += figure_of_merit_by_pol[ pol_idx ]

		generation_fom[ individual_idx ] = figure_of_merit_by_device[ 0 ]

		# We should just throw these devices away, maybe set their FOM to 0?
		if np.isnan( figure_of_merit_by_device[ 0 ] ):
			generation_fom[ individual_idx ] = 0
			np.save( projects_directory_location + '/nan_device_' + str( generation_idx ) + "_" + str( individual_idx ) + ".npy", get_index )

	search_fom.append( generation_fom )
	search_devices.append( generation_devices )

	sorted_generation_fom = sorted( generation_fom, reverse=True )
	cutoff_fom = sorted_generation_fom[ num_parents_new_generation ]

	new_parents = []
	new_parents_idxs = []
	new_parents_propagated = []
	optimized_parents_fom = []
	optimized_parents_final_fom = []

	for parent_idx in range( 0, len( generation_fom ) ):
		if ( generation_fom[ parent_idx ] > cutoff_fom ) and ( len( new_parents ) < num_parents_new_generation ):
			new_parents.append( individuals_by_generation[ generation_idx ][ parent_idx ] )
			new_parents_idxs.append( parent_idx )
		

	#
	# Let's optimize all the parents we are sending forward!
	#
	num_optimization_cycles_per_parent = 10

	# num_optimization_cycles_per_parent = 2#10

	for parent_idx in range( 0, len( new_parents ) ):

		if num_optimization_cycles_per_parent > 0:

			optimized_new_parent, fom_track = optimize_parent_locally( new_parents[ parent_idx ], num_optimization_cycles_per_parent )
			new_parents[ parent_idx ] = optimized_new_parent
			optimized_parents_fom.append( fom_track )
			optimized_parents_final_fom.append( fom_track[ -1 ] )

			search_fom[ generation_idx ][ new_parents_idxs[ parent_idx ] ] = fom_track[ -1 ]
			search_devices[ generation_idx ][ new_parents_idxs[ parent_idx ] ] = optimized_new_parent.assemble_index( 0 )

		else:
			optimized_parents_final_fom.append( generation_fom[ new_parents_idxs[ parent_idx ] ] )



	sorted_optimized_parents_fom = sorted( optimized_parents_final_fom, reverse=True )
	cutoff_fom_propagation = sorted_optimized_parents_fom[ num_parents_propagated ]

	for parent_idx in range( 0, len( sorted_optimized_parents_fom ) ):
		if ( optimized_parents_final_fom[ parent_idx ] > cutoff_fom_propagation ) and ( len( new_parents_propagated ) < num_parents_propagated ):
			new_parents_propagated.append( new_parents[ parent_idx ] )

	np.save( projects_directory_location + "/optimized_parents_fom_" + str( generation_idx ) + ".npy", optimized_parents_fom )

	log_file = open( projects_directory_location + "/log.txt", 'a' )
	log_file.write( "Number of new parents = " + str( len( new_parents ) ) + " and propagated parents = " + str( len( new_parents_propagated ) ) + "\n" )
	log_file.write( "FOM Cutoffs = " + str( cutoff_fom ) + ", " + str( cutoff_fom_propagation ) + "\n\n" )
	log_file.close()

	if generation_idx < ( num_generations - 1 ):
		new_generation = []

		for child_idx in range( 0, num_devices_per_generation ):

			if ( child_idx < num_parents_propagated ) and ( child_idx < len( new_parents_propagated ) ):
				new_generation.append( new_parents_propagated[ child_idx ] )
			else:
				parent_1_idx = np.minimum( int( np.random.uniform( 0, 1 ) * len( new_parents ) ), len( new_parents ) - 1 )
				parent_2_idx = np.minimum( int( np.random.uniform( 0, 1 ) * len( new_parents ) ), len( new_parents ) - 1 )

				new_generation.append( offspring(
					new_parents[ parent_1_idx ],
					new_parents[ parent_2_idx ],
					mutation_probability[ generation_idx ]
				) )

		individuals_by_generation[ generation_idx + 1 ] = new_generation

	np.save( projects_directory_location + "/search_fom.npy", search_fom )
	np.save( projects_directory_location + "/search_fom_by_wl.npy", search_fom_by_wl )
	np.save( projects_directory_location + "/search_devices.npy", search_devices )


np.save( projects_directory_location + "/search_fom.npy", search_fom )
np.save( projects_directory_location + "/search_fom_by_wl.npy", search_fom_by_wl )
np.save( projects_directory_location + "/search_devices.npy", search_devices )
