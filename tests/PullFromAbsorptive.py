import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# import imp
# imp.load_source( "lumapi", "/Applications/Lumerical 2020a.app/Contents/API/Python/lumapi.py" )
import lumapi

import functools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

import scipy
import scipy.optimize

is_lumerical_version_2020a = False

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

project_load_directory = projects_directory_location + "/optimize_absorptive_switch_states_all_absorptive"

preload = True
start_iter = 50
preload_loc = projects_directory_location + "/pull_from_absorptive_switch_states_from_start_higher_index_v6"

projects_directory_location += "/pull_from_absorptive_switch_states_from_start_higher_index_v7"


if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")

fdtd_region_size_lateral_um = 5
fdtd_region_minimum_vertical_um = -2.5
fdtd_region_maximum_vertical_um = 2.5

mesh_size_um = 0.02

fdtd_region_minimum_lateral_voxels = 1 + int ( fdtd_region_size_lateral_um / mesh_size_um )
fdtd_region_minimum_vertical_voxels = 1 + int( ( fdtd_region_maximum_vertical_um - fdtd_region_minimum_vertical_um ) / mesh_size_um )


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
fdtd['simulation time'] = 100000 * 1e-15
fdtd['background index'] = 1.0

lambda_min_um = 0.4
lambda_max_um = 0.7
num_design_frequency_points = 20
half_frequency_point = int( 0.5 * num_design_frequency_points )

lambda_values_um = np.linspace( lambda_min_um, lambda_max_um, num_design_frequency_points )

device_width_um = 2
device_height_um = 1
device_min_um = 0
device_max_um = device_min_um + device_height_um


#
# General polarized source information
#
xy_phi_rotations = [0, 90]
xy_names = ['x', 'y']


forward_src = fdtd_hook.addplane()
forward_src['name'] = 'forward_src'
forward_src['plane wave type'] = 'Diffracting'
forward_src['polarization angle'] = 90
forward_src['direction'] = 'Backward'
forward_src['x span'] = 1.1 * device_width_um * 1e-6
forward_src['y'] = ( device_max_um + 0.5 * device_height_um ) * 1e-6
forward_src['wavelength start'] = lambda_min_um * 1e-6
forward_src['wavelength stop'] = lambda_max_um * 1e-6


tbox_y_max_um = 2.0
tbox_y_min_um = -1.0
tbox_x_max_um = 1.5
tbox_x_min_um = -1.5

adjoint_monitor_top = fdtd_hook.addpower()
adjoint_monitor_top['name'] = 'adjoint_monitor_top'
adjoint_monitor_top['monitor type'] = 'Linear X'
adjoint_monitor_top['x span'] = 3 * 1e-6
adjoint_monitor_top['y'] = tbox_y_max_um * 1e-6
adjoint_monitor_top['override global monitor settings'] = 1
if is_lumerical_version_2020a:
	adjoint_monitor_top['use wavelength spacing'] = 1
else:
	adjoint_monitor_top['use linear wavelength spacing'] = 1

adjoint_monitor_top['use source limits'] = 0
adjoint_monitor_top['minimum wavelength'] = lambda_min_um * 1e-6
adjoint_monitor_top['maximum wavelength'] = lambda_max_um * 1e-6
adjoint_monitor_top['frequency points'] = num_design_frequency_points
adjoint_monitor_top['output Hx'] = 1
adjoint_monitor_top['output Hy'] = 1
adjoint_monitor_top['output Hz'] = 1


top_adjoint_source = fdtd_hook.addimportedsource()
top_adjoint_source['name'] = 'top_adjoint_source'
top_adjoint_source['injection axis'] = 'y-axis'
top_adjoint_source['direction'] = 'Backward'
top_adjoint_source['wavelength start'] = lambda_min_um * 1e-6
top_adjoint_source['wavelength stop'] = lambda_max_um * 1e-6
top_adjoint_source['x span'] = adjoint_monitor_top['x span']
top_adjoint_source['y'] = adjoint_monitor_top['y']


transmission_box_top = fdtd_hook.addpower()
transmission_box_top['name'] = 'transmission_box_top'
transmission_box_top['monitor type'] = 'Linear X'
transmission_box_top['x span'] = 3 * 1e-6
transmission_box_top['y'] = tbox_y_max_um * 1e-6
transmission_box_top['override global monitor settings'] = 1
if is_lumerical_version_2020a:
	transmission_box_top['use wavelength spacing'] = 1
else:
	transmission_box_top['use linear wavelength spacing'] = 1

transmission_box_top['use source limits'] = 0
transmission_box_top['minimum wavelength'] = lambda_min_um * 1e-6
transmission_box_top['maximum wavelength'] = lambda_max_um * 1e-6
transmission_box_top['frequency points'] = num_design_frequency_points
transmission_box_top['output Hx'] = 1
transmission_box_top['output Hy'] = 1
transmission_box_top['output Hz'] = 1



def compute_transmission_top( wavelength_indexes ):
	get_T = get_monitor_data( transmission_box_top[ 'name' ], 'T' )
	get_T *= 1.0
	get_T = get_T[ 0 ]

	select_data = get_T[ wavelength_indexes[ 0 ] : wavelength_indexes[ 1 ] ]
	total = np.mean( select_data )

	return total


#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#
def disable_all_sources():
	fdtd_hook.switchtolayout()

	top_adjoint_source.enabled = 0
	forward_src.enabled = 0



#
# Set up the volumetric electric field monitor inside the design region.  We will need this compute
# the adjoint gradient
#

design_efield_monitor = fdtd_hook.addprofile()
design_efield_monitor['name'] = 'design_efield_monitor'
design_efield_monitor['x span'] = device_width_um * 1e-6
design_efield_monitor['y min'] = device_min_um * 1e-6
design_efield_monitor['y max'] = device_max_um * 1e-6
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
design_index_monitor['x span'] = device_width_um * 1e-6
design_index_monitor['y min'] = device_min_um * 1e-6
design_index_monitor['y max'] = device_max_um * 1e-6



#
# Add device region and create device permittivity
#

device_import = fdtd_hook.addimport()
device_import['name'] = 'device_import'
device_import['x span'] = device_width_um * 1e-6
device_import['y min'] = device_min_um * 1e-6
device_import['y max'] = device_max_um * 1e-6
device_import['z min'] = -0.51 * 1e-6
device_import['z max'] = 0.51 * 1e-6
device_import["override mesh order from material database"] = 1
device_import['mesh order'] = 1


device_width_voxels = 2 + int( device_width_um / mesh_size_um )
device_height_voxels = 2 + int( device_height_um / mesh_size_um )

permittivity_max = 2.5**2
permittivity_min = 1.5**2
permittivity_mid = 0.5 * ( permittivity_min + permittivity_max )

# Load the all-absorptive device
# device_permittivity = np.load( project_load_directory + '/cur_device.npy' )
quarter_permittivity = permittivity_min + 0.25 * ( permittivity_max - permittivity_min )
device_permittivity = quarter_permittivity * np.ones( ( device_width_voxels, device_height_voxels, 2 ) )


device_x_range = 1e-6 * np.linspace( -0.5 * device_width_um, 0.5 * device_width_um, device_width_voxels )
device_y_range = 1e-6 * np.linspace( device_min_um, device_max_um, device_height_voxels )
device_z_range = 1e-6 * np.linspace( -0.51, 0.51, 2 )

# todo: If you were going for a quarter wave of the mid-wave this is not the right thickness
cavity_index = 1.5
# cavity_height_um = 0.2
cavity_height_um = 0.25 * 0.5 * ( lambda_min_um + lambda_max_um ) / cavity_index
cavity_max_um = device_min_um
cavity_min_um = cavity_max_um - cavity_height_um

cavity = fdtd_hook.addrect()
cavity['name'] = 'cavity'
cavity['x span'] = device_width_um * 1e-6
cavity['y max'] = cavity_max_um * 1e-6
cavity['y min'] = cavity_min_um * 1e-6
cavity['z min'] = -0.51 * 1e-6
cavity['z max'] = 0.51 * 1e-6
cavity['index'] = cavity_index


# gsst_n_states = [ 3.0, 4.5 ]
# gsst_k_states = [ 0.1, 0.25 ]

gsst_n_states = [ 4.2, 5.75 ]
gsst_k_states = [ 2.5, 3.75 ]


# note: may want an override mesh here around this interface because it is small and high index
gsst_height_um = 3 * mesh_size_um

num_gsst_states = len( gsst_n_states )

gsst_max_um = cavity_min_um
gsst_min_um = cavity_min_um - gsst_height_um

gsst_indexes = [ ( gsst_n_states[ idx ] + 1j * gsst_k_states[ idx ] ) * np.ones( ( 2, 2, 2 ), dtype=np.complex ) for idx in range( 0, len( gsst_n_states ) ) ]

gsst_import = fdtd_hook.addimport()
gsst_import['name'] = 'gsst_import'
gsst_import['x span'] = device_width_um * 1e-6
gsst_import['y min'] = gsst_min_um * 1e-6
gsst_import['y max'] = gsst_max_um * 1e-6
gsst_import['z min'] = -0.51 * 1e-6
gsst_import['z max'] = 0.51 * 1e-6
gsst_import["override mesh order from material database"] = 1
gsst_import['mesh order'] = 1

# gsst_override_mesh = fdtd_hook.addmesh()
# gsst_override_mesh['name'] = 'gsst_override_mesh'
# gsst_override_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
# gsst_override_mesh['y min'] = ( gsst_min_um - 0.05 ) * 1e-6
# gsst_override_mesh['y max'] = ( gsst_max_um + 0.05 ) * 1e-6
# gsst_override_mesh['z min'] = -0.51 * 1e-6
# gsst_override_mesh['z max'] = 0.51 * 1e-6
# # gsst_override_mesh['dx'] = 0.001 * 1e-6
# gsst_override_mesh['dy'] = 0.003 * 1e-6

gsst_x_range = 1e-6 * np.linspace( -0.5 * device_width_um, 0.5 * device_width_um, 2 )
gsst_y_range = 1e-6 * np.linspace( gsst_min_um, gsst_max_um, 2 )
gsst_z_range = 1e-6 * np.linspace( -0.51, 0.51, 2 )


mirror_max_um = gsst_min_um
mirror_height_um = 0.5
mirror_min_um = mirror_max_um - mirror_height_um

mirror = fdtd_hook.addrect()
mirror['name'] = 'mirror'
mirror['x span'] = device_width_um * 1e-6
mirror['y max'] = mirror_max_um * 1e-6
mirror['y min'] = mirror_min_um * 1e-6
mirror['z min'] = -0.51 * 1e-6
mirror['z max'] = 0.51 * 1e-6
mirror['material'] = 'Au (Gold) - Palik'

def lumapi_set_wavelength( wl_idx ):
	# lumerical script is one indexed so need to adjust from python indexing
	cmd = "wl_idx = " + str( wl_idx + 1 ) + ";"
	lumapi.evalScript( fdtd_hook.handle, cmd )

lumapi_pull_results = """
	E_field = getresult( "adjoint_monitor_top", "E" );
	H_field = getresult( "adjoint_monitor_top", "H" );
"""

lumapi_import_source = """
	Ex = E_field.Ex( :, :, :, wl_idx );
	Ey = E_field.Ey( :, :, :, wl_idx );
	Ez = E_field.Ez( :, :, :, wl_idx );
	Hx = H_field.Hx( :, :, :, wl_idx );
	Hy = H_field.Hy( :, :, :, wl_idx );
	Hz = H_field.Hz( :, :, :, wl_idx );
	get_f = E_field.f( wl_idx );
	get_lambda = c / get_f;
	EM = rectilineardataset("EM fields",E_field.x,E_field.y,E_field.z);
	EM.addparameter("lambda",get_lambda,"f",get_f);
	EM.addattribute("E",conj(Ex),conj(Ey),conj(Ez));
	EM.addattribute("H",conj(Hx),conj(Hy),conj(Hz));
	switchtolayout;
	select("top_adjoint_source");
	importdataset(EM);
"""


alpha_extremes = 10

def approx_max( v ):
	return ( 1. / alpha_extremes ) * np.log( np.sum( np.exp( alpha_extremes * v ) ) )
def approx_min( v ):
	return ( -1. / alpha_extremes ) * np.log( np.sum( np.exp( -alpha_extremes * v ) ) )


def approx_max_grad( v ):
	return np.exp( alpha_extremes * v ) / np.sum( np.exp( alpha_extremes * v ) )

def approx_min_grad( v ):
	return np.exp( -alpha_extremes * v ) / np.sum( np.exp( -alpha_extremes * v ) )


def fom_dark( transmission_by_wavelength ):
	return approx_max( transmission_by_wavelength )
def grad_dark( transmission_by_wavelength, gradient_by_wavelength ):
	gradient_max = approx_max_grad( transmission_by_wavelength )

	gradient = np.zeros( gradient_by_wavelength[ 0 ].shape )
	for wl_idx in range( 0, len( transmission_by_wavelength ) ):
		gradient += gradient_by_wavelength[ wl_idx ] * gradient_max[ wl_idx ]

	return gradient



choose_hot_color = int( 3 * num_design_frequency_points / 4. )
hot_colors = [ choose_hot_color - 1, choose_hot_color, choose_hot_color + 1 ]

def fom_color( transmission_by_wavelength ):
	trim_transmission = np.zeros( len( hot_colors ) )

	cur_idx = 0
	for color_idx in range( 0, len( transmission_by_wavelength ) ):
		if color_idx in hot_colors:
			trim_transmission[ cur_idx ] = transmission_by_wavelength[ color_idx ]
			cur_idx += 1

	return approx_min( trim_transmission )

def grad_color( transmission_by_wavelength, gradient_by_wavelength ):
	trim_transmission = np.zeros( len( hot_colors ) )
	trim_gradient = np.zeros( [ len( hot_colors ) ] + list( gradient_by_wavelength[ 0 ].shape ) )

	cur_idx = 0
	for color_idx in range( 0, len( transmission_by_wavelength ) ):
		if color_idx in hot_colors:
			trim_transmission[ cur_idx ] = transmission_by_wavelength[ color_idx ]
			trim_gradient[ cur_idx ] = gradient_by_wavelength[ color_idx ]
			cur_idx += 1

	gradient_min = approx_min_grad( trim_transmission )

	gradient = np.zeros( gradient_by_wavelength[ 0 ].shape )
	for wl_idx in range( 0, len( trim_transmission ) ):
		gradient += trim_gradient[ wl_idx ] * gradient_min[ wl_idx ]

	return gradient



def fom_dark_narrow( transmission_by_wavelength ):
	trim_transmission = np.zeros( len( hot_colors ) )

	cur_idx = 0
	for color_idx in range( 0, len( transmission_by_wavelength ) ):
		if color_idx in hot_colors:
			trim_transmission[ cur_idx ] = transmission_by_wavelength[ color_idx ]
			cur_idx += 1

	return approx_max( trim_transmission )

def grad_dark_narrow( transmission_by_wavelength, gradient_by_wavelength ):
	trim_transmission = np.zeros( len( hot_colors ) )
	trim_gradient = np.zeros( [ len( hot_colors ) ] + list( gradient_by_wavelength[ 0 ].shape ) )

	cur_idx = 0
	for color_idx in range( 0, len( transmission_by_wavelength ) ):
		if color_idx in hot_colors:
			trim_transmission[ cur_idx ] = transmission_by_wavelength[ color_idx ]
			trim_gradient[ cur_idx ] = gradient_by_wavelength[ color_idx ]
			cur_idx += 1

	gradient_min = approx_max_grad( trim_transmission )

	gradient = np.zeros( gradient_by_wavelength[ 0 ].shape )
	for wl_idx in range( 0, len( trim_transmission ) ):
		gradient += trim_gradient[ wl_idx ] * gradient_min[ wl_idx ]

	return gradient









# directional_weightings_by_state = [ np.ones( num_design_frequency_points ) for idx in range( 0, num_gsst_states ) ]
# # directional_weightings_by_state[ 1 ][ 0 : half_frequency_point ] = -1
# directional_weightings_by_state[ 0 ][ : ] = -1
# directional_weightings_by_state[ 1 ][ : ] = 0
# directional_weightings_by_state[ 1 ][ int( 3 * num_design_frequency_points / 4. ) ] = num_design_frequency_points

# choose_hot_color = int( 3 * num_design_frequency_points / 4. )
# hot_colors = [ choose_hot_color - 1, choose_hot_color, choose_hot_color + 1 ]
# cold_colors = []
# for idx in range( 0, num_design_frequency_points ):
# 	if not ( idx in hot_colors ):
# 		cold_colors.append( idx )

# num_hot_colors = len( hot_colors )
# num_cold_colors = len( cold_colors )

num_iterations = 100
# figure_of_merit_by_iteration_by_state_by_wavelength = np.zeros( ( num_iterations, num_gsst_states, num_design_frequency_points ) )
figure_of_merit_by_iteration = np.zeros( num_iterations )

num_iterations_just_hot = 10#50

if not preload:
	start_iter = 0

if preload:
	device_permittivity = np.load( preload_loc + '/cur_device.npy' )

for iteration in range( start_iter, num_iterations ):

	gradient_by_gsst_state = []
	fom_by_gsst_state = []
	# fom_by_temp = [ 0 for i in range( 0, 2 ) ]
	# gradient_by_temp = []

	for gsst_state in range( 0, num_gsst_states ):

		device_index = permittivity_to_index( device_permittivity )
		np.save( projects_directory_location + '/cur_device.npy', device_permittivity )
		fdtd_hook.switchtolayout()
		fdtd_hook.select( device_import['name'] )
		fdtd_hook.importnk2( device_index, device_x_range, device_y_range, device_z_range )
		fdtd_hook.select( gsst_import['name'] )
		fdtd_hook.importnk2( gsst_indexes[ gsst_state ], gsst_x_range, gsst_y_range, gsst_z_range )

		disable_all_sources()
		forward_src.enabled = 1
		fdtd_hook.run()

		# transmission_fom = 0#np.zeros( num_design_frequency_points )
		transmission_by_wavelength = np.zeros( num_design_frequency_points )
		for wl_idx in range( 0, num_design_frequency_points ):
			get_T_top = compute_transmission_top( [ wl_idx, wl_idx + 1 ] )

			transmission_by_wavelength[ wl_idx ] = get_T_top

			# fom_T = get_T_top
			# # if directional_weightings_by_state[ gsst_state ][ wl_idx ] < 0:
			# # 	fom_T = 1 + directional_weightings_by_state[ gsst_state ][ wl_idx ] * fom_T
			# # else:
			# # 	fom_T *= directional_weightings_by_state[ gsst_state ][ wl_idx ]

			# # fom_T = np.maximum( np.minimum( fom_T, 1.0 ), 0.0 )
			# fom_T = np.maximum( fom_T, 0.0 )

			# if wl_idx in hot_colors:
			# 	if gsst_state == 1:
			# 		fom_by_temp[ 0 ] += ( 1. / num_hot_colors ) * fom_T
			# 	else:
			# 		fom_by_temp[ 0 ] -= ( 1. / num_hot_colors ) * fom_T
			# else:
			# 	if gsst_state == 0:
			# 		fom_by_temp[ 1 ] += ( 1. / num_cold_colors ) * ( 1 - fom_T )

			# transmission_fom[ wl_idx ] = fom_T

		get_fom = 0
		if gsst_state == 0:
			if iteration >= num_iterations_just_hot:
				get_fom = fom_dark_narrow( transmission_by_wavelength )
			else:
				get_fom = fom_dark( transmission_by_wavelength )
		else:
			get_fom = fom_color( transmission_by_wavelength )

		fom_by_gsst_state.append( get_fom )

		# figure_of_merit_by_iteration_by_state_by_wavelength[ iteration, gsst_state, : ] = transmission_fom
		# fom_by_temp.append( transmission_fom )# np.mean( transmission_fom ) )

		forward_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )
		# if gsst_state == 0:
		# 	for i in range( 0, 2 ):
		# 		gradient_by_temp.append( np.zeros( forward_e_fields[ 0, 0 ].shape ) )

		adjoint_e_fields = np.zeros( forward_e_fields.shape, dtype=np.complex )

		lumapi.evalScript( fdtd_hook.handle,
			''.join( lumapi_pull_results.split() )
		)

		for wl_idx in range( 0, num_design_frequency_points ):
			fdtd_hook.switchtolayout()
			lumapi_set_wavelength( wl_idx )

			shutil.copy(
				projects_directory_location + "/optimization.fsp",
				projects_directory_location + "/optimization_gsst_state_" + str( gsst_state ) + ".fsp" )
			
			disable_all_sources()
			top_adjoint_source.enabled = 1

			lumapi.evalScript( fdtd_hook.handle,
				''.join( lumapi_import_source.split() )
			)

			fdtd_hook.run()

			single_wl_adjoint_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )
			adjoint_e_fields[ :, wl_idx, :, :, : ] = single_wl_adjoint_e_fields[ :, wl_idx, :, :, : ]

		gradient = -2 * np.real( np.sum( forward_e_fields * adjoint_e_fields, axis=0 ) / 1j )

		if gsst_state == 0:
			if iteration >= num_iterations_just_hot:
				gradient_by_gsst_state.append( grad_dark_narrow( transmission_by_wavelength, -gradient ) )
			else:
				gradient_by_gsst_state.append( grad_dark( transmission_by_wavelength, -gradient ) )
		else:
			gradient_by_gsst_state.append( grad_color( transmission_by_wavelength, gradient ) )

		# for wl_idx in range( 0, num_design_frequency_points ):

		# 	if wl_idx in hot_colors:
		# 		if gsst_state == 1:
		# 			gradient_by_temp[ 0 ] += ( 1. / num_hot_colors ) * gradient[ wl_idx ]
		# 		else:
		# 			gradient_by_temp[ 0 ] -= ( 1. / num_hot_colors ) * gradient[ wl_idx ]
		# 	else:
		# 		if gsst_state == 0:
		# 			gradient_by_temp[ 1 ] -= ( 1. / num_cold_colors ) * gradient[ wl_idx ]



			# gradient[ wl_idx, :, :, : ] *= directional_weightings_by_state[ gsst_state ][ wl_idx ]

		# Not the right solution, but ok for now
		# fom_weightings = np.ones( num_design_frequency_points )
		# if gsst_state == 0:
		# 	fom_weightings = ( 2. / num_design_frequency_points ) - transmission_fom**2 / np.sum( transmission_fom**2 )
		# 	fom_weightings = np.maximum( fom_weightings, 0 )
		# 	fom_weightings /= np.sum( fom_weightings )

		# weighted_gradient = np.zeros( gradient[ 0 ].shape )
		# for wl_idx in range( 0, num_design_frequency_points ):
		# 	weighted_gradient += fom_weightings[ wl_idx ] * gradient[ wl_idx ] 

		# weighted_gradient = np.swapaxes( weighted_gradient, 0, 2 )
		# gradient_by_gsst_state.append( weighted_gradient )

	# HAVE TO SWAP THE AXES!!
	# for gsst_state_idx in range( 0, len( gradient_by_temp ) ):
	# 	gradient_by_temp[ gsst_state_idx ] = np.swapaxes( gradient_by_temp[ gsst_state_idx ], 0, 2 )

	max_reflection = 1.0
	print( fom_by_gsst_state )
	fom_by_gsst_state[ 0 ] = np.maximum( max_reflection - fom_by_gsst_state[ 0 ], 0 )
	fom_by_gsst_state[ 1 ] = np.maximum( fom_by_gsst_state[ 1 ], 0 )
	fom_by_gsst_state = np.array( fom_by_gsst_state )

	# fom_by_gsst_state = np.array( fom_by_gsst_state )
	# fom_by_temp = np.array( fom_by_temp )
	# figure_of_merit_by_iteration[ iteration ] = np.mean( fom_by_temp )
	figure_of_merit_by_iteration[ iteration ] = np.mean( fom_by_gsst_state )

	if iteration >= num_iterations_just_hot:

		hot_normalized_gradient = gradient_by_gsst_state[ 1 ] / np.max( np.abs( gradient_by_gsst_state[ 1 ] ) )
		cold_normalized_gradient = -gradient_by_gsst_state[ 0 ] / np.max( np.abs( gradient_by_gsst_state[ 0 ] ) )

		hot_normalized_gradient = np.swapaxes( hot_normalized_gradient, 0, 2 )
		cold_normalized_gradient = np.swapaxes( cold_normalized_gradient, 0, 2 )

		# max_movement = 0.01
		flattened_dark_min_gradient = cold_normalized_gradient.flatten()
		flattened_color_max_gradient = hot_normalized_gradient.flatten()
		# flattened_dark_min_gradient = -( gradient_by_temp[ 1 ] / np.max( np.abs( gradient_by_temp[ 0 ] ) ) ).flatten()
		# flattened_color_max_gradient = ( gradient_by_temp[ 0 ] / np.max( np.abs( gradient_by_temp[ 0 ] ) ) ).flatten()
		flattened_device = ( device_permittivity[ :, :, 0 ] ).flatten()

		np.save( projects_directory_location + '/device_permittivity_' + str( iteration ) + '.npy', device_permittivity )
		np.save( projects_directory_location + '/gradient_dark_' + str( iteration ) + '.npy', gradient_by_gsst_state[ 0 ] )
		np.save( projects_directory_location + '/gradient_bright_' + str( iteration ) + '.npy', gradient_by_gsst_state[ 1 ] )
		# np.save( projects_directory_location + '/gradient_dark_' + str( iteration ) + '.npy', gradient_by_temp[ 0 ] )
		# np.save( projects_directory_location + '/gradient_bright_' + str( iteration ) + '.npy', gradient_by_temp[ 1 ] )

		desired_colored_fom_change = 0.0025 * np.product( device_permittivity[ :, :, 0 ].shape ) * ( 1.1 - ( iteration / ( num_iterations - 1 ) ) )
		# print("desired change = " + str( desired_colored_fom_change))

		# Let's not let any epsilon move by more than 0.25 percent in density per iteration
		beta = 0.0025 * ( permittivity_max - permittivity_min )
		projected_binarization_increase = 0

		# c = flatten_fom_gradients
		c = -flattened_color_max_gradient
		# c = flattened_dark_min_gradient
		dim = len(c)

		# initial_colored_fom = fom_by_temp[ 0 ]
		# print( "Starting colored FOM = " + str( initial_colored_fom ) )

		# b = np.real( extract_binarization_gradient )
		b = -flattened_dark_min_gradient
		# b = flattened_color_max_gradient
		cur_x = np.zeros( dim )

		lower_bounds = np.zeros( len( c ) )
		upper_bounds = np.zeros( len( c ) )

		for idx in range( 0, len( c ) ):
			# upper_bounds[ idx ] = np.maximum( np.minimum( beta, 1 - flatten_design_cuts[ idx ] ), 0 )
			# lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -flatten_design_cuts[ idx ] ), 0 )
			upper_bounds[ idx ] = np.maximum( np.minimum( beta, permittivity_max - flattened_device[ idx ] ), 0 )
			lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -( flattened_device[ idx ] - permittivity_min ) ), 0 )

		max_possible_colored_change = 0
		for idx in range( 0, len( c ) ):
			if b[ idx ] > 0:
				max_possible_colored_change += b[ idx ] * upper_bounds[ idx ]
			else:
				max_possible_colored_change += b[ idx ] * lower_bounds[ idx ]
		
		alpha = np.minimum( max_possible_colored_change / 3., desired_colored_fom_change )

		def ramp( x ):
			return np.maximum( x, 0 )

		def opt_function( nu ):
			lambda_1 = ramp( nu * b - c )
			lambda_2 = c + lambda_1 - nu * b

			return -( -np.dot( lambda_1, upper_bounds ) + np.dot( lambda_2, lower_bounds ) + nu * alpha )


		tolerance = 1e-12
		optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance, bounds=[ [ 0, np.inf ] ] )

		nu_star = optimization_solution_nu.x
		lambda_1_star = ramp( nu_star * b - c )
		lambda_2_star = c + lambda_1_star - nu_star * b
		x_star = np.zeros( dim )

		for idx in range( 0, dim ):
			if lambda_1_star[ idx ] > 0:
				x_star[ idx ] = upper_bounds[ idx ]
			else:
				x_star[ idx ] = lower_bounds[ idx ]


		proposed_device = flattened_device + x_star
		proposed_device = np.minimum( np.maximum( proposed_device, permittivity_min ), permittivity_max )
		stepped_permittivity = np.reshape( proposed_device, device_permittivity[ :, :, 0 ].shape )

		expected_color_max_change = np.dot( x_star, -c )
		expected_color_min_change = np.dot( x_star, b )
		# expected_color_max_change = np.dot( x_star, b )
		# expected_color_min_change = np.dot( x_star, -c )
		print( "Expected color min change = " + str( expected_color_min_change ) )
		print( "Expected color max change = " + str( expected_color_max_change ) )

	else:
		fom_weightings = ( 2. / num_gsst_states ) - fom_by_gsst_state**2 / np.sum( fom_by_gsst_state**2 )
		fom_weightings = np.maximum( fom_weightings, 0 )
		fom_weightings /= np.sum( fom_weightings )
		print( 'cur weightings = ' + str( fom_weightings ) )

		weighted_gradient = np.zeros( gradient_by_gsst_state[ 0 ].shape )
		for gsst_state_idx in range( 0, num_gsst_states ):
			weighted_gradient += gradient_by_gsst_state[ gsst_state_idx ] * fom_weightings[ gsst_state_idx ]

		weighted_gradient = np.swapaxes( weighted_gradient, 0, 2 )

		step_magnitude = 0.05 - ( iteration / ( num_iterations - 1 ) ) * ( 0.05 - 0.01 )
		# step = 0.01 * ( gradient_by_temp[ 0 ] / np.max( np.abs( gradient_by_temp[ 0 ] ) ) )
		step = step_magnitude * ( permittivity_max - permittivity_min ) * ( weighted_gradient / np.max( np.abs( weighted_gradient ) ) )
		stepped_permittivity = device_permittivity[ :, :, 0 ] + step[ :, :, 0 ]
		stepped_permittivity = np.minimum( np.maximum( stepped_permittivity, permittivity_min ), permittivity_max )

	print( "On iteration " + str( iteration ) + " fom by gsst state = " + str( fom_by_gsst_state ) )

	# total_gradient = np.zeros( gradient_by_gsst_state[ 0 ].shape )
	# for gsst_state in range( 0, num_gsst_states ):
	# 	total_gradient += weightings_by_state[ gsst_state ] * gradient_by_gsst_state[ gsst_state ]

	# step_size_rel = 0.05 - ( iteration / ( num_iterations - 1 ) ) * ( 0.05 - 0.025 )
	# step = step_size_rel * total_gradient / np.max( np.abs( total_gradient ) )

	# stepped_permittivity = device_permittivity[ :, :, 0 ] + step[ :, :, 0 ]
	# stepped_permittivity = np.maximum( np.minimum( stepped_permittivity, permittivity_max ), permittivity_min )
	device_permittivity[ :, :, 0 ] = stepped_permittivity
	device_permittivity[ :, :, 1 ] = stepped_permittivity

