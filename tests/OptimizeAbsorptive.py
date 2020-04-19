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

projects_directory_location += "/optimize_absorptive"

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
num_design_frequency_points = 30
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
adjoint_monitor_top['maximum wavelength'] = lambda_min_um * 1e-6
adjoint_monitor_top['frequency points'] = 1
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

permittivity_max = 2.5
permittivity_min = 1.5
permittivity_mid = 0.5 * ( permittivity_min + permittivity_max )

device_permittivity = permittivity_mid * np.ones( ( device_width_voxels, device_height_voxels, 2 ) )

device_x_range = 1e-6 * np.linspace( -0.5 * device_width_um, 0.5 * device_width_um, device_width_voxels )
device_y_range = 1e-6 * np.linspace( device_min_um, device_max_um, device_height_voxels )
device_z_range = 1e-6 * np.linspace( -0.51, 0.51, 2 )

cavity_height_um = 0.2
cavity_index = 1.5
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


gsst_n_states = [ 3.0, 4.5 ]
gsst_k_states = [ 0.1, 0.25 ]

# note: may want an override mesh here around this interface because it is small and high index
gsst_height_um = 3 * mesh_size_um

num_gsst_states = len( gsst_n_states )

gsst_indexes = [ ( gsst_n_states[ idx ] + 1j * gsst_k_states[ idx ] ) * np.ones( ( 2, 2, 2 ), dtype=np.complex ) for idx in range( 0, len( gsst_n_states ) ) ]
gsst_imports = []

gsst_max_um = cavity_min_um
gsst_min_um = cavity_min_um - gsst_height_um

gsst_x_range = 1e-6 * np.linspace( -0.5 * device_width_um, 0.5 * device_width_um, 2 )
gsst_y_range = 1e-6 * np.linspace( gsst_min_um, gsst_max_um, 2 )
gsst_z_range = 1e-6 * np.linspace( -0.51, 0.51, 2 )


for idx in range( 0, num_gsst_states ):
	gsst_import = fdtd_hook.addimport()
	gsst_import['name'] = 'gsst_import_' + str( idx )
	gsst_import['x span'] = device_width_um * 1e-6
	gsst_import['y min'] = device_min_um * 1e-6
	gsst_import['y max'] = device_max_um * 1e-6
	gsst_import['z min'] = -0.51 * 1e-6
	gsst_import['z max'] = 0.51 * 1e-6
	gsst_import["override mesh order from material database"] = 1
	gsst_import['mesh order'] = 1

	gsst_imports.append( gsst_import )


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


lumapi_cmd = """
	E_field = getresult( "adjoint_monitor_top", "E" );
	H_field = getresult( "adjoint_monitor_top", "H" );
	EM = rectilineardataset("EM fields",E_field.x,E_field.y,E_field.z);
	EM.addparameter("lambda",c/E_field.f,"f",E_field.f);
	EM.addattribute("E",conj(E_field.Ex),conj(E_field.Ey),conj(E_field.Ez));
	EM.addattribute("H",conj(H_field.Hx),conj(H_field.Hy),conj(H_field.Hz));
	select("top_adjoint_source");
	switchtolayout;
	importdataset(EM);
"""

directional_weightings_by_state = [ np.ones( num_design_frequency_points ) for idx in range( 0, num_gsst_states ) ]
directional_weightings_by_state[ 0 ][ 0 : half_frequency_point ] = -1
directional_weightings_by_state[ 1 ][ : ] = -1

num_iterations = 100
figure_of_merit_by_iteration_by_state_by_wavelength = np.zeros( ( num_iterations, num_gsst_states, num_design_frequency_points ) )
figure_of_merit_by_iteration = np.zeros( num_iterations )

for iteration in range( 0, num_iterations ):

	gradient_by_gsst_state = []
	fom_by_gsst_state = []

	for gsst_state in range( 0, num_gsst_states ):

		device_index = permittivity_to_index( device_permittivity )
		fdtd_hook.switchtolayout()
		fdtd_hook.select( device_import['name'] )
		fdtd_hook.importnk2( device_index, device_x_range, device_y_range, device_z_range )
		fdtd_hook.select( gsst_imports[ gsst_state ]['name'] )
		fdtd_hook.importnk2( gsst_indexes[ gsst_state ], gsst_x_range, gsst_y_range, gsst_z_range )

		disable_all_sources()
		forward_src.enabled = 1
		fdtd_hook.run()

		transmission_fom = np.zeros( num_design_frequency_points )
		for wl_idx in range( 0, num_design_frequency_points ):
			get_T_top = compute_transmission_top( [ wl_idx, wl_idx + 1 ] )
			fom_T = get_T_top
			if directional_weightings_by_state[ gsst_state ][ wl_idx ] < 0:
				fom_T = 1 + directional_weightings_by_state[ gsst_state ][ wl_idx ] * fom_T

			fom_T = np.maximum( np.minimum( fom_T, 1.0 ), 0.0 )

			transmission_fom[ wl_idx ] = fom_T

		figure_of_merit_by_iteration_by_state_by_wavelength[ iteration, gsst_state, : ] = transmission_fom
		fom_by_gsst_state.append( np.mean( transmission_fom ) )

		forward_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )
		adjoint_e_fields = np.zeros( forward_e_fields.shape, dtype=np.complex )

		for wl_idx in range( 0, num_design_frequency_points ):

			fdtd_hook.switchtolayout()
			get_lambda_um = lambda_values_um[ wl_idx ]
			adjoint_monitor_top['minimum wavelength'] = get_lambda_um * 1e-6
			adjoint_monitor_top['maximum wavelength'] = get_lambda_um * 1e-6

			disable_all_sources()
			forward_src.enabled = 1
			fdtd_hook.run()

			shutil.copy(
				projects_directory_location + "/optimization.fsp",
				projects_directory_location + "/" + my_optimization_state.filename_prefix + "optimization_gsst_state_" + str( gsst_state ) + ".fsp" )

			lumapi.evalScript( fdtd_hook.handle,
				''.join( lumapi_cmd.split() )
			)
			disable_all_sources()
			top_adjoint_source.enabled = 1

			fdtd_hook.run()

			single_wl_adjoint_e_fields = get_complex_monitor_data( design_efield_monitor[ 'name' ], 'E' )
			adjoint_e_fields[ :, wl_idx, :, :, : ] = single_wl_adjoint_e_fields[ :, wl_idx, :, :, : ]

		gradient = -2 * np.real( np.sum( forward_e_fields * adjoint_e_fields, axis=0 ) / 1j )

		for wl_idx in range( 0, num_design_frequency_points ):
			gradient[ wl_idx, :, :, : ] *= directional_weightings_by_state[ gsst_state ][ wl_idx ]

		fom_weightings = ( 2. / num_design_frequency_points ) - transmission_fom**2 / np.sum( transmission_fom**2 )
		fom_weightings = np.maximum( fom_weightings, 0 )
		fom_weightings /= np.sum( fom_weightings )

		weighted_gradient = np.zeros( gradient[ 0 ].shape )
		for wl_idx in range( 0, num_design_frequency_points ):
			weighted_gradient += fom_weightings[ wl_idx ] * gradient[ wl_idx ] 

		weighted_gradient = np.swapaxes( weighted_gradient, 0, 2 )
		gradient_by_gsst_state.append( weighted_gradient )

	fom_by_gsst_state = np.array( fom_by_gsst_state )
	figure_of_merit_by_iteration[ iteration ] = np.mean( fom_by_gsst_state )
	weightings_by_state = 1. - fom_by_gsst_state**2 / np.sum( fom_by_gsst_state )
	weightings_by_state = np.maximum( weightings_by_state, 0 )
	weightings_by_state /= np.sum( weightings_by_state )
	print( "On iteration " + str( iteration ) + " fom by gsst state = " + str( fom_by_gsst_state ) )

	total_gradient = np.zeros( gradient_by_gsst_state[ 0 ].shape )
	for gsst_state in range( 0, num_gsst_states ):
		total_gradient += weightings_by_state[ gsst_state ] * gradient_by_gsst_state[ gsst_state ]

	step_size_rel = 0.05 - ( iteration / ( num_iterations - 1 ) ) * ( 0.05 - 0.025 )
	step = step_size_rel * total_gradient / np.max( np.abs( total_gradient ) )

	stepped_permittivity = device_permittivity[ :, :, 0 ] + step[ :, :, 0 ]
	stepped_permittivity = np.maximum( np.minimum( stepped_permittivity, permittivity_max ), permittivity_min )
	device_permittivity[ :, :, 0 ] = stepped_permittivity
	device_permittivity[ :, :, 1 ] = stepped_permittivity

