import numpy as np
import matplotlib.pyplot as plt

from LayeredMWIRBridgesBayerFilterAngularNormParameters import *

import os


#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


projects_directory_location_base = "/central/groups/Faraon_Computing/projects" 
projects_directory_location_base += "/" + project_name
projects_directory_location = projects_directory_location_base + '_angular_bfast_32_snell_large_focal_norm_xpol_v3'#_dilated_250nm'


angular_efields = np.load( projects_directory_location + "/device_input_efields.npy" )
angular_hfields = np.load( projects_directory_location + "/device_input_hfields.npy" )


coherent_efields_by_wl = np.zeros( ( num_design_frequency_points, 3, 1 + fdtd_region_minimum_lateral_voxels, 1 + fdtd_region_minimum_lateral_voxels ), dtype=np.complex )
coherent_hfields_by_wl = np.zeros( ( num_design_frequency_points, 3, 1 + fdtd_region_minimum_lateral_voxels, 1 + fdtd_region_minimum_lateral_voxels ), dtype=np.complex )

phase_by_prop_prefactor_lambda = -2 * np.pi * 0.5 * silicon_thickness_um * 3.42 / lambda_values_um
# phase_by_prop_prefactor_theta = 1.0 / np.cos( eval_theta_radians )
phase_by_prop_prefactor_theta = np.cos( eval_theta_radians )
phase_weighting_by_phi = np.zeros( num_phi, dtype=np.complex )

half_phi_idx = num_phi // 2
for phi_idx in range( 0, num_phi ):
	get_phi_value_radians = np.pi * eval_phi_degrees[ phi_idx % half_phi_idx ] / 180.

	phase_weighting_by_phi[ phi_idx ] = 1.0#np.exp( 1j * get_phi_value_radians )


weighting = np.ones( ( num_phi, num_theta ) )

for theta_idx in range( 0, num_theta ):
	f_by_f0 = np.sin( pre_substrate_theta_radians[ theta_idx ] ) / np.sin( np.max( pre_substrate_theta_radians ) )
	scale_f_by_f0 = f_by_f0 / 0.48
	Ta = np.exp( -( ( scale_f_by_f0**( 1.5 ) ) ) )

	weighting[ :, theta_idx ] = Ta#1.0#Ta

for wl_idx in range( 0, num_design_frequency_points ):

	efields_by_wl = np.squeeze( angular_efields[ :, :, :, :, :, wl_idx ] )
	hfields_by_wl = np.squeeze( angular_hfields[ :, :, :, :, :, wl_idx ] )
	coherent_efields = np.squeeze( np.sum( np.squeeze( np.sum( fields_by_wl, axis=0 ) ), axis=0 ) )
	coherent_hfields = np.squeeze( np.sum( np.squeeze( np.sum( fields_by_wl, axis=0 ) ), axis=0 ) )

	coherent_efields = np.zeros( coherent_efields.shape, dtype=np.complex )
	coherent_hfields = np.zeros( coherent_hfields.shape, dtype=np.complex )
	for phi_idx in range( 0, num_phi ):

		quadurant_weighting = 1.0
		# if phi_idx >= ( num_phi // 2 ):
		# 	quadurant_weighting = -1.0

		for theta_idx in range( 0, num_theta ):
			get_phase = phase_weighting_by_phi[ phi_idx ] * np.exp( 1j * phase_by_prop_prefactor_lambda[ wl_idx ] * phase_by_prop_prefactor_theta[ theta_idx ] )

			# get_phase = np.exp( 1j * random_phases[ 0, theta_idx ] )
			coherent_efields += quadurant_weighting * weighting[ phi_idx, theta_idx ] * np.squeeze( get_phase * efields_by_wl[ phi_idx, theta_idx, :, :, : ] )
			coherent_hfields += quadurant_weighting * weighting[ phi_idx, theta_idx ] * np.squeeze( get_phase * hfields_by_wl[ phi_idx, theta_idx, :, :, : ] )
	
	coherent_efields_by_wl[ wl_idx ] = coherent_efields
	coherent_hfields_by_wl[ wl_idx ] = coherent_hfields


np.save( projects_directory_location + "/coherent_efields_by_wl.npy", coherent_efields_by_wl )
np.save( projects_directory_location + "/coherent_hfields_by_wl.npy", coherent_hfields_by_wl )
