import numpy as np
import matplotlib.pyplot as plt

from LayeredMWIRBridgesBayerFilterAngularEvalParameters import *

import os


#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


projects_directory_location_base = "/central/groups/Faraon_Computing/projects" 
projects_directory_location_base += "/" + project_name
projects_directory_location = projects_directory_location_base + '_angular_bfast_32_snell_large_focal'


angular_focal_fields = np.load( projects_directory_location + "/angular_focal_fields.npy" )

# num_incoherent_sums = 500#1000#500#1000

# coherent_intensity_by_wl = np.zeros( ( num_design_frequency_points, device_voxels_lateral, device_voxels_lateral ) )
# incoherent_intensity_by_wl = np.zeros( ( num_design_frequency_points, device_voxels_lateral, device_voxels_lateral ) )

coherent_intensity_by_wl = np.zeros( ( num_design_frequency_points, 1 + fdtd_region_minimum_lateral_voxels, 1 + fdtd_region_minimum_lateral_voxels ) )
# incoherent_intensity_by_wl = np.zeros( ( num_design_frequency_points, 1 + fdtd_region_minimum_lateral_voxels, 1 + fdtd_region_minimum_lateral_voxels ) )

phase_by_prop_prefactor_lambda = -2 * np.pi * ( src_maximum_vertical_um - 0.5 * silicon_thickness_um ) * 3.42 / lambda_values_um
phase_by_prop_prefactor_theta = 1.0 / np.cos( eval_theta_radians )

for wl_idx in range( 0, num_design_frequency_points ):

	fields_by_wl = np.squeeze( angular_focal_fields[ :, :, :, :, :, wl_idx ] )
	coherent_fields = np.squeeze( np.sum( np.squeeze( np.sum( fields_by_wl, axis=0 ) ), axis=0 ) )

	coherent_fields = np.zeros( coherent_fields.shape, dtype=np.complex )
	for phi_idx in range( 0, num_phi ):
		for theta_idx in range( 0, num_theta ):
			get_phase = np.exp( 1j * phase_by_prop_prefactor_lambda[ wl_idx ] * phase_by_prop_prefactor_theta[ theta_idx ] )
			coherent_fields += np.squeeze( get_phase * fields_by_wl[ phi_idx, theta_idx, :, :, : ] )
	
	coherent_intensity = np.squeeze( np.sum( np.abs( coherent_fields )**2, axis=0 ) )
	coherent_intensity_by_wl[ wl_idx ] = coherent_intensity


	# averaged_incoherent_intensity = np.zeros( coherent_intensity.shape )
	# for avg_idx in range( 0, num_incoherent_sums ):
	# 	random_phases = 2 * np.pi * np.random.random( ( num_phi, num_theta ) )

	# 	averaged_incoherent_fields = np.zeros( coherent_fields.shape, dtype=np.complex )
	# 	for phi_idx in range( 0, num_phi ):
	# 		for theta_idx in range( 0, num_theta ):
	# 			averaged_incoherent_fields += np.squeeze( np.exp( 1j * random_phases[ phi_idx, theta_idx ] ) * fields_by_wl[ phi_idx, theta_idx, :, :, : ] )
		
	# 	averaged_incoherent_intensity += ( 1. / num_incoherent_sums ) * np.squeeze( np.sum( np.abs( averaged_incoherent_fields )**2, axis=0 ) )

	# incoherent_intensity_by_wl[ wl_idx ] = averaged_incoherent_intensity


np.save( projects_directory_location + "/coherent_intensity_by_wl.npy", coherent_intensity_by_wl )
# np.save( projects_directory_location + "/incoherent_intensity_by_wl.npy", incoherent_intensity_by_wl )
