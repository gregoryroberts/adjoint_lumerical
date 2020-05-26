#
# Housekeeping
#
import os
import shutil
import sys

#
# Math
#
import numpy as np

#
# Plotting
#
import matplotlib as mpl
import matplotlib.pylab as plt

#
# Electromagnetics
#
run_on_cluster = True
if run_on_cluster:
	sys.path.append( '/central/home/gdrobert/Develompent/ceviche' )
import ceviche


if len( sys.argv ) < 5:
	print( "Usage: python " + sys.argv[ 0 ] + " { random seed } { number of simulations } { simulation ID number } { base folder }" )
	sys.exit( 1 )


random_seed = int( sys.argv[ 1 ] )
num_simulations = int( sys.argv[ 2 ] )
simulation_id_number = int( sys.argv[ 3 ] )
base_folder = sys.argv[ 4 ]
np.random.seed( random_seed )

save_location = base_folder + '/simulations_' + str( simulation_id_number )
np.save( save_location + '/random_seed.npy', np.array( random_seed ) )

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
shutil.copy2(
	python_src_directory + "/degrees_of_freedom.py",
	save_location + "/degrees_of_freedom.py")

eps_nought = 8.854 * 1e-12
c = 3.0 * 1e8

def compute_fom_and_gradient( omega, mesh_size_m, relative_permittivity, pml_cells, fwd_src_y_loc, focal_point_y_loc ):
	simulation_width_cells = relative_permittivity.shape[ 0 ]
	simulation_height_cells = relative_permittivity.shape[ 1 ]
	simulation = ceviche.fdfd_ez( omega, mesh_size_m, relative_permittivity, pml_cells )

	fwd_src_x = np.arange( 0, simulation_width_cells )
	fwd_src_y = fwd_src_y_loc * np.ones( fwd_src_x.shape, dtype=int )

	fwd_source = np.zeros( ( simulation_width_cells, simulation_height_cells ), dtype=np.complex )
	fwd_source[ fwd_src_x, fwd_src_y ] = 1

	fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( fwd_source )

	focal_point_y = focal_point_y_loc
	focal_point_x = int( 0.5 * simulation_width_cells )

	fom = np.abs( fwd_Ez[ focal_point_x, focal_point_y ] )**2
	
	adj_source = np.zeros( ( simulation_width_cells, simulation_height_cells ), dtype=np.complex )
	adj_source[ focal_point_x, focal_point_y ] = np.conj( fwd_Ez[ focal_point_x, focal_point_y ] )

	adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

	gradient = 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

	return fom, gradient


mesh_size_nm = 25
min_feature_size = 2 * mesh_size_nm
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_nm = 450
lambda_max_nm = 500
num_lambda_values = 5

min_relative_permittivity = 1.0
max_relative_permittivity = 2.5**2

lambda_values_nm = np.linspace( lambda_min_nm, lambda_max_nm, num_lambda_values )
omega_values = 2 * np.pi * c / ( 1e-9 * lambda_values_nm )

pml_voxels = 20
device_width_voxels = 81
device_height_voxels = 41
mid_width_voxel = int( 0.5 * device_width_voxels )
mid_height_voxel = int( 0.5 * device_height_voxels )
width_gap_voxels = 40
height_gap_voxels_top = 80
height_gap_voxels_bottom = 40
focal_length_voxels = 40
simluation_width_voxels = device_width_voxels + 2 * width_gap_voxels + 2 * pml_voxels
simulation_height_voxels = device_height_voxels + height_gap_voxels_bottom + height_gap_voxels_top + 2 * pml_voxels

device_width_start = int( 0.5 * ( simluation_width_voxels - device_width_voxels ) )
device_width_end = device_width_start + device_width_voxels
device_height_start = int( pml_voxels + height_gap_voxels_bottom + focal_length_voxels )
device_height_end = device_height_start + device_height_voxels

fwd_src_y = int( pml_voxels + height_gap_voxels_bottom + focal_length_voxels + device_height_voxels + 0.5 * height_gap_voxels_top )
focal_point_y = int( pml_voxels + height_gap_voxels_bottom )

#
# Verify finite difference gradient
#
verify_fd_grad = False

if verify_fd_grad:
	fd_x = int( 0.5 * simluation_width_voxels )
	fd_y = np.arange( device_height_start, device_height_end )
	compute_fd = np.zeros( len( fd_y ) )
	fd_omega = omega_values[ int( 0.5 * len( omega_values ) ) ]

	fd_init_device = 1.5 * np.ones( ( device_width_voxels, device_height_voxels ) )
	rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = test_device

	get_fom, get_grad = compute_fom_and_gradient( fd_omega, mesh_size_m, rel_eps_simulation, [ pml_voxels, pml_voxels ], fwd_src_y, focal_point_y )

	fd_step_eps = 1e-4

	for fd_y_idx in range( 0, len( fd_y ) ):
		fd_permittivity = rel_eps_simulation.copy()
		fd_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps

		get_fom_step, get_grad_step = compute_fom_and_gradient( fd_omega, mesh_size_m, fd_permittivity, [ pml_voxels, pml_voxels ], fwd_src_y, focal_point_y )

		compute_fd[ fd_y_idx ] = ( get_fom_step - get_fom ) / fd_step_eps

	plt.plot( compute_fd, color='g', linewidth=2 )
	plt.plot( get_grad[ fd_x, device_height_start : device_height_end ], color='b', linewidth=2, linestyle='--' )
	plt.show()


for sim_idx in range( 0, num_simulations ):
	pad_width = 10

	assert ( ( device_width_voxels - device_height_voxels ) % 2 ) == 0, "The padding assumption is not quite right!"

	pad_height = int( 0.5 * ( device_width_voxels - device_height_voxels ) ) + pad_width

	padded_width = 2 * pad_width + device_width_voxels
	padded_height = 2 * pad_height + device_height_voxels

	assert ( padded_width == padded_height ), "Expected the padded dimensions to be the same!"

	pad_mid_width_voxel = mid_width_voxel + pad_width
	pad_mid_height_voxel = mid_height_voxel + pad_height

	device_k = np.random.random( ( padded_width, padded_height ) ) + 1j * np.random.random( ( padded_width, padded_height ) )
	device_k[ pad_mid_width_voxel, pad_mid_height_voxel ] = np.real( device_k[ pad_mid_width_voxel, pad_mid_height_voxel ] )
	device_k_symmetric = device_k.copy()
	for x_idx in range( 0, padded_width ):
		for y_idx in range( 0, padded_height ):
			device_k_symmetric[ x_idx, y_idx ] = np.conj( device_k_symmetric[ padded_width - 1 - x_idx, padded_height - 1 - y_idx ] )

	k_max = np.floor( 0.5 * ( padded_width - 1 ) / ( 2 * min_feature_size / mesh_size_nm ) )
	k_taper_length = 2

	device_k_filtered = device_k_symmetric.copy()
	for kx_value in range( 0, padded_width ):
		for ky_value in range( 0, padded_height ):
			kx = kx_value - pad_mid_width_voxel
			ky = ky_value - pad_mid_height_voxel

			k_abs = np.sqrt( kx**2 + ky**2 )
			if k_abs > k_max:
				device_k_filtered[ kx_value, ky_value ] *= np.exp( -( k_abs - k_max ) / k_taper_length )


	filtered_device = np.real( np.fft.ifft2( np.fft.ifftshift( device_k_filtered ) ) )
	filtered_device_complex = np.fft.ifft2( np.fft.ifftshift( device_k_filtered ) )
	filtered_device = np.real( filtered_device_complex )
	filtered_device_imag = np.imag( filtered_device_complex )

	assert ( np.max( np.abs( filtered_device_imag ) ) / np.max( np.abs( filtered_device ) ) ) < 1e-12, "We didn't expect a device with a significant imaginary component"

	filtered_device = filtered_device[ pad_width : ( pad_width + device_width_voxels ), pad_height : ( pad_height + device_height_voxels ) ]

	random_center_x = device_width_voxels * np.random.random( 1 )[ 0 ]
	random_center_y = device_height_voxels * np.random.random( 1 )[ 0 ]

	random_width_x = device_width_voxels * np.random.random( 1 )[ 0 ]
	random_width_y = device_height_voxels * np.random.random( 1 )[ 0 ]

	weights = np.zeros( ( device_width_voxels, device_height_voxels ) )

	reweight_device = filtered_device.copy()
	for x_idx in range( 0, device_width_voxels ):
		for y_idx in range( 0, device_height_voxels ):

			x_delta = ( x_idx - random_center_x )**2
			y_delta = ( y_idx - random_center_y )**2

			weighting = np.exp( -0.5 * ( ( x_delta / random_width_x**2 ) + ( y_delta / random_width_y**2 ) ) )
			weights[ x_idx, y_idx ] = weighting

			reweight_device[ x_idx, y_idx ] *= weighting

	rescale_device = reweight_device.copy()
	rescale_device -= np.min( rescale_device )
	rescale_device /= np.max( rescale_device )

	min_max = 0.6
	max_min = 0.4

	choose_min = max_min * np.random.random( 1 )[ 0 ]
	choose_max = min_max + ( 1 - min_max ) * np.random.random( 1 )[ 0 ]

	rescale_device *= ( choose_max - choose_min )
	rescale_device += choose_min

	device_permittivity = min_relative_permittivity + ( max_relative_permittivity - min_relative_permittivity ) * rescale_device

	rel_eps_simulation = np.ones( ( simluation_width_voxels, simulation_height_voxels ) )
	rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = device_permittivity

	gradients = np.zeros( ( num_lambda_values, device_width_voxels, device_height_voxels ) )
	figures_of_merit = np.zeros( num_lambda_values )

	for wl_idx in range( 0, num_lambda_values ):
		figure_of_merit, gradient = compute_fom_and_gradient( omega_values[ wl_idx ], mesh_size_m, rel_eps_simulation, [ pml_voxels, pml_voxels ], fwd_src_y, focal_point_y )

		figures_of_merit[ wl_idx ] = figure_of_merit
		gradients[ wl_idx ] = gradient[ device_width_start : device_width_end, device_height_start : device_height_end ]

	np.save( save_location + '/device_permittivity_' + str( sim_idx ) + '.npy', device_permittivity )
	np.save( save_location + '/figures_of_merit_' + str( sim_idx ) + '.npy', figures_of_merit )
	np.save( save_location + '/gradients_' + str( sim_idx ) + '.npy', gradients )


