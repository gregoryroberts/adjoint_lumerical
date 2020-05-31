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

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { random seed } { save folder }" )
	sys.exit( 1 )

random_seed = int( sys.argv[ 1 ] )
save_folder = sys.argv[ 2 ]
np.random.seed( random_seed )

np.save( save_folder + '/random_seed.npy', np.array( random_seed ) )

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
shutil.copy2(
	python_src_directory + "/loss_landscape.py",
	save_folder + "/loss_landscape.py")

eps_nought = 8.854 * 1e-12
c = 3.0 * 1e8

def vector_norm( v_in ):
	return np.sqrt( np.sum( np.abs( v_in )**2 ) )

def compute_fom_and_gradient( omega, mesh_size_m, relative_permittivity, pml_cells, fwd_src_y_loc, focal_point_x_loc, focal_point_y_loc ):
	simulation_width_cells = relative_permittivity.shape[ 0 ]
	simulation_height_cells = relative_permittivity.shape[ 1 ]
	simulation = ceviche.fdfd_ez( omega, mesh_size_m, relative_permittivity, pml_cells )

	fwd_src_x = np.arange( 0, simulation_width_cells )
	fwd_src_y = fwd_src_y_loc * np.ones( fwd_src_x.shape, dtype=int )

	fwd_source = np.zeros( ( simulation_width_cells, simulation_height_cells ), dtype=np.complex )
	fwd_source[ fwd_src_x, fwd_src_y ] = 1

	fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( fwd_source )

	focal_point_y = focal_point_y_loc
	focal_point_x = focal_point_x_loc

	fom = np.abs( fwd_Ez[ focal_point_x, focal_point_y ] )**2
	
	adj_source = np.zeros( ( simulation_width_cells, simulation_height_cells ), dtype=np.complex )
	adj_source[ focal_point_x, focal_point_y ] = np.conj( fwd_Ez[ focal_point_x, focal_point_y ] )

	adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

	gradient = 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

	return fom, gradient


mesh_size_nm = 20
min_feature_size = 2 * mesh_size_nm
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_nm = 450
lambda_max_nm = 550
num_lambda_values = 2

min_relative_permittivity = 1.0
max_relative_permittivity = 1.5**2

lambda_values_nm = np.linspace( lambda_min_nm, lambda_max_nm, num_lambda_values )
omega_values = 2 * np.pi * c / ( 1e-9 * lambda_values_nm )

pml_voxels = 20
device_width_voxels = 101
device_height_voxels = 51
mid_width_voxel = int( 0.5 * device_width_voxels )
mid_height_voxel = int( 0.5 * device_height_voxels )
width_gap_voxels = 50
height_gap_voxels_top = 90
height_gap_voxels_bottom = 50
focal_length_voxels = 75
simluation_width_voxels = device_width_voxels + 2 * width_gap_voxels + 2 * pml_voxels
simulation_height_voxels = device_height_voxels + focal_length_voxels + height_gap_voxels_bottom + height_gap_voxels_top + 2 * pml_voxels

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
	rel_eps_simulation = np.ones( ( simluation_width_voxels, simulation_height_voxels ) )
	rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = (
		0.5 * ( min_relative_permittivity + max_relative_permittivity ) *  np.ones( ( device_width_voxels, device_height_voxels ) ) )

	focal_point_x = int( 0.5 * simulation_width_cells )

	get_fom, get_grad = compute_fom_and_gradient( fd_omega, mesh_size_m, rel_eps_simulation, [ pml_voxels, pml_voxels ], fwd_src_y, focal_point_x, focal_point_y )

	fd_step_eps = 1e-4

	for fd_y_idx in range( 0, len( fd_y ) ):
		fd_permittivity = rel_eps_simulation.copy()
		fd_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps

		get_fom_step, get_grad_step = compute_fom_and_gradient( fd_omega, mesh_size_m, fd_permittivity, [ pml_voxels, pml_voxels ], fwd_src_y, focal_point_x, focal_point_y )

		compute_fd[ fd_y_idx ] = ( get_fom_step - get_fom ) / fd_step_eps

	plt.plot( compute_fd, color='g', linewidth=2 )
	plt.plot( get_grad[ fd_x, device_height_start : device_height_end ], color='b', linewidth=2, linestyle='--' )
	plt.show()

	sys.exit( 0 )

def density_to_permittivity( density_in ):
	return ( min_relative_permittivity + ( max_relative_permittivity - min_relative_permittivity ) * density_in )

init_density = 0.5
device_density = init_density * np.ones( ( device_width_voxels, device_height_voxels ) )

np.save( save_folder + "/init_device_density.npy", device_density )

focal_points_x = [
	int( device_width_start + 0.25 * device_width_voxels ),
	int( device_width_start + 0.75 * device_width_voxels )
]

num_iterations = 1000

max_density_change_per_iteration_start = 0.05
max_density_change_per_iteration_end = 0.005

rel_eps_simulation = np.ones( ( simluation_width_voxels, simulation_height_voxels ) )

wavelength_intensity_scaling = lambda_max_nm**2 / lambda_values_nm**2

gradient_norm_evolution = np.zeros( num_iterations )
fom_evolution = np.zeros( num_iterations )
fom_by_wl_evolution = np.zeros( ( num_iterations, num_lambda_values ) )

for iter_idx in range( 0, num_iterations ):
	
	device_permittivity = density_to_permittivity( device_density )

	rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = device_permittivity

	gradient_by_wl = []
	fom_by_wl = []

	for wl_idx in range( 0, num_lambda_values ):
		get_fom, get_grad = compute_fom_and_gradient(
			omega_values[ wl_idx ],
			mesh_size_m, rel_eps_simulation,
			[ pml_voxels, pml_voxels ],
			fwd_src_y,
			focal_points_x[ wl_idx ], focal_point_y )

		device_grad = get_grad[ device_width_start : device_width_end, device_height_start : device_height_end ]

		scale_fom_for_wl = get_fom * wavelength_intensity_scaling[ wl_idx ]
		scale_gradient_for_wl = device_grad * wavelength_intensity_scaling[ wl_idx ]

		gradient_by_wl.append( scale_gradient_for_wl )
		fom_by_wl.append( scale_fom_for_wl )

	net_fom = np.product( fom_by_wl )
	net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

	for wl_idx in range( 0, num_lambda_values ):
		wl_gradient = ( max_relative_permittivity - min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
		weighting = net_fom / fom_by_wl[ wl_idx ]

		net_gradient += ( weighting * wl_gradient )

	gradient_norm = vector_norm( net_gradient )

	fom_evolution[ iter_idx ] = net_fom
	fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
	gradient_norm_evolution[ iter_idx ] = gradient_norm

	norm_scaled_gradient = net_gradient / gradient_norm

	max_density_change = (
		max_density_change_per_iteration_start +
		( iter_idx / ( num_iterations - 1 ) ) * ( max_density_change_per_iteration_end - max_density_change_per_iteration_start )
	)

	device_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
	device_density = np.maximum( 0, np.minimum( device_density, 1 ) )

	np.save( save_folder + "/figure_of_merit.npy", fom_evolution )
	np.save( save_folder + "/figure_of_merit_by_wavelength.npy", fom_by_wl_evolution )
	np.save( save_folder + "/device_density.npy", device_density )
