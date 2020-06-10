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
use_previous_opt = False

if run_on_cluster:
	sys.path.append( '/central/home/gdrobert/Develompent/ceviche' )
import ceviche

if len( sys.argv ) < 3:
	print( "Usage: python " + sys.argv[ 0 ] + " { random seed } { save folder }" )
	sys.exit( 1 )

random_seed = int( sys.argv[ 1 ] )
save_folder = sys.argv[ 2 ]

if use_previous_opt:
	random_seed = int( np.load( save_folder + '/random_seed.npy' ) )
else:
	np.save( save_folder + '/random_seed.npy', np.array( random_seed ) )

np.random.seed( random_seed )

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


def compute_fom( omega, mesh_size_m, relative_permittivity, pml_cells, fwd_src_y_loc, focal_point_x_loc, focal_point_y_loc ):
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
	
	return fom

mesh_size_nm = 15
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_nm = 500
lambda_max_nm = 600#700
num_lambda_values = 2#10

min_relative_permittivity = 1.5**2
max_relative_permittivity = 2.5**2

lambda_values_nm = np.linspace( lambda_min_nm, lambda_max_nm, num_lambda_values )
omega_values = 2 * np.pi * c / ( 1e-9 * lambda_values_nm )

pml_voxels = 40
device_width_voxels = 140
# device_height_voxels = 80
device_height_voxels = 60
device_voxels_total = device_width_voxels * device_height_voxels
mid_width_voxel = int( 0.5 * device_width_voxels )
mid_height_voxel = int( 0.5 * device_height_voxels )
width_gap_voxels = 50
height_gap_voxels_top = 75
height_gap_voxels_bottom = 50
focal_length_voxels = 100
simluation_width_voxels = device_width_voxels + 2 * width_gap_voxels + 2 * pml_voxels
simulation_height_voxels = device_height_voxels + focal_length_voxels + height_gap_voxels_bottom + height_gap_voxels_top + 2 * pml_voxels

device_width_start = int( 0.5 * ( simluation_width_voxels - device_width_voxels ) )
device_width_end = device_width_start + device_width_voxels
device_height_start = int( pml_voxels + height_gap_voxels_bottom + focal_length_voxels )
device_height_end = device_height_start + device_height_voxels

fwd_src_y = int( pml_voxels + height_gap_voxels_bottom + focal_length_voxels + device_height_voxels + 0.75 * height_gap_voxels_top )
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

def reinterpolate_average( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k / factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		start_x = int( factor * x_idx )
		end_x = start_x + factor
		for y_idx in range( 0, output_block_size[ 1 ] ):
			start_y = int( factor * y_idx )
			end_y = start_y + factor

			average = 0.0

			for sweep_x in range( start_x, end_x ):
				for sweep_y in range( start_y, end_y ):
					average += ( 1. / factor**2 ) * input_block[ sweep_x, sweep_y ]
			
			output_block[ x_idx, y_idx ] = average

	return output_block

def upsample( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k * factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		for y_idx in range( 0, output_block_size[ 1 ] ):
			output_block[ x_idx, y_idx ] = input_block[ int( x_idx / factor ), int( y_idx / factor ) ]

	return output_block

def density_to_permittivity( density_in ):
	return ( min_relative_permittivity + ( max_relative_permittivity - min_relative_permittivity ) * density_in )

rel_eps_simulation = np.ones( ( simluation_width_voxels, simulation_height_voxels ) )

focal_points_x = [
	int( device_width_start + 0.25 * device_width_voxels ),
	int( device_width_start + 0.75 * device_width_voxels )
]

wavelength_intensity_scaling = lambda_max_nm**2 / lambda_values_nm**2

if use_previous_opt:
	device_density = np.load( save_folder + "/device_density.npy" )
else:

	init_density = 0.5
	device_dense_density = init_density * np.ones( ( device_width_voxels, device_height_voxels ) )
	device_density = reinterpolate_average( device_dense_density, density_coarsen_factor )

	np.save( save_folder + "/init_device_dense_density.npy", device_dense_density )
	np.save( save_folder + "/init_device_density.npy", device_density )

	num_iterations = 4#30#0#250

	max_density_change_per_iteration_start = 0.05
	max_density_change_per_iteration_end = 0.005

	gradient_norm_evolution = np.zeros( num_iterations )
	fom_evolution = np.zeros( num_iterations )
	fom_by_wl_evolution = np.zeros( ( num_iterations, num_lambda_values ) )
	gradient_directions = np.zeros( ( num_iterations, device_density.shape[ 0 ], device_density.shape[ 1 ] ) )

	for iter_idx in range( 0, num_iterations ):
		
		import_density = upsample( device_density, density_coarsen_factor )
		device_permittivity = density_to_permittivity( import_density )

		rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = device_permittivity

		gradient_by_wl = []
		fom_by_wl = []

		for wl_idx in range( 0, num_lambda_values ):
			get_focal_point = focal_points_x[ 0 ]
			if wl_idx >= int( 0.5 * num_lambda_values ):
				get_focal_point = focal_points_x[ 1 ]
			get_fom, get_grad = compute_fom_and_gradient(
				omega_values[ wl_idx ],
				mesh_size_m, rel_eps_simulation,
				[ pml_voxels, pml_voxels ],
				fwd_src_y,
				get_focal_point, focal_point_y )

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

		net_gradient = reinterpolate_average( net_gradient, density_coarsen_factor )

		gradient_norm = vector_norm( net_gradient )

		fom_evolution[ iter_idx ] = net_fom
		fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
		gradient_norm_evolution[ iter_idx ] = gradient_norm

		norm_scaled_gradient = net_gradient / gradient_norm

		gradient_directions[ iter_idx ] = norm_scaled_gradient

		max_density_change = (
			max_density_change_per_iteration_start +
			( iter_idx / ( num_iterations - 1 ) ) * ( max_density_change_per_iteration_end - max_density_change_per_iteration_start )
		)

		device_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
		device_density = np.maximum( 0, np.minimum( device_density, 1 ) )

		np.save( save_folder + "/figure_of_merit.npy", fom_evolution )
		np.save( save_folder + "/gradient_directions.npy", gradient_norm_evolution )
		np.save( save_folder + "/figure_of_merit_by_wavelength.npy", fom_by_wl_evolution )
		np.save( save_folder + "/device_density.npy", device_density )


random_direction_mean = 0.0
random_direction_sigma = 1.0

device_density_flattened = device_density.flatten()

#
# Should also eventually consider feature size in here! Maybe just do this whole thing on a coarser grid including the optimization
#

direction_delta = np.random.normal( loc=random_direction_mean, scale=random_direction_sigma, size=[ int( device_voxels_total / density_coarsen_factor**2 ) ] )
direction_delta /= vector_norm( direction_delta )

direction_eta = np.random.normal( loc=random_direction_mean, scale=random_direction_sigma, size=[ int( device_voxels_total / density_coarsen_factor**2 ) ] )
direction_eta /= vector_norm( direction_eta )

np.save( save_folder + "/direction_delta.npy", direction_delta )
np.save( save_folder + "/direction_eta.npy", direction_eta )

num_steps_per_direction = 51

# alpha = np.linspace( -1, 1, num_steps_per_direction )
# beta = np.linspace( -1, 1, num_steps_per_direction )

alpha = np.linspace( -5, 5, num_steps_per_direction )
beta = np.linspace( -5, 5, num_steps_per_direction )

np.save( save_folder + "/alpha.npy", alpha )
np.save( save_folder + "/beta.npy", beta )

landscape = np.zeros( ( num_steps_per_direction, num_steps_per_direction ) )
landscape_valid = np.ones( ( num_steps_per_direction, num_steps_per_direction ) )

#
# How do we deal with the bounds of the problem on permittivity?
#
def density_bound_from_eps( eps_val ):
	return ( eps_val - min_relative_permittivity ) / ( max_relative_permittivity - min_relative_permittivity )
eps_max_landscape = 3.0**2
eps_min_landscape = 1.0**2

density_max_landscape = density_bound_from_eps( eps_max_landscape )
density_min_landscape = density_bound_from_eps( eps_min_landscape )

for landscape_x in range( 0, num_steps_per_direction ):
	for landscape_y in range( 0, num_steps_per_direction ):

		landscape_density_flattened = device_density_flattened + alpha[ landscape_x ] * direction_delta + beta[ landscape_y ] * direction_eta
		landscape_density = np.reshape( landscape_density_flattened, device_density.shape )

		if ( np.min( landscape_density ) < density_min_landscape ) or ( np.max( landscape_density ) > density_max_landscape ):
			landscape[ landscape_x, landscape_y ] = -1
			landscape_valid[ landscape_x, landscape_y ] = 0

		else:
			import_landscape_density = upsample( landscape_density, density_coarsen_factor )

			landscape_device_permittivity = density_to_permittivity( import_landscape_density )

			rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = landscape_device_permittivity

			fom_by_wl = []

			for wl_idx in range( 0, num_lambda_values ):
				get_focal_point = focal_points_x[ 0 ]
				if wl_idx >= int( 0.5 * num_lambda_values ):
					get_focal_point = focal_points_x[ 1 ]
				get_fom = compute_fom(
					omega_values[ wl_idx ],
					mesh_size_m, rel_eps_simulation,
					[ pml_voxels, pml_voxels ],
					fwd_src_y,
					get_focal_point, focal_point_y )

				scale_fom_for_wl = get_fom * wavelength_intensity_scaling[ wl_idx ]
				fom_by_wl.append( scale_fom_for_wl )

			landscape[ landscape_x, landscape_y ] = np.product( fom_by_wl )


		np.save( save_folder + "/landscape.npy", landscape )
		np.save( save_folder + "/landscape_valid.npy", landscape_valid )



