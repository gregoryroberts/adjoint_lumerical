#
# Math
#
import numpy as np
import scipy.optimize

#
# System
#
import sys

#
# Electromagnetics
#
run_on_cluster = True

if run_on_cluster:
	sys.path.append( '/central/home/gdrobert/Develompent/ceviche' )
import ceviche

eps_nought = 8.854 * 1e-12
c = 3.0 * 1e8

# def compute_binarization( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 2 / np.sqrt( total_shape ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
# def compute_binarization_gradient( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 4 / np.sqrt( total_shape ) ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )

def compute_binarization( input_variable ):
	total_shape = np.product( input_variable.shape )
	return ( 1. / total_shape ) * np.sum( np.sqrt( ( input_variable - 0.5 )**2 ) )
def compute_binarization_gradient( input_variable ):
	total_shape = np.product( input_variable.shape )
	return ( 1. / total_shape ) * ( input_variable - 0.5 ) / np.sqrt( ( input_variable - 0.5 )**2 )	

def vector_norm( v_in ):
	return np.sqrt( np.sum( np.abs( v_in )**2 ) )

def upsample( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k * factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		for y_idx in range( 0, output_block_size[ 1 ] ):
			output_block[ x_idx, y_idx ] = input_block[ int( x_idx / factor ), int( y_idx / factor ) ]

	return output_block

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

class ColorSplittingOptimization2D():

	def __init__( self,
		device_size_voxels, coarsen_factor, mesh_size_nm,
		permittivity_bounds, focal_spots_x_relative, focal_length_y_voxels,
		wavelengths_um, wavelength_idx_to_focal_idx, random_seed,
		num_layers, designable_layer_indicators, non_designable_permittivity,
		save_folder ):
		
		self.device_width_voxels = device_size_voxels[ 0 ]
		self.device_height_voxels = device_size_voxels[ 1 ]

		self.coarsen_factor = coarsen_factor
		assert ( self.device_width_voxels % coarsen_factor ) == 0, "The specified coarsening factor does not evenly divide the device width in voxels!"
		assert ( self.device_height_voxels % coarsen_factor ) == 0, "The specified coarsening factor does not evenly divide the device height in voxels!"

		self.design_width_voxels = int( device_size_voxels[ 0 ] / coarsen_factor )
		self.design_height_voxels = int( device_size_voxels[ 1 ] / coarsen_factor )

		self.design_density = None

		self.mesh_size_nm = mesh_size_nm
		self.mesh_size_um = 1e-3 * mesh_size_nm
		self.mesh_size_m = 1e-9 * mesh_size_nm

		self.device_size_um = [ self.mesh_size_um * device_size_voxels[ idx ] for idx in range( 0, len( device_size_voxels ) ) ]

		self.permittivity_bounds = permittivity_bounds
		self.min_relative_permittivity = permittivity_bounds[ 0 ]
		self.max_relative_permittivity = permittivity_bounds[ 1 ]

		self.focal_spots_x_relative = focal_spots_x_relative
		self.focal_length_y_voxels = focal_length_y_voxels
		self.wavelengths_um = wavelengths_um
		self.wavelength_intensity_scaling = self.wavelengths_um**2 / np.max( self.wavelengths_um )**2

		self.num_wavelengths = len( wavelengths_um )

		self.omega_values = 2 * np.pi * c / ( 1e-6 * wavelengths_um )

		self.wavelength_idx_to_focal_idx = wavelength_idx_to_focal_idx

		self.random_seed = random_seed
		np.random.seed( self.random_seed )

		assert( self.design_height_voxels % num_layers ) == 0, "Expected the number of layers to evenly divide the design region"

		self.num_layers = num_layers
		self.design_voxels_per_layer = int( self.design_height_voxels / num_layers )

		assert ( len( designable_layer_indicators ) == self.num_layers ), "The layer designability indicator length does not make sense!"
		assert ( len( non_designable_permittivity ) == len( designable_layer_indicators ) ), "Expected a different length for the non designable permittivity "

		self.designable_layer_indicators = np.array( designable_layer_indicators )
		self.non_designable_permittivity = np.array( non_designable_permittivity )
		self.non_designable_density = ( self.non_designable_permittivity - self.min_relative_permittivity ) / ( self.max_relative_permittivity - self.min_relative_permittivity )

		self.save_folder = save_folder

		self.setup_simulation()

	def init_density_with_random( self, mean_density, sigma_density ):
		num_random_values = self.design_width_voxels * self.num_layers

		random_array_normal_distribution = np.random.normal(
			loc=mean_density,
			scale=sigma_density, size=[ num_random_values ] )

		self.design_density = np.ones( [ self.design_width_voxels, self.design_height_voxels ] )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			random_values_start = layer_idx * self.design_width_voxels
			random_values_end = random_values_start + self.design_width_voxels

			fill_data = self.non_designable_density[ layer_idx ] * np.ones( self.design_width_voxels )

			if self.designable_layer_indicators[ layer_idx ]:
				fill_data = random_array_normal_distribution[ random_values_start : random_values_end ]

			for internal_layer_idx in range( layer_start, layer_end ):
				self.design_density[ :, internal_layer_idx ] = fill_data

		self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

	def init_density_with_uniform( self, density_value ):
		assert ( ( density_value <= 1.0 ) and ( density_value >= 0.0 ) ), "Invalid density value specified!"

		self.design_density = np.ones( [ self.design_width_voxels, self.design_height_voxels ] )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			random_values_start = layer_idx * self.design_width_voxels
			random_values_end = random_values_start + self.design_width_voxels

			choose_density = self.non_designable_density[ layer_idx ]

			if self.designable_layer_indicators[ layer_idx ]:
				choose_density = density_value

			for internal_layer_idx in range( layer_start, layer_end ):
				self.design_density[ :, internal_layer_idx ] = choose_density

	def init_density_directly( self, input_density ):
		assert ( ( input_density.shape[ 0 ] == self.design_width_voxels ) and ( input_density.shape[ 1 ] == self.design_height_voxels ) ), "Specified design has the wrong shape"

		self.design_density = input_density.copy()

	def init_density_with_this_class( self, this_class ):
		self.init_density_directly( this_class.design_density )

	def setup_simulation( self ):
		self.width_gap_voxels = int( 1.0 * np.max( self.wavelengths_um ) / self.mesh_size_um )
		self.height_gap_voxels_top = int( 1.5 * np.max( self.wavelengths_um ) / self.mesh_size_um )
		self.height_gap_voxels_bottom = self.width_gap_voxels
		self.pml_voxels = int( 1.0 * np.max( self.wavelengths_um ) / self.mesh_size_um )

		self.simulation_width_voxels = self.device_width_voxels + 2 * self.width_gap_voxels + 2 * self.pml_voxels
		self.simulation_height_voxels = self.device_height_voxels + self.focal_length_y_voxels + self.height_gap_voxels_bottom + self.height_gap_voxels_top + 2 * self.pml_voxels

		self.device_width_start = int( 0.5 * ( self.simulation_width_voxels - self.device_width_voxels ) )
		self.device_width_end = self.device_width_start + self.device_width_voxels
		self.device_height_start = int( self.pml_voxels + self.height_gap_voxels_bottom + self.focal_length_y_voxels )
		self.device_height_end = self.device_height_start + self.device_height_voxels

		self.focal_spots_x_voxels = [
			int( self.device_width_start + self.focal_spots_x_relative[ idx ] * self.device_width_voxels ) for idx in range( 0, len( self.focal_spots_x_relative ) )
		]

		self.fwd_src_y = int( self.pml_voxels + self.height_gap_voxels_bottom + self.focal_length_y_voxels + self.device_height_voxels + 0.75 * self.height_gap_voxels_top )
		self.focal_point_y = int( self.pml_voxels + self.height_gap_voxels_bottom )

		self.rel_eps_simulation = np.ones( ( self.simulation_width_voxels, self.simulation_height_voxels ) )

		fwd_src_x_range = np.arange( 0, self.simulation_width_voxels )
		fwd_src_y_range = self.fwd_src_y * np.ones( fwd_src_x_range.shape, dtype=int )

		self.fwd_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		self.fwd_source[ fwd_src_x_range, fwd_src_y_range ] = 1

	def compute_forward_fields( self, omega, device_permittivity ):
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( self.fwd_source )

		return fwd_Ez

	def compute_fom( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		# import matplotlib.pyplot as plt
		# plt.imshow( np.abs( fwd_Ez )**2 )
		# plt.show()

		return fom

	def compute_fom_and_gradient( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		return fom, gradient

	# CODE DUPLICATION! FIX
	def compute_net_fom( self ):
		fom_by_wl = []

		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		for wl_idx in range( 0, self.num_wavelengths ):
			get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]
			get_fom = self.compute_fom(
				self.omega_values[ wl_idx ], device_permittivity,
				self.focal_spots_x_voxels[ get_focal_point_idx ], self.wavelength_intensity_scaling[ wl_idx ] )
			fom_by_wl.append( scale_fom_for_wl )

		net_fom = np.product( fom_by_wl )

		return net_fom


	def verify_adjoint_against_finite_difference( self ):
		fd_x = int( 0.5 * self.simulation_width_voxels )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd = np.zeros( len( fd_y ) )
		omega_idx = int( 0.5 * len( self.omega_values ) )
		fd_omega = self.omega_values[ omega_idx ]

		fd_init_device = 1.5 * np.ones( ( self.device_width_voxels, self.device_height_voxels ) )

		focal_point_x = self.focal_spots_x_voxels[ 0 ]

		get_fom, get_grad = self.compute_fom_and_gradient(
			fd_omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ],
			self.fwd_src_y, focal_point_x, self.focal_point_y, self.wavelength_intensity_scaling[ omega_idx ] )

		fd_step_eps = 1e-4

		for fd_y_idx in range( 0, len( fd_y ) ):
			fd_device_permittivity = fd_init_device.copy()
			fd_device_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps

			get_fom_step = self.compute_fom( fd_omega, fd_device_permittivity, focal_point_x, self.wavelength_intensity_scaling[ omega_idx ] )

			compute_fd[ fd_y_idx ] = ( get_fom_step - get_fom ) / fd_step_eps

		self.average_adjoint_finite_difference_error = np.sqrt( np.mean( np.abs( get_grad[ fd_x, device_height_start : device_height_end ] - compute_fd )**2 ) )

	def density_to_permittivity( self, density ):
		return ( self.min_relative_permittivity + ( self.max_relative_permittivity - self.min_relative_permittivity ) * density )

	def layer_spacer_averaging( self, gradient_input ):
		gradient_output = np.zeros( gradient_input.shape )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			fill_gradient = np.zeros( self.design_width_voxels )

			if self.designable_layer_indicators[ layer_idx ]:
				fill_gradient = np.mean( gradient_input[ :, layer_start : layer_end ], axis=1 )

			for internal_layer_idx in range( layer_start, layer_end ):
				gradient_output[ :, internal_layer_idx ] = fill_gradient

		return gradient_output

	def step_binarize( self, gradient, binarize_amount, binarize_max_movement ):

		density_for_binarizing = self.design_density.flatten()
		flatten_gradient = gradient.flatten()

		flatten_design_cuts = density_for_binarizing.copy()
		extract_binarization_gradient = compute_binarization_gradient( flatten_design_cuts )
		flatten_fom_gradients = flatten_gradient.copy()

		beta = binarize_max_movement
		projected_binarization_increase = 0

		c = flatten_fom_gradients

		initial_binarization = compute_binarization( flatten_design_cuts )

		b = np.real( extract_binarization_gradient )

		lower_bounds = np.zeros( len( c ) )
		upper_bounds = np.zeros( len( c ) )

		for idx in range( 0, len( c ) ):
			upper_bounds[ idx ] = np.maximum( np.minimum( beta, 1 - flatten_design_cuts[ idx ] ), 0 )
			lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -flatten_design_cuts[ idx ] ), 0 )

		max_possible_binarization_change = 0
		for idx in range( 0, len( c ) ):
			if b[ idx ] > 0:
				max_possible_binarization_change += b[ idx ] * upper_bounds[ idx ]
			else:
				max_possible_binarization_change += b[ idx ] * lower_bounds[ idx ]
		
		# Try this! Not sure how well it will work
		alpha = np.minimum( initial_binarization * max_possible_binarization_change, binarize_amount )

		def ramp( x ):
			return np.maximum( x, 0 )

		def opt_function( nu ):
			lambda_1 = ramp( nu * b - c )
			lambda_2 = c + lambda_1 - nu * b

			return -( -np.dot( lambda_1, upper_bounds ) + np.dot( lambda_2, lower_bounds ) + nu * alpha )

		tolerance = 1e-12
		optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance )

		nu_star = optimization_solution_nu.x
		lambda_1_star = ramp( nu_star * b - c )
		lambda_2_star = c + lambda_1_star - nu_star * b
		x_star = np.zeros( len( c ) )

		for idx in range( 0, len( c ) ):
			if lambda_1_star[ idx ] > 0:
				x_star[ idx ] = upper_bounds[ idx ]
			else:
				x_star[ idx ] = lower_bounds[ idx ]

		proposed_design_variable = flatten_design_cuts + x_star
		proposed_design_variable = np.minimum( np.maximum( proposed_design_variable, 0 ), 1 )

		print( initial_binarization )
		print( compute_binarization( proposed_design_variable.flatten() ) )

		return np.reshape( proposed_design_variable, self.design_density.shape )

	def optimize( self, num_iterations, binarize=False, binarize_movement_per_step=0.01, binarize_max_movement_per_voxel=0.025 ):
		self.max_density_change_per_iteration_start = 0.03#0.05
		self.max_density_change_per_iteration_end = 0.01#0.005

		self.gradient_norm_evolution = np.zeros( num_iterations )
		self.fom_evolution = np.zeros( num_iterations )
		self.binarization_evolution = np.zeros( num_iterations )
		self.fom_by_wl_evolution = np.zeros( ( num_iterations, self.num_wavelengths ) )
		self.gradient_directions = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		log_file = open( self.save_folder + "/log.txt", 'a' )
		for iter_idx in range( 0, num_iterations ):
			log_file.write( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) + "\n")

			import_density = upsample( self.design_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )

			gradient_by_wl = []
			fom_by_wl = []

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				get_fom, get_grad = self.compute_fom_and_gradient(
					self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
					self.wavelength_intensity_scaling[ wl_idx ] )

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

				scale_fom_for_wl = get_fom * self.wavelength_intensity_scaling[ wl_idx ]
				scale_gradient_for_wl = upsampled_device_grad * self.wavelength_intensity_scaling[ wl_idx ]

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )

			net_fom = np.product( fom_by_wl )
			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

			# We are currently not doing a performance based weighting here, but we can add it in
			for wl_idx in range( 0, self.num_wavelengths ):
				wl_gradient = ( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			net_gradient = reinterpolate_average( net_gradient, self.coarsen_factor )

			#
			# Now, we should zero out non-designable regions and average over designable layers
			#
			net_gradient = self.layer_spacer_averaging( net_gradient )

			gradient_norm = vector_norm( net_gradient )

			self.fom_evolution[ iter_idx ] = net_fom
			self.fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
			self.gradient_norm_evolution[ iter_idx ] = gradient_norm

			norm_scaled_gradient = net_gradient / gradient_norm

			self.gradient_directions[ iter_idx ] = norm_scaled_gradient

			max_density_change = (
				self.max_density_change_per_iteration_start +
				( iter_idx / ( num_iterations - 1 ) ) * ( self.max_density_change_per_iteration_end - self.max_density_change_per_iteration_start )
			)

			self.binarization_evolution[ iter_idx ] = compute_binarization( self.design_density.flatten() )

			if binarize:
				self.design_density = self.step_binarize( -norm_scaled_gradient, binarize_movement_per_step, binarize_max_movement_per_voxel )
			else:
				self.design_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
				self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

		log_file.close()

	def save_optimization_data( self, file_base ):
		np.save( file_base + "_gradient_norm_evolution.npy", self.gradient_norm_evolution )
		np.save( file_base + "_fom_evolution.npy", self.fom_evolution )
		np.save( file_base + "_binarization_evolution.npy", self.binarization_evolution )
		np.save( file_base + "_fom_by_wl_evolution.npy", self.fom_by_wl_evolution )
		np.save( file_base + "_gradient_directions.npy", self.gradient_directions )
		np.save( file_base + "_optimized_density.npy", self.design_density )



