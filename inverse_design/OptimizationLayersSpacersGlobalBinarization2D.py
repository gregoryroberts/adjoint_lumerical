import numpy as np
import FreeBayerFilter2D
import OptimizationState

import matplotlib.pyplot as plt

import scipy.ndimage.morphology

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )

class OptimizationLayersSpacersGlobalBinarization2D( OptimizationState.OptimizationState ):

	def __init__(
		self,
		num_iterations, num_epochs, max_binarize_movement, desired_binarize_change, erode_dilate_start_iter, erode_dilate_iter_gap, num_iterations_erode_dilate,
		device_size_um, optimization_mesh_step_um,
		layer_start_coordinates_um, layer_thicknesses_um, layer_designability, layer_background_index, simulation_background_index,
		bayer_filter_creator_fn,
		bayer_filter_update_fn,
		filename_prefix,
		has_fabrication_penalty=False ):

		super(OptimizationLayersSpacersGlobalBinarization2D, self).__init__( num_iterations, num_epochs, filename_prefix )

		self.device_size_um = device_size_um
		self.optimization_mesh_step_um = optimization_mesh_step_um

		self.layer_start_coordinates_um = layer_start_coordinates_um
		self.layer_thicknesses_um = layer_thicknesses_um
		self.layer_designability = layer_designability
		self.layer_background_index = layer_background_index

		self.opt_device_size_voxels = [ 1 + int( self.device_size_um[ i ] / self.optimization_mesh_step_um ) for i in range( 0, len( self.device_size_um ) ) ]

		self.index = simulation_background_index * np.ones( self.opt_device_size_voxels, dtype=np.complex )

		self.bayer_filter_creator_fn = bayer_filter_creator_fn
		self.bayer_filter_update_fn = bayer_filter_update_fn

		self.max_binarize_movement = max_binarize_movement
		self.desired_binarize_change = desired_binarize_change

		self.erode_dilate_start_iter = erode_dilate_start_iter
		self.erode_dilate_iter_gap = erode_dilate_iter_gap
		self.num_iterations_erode_dilate = num_iterations_erode_dilate

		self.has_fabrication_penalty = has_fabrication_penalty

		self.create_bayer_filters()

	def create_bayer_filters( self ):
		self.num_total_layers = len( self.layer_thicknesses_um )

		self.bayer_filters = []

		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			layer_size_voxels = [ self.opt_device_size_voxels[ 0 ], layer_top_voxel - layer_bottom_voxel ]

			if self.layer_designability[ layer_idx ]:
				num_internal_layers = layer_size_voxels[ 1 ]

				bayer_filter = self.bayer_filter_creator_fn( layer_size_voxels, num_internal_layers, bayer_idx )
				self.bayer_filters.append( bayer_filter )

				bayer_idx += 1

	def assemble_index( self ):
		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:
				self.index[ :, layer_bottom_voxel : layer_top_voxel ] = permittivity_to_index( self.bayer_filters[ bayer_idx ].get_permittivity() )
				bayer_idx += 1
			else:
				self.index[ :, layer_bottom_voxel : layer_top_voxel ] = self.layer_background_index[ layer_idx ]

		return self.index

	def convert_permittivity_to_density( self, index, min_real_index, max_real_index ):
		permittivity =  ( np.real( index ) )**2
		min_real_permittivity = ( min_real_index )**2
		max_real_permittivity = ( max_real_index )**2
		density = ( permittivity - min_real_permittivity ) / ( max_real_permittivity - min_real_permittivity )

		reinterpolate_density = self.reinterpolate( density, self.index.shape )

		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:
				self.bayer_filters[ bayer_idx ].set_design_variable(
					reinterpolate_density[ :, layer_bottom_voxel : layer_top_voxel ]
				)

				bayer_idx += 1

	def convert_permittivity_to_layered_density( self, index, min_real_index, max_real_index ):
		permittivity =  ( np.real( index ) )**2
		min_real_permittivity = ( min_real_index )**2
		max_real_permittivity = ( max_real_index )**2
		density = ( permittivity - min_real_permittivity ) / ( max_real_permittivity - min_real_permittivity )

		reinterpolate_density = self.reinterpolate( density, self.index.shape )

		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			layer_thickness_voxels = layer_top_voxel - layer_bottom_voxel
			average_layer_density = np.zeros( ( reinterpolate_density.shape[ 0 ], layer_thickness_voxels ) )
			
			get_average = np.squeeze( np.mean( reinterpolate_density[ :, layer_bottom_voxel : layer_top_voxel ], axis=1 ) )

			for sublayer in range( layer_thickness_voxels ):
				average_layer_density[ :, sublayer ] = get_average

			if self.layer_designability[ layer_idx ]:
				self.bayer_filters[ bayer_idx ].set_design_variable(
					average_layer_density
				)

				bayer_idx += 1

	def convert_permittivity_to_kspace( self, index, min_real_index, max_real_index, control_feature_size_voxels ):
		permittivity =  ( np.real( index ) )**2
		min_real_permittivity = ( min_real_index )**2
		max_real_permittivity = ( max_real_index )**2
		density = ( permittivity - min_real_permittivity ) / ( max_real_permittivity - min_real_permittivity )

		reinterpolate_density = self.reinterpolate( density, self.index.shape )

		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			layer_thickness_voxels = layer_top_voxel - layer_bottom_voxel
			get_average = np.squeeze( np.mean( reinterpolate_density[ :, layer_bottom_voxel : layer_top_voxel ], axis=1 ) )

			dim = len( get_average )
			fourier_dim = dim + 2 + ( 1 - ( dim % 2 ) )
			middle_point = fourier_dim // 2
			fourier_dim_difference = fourier_dim - dim
			left_pad = fourier_dim_difference // 2
			right_pad = ( 1 + fourier_dim_difference ) // 2

			pad_density = np.pad( get_average, ((left_pad, right_pad)), mode='constant' )

			k_limit = int( np.floor( fourier_dim / ( 2 * control_feature_size_voxels ) ) )

			density_k_unshifted = np.fft.fft( pad_density )
			density_k = np.fft.fftshift( density_k_unshifted )

			density_k[ 0 : ( middle_point - k_limit ) ] = 0
			density_k[ ( middle_point + k_limit + 1 ) : ] = 0

			if self.layer_designability[ layer_idx ]:
				self.bayer_filters[ bayer_idx ].set_design_variable(
					density_k
				)

				bayer_idx += 1


	def load_other_design( self, other_filename_prefix, filebase, epoch, shift=0 ):
		for bayer_idx in range( 0, len( self.bayer_filters ) ):
			self.bayer_filters[ bayer_idx ].set_design_variable(
				np.load( filebase + "/" + other_filename_prefix + str( bayer_idx ) + "_" + str( epoch ) + ".npy" ) - shift )

	def load_design( self, filebase, epoch ):
		for bayer_idx in range( 0, len( self.bayer_filters ) ):
			self.bayer_filters[ bayer_idx ].set_design_variable(
				np.load( filebase + "/" + self.filename_prefix + str( bayer_idx ) + "_" + str( epoch ) + ".npy" ) )

	def save_design( self, filebase, epoch ):
		for bayer_idx in range( 0, len( self.bayer_filters ) ):
			np.save( filebase + "/" + self.filename_prefix + str( bayer_idx ) + "_" + str( epoch ) + ".npy", self.bayer_filters[ bayer_idx ].get_design_variable() )

	def update_epoch( self, epoch ):
		for bayer_idx in range( 0, len( self.bayer_filters ) ):
			self.bayer_filters[ bayer_idx ].update_filters( epoch )

	def update( self, gradient_real, gradient_imag, lsf_gradient_real, lsf_gradient_imag, epoch, iteration ):
		gradient_real_interpolate = self.reinterpolate( np.squeeze( gradient_real ), self.opt_device_size_voxels )
		gradient_imag_interpolate = self.reinterpolate( np.squeeze( gradient_imag ), self.opt_device_size_voxels )

		lsf_gradient_real_interpolate = self.reinterpolate( np.squeeze( lsf_gradient_real ), self.opt_device_size_voxels )
		lsf_gradient_imag_interpolate = self.reinterpolate( np.squeeze( lsf_gradient_imag ), self.opt_device_size_voxels )

		max_abs_gradient_step = -1
		max_abs_lsf_gradient_step = -1
		max_abs_fab_penalty_step = -1

		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:
				gradient_step = self.bayer_filters[ bayer_idx ].backpropagate(
					gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
					gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ] )
				
				lsf_gradient_step = self.bayer_filters[ bayer_idx ].backpropagate(
					lsf_gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
					lsf_gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ] )
				
				max_abs_gradient_step = np.maximum( max_abs_gradient_step, np.max( np.abs( gradient_step ) ) )
				max_abs_lsf_gradient_step = np.maximum( max_abs_lsf_gradient_step, np.max( np.abs( lsf_gradient_step ) ) )

				if self.has_fabrication_penalty:
					penalty_gradient, fab_penalty = self.bayer_filters[ bayer_idx ].compute_penalty()

					max_abs_fab_penalty_step = np.maximum( max_abs_fab_penalty_step, np.max( np.abs( penalty_gradient ) ) )

				bayer_idx += 1

		bayer_idx = 0
		collect_fom_gradients = []
		collect_design_cuts = []
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:
			
				backprop = self.bayer_filters[ bayer_idx ].backpropagate(
					gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
					gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ] )
				get_design_variable = self.bayer_filters[ bayer_idx ].w[ 0 ]

				assert len( get_design_variable.shape ) == 2, "Not 2-dimensional design space!"
				collect_fom_gradients.append( backprop[ :, 0 ] )
				collect_design_cuts.append( get_design_variable[ :, 0 ] )

				bayer_idx += 1
		
		flatten_fom_gradients = []
		flatten_design_cuts = []
		gradient_lengths = []

		for idx in range( 0, len( collect_fom_gradients ) ):
			flatten_fom_gradients.extend( collect_fom_gradients[ idx ] )
			flatten_design_cuts.extend( collect_design_cuts[ idx ] )
			gradient_lengths.append( len( collect_fom_gradients[ idx ] ) )

		flatten_fom_gradients = np.array( flatten_fom_gradients )
		flatten_design_cuts = np.array( flatten_design_cuts )
		
		def compute_binarization( input_variable ):
			return ( 2 / np.sqrt( len( input_variable ) ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
		def compute_binarization_gradient( input_variable ):
			return ( 4 / len( input_variable ) ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )


		starting_binarization = compute_binarization( flatten_design_cuts )

		# Desired binarization increase
		# alpha = np.minimum( self.desired_binarize_change, 1 - starting_binarization )
		# Maximum movement for each density variable
		beta = self.max_binarize_movement
		
		# For now, ignore the beta specified because we are going to ensure
		# that we get the requested binarization improvement
		beta_low = 0
		beta_high = self.max_binarize_movement#2 * self.max_binarize_movement
		projected_binarization_increase = 0

		c = flatten_fom_gradients
		dim = len(c)

		print( "Starting binarization = " + str( starting_binarization ) )

		b = compute_binarization_gradient( flatten_design_cuts )
		cur_x = np.zeros( dim )

		lower_bounds = np.zeros( len( c ) )
		upper_bounds = np.zeros( len( c ) )

		np.save( 'c.npy', c )
		np.save( 'b.npy', b )

		for idx in range( 0, len( c ) ):
			upper_bounds[ idx ] = np.maximum( np.minimum( beta, 1 - flatten_design_cuts[ idx ] ), 0 )
			lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -flatten_design_cuts[ idx ] ), 0 )

		np.save( 'lower_bounds.npy', lower_bounds )
		np.save( 'upper_bounds.npy', upper_bounds )

		max_possible_binarization_change = 0
		for idx in range( 0, len( c ) ):
			if b[ idx ] > 0:
				max_possible_binarization_change += b[ idx ] * upper_bounds[ idx ]
			else:
				max_possible_binarization_change += b[ idx ] * lower_bounds[ idx ]
		
		alpha = np.minimum( max_possible_binarization_change / 3., self.desired_binarize_change )


		def ramp( x ):
			return np.maximum( x, 0 )

		def opt_function( nu ):
			lambda_1 = ramp( nu * b - c )
			lambda_2 = c + lambda_1 - nu * b

			return -( -np.dot( lambda_1, upper_bounds ) + np.dot( lambda_2, lower_bounds ) + nu * alpha )



		tolerance = 1e-12
		# nu_bounds = np.zeros( ( 1, 2 ) )
		# nu_bounds[ 0, : ] = [ 0, np.inf ]
		# optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance, bounds=nu_bounds )
		optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance )


		nu_star = optimization_solution_nu.x
		lambda_1_star = ramp( nu_star * b - c )
		lambda_2_star = c + lambda_1_star - nu_star * b
		x_star = np.zeros( dim )

		for idx in range( 0, dim ):
			if lambda_1_star[ idx ] > 0:
				x_star[ idx ] = upper_bounds[ idx ]
			else:
				x_star[ idx ] = lower_bounds[ idx ]



		proposed_design_variable = flatten_design_cuts + x_star
		proposed_design_variable = np.minimum( np.maximum( proposed_design_variable, 0 ), 1 )

		ending_binarization = compute_binarization( proposed_design_variable )

		# import matplotlib.pyplot as plt
		# plt.plot( cur_x, color='b', linewidth=2 )
		# plt.show()
		expected_fom_change = np.dot( x_star, -c )
		print( "Expected delta = " + str( np.dot( x_star, b ) ) )
		print( "Desired delta = " + str( self.desired_binarize_change ) )
		print( "Limit on delta = " + str( max_possible_binarization_change ) )
		print( "Expected scaled FOM change = " + str( expected_fom_change ) )
		print( "Ending binarization = " + str( ending_binarization ) )
		print( "Achieved delta = " + str( ending_binarization - starting_binarization ) )

		cur_reshape_idx = 0
		for bayer_idx in range( 0, len( self.bayer_filters ) ):
			get_design_variable = self.bayer_filters[ bayer_idx ].w[ 0 ]
			new_design_variable = np.zeros( get_design_variable.shape )

			get_design_cut = proposed_design_variable[ cur_reshape_idx : ( cur_reshape_idx + gradient_lengths[ bayer_idx ] ) ]

			if iteration >= self.erode_dilate_start_iter:
				if ( ( iteration - self.erode_dilate_start_iter ) % self.erode_dilate_iter_gap ) == 0:
					binarize_rho = 1.0 * np.greater( get_design_cut, 0.5 )
					# Replace this with other perturbation insensitivity (i.e. - eroded and dilated design gradients)
					# Can remove this part
					erode = 1.0 * scipy.ndimage.morphology.binary_erosion( binarize_rho, iterations=self.num_iterations_erode_dilate )
					redilate = 1.0 * scipy.ndimage.morphology.binary_dilation( erode, iterations=self.num_iterations_erode_dilate )
					redilate_again = 1.0 * scipy.ndimage.morphology.binary_dilation( redilate, iterations=self.num_iterations_erode_dilate )
					re_erode = 1.0 * scipy.ndimage.morphology.binary_erosion( redilate_again, iterations=self.num_iterations_erode_dilate )
					

					# import matplotlib.pyplot as plt
					changed_pixels = np.greater( ( 1 - re_erode ) * binarize_rho + re_erode * ( 1 - binarize_rho ), 0.5 )

					# plt.plot( re_erode, color='g', linewidth=2 )
					# plt.plot( binarize_rho, color='b', linestyle='--', linewidth=2 )
					# plt.plot( changed_pixels, color='r', linestyle=':', linewidth=2 )
					# plt.show()

					new_design_cut = get_design_cut.copy()

					for idx in range( 0, len( new_design_cut ) ):
						if changed_pixels[ idx ]:
							if new_design_cut[ idx ] > 0.5:
								new_design_cut[ idx ] = 0.25
							else:
								new_design_cut[ idx ] = 0.75

					get_design_cut = new_design_cut

			for y_idx in range( 0, get_design_variable.shape[ 1 ] ):
				new_design_variable[ :, y_idx ] = get_design_cut
			
			cur_reshape_idx += gradient_lengths[ bayer_idx ]

			self.bayer_filters[ bayer_idx ].set_design_variable( new_design_variable )









