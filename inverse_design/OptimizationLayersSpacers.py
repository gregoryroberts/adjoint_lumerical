import numpy as np
import FreeBayerFilter2D
import OptimizationState

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )

class OptimizationLayersSpacers( OptimizationState.OptimizationState ):

	def __init__(
		self,
		num_iterations, num_epochs, step_size,
		device_size_um, optimization_mesh_step_um,
		layer_start_coordinates_um, layer_thicknesses_um, layer_designability, layer_background_index, simulation_background_index,
		bayer_filter_creator_fn,
		bayer_filter_update_fn,
		filename_prefix,
		has_fabrication_penalty=False ):

		super(OptimizationLayersSpacers, self).__init__( num_iterations, num_epochs, filename_prefix )

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

		self.step_size = step_size

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
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:

				if self.has_fabrication_penalty:
					self.bayer_filter_update_fn(
						self.bayer_filters[ bayer_idx ],
						gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						lsf_gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						lsf_gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						epoch, iteration,
						max_abs_gradient_step, max_abs_lsf_gradient_step, max_abs_fab_penalty_step,
						self.step_size )
				else:
					self.bayer_filter_update_fn(
						self.bayer_filters[ bayer_idx ],
						gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						lsf_gradient_real_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						lsf_gradient_imag_interpolate[ :, layer_bottom_voxel : layer_top_voxel ],
						epoch, iteration,
						max_abs_gradient_step, max_abs_lsf_gradient_step, None,
						self.step_size )

				bayer_idx += 1
