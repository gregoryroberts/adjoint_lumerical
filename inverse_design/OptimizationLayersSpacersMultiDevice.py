import numpy as np
import FreeBayerFilter2D
import OptimizationStateMultiDevice

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )

#
# All of the devices need to share the same density
#
class OptimizationLayersSpacersMultiDevice( OptimizationStateMultiDevice.OptimizationStateMultiDevice ):

	def __init__(
		self,
		num_iterations, num_epochs, step_size,
		device_size_um, optimization_mesh_step_um,
		layer_start_coordinates_um, layer_thicknesses_um, layer_designability, layer_background_index, simulation_background_index,
		bayer_filter_creator_fns,
		filename_prefix,
		num_devices ):
		super(OptimizationLayersSpacersMultiDevice, self).__init__( num_iterations, num_epochs, filename_prefix, num_devices )

		self.device_size_um = device_size_um
		self.optimization_mesh_step_um = optimization_mesh_step_um

		self.layer_start_coordinates_um = layer_start_coordinates_um
		self.layer_thicknesses_um = layer_thicknesses_um
		self.layer_designability = layer_designability
		self.layer_background_index = layer_background_index

		self.opt_device_size_voxels = [ 1 + int( self.device_size_um[ i ] / self.optimization_mesh_step_um ) for i in range( 0, len( self.device_size_um ) ) ]

		self.index = simulation_background_index * np.ones( self.opt_device_size_voxels, dtype=np.complex )

		self.bayer_filter_creator_fns = bayer_filter_creator_fns

		self.step_size = step_size

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

				device_bayer_filters = []
				for device_idx in range( 0, self.num_devices ):
					bayer_filter = self.bayer_filter_creator_fns[ device_idx ]( layer_size_voxels, num_internal_layers, bayer_idx )
					device_bayer_filters.append( bayer_filter )

				self.bayer_filters.append( device_bayer_filters )
				bayer_idx += 1

	def assemble_index( self, device_idx ):
		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:
				self.index[ :, layer_bottom_voxel : layer_top_voxel ] = permittivity_to_index( self.bayer_filters[ bayer_idx ][ device_idx ].get_permittivity() )
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

		for device_idx in range( 0, self.num_devices ):
			bayer_idx = 0
			for layer_idx in range( 0, self.num_total_layers ):
				layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
				layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

				if self.layer_designability[ layer_idx ]:
					self.bayer_filters[ bayer_idx ][ device_idx ].set_design_variable(
						reinterpolate_density[ :, layer_bottom_voxel : layer_top_voxel ]
					)

					bayer_idx += 1

	def convert_permittivity_to_layered_density( self, index, min_real_index, max_real_index ):
		permittivity =  ( np.real( index ) )**2
		min_real_permittivity = ( min_real_index )**2
		max_real_permittivity = ( max_real_index )**2
		density = ( permittivity - min_real_permittivity ) / ( max_real_permittivity - min_real_permittivity )

		reinterpolate_density = self.reinterpolate( density, self.index.shape )

		for device_idx in range( 0, self.num_devices ):
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
					self.bayer_filters[ bayer_idx ][ device_idx ].set_design_variable(
						average_layer_density
					)

					bayer_idx += 1

	#
	# We should only have saved one density variable
	#
	def load_other_design( self, other_filename_prefix, filebase, epoch, shift=0 ):
		for populate_design_idx in range( 0, self.num_devices ):
			for bayer_idx in range( 0, len( self.bayer_filters ) ):
				self.bayer_filters[ bayer_idx ][ populate_design_idx ].set_design_variable(
					np.load( filebase + "/" + other_filename_prefix + str( bayer_idx ) + "_" + str( epoch ) + ".npy" ) - shift )

	def load_design( self, filebase, epoch ):
		for populate_design_idx in range( 0, self.num_devices ):
			for bayer_idx in range( 0, len( self.bayer_filters ) ):
				self.bayer_filters[ bayer_idx ][ populate_design_idx ].set_design_variable(
					np.load( filebase + "/" + self.filename_prefix + str( bayer_idx ) + "_" + str( epoch ) + ".npy" ) )

	#
	# We should only save one density variable (these density variables are shared)
	#
	def save_design( self, filebase, epoch ):
		design_idx_to_save = 0
		for bayer_idx in range( 0, len( self.bayer_filters ) ):
			np.save( filebase + "/" + self.filename_prefix + str( bayer_idx ) + "_" + str( epoch ) + ".npy", self.bayer_filters[ bayer_idx ][ design_idx_to_save ].get_design_variable() )

	def update_epoch( self, epoch ):
		for design_idx in range( 0, self.num_devices ):
			for bayer_idx in range( 0, len( self.bayer_filters ) ):
				self.bayer_filters[ bayer_idx ][ design_idx ].update_filters( epoch )

	def update( self, gradients_real, gradients_imag, lsf_gradients_real, lsf_gradients_imag, epoch, iteration ):
		#
		# First, we need to know how to weight the contributions from each device.  We aren't doing this for the fabrication penalty because we need
		# to evaluate fabrication performance for each device.  But this code should replace the fabrication piece so we should probably take that
		# out of there.
		#
		performance_by_device = self.figure_of_merit[ epoch, iteration, : ]
		weighting_by_device = ( 2. / self.num_devices ) - performance_by_device**2 / np.sum( performance_by_device**2 )
		weighting_by_device = np.maximum( weighting_by_device, 0 )
		weighting_by_device /= np.sum( weighting_by_device )

		gradients_real_interpolate = []
		gradients_imag_interpolate = []

		lsf_gradients_real_interpolate = []
		lsf_gradients_imag_interpolate = []

		for device_idx in range( 0, self.num_devices ):
			gradients_real_interpolate.append(
				self.reinterpolate( np.squeeze( gradients_real[ device_idx ] ), self.opt_device_size_voxels )
			)
			gradients_imag_interpolate.append(
				self.reinterpolate( np.squeeze( gradients_imag[ device_idx ] ), self.opt_device_size_voxels )
			)

			lsf_gradients_real_interpolate.append(
				self.reinterpolate( np.squeeze( lsf_gradients_real[ device_idx ] ), self.opt_device_size_voxels )
			)
			lsf_gradients_imag_interpolate.append(
				self.reinterpolate( np.squeeze( lsf_gradients_imag[ device_idx ] ), self.opt_device_size_voxels )
			)

		max_abs_gradient_step = -1
		max_abs_lsf_gradient_step = -1

		combined_density_layer_gradients = []
		combined_lsf_layer_gradients = []

		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:

				combined_density_layer_gradient = np.zeros( self.bayer_filters[ bayer_idx ][ 0 ].size )
				combined_lsf_layer_gradient = np.zeros( self.bayer_filters[ bayer_idx ][ 0 ].size )

				for device_idx in range( 0, self.num_devices ):
					combined_density_layer_gradient += weighting_by_device[ device_idx ] * self.bayer_filters[ bayer_idx ][ device_idx ].backpropagate(
						gradients_real_interpolate[ device_idx ][ :, layer_bottom_voxel : layer_top_voxel ],
						gradients_imag_interpolate[ device_idx ][ :, layer_bottom_voxel : layer_top_voxel ] )

					combined_lsf_layer_gradient += weighting_by_device[ device_idx ] * self.bayer_filters[ bayer_idx ][ device_idx ].backpropagate(
						lsf_gradients_real_interpolate[ device_idx ][ :, layer_bottom_voxel : layer_top_voxel ],
						lsf_gradients_imag_interpolate[ device_idx ][ :, layer_bottom_voxel : layer_top_voxel ] )

				combined_density_layer_gradients.append( combined_density_layer_gradient )
				combined_lsf_layer_gradients.append( combined_lsf_layer_gradient )

				gradient_step = combined_density_layer_gradient
				lsf_gradient_step = combined_lsf_layer_gradient
				
				max_abs_gradient_step = np.maximum( max_abs_gradient_step, np.max( np.abs( gradient_step ) ) )
				max_abs_lsf_gradient_step = np.maximum( max_abs_lsf_gradient_step, np.max( np.abs( lsf_gradient_step ) ) )

				bayer_idx += 1

		#
		# We need to get the net effect on the density variables here and then use this to update.  How are the update functions handling this.
		# You may need to just do the design variable step in here.  However, we need to take into account a potential binarization forcing
		# in which case the amount of binarization is dependent on which device we are looking at (because each is related to a blur) - and getting
		# the blurred ones to be binary might be tricky? Do we just want to aim for a binarized density in the non-blurred version?  Maybe
		# we can pass all the bayer filters as well as the combined density gradient out to a single update function?  Because then it has
		# control over what happens.
		#

		#
		# Here is where we have to take into account that they share the same density variable
		# We need to backpropagate all of them and then performance weight their effect on the
		# shared density variable.
		#
		bayer_idx = 0
		for layer_idx in range( 0, self.num_total_layers ):
			layer_bottom_voxel = int( self.layer_start_coordinates_um[ layer_idx ] / self.optimization_mesh_step_um )
			layer_top_voxel = int( ( self.layer_start_coordinates_um[ layer_idx ] + self.layer_thicknesses_um[ layer_idx ] ) / self.optimization_mesh_step_um )

			if self.layer_designability[ layer_idx ]:
				proposed_design_variable = self.bayer_filters[ bayer_idx ][ 0 ].get_design_variable() - self.step_size * combined_density_layer_gradients[ bayer_idx ] / max_abs_gradient_step
				proposed_design_variable = np.minimum( np.maximum( proposed_design_variable, 0 ), 1 )

				for device_idx in range( 0, self.num_devices ):
					self.bayer_filters[ bayer_idx ][ device_idx ].set_design_variable( proposed_design_variable )

				bayer_idx += 1
