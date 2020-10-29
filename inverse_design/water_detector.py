import numpy as np
import matplotlib.pyplot as plt
from LevelSet import LevelSet
import OptimizationState
from scipy import ndimage


def upsample_nearest( density, upsample_shape ):
	print( density.shape )
	print( upsample_shape )
	assert ( upsample_shape[ 0 ] % density.shape[ 0 ] ) == 0, "Expected an even nearest upsampling"
	assert ( upsample_shape[ 1 ] % density.shape[ 1 ] ) == 0, "Expected an even nearest upsampling"

	upsample_ratio = 1.0 * np.array( upsample_shape ) / np.array( density.shape )

	upsampled = np.zeros( upsample_shape )

	for x in range( 0, upsample_shape[ 0 ] ):
		down_x = np.minimum( int( x / upsample_ratio[ 0 ] ), density.shape[ 0 ] - 1 )
		for y in range( 0, upsample_shape[ 1 ] ):
			down_y = np.minimum( int( y / upsample_ratio[ 1 ] ), density.shape[ 1 ] - 1 )

			upsampled[ x, y ] = density[ down_x, down_y ]

	return upsampled

# def downsample_nearest( profile, downsampled_length ):
# 	cur_length = len( profile )

# 	assert ( cur_length % downsampled_length ) == 0, "Expected an even nearest downsampling"

# 	downsample_ratio = int( cur_length / downsampled_length )

# 	downsampled_profile = np.zeros( downsampled_length )

# 	for idx in range( 0, downsampled_length ):
# 		up_idx = int( idx * downsample_ratio )

# 		downsampled_profile[ idx ] = profile[ up_idx ]

# 	return downsampled_profile


def downsample_average( density, downsample_shape ):
	print( density.shape[ 0 ] )
	print( downsample_shape[ 0 ] )
	print( density.shape[ 1 ] )
	print( downsample_shape[ 1 ] )

	assert ( density.shape[ 0 ] % downsample_shape[ 0 ] ) == 0, "Expected an even average downsampling"
	assert ( density.shape[ 1 ] % downsample_shape[ 1 ] ) == 0, "Expected an even average downsampling"

	downsample_ratio = 1.0 * np.array( density.shape ) / np.array( downsample_shape )

	downsampled = np.zeros( downsample_shape )

	for x in range( 0, downsample_shape[ 0 ] ):
		start_x = int( x * downsample_ratio )
		end_x = int( start_profile_idx + downsample_ratio )

		for y in range( 0, downsample_shape[ 1 ] ):
			start_y = int( y * downsample_ratio )
			end_y = int( start_profile_idx + downsample_ratio )

			downsampled[ x, y ] = np.mean(
				density[ start_x : end_x, start_y : end_y ]
			)

	return downsampled



class WaterDetector( OptimizationState.OptimizationState ):

	#
	# TODO: Is it bad to have a default value of 1.0 for background density?
	#
	def __init__(
		self,
		permittivity_bounds,
		opt_mesh_size_um, opt_height_um, opt_width_um,
		num_iterations, num_epochs, filename_prefix, device_background_density ):

		super( WaterDetector, self ).__init__( num_iterations, num_epochs, filename_prefix )

		self.permittivity_bounds = permittivity_bounds
		self.opt_mesh_size_um = opt_mesh_size_um

		self.num_devices = 1

		self.opt_width_um = opt_width_um
		self.opt_height_um = opt_width_um

		self.opt_height_num_voxels = int( opt_height_um / self.opt_mesh_size_um )
		self.opt_width_num_voxels = int( self.opt_width_um / self.opt_mesh_size_um )

		self.design_height = int( self.opt_height_num_voxels / 5.0 )
		self.design_width = int( self.opt_width_num_voxels / 5.0 )
		# self.design_height = int( self.opt_height_num_voxels / 4.0 )
		# self.design_width = int( self.opt_width_num_voxels / 4.0 )

		self.cur_density = np.zeros( ( self.design_width, self.design_height ) )

		self.device_background_density = device_background_density

	def init_uniform( self, uniform_density ):
		self.cur_density = uniform_density * np.ones( self.cur_density.shape )

	def assemble_index( self, device_idx=1 ):
		upsampled_density = upsample_nearest( self.cur_density, np.array( [ self.opt_width_num_voxels, self.opt_height_num_voxels ] ) )

		return np.sqrt(
					self.permittivity_bounds[ 0 ] +
					( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * upsampled_density )


	def update( self, gradient_real, graident_imag, gradient_real_lsf, gradient_imag_lsf, epoch, iteration ):
		# gradient_real_interpolate = self.reinterpolate( np.squeeze( gradient_real ), [ self.design_width, self.design_height ] )
		# gradient_real_interpolate = ( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * gradient_real_interpolate

		gradient_real_interpolate = np.squeeze( gradient_real )
		gradient_real_interpolate = 0.25 * ( 
			gradient_real_interpolate[ 0 : gradient_real_interpolate.shape[ 0 ] - 1, 0 : gradient_real_interpolate.shape[ 1 ] - 1 ] +
			gradient_real_interpolate[ 1 : gradient_real_interpolate.shape[ 0 ], 0 : gradient_real_interpolate.shape[ 1 ] - 1 ] +
			gradient_real_interpolate[ 0 : gradient_real_interpolate.shape[ 0 ] - 1, 1 : gradient_real_interpolate.shape[ 1 ] ] +
			gradient_real_interpolate[ 1 : gradient_real_interpolate.shape[ 0 ], 1 : gradient_real_interpolate.shape[ 1 ] ]
		)

		gradient_real_interpolate = downsample_average( gradient_real_interpolate, [ self.design_width, self.design_height ] )
		gradient_real_interpolate = ( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * gradient_real_interpolate

		scaled_gradient = gradient_real_interpolate / np.max( np.abs( gradient_real_interpolate ) )
		scaled_step_size = 0.025
		# scaled_step_size = 0.01

		self.cur_density -= scaled_step_size * scaled_gradient
		self.cur_density = np.minimum( 1.0, np.maximum( 0.0, self.cur_density ) )

	def save_design( self, filebase, epoch ):
		return





