import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

import scipy.optimize

import numpy as np

from LayeredLithographyIRPolarizationRotationParameters import *

def bilinear_interpolation_rotate( input, theta_radians ):
	def remap_x( x_coord, y_coord ):
		return ( x_coord * np.cos( theta_radians ) + y_coord * np.sin( theta_radians ) )
	def remap_y( x_coord, y_coord ):
		return ( -x_coord * np.sin( theta_radians ) + y_coord * np.cos( theta_radians ) )

	output = np.zeros( input.shape )

	input_shape = input.shape
	mid_pt_x = int( 0.5 * input_shape[ 0 ] )
	mid_pt_y = int( 0.5 * input_shape[ 1 ] )
	for x_idx in range( 0, input_shape[ 0 ] ):
		offset_x = x_idx - mid_pt_x
		for y_idx in range( 0, input_shape[ 1 ] ):
			offset_y = y_idx - mid_pt_y

			remapped_x = remap_x( offset_x, offset_y )
			remapped_y = remap_y( offset_x, offset_y )

			reoffset_x = remapped_x + mid_pt_x
			reoffset_y = remapped_y + mid_pt_y

			if ( reoffset_x >= input_shape[ 0 ] or reoffset_x < 0 ) or ( reoffset_y >= input_shape[ 1 ] or reoffset_y < 0 ):
				continue

			lower_x = np.maximum( int( reoffset_x ), 0 )
			upper_x = np.minimum( int( reoffset_x + 1 ), input_shape[ 0 ] - 1 )
			weight_high_x = reoffset_x - lower_x
			weight_low_x = 1 - weight_high_x

			lower_y = np.maximum( int( reoffset_y ), 0 )
			upper_y = np.minimum( int( reoffset_y + 1 ), input_shape[ 1 ] - 1 )
			weight_high_y = reoffset_y - lower_y
			weight_low_y = 1 - weight_high_y

			interpolate_x_low = weight_low_x * input[ lower_x, lower_y ] + weight_high_x * input[ upper_x, lower_y ]
			interpolate_x_high = weight_low_x * input[ lower_x, upper_y ] + weight_high_x * input[ upper_x, upper_y ]
			output[ x_idx, y_idx ] = weight_low_y * interpolate_x_low + weight_high_y * interpolate_x_high

	return output

class LayeredLithographyIRPolarizationRotationDevice(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers, spacer_height_voxels, last_layer_permittivity, max_binarize_movement, desired_binarize_change):
		super(LayeredLithographyIRPolarizationRotationDevice, self).__init__(size, permittivity_bounds, init_permittivity)

		self.num_z_layers = num_z_layers
		self.flip_threshold = 0.5
		self.minimum_design_value = 0
		self.maximum_design_value = 1
		self.spacer_height_voxels = spacer_height_voxels
		self.layer_height_voxels = int( ( self.size[ 2 ] / self.num_z_layers ) - self.spacer_height_voxels )
		self.init_filters_and_variables()
		self.last_layer_permittivity = last_layer_permittivity

		self.max_binarize_movement = max_binarize_movement
		self.desired_binarize_change = desired_binarize_change

		self.circle_dimension = ( size[ 0 ] / 2. ) - 1

		self.cut_circle = np.zeros( ( size[ 0 ], size[ 1 ] ) )
		self.mid_pt_x = int( 0.5 * size[ 0 ] )
		self.mid_pt_y = int( 0.5 * size[ 1 ] )
		for x_idx in range( 0, size[ 0 ] ):
			for y_idx in range( 0, size[ 1 ] ):
				r_sq = ( x_idx - self.mid_pt_x )**2 + ( y_idx - self.mid_pt_y )**2
				if r_sq < self.circle_dimension**2:
					self.cut_circle[ x_idx, y_idx ] = 1

		self.update_permittivity()


	#
	# Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
	#
	def update_permittivity(self):
		var0 = self.w[0]

		cut_and_rotate_var0 = var0.copy()
		density_height = cut_and_rotate_var0.shape[ 2 ]
		for z_idx in range( 0, density_height ):
			cut_and_rotate_var0[ :, :, z_idx ] *= self.cut_circle
			cut_and_rotate_var0[ :, :, z_idx ] = ( 1 / 3. ) * (
				cut_and_rotate_var0[ :, :, z_idx ] +
				bilinear_interpolation_rotate( cut_and_rotate_var0[ :, :, z_idx ], 2 * np.pi / 3. ),
				bilinear_interpolation_rotate( cut_and_rotate_var0[ :, :, z_idx ], 2 * 2 * np.pi / 3. )
			)

		var1 = self.layering_z_0.forward(cut_and_rotate_var0)
		self.w[1] = var1

		var2 = self.max_blur_xy_1.forward(var1)
		self.w[2] = var2

		var3 = self.scale_2.forward(var2)
		get_last_layer_var2 = var2[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ]
		var3[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ] = (
			self.last_layer_permittivity[ 0 ] + ( self.last_layer_permittivity[ 1 ] - self.last_layer_permittivity[ 0 ] ) * get_last_layer_var2
		)
		var3[ :, :, self.layering_z_0.last_layer_end : var3.shape[ 2 ] ] = self.last_layer_permittivity[ 0 ]

		self.w[3] = var3


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient):
		get_last_layer_gradient = gradient[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ]

		gradient = self.scale_2.chain_rule(gradient, self.w[3], self.w[2])
		gradient[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ] = ( self.last_layer_permittivity[ 1 ] - self.last_layer_permittivity[ 0 ] ) * get_last_layer_gradient

		gradient = self.max_blur_xy_1.chain_rule(gradient, self.w[2], self.w[1])
		gradient = self.layering_z_0.chain_rule(gradient, self.w[1], self.w[0])

		density_height = self.w[0].shape[ 2 ]
		for z_idx in range( 0, density_height ):
			gradient[ :, :, z_idx ] *= self.cut_circle
			gradient[ :, :, z_idx ] = ( 1 / 3. ) * (
				gradient[ :, :, z_idx ] +
				bilinear_interpolation_rotate( gradient[ :, :, z_idx ], 2 * np.pi / 3. ),
				bilinear_interpolation_rotate( gradient[ :, :, z_idx ], 2 * 2 * np.pi / 3. )
			)

		return gradient

	def update_filters(self, epoch):
		return

	def init_filters_and_variables(self):
		self.num_filters = 3
		self.num_variables = 1 + self.num_filters

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		z_voxel_layers = self.size[2]
		self.layering_z_0 = layering.Layering(z_dimension_idx, self.num_z_layers, [0, 1], self.spacer_height_voxels, 0.0)

		alpha = 8
		self.blur_half_width = blur_half_width_voxels
		#
		# This notation is slightly confusing, but it is meant to be the
		# direction you blur when you are on the layer corresponding to x-
		# or y-layering.  So, if you are layering in x, then you blur in y
		# and vice versa.
		#
		self.max_blur_xy_1 = square_blur.SquareBlur(
			alpha,
			[self.blur_half_width, self.blur_half_width, 0])

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_2 = scale.Scale([scale_min, scale_max])

		# Initialize the filter chain
		self.filters = [self.layering_z_0, self.max_blur_xy_1, self.scale_2]

		self.init_variables()

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient, step_size, enforce_binarization=False, save_location=None):
		if enforce_binarization:

			def compute_binarization( input_variable ):
				total_shape = np.product( input_variable.shape )
				return ( 2 / np.sqrt( total_shape ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
			def compute_binarization_gradient( input_variable ):
				total_shape = np.product( input_variable.shape )
				return ( 4 / np.sqrt( total_shape ) ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )

			#
			# This is after the feature size blurring
			#
			density_for_binarizing = np.real( self.w[2] )


			get_binarization_gradient = compute_binarization_gradient( density_for_binarizing )
			backprop_binarization_gradient = self.max_blur_xy_1.chain_rule(get_binarization_gradient, self.w[2], self.w[1])
			backprop_binarization_gradient = self.layering_z_0.chain_rule(backprop_binarization_gradient, self.w[1], self.w[0])

			backprop_beta = self.max_blur_xy_1.chain_rule( self.max_binarize_movement * np.ones( self.w[0].shape ), self.w[2], self.w[1] )
			backprop_beta = self.layering_z_0.chain_rule( backprop_beta, self.w[1], self.w[0] )

			layer_start_idxs = self.layering_z_0.get_layer_idxs( self.w[0].shape )

			backprop_photonic_gradient = self.backpropagate( gradient )

			extract_layers = []
			extract_photonic_gradient = []
			extract_binarization_gradient = []
			extract_w2_init = []
			extract_beta = []
			layer_lengths = []
			original_shapes = []

			import matplotlib.pyplot as plt

			for layer_start_idx in range( 0, len( layer_start_idxs ) ):
				layer_start = layer_start_idxs[ layer_start_idx ]

				extract = np.real( self.w[0][ :, :, layer_start ].flatten() )
				extract_layers.extend( extract )
				layer_lengths.append( len( extract ) )

				original_shapes.append( self.w[0][ :, :, layer_start ].shape )

				extract_photonic_gradient.extend( np.real( backprop_photonic_gradient[ :, :, layer_start ].flatten() ) )
				extract_binarization_gradient.extend( np.real( backprop_binarization_gradient[ :, :, layer_start ].flatten() ) )

				extract_w2_init.extend( np.real( self.w[2][ :, :, layer_start ].flatten() ) )

				extract_beta.extend( np.real( backprop_beta[ :, :, layer_start ].flatten() ) )

			beta_factor = 1#np.mean( np.abs( extract_beta ) )

			flatten_design_cuts = np.real( extract_layers )
			flatten_fom_gradients = np.real( extract_photonic_gradient )

			beta = self.max_binarize_movement / beta_factor
			print( 'beta factor = ' + str( beta ) )
			projected_binarization_increase = 0

			c = flatten_fom_gradients
			dim = len(c)

			initial_binarization = compute_binarization( np.array( extract_w2_init ) )
			print( "Starting binarization = " + str( initial_binarization ) )

			b = np.real( extract_binarization_gradient )
			cur_x = np.zeros( dim )

			lower_bounds = np.zeros( len( c ) )
			upper_bounds = np.zeros( len( c ) )

			np.save( save_location + '/c.npy', c )
			np.save( save_location + '/b.npy', b )

			for idx in range( 0, len( c ) ):
				upper_bounds[ idx ] = np.maximum( np.minimum( beta, 1 - flatten_design_cuts[ idx ] ), 0 )
				lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -flatten_design_cuts[ idx ] ), 0 )

			# plt.plot( extract_beta, color='r' )
			# plt.plot( upper_bounds, color='g', linestyle='--' )
			# plt.plot( lower_bounds, color='b', linestyle='--' )
			# plt.show()

			# import matplotlib.pyplot as plt
			# plt.plot( extract_beta, color='g', linewidth=2 )
			# plt.plot( beta * np.ones( len( extract_beta ) ), color='b', linewidth=2, linestyle='--' )
			# plt.show()

			np.save( save_location + '/lower_bounds.npy', lower_bounds )
			np.save( save_location + '/upper_bounds.npy', upper_bounds )


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

			reassemble = self.w[0].copy()
			layer_start_idxs = self.layering_z_0.get_layer_idxs( self.w[0].shape )
			layer_start_idxs.append( self.w[0].shape[2] )
			cur_location = 0
			for layer_start_idx in range( 0, len( layer_start_idxs ) - 1 ):
				layer_start = layer_start_idxs[ layer_start_idx ]
				layer_end = layer_start_idxs[ layer_start_idx + 1 ] - self.layering_z_0.spacer_height_voxels
				
				pull_design = proposed_design_variable[ cur_location : ( cur_location + layer_lengths[ layer_start_idx ] ) ]
				cur_location += layer_lengths[ layer_start_idx ]
				for internal in range( layer_start, layer_end ):
					reassemble[ :, :, internal ] = pull_design.reshape( original_shapes[ layer_start_idx ] )

			self.w[0] = reassemble
			self.update_permittivity()

			extract_w2_final = []

			layer_start_idxs = self.layering_z_0.get_layer_idxs( self.w[0].shape )
			for layer_start_idx in range( 0, len( layer_start_idxs ) ):
				layer_start = layer_start_idxs[ layer_start_idx ]
				extract_w2_final.extend( np.real( self.w[2][ :, :, layer_start ].flatten() ) )

			final_binarization = compute_binarization( np.array( extract_w2_final ) )

			print( "Ending binarization = " + str( final_binarization ) )

			expected_binarization_change = np.dot( x_star, b )
			actual_binarization_change = final_binarization - initial_binarization

			if actual_binarization_change < 0:
				np.save( save_location + '/fom_gradients_debug.npy', c )
				np.save( save_location + '/binarization_gradients_debug.npy', b )
				np.save( save_location + '/upper_bounds_debug.npy', upper_bounds )
				np.save( save_location + '/lower_bounds_debug.npy', lower_bounds )
				np.save( save_location + '/beta_debug.npy', beta )

			expected_fom_change = np.dot( x_star, -c )
			print( "Expected delta = " + str( np.dot( x_star, b ) ) )
			print( "Desired delta = " + str( self.desired_binarize_change ) )
			print( "Limit on delta = " + str( max_possible_binarization_change ) )
			print( "Expected scaled FOM change = " + str( expected_fom_change ) )
			print( "Achieved binarization delta = " + str( actual_binarization_change ) )

		else:
			self.w[0] = self.proposed_design_step(gradient, step_size)
			# Update the variable stack including getting the permittivity at the w[-1] position
			self.update_permittivity()

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
