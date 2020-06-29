import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
# import square_blur as square_blur

import scipy.optimize

import numpy as np

from LayeredLithographyIRParameters import *

class LayeredLithographyIRBayerFilter(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers, spacer_height_voxels, last_layer_permittivity, max_binarize_movement, desired_binarize_change):
		super(LayeredLithographyIRBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

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

		self.update_permittivity()


	#
	# Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
	#
	def update_permittivity(self):
		var0 = self.w[0]

		var1 = self.layering_z_0.forward(var0)
		self.w[1] = var1

		# var2 = self.max_blur_xy_1.forward(var1)
		# self.w[2] = var2

		var2 = self.scale_1.forward(var1)
		get_last_layer_var2 = var1[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ]
		var2[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ] = (
			self.last_layer_permittivity[ 0 ] + ( self.last_layer_permittivity[ 1 ] - self.last_layer_permittivity[ 0 ] ) * get_last_layer_var2
		)
		var2[ :, :, self.layering_z_0.last_layer_end : var2.shape[ 2 ] ] = self.last_layer_permittivity[ 0 ]

		self.w[2] = var2


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient):
		get_last_layer_gradient = gradient[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ]

		gradient = self.scale_1.chain_rule(gradient, self.w[2], self.w[1])
		gradient[ :, :, self.layering_z_0.last_layer_start : self.layering_z_0.last_layer_end ] = ( self.last_layer_permittivity[ 1 ] - self.last_layer_permittivity[ 0 ] ) * get_last_layer_gradient

		gradient = self.layering_z_0.chain_rule(gradient, self.w[1], self.w[0])

		return gradient

	def update_filters(self, epoch):
		return

	def init_filters_and_variables(self):
		self.num_filters = 2
		self.num_variables = 1 + self.num_filters

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		z_voxel_layers = self.size[2]
		self.layering_z_0 = layering.Layering(z_dimension_idx, self.num_z_layers, [0, 1], self.spacer_height_voxels, 0.0)

		# alpha = 8
		# self.blur_half_width = blur_half_width_voxels
		#
		# This notation is slightly confusing, but it is meant to be the
		# direction you blur when you are on the layer corresponding to x-
		# or y-layering.  So, if you are layering in x, then you blur in y
		# and vice versa.
		#
		# self.max_blur_xy_1 = square_blur.SquareBlur(
		# 	alpha,
		# 	[self.blur_half_width, self.blur_half_width, 0])

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_1 = scale.Scale([scale_min, scale_max])

		# Initialize the filter chain
		self.filters = [self.layering_z_0, self.scale_1]

		self.init_variables()

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient, step_size, enforce_binarization=False, save_location=None, enforce_xy_symmetry=True):
		if enforce_binarization:

			#
			# todo: consider a differently shaped binarization function here
			def compute_binarization( input_variable ):
				total_shape = np.product( input_variable.shape )
				return ( 2 / np.sqrt( total_shape ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
			def compute_binarization_gradient( input_variable ):
				total_shape = np.product( input_variable.shape )
				return ( 4 / total_shape ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )

			#
			# This is after the feature size blurring
			#
			density_for_binarizing = np.real( self.w[0] )

			get_binarization_gradient = compute_binarization_gradient( density_for_binarizing )
			backprop_photonic_gradient = self.backpropagate( gradient )

			extract_layers = []
			extract_photonic_gradient = []
			extract_binarization_gradient = []
			# extract_w2_init = []
			extract_beta = []
			layer_lengths = []
			original_shapes = []

			layer_start_idxs = self.layering_z_0.get_layer_idxs( self.w[0].shape )

			for layer_start_idx in range( 0, len( layer_start_idxs ) ):
				layer_start = layer_start_idxs[ layer_start_idx ]

				extract = np.real( self.w[0][ :, :, layer_start ].flatten() )
				extract_layers.extend( extract )
				layer_lengths.append( len( extract ) )

				original_shapes.append( self.w[0][ :, :, layer_start ].shape )

				extract_photonic_gradient.extend( np.real( backprop_photonic_gradient[ :, :, layer_start ].flatten() ) )
				extract_binarization_gradient.extend( np.real( get_binarization_gradient[ :, :, layer_start ].flatten() ) )


			flatten_design_cuts = np.real( extract_layers )
			flatten_fom_gradients = np.real( extract_photonic_gradient )

			beta = self.max_binarize_movement
			print( 'beta factor = ' + str( beta ) )
			projected_binarization_increase = 0

			c = flatten_fom_gradients
			dim = len(c)

			initial_binarization = compute_binarization( np.array( flatten_design_cuts ) )
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

			if enforce_xy_symmetry:
				# Note: with this way of enforcing the symmetry, if the two symmetric voxels are attempting to move in opposite directions, then
				# this will not move that voxel.  We can revisit this if it seems to be a problem.  The changes may very likely come out symmetric
				# anyway.
				reassemble = 0.5 * ( reassemble + np.swapaxes( reassemble, 0, 1 ) )

			self.w[0] = reassemble
			self.update_permittivity()

			extract_layers_final = []

			layer_start_idxs = self.layering_z_0.get_layer_idxs( self.w[0].shape )
			for layer_start_idx in range( 0, len( layer_start_idxs ) ):
				layer_start = layer_start_idxs[ layer_start_idx ]
				extract_layers_final.extend( np.real( self.w[0][ :, :, layer_start ].flatten() ) )

			final_binarization = compute_binarization( np.array( extract_layers_final ) )

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
			# x=y symmetry should be preserved here, but since I'm not sure what to conclude for the binarization part, I need
			# to make sure an x=y symmetric change gets made to the structure (it may happen automatically with a symmetric 
			# gradient and a symmetric device leading to a symmetric binarization gradient)
			if enforce_xy_symmetry:
				gradient = 0.5 * ( gradient + np.swapaxes( gradient, 0, 1 ) )
			self.w[0] = self.proposed_design_step(gradient, step_size)
			# Update the variable stack including getting the permittivity at the w[-1] position
			self.update_permittivity()

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
