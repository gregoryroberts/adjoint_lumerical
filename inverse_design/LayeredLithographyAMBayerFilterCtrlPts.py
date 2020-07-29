import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

from scipy.ndimage import gaussian_filter

import numpy as np

from LayeredLithographyAMCtrlPtsParameters import *

class GaussianBlur():
	def __init__( self, blur_sigma ):
		self.blur_sigma = blur_sigma

	def forward( self, variable_in ):
		# return variable_in
		z_shape = variable_in.shape[ 2 ]

		variable_out = np.zeros( variable_in.shape, dtype=variable_in.dtype )

		for z_idx in range( 0, z_shape ):
			get_layer = np.squeeze( variable_in[ :, :, z_idx ] )
			# blurred_layer = 2 * np.pi * self.blur_sigma**2 * gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )
			blurred_layer = gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )

			variable_out[ :, :, z_idx ] = blurred_layer

		return np.maximum( np.minimum( variable_out, 1.0 ), 0.0 )

	def chain_rule( self, gradient_out, variable_out, variable_in ):
		# return gradient_out
		z_shape = gradient_out.shape[ 2 ]

		gradient_in = np.zeros( gradient_out.shape, dtype=gradient_out.dtype )

		for z_idx in range( 0, z_shape ):
			get_layer = np.squeeze( gradient_out[ :, :, z_idx ] )
			# blurred_layer = 2 * np.pi * self.blur_sigma**2 * gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )
			blurred_layer = gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )

			gradient_in[ :, :, z_idx ] = blurred_layer

		return gradient_in	

class LayeredLithographyAMBayerFilterCtrlPts(device.Device):

	def __init__(self, size, lateral_subsampling, feature_size_sigma, permittivity_bounds, init_permittivity, num_z_layers, spacer_height_voxels):
		super(LayeredLithographyAMBayerFilterCtrlPts, self).__init__(size, permittivity_bounds, init_permittivity)

		self.num_z_layers = num_z_layers
		self.flip_threshold = 0.5
		self.minimum_design_value = 0
		self.maximum_design_value = 1
		self.spacer_height_voxels = spacer_height_voxels
		self.layer_height_voxels = int( ( self.size[ 2 ] / self.num_z_layers ) - self.spacer_height_voxels )
		self.init_filters_and_variables()

		self.lateral_subsampling = lateral_subsampling

		# May want to step this up through optimization
		self.middle_gaussian_weight = 0.75

		self.feature_size_sigma = feature_size_sigma
		self.control_points_per_box = 5
		self.box_counts = np.zeros( 2, dtype=np.int )
		for dim in range( 0, 2 ):
			# todo: these should be errors or exceptions or some return code, not assert
			assert ( size[ dim ] % lateral_subsampling[ dim ] ) == 0, "The subsampling does not divide the size equally"
			self.box_counts[ dim ] = int( size[ dim ] / lateral_subsampling[ dim ] )

		self.init_control_boxes()
		
		self.update_permittivity()

	#
	# todo: Swapping control points maintains the same distribution.  This may affect the convergence of this and could be
	# something to consider.
	#
	def init_control_boxes( self ):
		self.control_points = np.zeros( ( self.num_z_layers, self.box_counts[ 0 ], self.box_counts[ 1 ], self.control_points_per_box, 2 ) )

		assert ( self.box_counts[ 0 ] % 2 ) == 1, "Currently expecting an odd number of boxes for symmetry reasons"
		assert ( self.box_counts[ 1 ] % 2 ) == 1, "Currently expecting an odd number of boxes for symmetry reasons"

		for layer_idx in range( 0, self.num_z_layers ):
			for xbox in range( 0, self.box_counts[ 0 ] ):
				for ybox in range( 0, self.box_counts[ 1 ] ):

					offset_x = xbox * self.lateral_subsampling[ 0 ] + int( 0.25 * self.lateral_subsampling[ 0 ] )
					offset_y = ybox * self.lateral_subsampling[ 1 ] + int( 0.25 * self.lateral_subsampling[ 1 ] )

					random_control_pts_x = np.random.random( self.control_points_per_box ) * 0.5 * self.lateral_subsampling[ 0 ]
					random_control_pts_y = np.random.random( self.control_points_per_box ) * 0.5 * self.lateral_subsampling[ 1 ]

					for pt_idx in range( 0, self.control_points_per_box ):
						self.control_points[ layer_idx, xbox, ybox, pt_idx ] = np.array( [ offset_x + random_control_pts_x[ pt_idx ], offset_y + random_control_pts_y[ pt_idx ] ] )

			symmetry_range_x = int( np.ceil( 0.5 * self.box_counts[ 0 ] ) )
			symmetry_range_y = int( np.ceil( 0.5 * self.box_counts[ 1 ] ) )

			for xbox in range( 0, self.box_counts[ 0 ] ):
				for ybox in range( 0, xbox + 1 ):
					symmetric_x = ybox
					symmetric_y = xbox
					for pt_idx in range( 0, self.control_points_per_box ):

						if xbox == ybox:
							self.control_points[ layer_idx, symmetric_x, symmetric_y, pt_idx ] = np.array( [
								self.control_points[ layer_idx, xbox, ybox, pt_idx, 0 ],
								self.control_points[ layer_idx, xbox, ybox, pt_idx, 0 ]
							] )
						else:
							self.control_points[ layer_idx, symmetric_x, symmetric_y, pt_idx ] = np.array( [
								self.control_points[ layer_idx, xbox, ybox, pt_idx, 1 ],
								self.control_points[ layer_idx, xbox, ybox, pt_idx, 0 ]
							] )


	def make_layer( self, layer_idx ):
		layer = np.zeros( ( self.size[ 0 ], self.size[ 1 ] ) )

		for x_coarse in range( 0, self.box_counts[ 0 ] ):
			fine_x_offset = self.lateral_subsampling[ 0 ] * x_coarse

			for y_coarse in range( 0, self.box_counts[ 1 ] ):
				fine_y_offset = self.lateral_subsampling[ 1 ] * y_coarse

				for step_x in range( 0, self.lateral_subsampling[ 0 ] ):
					fine_x = fine_x_offset + step_x

					for step_y in range( 0, self.lateral_subsampling[ 1 ] ):
						fine_y = fine_y_offset + step_y

						for pt_idx in range( 0, self.control_points_per_box ):
							ctrl_x = self.control_points[ layer_idx, x_coarse, y_coarse, pt_idx, 0 ]
							ctrl_y = self.control_points[ layer_idx, x_coarse, y_coarse, pt_idx, 1 ]

							layer[ fine_x, fine_y ] += self.middle_gaussian_weight * np.exp( -( 1. / ( 2. * self.feature_size_sigma**2 ) ) * ( ( ( fine_x - ctrl_x )**2 + ( fine_y - ctrl_y )**2 ) ) )

		return layer

	def layer_to_control_points_grad( self, layer_idx, gradient ):
		control_points_grad = np.zeros( ( self.box_counts[ 0 ], self.box_counts[ 1 ], self.control_points_per_box, 2 ) )

		for x_coarse in range( 0, self.box_counts[ 0 ] ):
			fine_x_offset = self.lateral_subsampling[ 0 ] * x_coarse

			for y_coarse in range( 0, self.box_counts[ 1 ] ):
				fine_y_offset = self.lateral_subsampling[ 1 ] * y_coarse

				for step_x in range( 0, self.lateral_subsampling[ 0 ] ):
					fine_x = fine_x_offset + step_x

					for step_y in range( 0, self.lateral_subsampling[ 1 ] ):
						fine_y = fine_y_offset + step_y

						for pt_idx in range( 0, self.control_points_per_box ):
							ctrl_x = self.control_points[ layer_idx, x_coarse, y_coarse, pt_idx, 0 ]
							ctrl_y = self.control_points[ layer_idx, x_coarse, y_coarse, pt_idx, 1 ]

							control_points_grad[ x_coarse, y_coarse, pt_idx, 0 ] += (
								gradient[ fine_x, fine_y ] *
								self.middle_gaussian_weight *
								np.exp( -( 1. / ( 2. * self.feature_size_sigma**2 ) ) * ( ( ( fine_x - ctrl_x )**2 + ( fine_y - ctrl_y )**2 ) ) ) *
								( 1. / self.feature_size_sigma**2 ) *
								( fine_x - ctrl_x )
							)

							control_points_grad[ x_coarse, y_coarse, pt_idx, 1 ] += (
								gradient[ fine_x, fine_y ] *
								self.middle_gaussian_weight *
								np.exp( -( 1. / ( 2. * self.feature_size_sigma**2 ) ) * ( ( ( fine_x - ctrl_x )**2 + ( fine_y - ctrl_y )**2 ) ) ) *
								( 1. / self.feature_size_sigma**2 ) *
								( fine_y - ctrl_y )
							)

		return control_points_grad

	#
	# Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
	#
	def update_permittivity( self ):
		background_density = np.zeros( self.size )

		layer_indices = self.layering_z.get_layer_idxs( self.size )

		for layer_idx in range( 0, self.num_z_layers ):
			layer_start = layer_indices[ layer_idx ]
			layer_end = np.minimum( self.size[ 2 ] - self.spacer_height_voxels, layer_start + self.layer_height_voxels )

			create_layer = self.make_layer( layer_idx )

			for internal_idx in range( layer_start, layer_end ):
				background_density[ :, :, internal_idx ] = create_layer


		sigmoid_density = self.sigmoid_1.forward( background_density )
		scale_density = self.scale_2.forward( sigmoid_density )

		self.w[ 1 ] = background_density
		self.w[ 2 ] = sigmoid_density
		self.w[ 3 ] = scale_density


	#
	# Need to also override the backpropagation function
	#
	def backpropagate( self, gradient ):
		gradient = self.scale_2.chain_rule( gradient, self.w[ 3 ], self.w[ 2 ] )
		gradient = self.sigmoid_1.chain_rule( gradient, self.w[ 2 ], self.w[ 1 ] )

		layer_indices = self.layering_z.get_layer_idxs( self.size )

		average_gradient_layers = np.zeros( ( self.size[ 0 ], self.size[ 1 ], self.num_z_layers ), dtype=gradient.dtype )

		control_points_gradient = np.zeros( ( self.num_z_layers, self.box_counts[ 0 ], self.box_counts[ 1 ], self.control_points_per_box, 2 ) )

		for layer_idx in range( 0, self.num_z_layers ):
			layer_start = layer_indices[ layer_idx ]
			layer_end = np.minimum( self.size[ 2 ] - self.spacer_height_voxels, layer_start + self.layer_height_voxels )

			average_gradient_layer = np.zeros( ( self.size[ 0 ], self.size[ 1 ] ), dtype=gradient.dtype )
			num_voxels_averaged = layer_end - layer_start
			averaging_normalization = 1. / num_voxels_averaged

			for internal_idx in range( layer_start, layer_end ):
				average_gradient_layers[ :, :, layer_idx ] += ( averaging_normalization * gradient[ :, :, internal_idx ] )

			control_points_gradient[ layer_idx ] = self.layer_to_control_points_grad( layer_idx, average_gradient_layers[ :, :, layer_idx ] )

		return control_points_gradient

	def update_filters(self, epoch):
		self.sigmoid_beta = 0.0625 * ( 2**epoch )

		self.sigmoid_1 = sigmoid.Sigmoid( self.sigmoid_beta, self.sigmoid_eta )

	def init_filters_and_variables(self):
		self.num_filters = 3
		self.num_variables = 1 + self.num_filters

		# Start the sigmoids at weak strengths
		self.sigmoid_beta = 0.0625
		self.sigmoid_eta = 0.5
		self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		z_voxel_layers = self.size[2]
		self.layering_z = layering.Layering( z_dimension_idx, self.num_z_layers, [ 0, 1 ], self.spacer_height_voxels, 0.0 )

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_2 = scale.Scale([scale_min, scale_max])

		self.init_variables()

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient, step_size):
		control_points_gradient = self.backpropagate( gradient )
		proposed_step = self.control_points - step_size * control_points_gradient

		for layer_idx in range( 0, self.num_z_layers ):
			for x_coarse in range( 0, self.box_counts[ 0 ] ):
				x_bound_low = self.lateral_subsampling[ 0 ] * ( x_coarse + 0.25 )
				x_bound_high = self.lateral_subsampling[ 0 ] * ( x_coarse + 0.75 )

				for y_coarse in range( 0, self.box_counts[ 1 ] ):
					y_bound_low = self.lateral_subsampling[ 1 ] * ( y_coarse + 0.25 )
					y_bound_high = self.lateral_subsampling[ 1 ] * ( y_coarse + 0.75 )

					for pt_idx in range( 0, self.control_points_per_box ):
						proposed_step[ layer_idx, x_coarse, y_coarse, pt_idx, 0 ] = np.minimum(
							x_bound_high,
							np.maximum(
								proposed_step[ layer_idx, x_coarse, y_coarse, pt_idx, 0 ],
								x_bound_low )
							)

						proposed_step[ layer_idx, x_coarse, y_coarse, pt_idx, 1 ] = np.minimum(
							y_bound_high,
							np.maximum(
								proposed_step[ layer_idx, x_coarse, y_coarse, pt_idx, 1 ],
								y_bound_low )
							)

		self.control_points = proposed_step.copy()

		# self.w[0] = self.proposed_design_step(gradient, step_size)
		# Update the variable stack including getting the permittivity at the w[-1] position
		self.update_permittivity()

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
