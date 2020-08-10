import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

from scipy.ndimage import gaussian_filter

import numpy as np

from LayeredLithographyAMPostprocessParameters import *

# X=Y SYMMETRY!!! HERE AND IN CONTROL POINTS! - IF YOU ARE DOING RANDOM INITIALIZATIONS - put into optimization code for this
# one and control points will put into the filter initilialization

class GaussianBlur():
	def __init__( self, blur_sigma ):
		self.blur_sigma = blur_sigma

	def forward( self, variable_in ):
		if self.blur_sigma == 0:
			variable_out = variable_in.copy()
		else:

			# return variable_in
			z_shape = variable_in.shape[ 2 ]

			variable_out = np.zeros( variable_in.shape, dtype=variable_in.dtype )

			for z_idx in range( 0, z_shape ):
				get_layer = np.squeeze( variable_in[ :, :, z_idx ] )
				blurred_layer = 0.5 * 2 * np.pi * self.blur_sigma**2 * gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )
				# blurred_layer = gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )

				variable_out[ :, :, z_idx ] = blurred_layer

		return np.maximum( np.minimum( variable_out, 1.0 ), 0.0 )

	def chain_rule( self, gradient_out, variable_out, variable_in ):
		if self.blur_sigma == 0:
			return gradient_out
		# return gradient_out
		z_shape = gradient_out.shape[ 2 ]

		gradient_in = np.zeros( gradient_out.shape, dtype=gradient_out.dtype )

		for z_idx in range( 0, z_shape ):
			get_layer = np.squeeze( gradient_out[ :, :, z_idx ] )
			blurred_layer = 0.5 * 2 * np.pi * self.blur_sigma**2 * gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )
			# blurred_layer = gaussian_filter( np.real( get_layer ), sigma=self.blur_sigma )

			gradient_in[ :, :, z_idx ] = blurred_layer

		return gradient_in	

class LayeredLithographyAMBayerFilter(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers, spacer_height_voxels, last_layer_permittivity, gaussian_blur_sigma=gaussian_blur_filter_sigma ):
		super(LayeredLithographyAMBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

		self.num_z_layers = num_z_layers
		self.flip_threshold = 0.5
		self.minimum_design_value = 0
		self.maximum_design_value = 1
		self.spacer_height_voxels = spacer_height_voxels
		self.layer_height_voxels = int( ( self.size[ 2 ] / self.num_z_layers ) - self.spacer_height_voxels )
		self.gaussian_blur_sigma = gaussian_blur_sigma
		self.init_filters_and_variables()
		self.last_layer_permittivity = last_layer_permittivity

		self.update_permittivity()


	#
	# Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
	#
	def update_permittivity(self):
		var0 = self.w[0]

		var1 = self.sigmoid_0.forward(var0)
		self.w[1] = var1

		var2 = self.max_blur_xy_1.forward(var1)
		self.w[2] = var2

		var3 = self.sigmoid_2.forward(var2)
		self.w[3] = var3

		var4 = self.gaussian_blur_3.forward(var3)
		self.w[4] = var4

		var4 = 1.0 * np.greater_equal( var4, 0.5 )

		var5 = self.layering_z_4.forward(var4)
		self.w[5] = var5

		var6 = self.scale_5.forward(var5)

		get_last_layer_var5 = var5[ :, :, self.layering_z_4.last_layer_start : self.layering_z_4.last_layer_end ]
		var6[ :, :, self.layering_z_4.last_layer_start : self.layering_z_4.last_layer_end ] = (
			self.last_layer_permittivity[ 0 ] + ( self.last_layer_permittivity[ 1 ] - self.last_layer_permittivity[ 0 ] ) * get_last_layer_var5
		)
		var6[ :, :, self.layering_z_4.last_layer_end : var6.shape[ 2 ] ] = self.last_layer_permittivity[ 0 ]

		self.w[6] = var6


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient):
		get_last_layer_gradient = gradient[ :, :, self.layering_z_4.last_layer_start : self.layering_z_4.last_layer_end ]
		gradient = self.scale_5.chain_rule(gradient, self.w[6], self.w[5])
		gradient[ :, :, self.layering_z_4.last_layer_start : self.layering_z_4.last_layer_end ] = (
			( self.last_layer_permittivity[ 1 ] - self.last_layer_permittivity[ 0 ] ) * get_last_layer_gradient )

		gradient = self.layering_z_4.chain_rule(gradient, self.w[5], self.w[4])
		gradient = self.gaussian_blur_3.chain_rule(gradient, self.w[4], self.w[3])
		gradient = self.sigmoid_2.chain_rule(gradient, self.w[3], self.w[2])
		gradient = self.max_blur_xy_1.chain_rule(gradient, self.w[2], self.w[1])
		gradient = self.sigmoid_0.chain_rule(gradient, self.w[1], self.w[0])

		return gradient

	def update_filters(self, epoch):
		self.sigmoid_beta = 0.0625 * (2**epoch)

		self.sigmoid_0 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_2 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.filters = [self.sigmoid_0, self.max_blur_xy_1, self.sigmoid_2, self.gaussian_blur_3, self.layering_z_4, self.scale_5]

	def init_filters_and_variables(self):
		self.num_filters = 6
		self.num_variables = 1 + self.num_filters

		# Start the sigmoids at weak strengths
		self.sigmoid_beta = 0.0625
		self.sigmoid_eta = 0.5
		self.sigmoid_0 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_2 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		alpha = 8
		# I truly dislike this silent parameter passing, like truly scarily bad coding
		self.blur_half_width = max_blur_filter_half_width
		
		# This notation is slightly confusing, but it is meant to be the
		# direction you blur when you are on the layer corresponding to x-
		# or y-layering.  So, if you are layering in x, then you blur in y
		# and vice versa.
		
		self.max_blur_xy_1 = square_blur.SquareBlur(
			alpha,
			[self.blur_half_width, self.blur_half_width, 0])

		self.gaussian_blur_3 = GaussianBlur( self.gaussian_blur_sigma )


		z_voxel_layers = self.size[2]
		self.layering_z_4 = layering.Layering( z_dimension_idx, self.num_z_layers, [ 0, 1 ], self.spacer_height_voxels, 0.0 )

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_5 = scale.Scale([scale_min, scale_max])


		# Initialize the filter chain
		self.filters = [self.sigmoid_0, self.max_blur_xy_1, self.sigmoid_2, self.gaussian_blur_3, self.layering_z_4, self.scale_5]

		self.init_variables()

	def plot_layers( self, subplot_rows, subplot_cols, colormap, filename ):
		import matplotlib.pyplot as plt

		layer_indices = self.layering_z_4.get_layer_idxs( self.size )
		cur_permittivity = self.get_permittivity()
		plt.clf()
		for layer_idx in range( 0, len( layer_indices ) ):
			get_layer = cur_permittivity[ :, :, layer_indices[ layer_idx ] ]
			plt.subplot( subplot_rows, subplot_cols, layer_idx + 1 )
			plt.imshow( get_layer, cmap=colormap )
		plt.savefig( filename )

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient, step_size):
		self.w[0] = self.proposed_design_step(gradient, step_size)
		# Update the variable stack including getting the permittivity at the w[-1] position
		self.update_permittivity()

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
