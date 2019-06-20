import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

import numpy as np

from LayeredLithographyParameters import *

class LayeredLithographyBayerFilter(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers):
		super(CMOSBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

		self.num_z_layers = num_z_layers
		self.flip_threshold = 0.5
		self.minimum_design_value = 0
		self.maximum_design_value = 1
		self.init_filters_and_variables()

		self.update_permittivity()


	#
	# Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
	#
	def update_permittivity(self):
		var0 = self.w[0]

		var1 = self.sigmoid_0.forward(var0)
		self.w[1] = var1

		var2 = self.layering_z_1.forward(var1)
		self.w[2] = var2

		var3 = self.max_blur_xy_2.forward(var2)
		self.w[3] = var3

		var4 = self.sigmoid_3.forward(var3)
		self.w[4] = var4

		var5 = self.scale_4.forward(var4)
		self.w[5] = var5


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient):
		gradient = self.scale_4.chain_rule(gradient, self.w[5], self.w[4])
		gradient = self.sigmoid_3.chain_rule(gradient, self.w[4], self.w[3])
		gradient = self.max_blur_xy_2.chain_rule(gradient, self.w[3], self.w[2])
		gradient = self.layering_z_1.chain_rule(gradient, self.w[2], self.w[1])
		gradient = self.sigmoid_0.chain_rule(gradient, self.w[1], self.w[0])

		return gradient

	def update_filters(self, epoch):
		self.sigmoid_beta = 0.25 * (2**epoch)

		self.sigmoid_0 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_4 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.filters = [self.sigmoid_0, self.layering_z_1, self.max_blur_xy_2, self.layering_xy_3, self.sigmoid_4, self.scale_5]

	def init_filters_and_variables(self):
		self.num_filters = 6
		self.num_variables = 1 + self.num_filters

		# Start the sigmoids at weak strengths
		self.sigmoid_beta = 0.0625
		self.sigmoid_eta = 0.5
		self.sigmoid_0 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		z_voxel_layers = self.size[2]
		self.layering_z_1 = layering.Layering(z_dimension_idx, self.num_z_layers)

		alpha = 8
		self.blur_half_width = blur_half_width_voxels
		#
		# This notation is slightly confusing, but it is meant to be the
		# direction you blur when you are on the layer corresponding to x-
		# or y-layering.  So, if you are layering in x, then you blur in y
		# and vice versa.
		#
		max_blur_xy_2 = square_blur.SquareBlur(
			alpha,
			[self.blur_half_width, self.blur_half_width, 0])

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_4 = scale.Scale([scale_min, scale_max])

		# Initialize the filter chain
		self.filters = [self.sigmoid_0, self.layering_z_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4]

		self.init_variables()

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
