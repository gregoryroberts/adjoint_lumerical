import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

import numpy as np

from LayeredMWIRBayerFilterParameters import *

class LayeredMWIRBayerFilter(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers):
		super(LayeredMWIRBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

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

		num_z_layers = self.layering_z_1.num_layers
		num_z_voxels_total = self.size[2]

		num_z_voxels_per_layer = int(num_z_voxels_total / num_z_layers)
		num_z_voxels_last_layer = num_z_voxels_total - (num_z_layers - 1) * num_z_voxels_per_layer

		var3 = np.zeros(var2.shape)
		var4 = np.zeros(var2.shape)
		for layer_idx in range(0, num_z_layers - 1):
			layer_start = layer_idx * num_z_voxels_per_layer

			pull_layer_data = var2[:, :, layer_start : (layer_start + num_z_voxels_per_layer)]

			var3[:, :, layer_start : (layer_start + num_z_voxels_per_layer)] = (self.max_blur_xy_2[layer_idx % 2]).forward(pull_layer_data)
			var4[:, :, layer_start : (layer_start + num_z_voxels_per_layer)] = (
				self.layering_xy_3[layer_idx % 2]).forward(
				var3[:, :, layer_start : (layer_start + num_z_voxels_per_layer)])

		last_layer_start = (num_z_layers - 1) * num_z_voxels_per_layer

		pull_last_layer_data = var2[:, :, last_layer_start : num_z_voxels_total]
		var3[:, :, last_layer_start : num_z_voxels_total] = (self.max_blur_xy_2[(num_z_layers - 1) % 2]).forward(pull_last_layer_data)
		var4[:, :, last_layer_start : num_z_voxels_total] = (
			self.layering_xy_3[(num_z_layers - 1) % 2]).forward(
			var3[:, :, last_layer_start : num_z_voxels_total])

		self.w[3] = var3
		self.w[4] = var4

		var5 = self.sigmoid_4.forward(var4)
		self.w[5] = 5

		var6 = self.scale_5.forward(var5)
		self.w[6] = var6


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient):
		gradient = self.scale_5.chain_rule(gradient, self.w[6], self.w[5])
		gradient = self.sigmoid_4.chain_rule(gradient, self.w[5], self.w[4])

		num_z_layers = self.layering_z_1.num_layers
		num_z_voxels_total = self.size[2]

		num_z_voxels_per_layer = int(num_z_voxels_total / num_z_layers)
		num_z_voxels_last_layer = num_z_voxels_total - (num_z_layers - 1) * num_z_voxels_per_layer

		for layer_idx in range(0, num_z_layers - 1):
			layer_start = layer_idx * num_z_voxels_per_layer

			pull_var4_data = self.w[4][:, :, layer_start : (layer_start + num_z_voxels_per_layer)]
			pull_var3_data = self.w[3][:, :, layer_start : (layer_start + num_z_voxels_per_layer)]
			pull_var2_data = self.w[2][:, :, layer_start : (layer_start + num_z_voxels_per_layer)]
			pull_grad_data = gradient[:, :, layer_start : (layer_start + num_z_voxels_per_layer)]

			gradient[:, :, layer_start : (layer_start + num_z_voxels_per_layer)] = (
				self.layering_xy_3[layer_idx % 2]).chain_rule(
				gradient[:, :, layer_start : (layer_start + num_z_voxels_per_layer)],
				pull_var4_data,
				pull_var3_data)

			gradient[:, :, layer_start : (layer_start + num_z_voxels_per_layer)] = (
				self.max_blur_xy_2[layer_idx % 2]).chain_rule(
				gradient[:, :, layer_start : (layer_start + num_z_voxels_per_layer)],
				pull_var3_data,
				pull_var2_data)

		last_layer_start = (num_z_layers - 1) * num_z_voxels_per_layer

		pull_last_layer_var4_data = self.w[4][:, :, last_layer_start : num_z_voxels_total]
		pull_last_layer_var3_data = self.w[3][:, :, last_layer_start : num_z_voxels_total]
		pull_last_layer_var2_data = self.w[2][:, :, last_layer_start : num_z_voxels_total]

		gradient[:, :, last_layer_start : num_z_voxels_total] = (self.layering_xy_3[(num_z_layers - 1) % 2]).chain_rule(
			gradient[:, :, last_layer_start : num_z_voxels_total],
			pull_last_layer_var4_data,
			pull_last_layer_var3_data)

		gradient[:, :, last_layer_start : num_z_voxels_total] = (self.max_blur_xy_2[(num_z_layers - 1) % 2]).chain_rule(
			gradient[:, :, last_layer_start : num_z_voxels_total],
			pull_last_layer_var3_data,
			pull_last_layer_var2_data)

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
		self.sigmoid_4 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

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
		max_blur_x_2 = square_blur.SquareBlur(
			alpha,
			[0, self.blur_half_width, 0])
		max_blur_y_2 = square_blur.SquareBlur(
			alpha,
			[self.blur_half_width, 0, 0])
		self.max_blur_xy_2 = [max_blur_x_2, max_blur_y_2]

		single_layer = 1
		layering_x_3 = layering.Layering(x_dimension_idx, single_layer)
		layering_y_3 = layering.Layering(y_dimension_idx, single_layer)
		self.layering_xy_3 = [layering_x_3, layering_y_3]

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_5 = scale.Scale([scale_min, scale_max])

		# Initialize the filter chain
		self.filters = [self.sigmoid_0, self.layering_z_1, self.max_blur_xy_2, self.layering_xy_3, self.sigmoid_4, self.scale_5]

		self.init_variables()

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
