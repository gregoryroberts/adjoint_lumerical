import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import generic_blur_2d as generic_blur_2d

import numpy as np

class FreeBayerFilterWithBlur2D(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_y_layers, dilate_size_voxels):
		super(FreeBayerFilterWithBlur2D, self).__init__(size, permittivity_bounds, init_permittivity)

		self.x_dimension_idx = 0
		self.y_dimension_idx = 1
		self.z_dimension_idx = 2

		self.num_y_layers = num_y_layers
		# This class interprets a negative dilation as an erosion and a zero dilation as doing nothing.
		# The dilation size should be a half width, so the number of voxels involved in a dilation
		# is given by ( 2 * dilate_size_voxels + 1 )
		self.dilate_size_voxels = dilate_size_voxels
		self.should_erode = ( self.dilate_size_voxels < 0 )

		self.minimum_design_value = 0
		self.maximum_design_value = 1
		self.init_filters_and_variables()

		self.update_permittivity()

	#
	# Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
	#
	def update_permittivity(self):
		var0 = self.w[0]

		var1 = self.layering_y_0.forward( var0 )
		self.w[1] = var1

		var2 = np.zeros( var1.shape, dtype=var1.dtype )
		if self.should_erode:
			var2 = 1 - self.blur_horizontal_1.forward( 1 - var1 )
		else:
			var2 = self.blur_horizontal_1.forward( var1 )
		self.w[2] = var2


		scale_real_2 = self.scale_2[ 0 ]
		scale_imag_2 = self.scale_2[ 1 ]

		var3 = scale_real_2.forward( var2 ) + 1j * scale_imag_2.forward( var2 )
		self.w[3] = var3

	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient_real, gradient_imag):
		scale_real_2 = self.scale_2[ 0 ]
		scale_imag_2 = self.scale_2[ 1 ]

		gradient = (
			scale_real_2.chain_rule( gradient_real, self.w[3], self.w[2] ) +
			scale_imag_2.chain_rule( gradient_imag, self.w[3], self.w[2] )
		)

		if self.should_erode:
			gradient = self.blur_horizontal_1.chain_rule( gradient, 1 - self.w[2], 1 - self.w[1] )
		else:
			gradient = self.blur_horizontal_1.chain_rule( gradient, self.w[2], self.w[1] )

		gradient = self.layering_y_0.chain_rule( gradient, self.w[1], self.w[0] )

		return gradient

	def init_variables(self):
		self.w = [np.zeros(self.size, dtype=np.complex) for i in range(0, self.num_variables)]

		self.w[0] = np.multiply(self.init_permittivity, np.ones(self.size, dtype=np.complex))

	def update_filters(self, epoch):
		return

	def init_filters_and_variables(self):
		self.num_filters = 3
		self.num_variables = 1 + self.num_filters

		self.layering_y_0 = layering.Layering(self.y_dimension_idx, self.num_y_layers)

		blur_alpha = 8.0
		self.blur_horizontal_1 = generic_blur_2d.make_rectangular_blur( blur_alpha, np.abs( self.dilate_size_voxels ), 0 )

		scale_real_min = np.real( self.permittivity_bounds[0] )
		scale_real_max = np.real( self.permittivity_bounds[1] )
		scale_real_2 = scale.Scale([scale_real_min, scale_real_max])

		scale_imag_min = np.imag( self.permittivity_bounds[0] )
		scale_imag_max = np.imag( self.permittivity_bounds[1] )
		scale_imag_2 = scale.Scale([scale_imag_min, scale_imag_max])

		self.scale_2 = [ scale_real_2, scale_imag_2 ]

		self.update_filters( 0 )

		self.init_variables()

	def proposed_design_step(self, gradient_real, gradient_imag, step_size):
		gradient = self.backpropagate( gradient_real, gradient_imag )

		proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)

		proposed_design_variable = np.maximum(
									np.minimum(
										proposed_design_variable,
										self.maximum_design_value),
									self.minimum_design_value)

		return proposed_design_variable


	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient_real, gradient_imag, step_size):
		self.w[0] = self.proposed_design_step(gradient_real, gradient_imag, step_size)
		# Update the variable stack including getting the permittivity at the w[-1] position
		self.update_permittivity()
