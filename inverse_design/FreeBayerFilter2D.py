import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

import numpy as np

class FreeBayerFilter2D(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_y_layers):
		super(FreeBayerFilter2D, self).__init__(size, permittivity_bounds, init_permittivity)

		self.x_dimension_idx = 0
		self.y_dimension_idx = 1
		self.z_dimension_idx = 2

		self.num_y_layers = num_y_layers

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

		scale_real_1 = self.scale_1[ 0 ]
		scale_imag_1 = self.scale_1[ 1 ]

		var2 = scale_real_1.forward( var1 ) + 1j * scale_imag_1.forward( var1 )
		self.w[2] = var2

	# def get_permittivity( self ):
	# 	self.w[0] = 1.0 * np.greater( self.w[0], 0.065 )
	# 	self.update_permittivity()
	# 	return self.w[2]

	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient_real, gradient_imag):
		scale_real_1 = self.scale_1[ 0 ]
		scale_imag_1 = self.scale_1[ 1 ]

		gradient = (
			scale_real_1.chain_rule( gradient_real, self.w[2], self.w[1] ) +
			scale_imag_1.chain_rule( gradient_imag, self.w[2], self.w[1] )
		)	

		gradient = self.layering_y_0.chain_rule( gradient, self.w[1], self.w[0] )

		return gradient

	def init_variables(self):
		self.w = [np.zeros(self.size, dtype=np.complex) for i in range(0, self.num_variables)]

		self.w[0] = np.multiply(self.init_permittivity, np.ones(self.size, dtype=np.complex))

	def update_filters(self, epoch):
		return

	def init_filters_and_variables(self):
		self.num_filters = 2
		self.num_variables = 1 + self.num_filters

		self.layering_y_0 = layering.Layering(self.y_dimension_idx, self.num_y_layers)

		scale_real_min = np.real( self.permittivity_bounds[0] )
		scale_real_max = np.real( self.permittivity_bounds[1] )
		scale_real_1 = scale.Scale([scale_real_min, scale_real_max])

		scale_imag_min = np.imag( self.permittivity_bounds[0] )
		scale_imag_max = np.imag( self.permittivity_bounds[1] )
		scale_imag_1 = scale.Scale([scale_imag_min, scale_imag_max])

		self.scale_1 = [ scale_real_1, scale_imag_1 ]

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
