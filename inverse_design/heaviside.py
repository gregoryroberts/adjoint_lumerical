import filter as filter

import numpy as np

class Heaviside(filter.Filter):

	#
	# bandwidth: width of step crossing over zero in heaviside approximation
	#
	def __init__(self, bandwidth):
		variable_bounds = [0.0, 1.0]
		super(Heaviside, self).__init__(variable_bounds)

		self.bandwidth = bandwidth

	def forward(self, variable_in):
		greater_values = 1.0 * np.greater( variable_in, self.bandwidth )
		lesser_values = 1.0 * np.less( variable_in, -self.bandwidth )
		middle_values = np.ones( variable_in.shape ) - greater_values - lesser_values

		middle = -0.25 * ( variable_in / self.bandwidth )**3 + 0.75 * ( variable_in / self.bandwidth ) + 0.5
		
		return ( 1.0 * greater_values ) + ( 0.0 * lesser_values ) + ( middle * middle_values )


	def chain_rule(self, derivative_out, variable_out, variable_in):
		greater_values = 1.0 * np.greater( variable_in, self.bandwidth )
		lesser_values = 1.0 * np.less( variable_in, -self.bandwidth )
		middle_values = np.ones( variable_in.shape ) - greater_values - lesser_values

		middle = derivative_out * ( ( -0.75 / self.bandwidth ) * ( variable_in / self.bandwidth )**2 + ( 0.75 / self.bandwidth ) )

		return ( 0.0 * greater_values ) + ( 0.0 * lesser_values ) + ( middle * middle_values )

