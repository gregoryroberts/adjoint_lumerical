import filter as filter

import numpy as np

class Ramp(filter.Filter):

	def __init__(self):
		variable_bounds = [0.0, 1.0]
		super(Ramp, self).__init__(variable_bounds)

		self.bandwidth = 0.5

	def forward(self, variable_in):
		offset = variable_in - self.bandwidth

		greater_values = 1.0 * np.greater( offset, self.bandwidth )
		lesser_values = 1.0 * np.less( offset, -self.bandwidth )
		middle_values = np.ones( offset.shape ) - greater_values - lesser_values

		# middle = -0.25 * ( offset / self.bandwidth )**3 + 0.75 * ( offset / self.bandwidth ) + 0.5
		
		return ( 1.0 * greater_values ) + ( 0.0 * lesser_values ) + ( variable_in * middle_values )


		# greater_values = 1.0 * np.greater( variable_in, 1 )
		# lesser_values = 1.0 * np.less( variable_in, 0 )
		# middle_values = np.ones( variable_in.shape ) - greater_values - lesser_values

		# middle = variable_in
		
		# return ( 1.0 * greater_values ) + ( 0.0 * lesser_values ) + ( middle * middle_values )

	def chain_rule(self, derivative_out, variable_out, variable_in):
		# offset = variable_in - self.bandwidth

		# greater_values = 1.0 * np.greater( offset, self.bandwidth )
		# lesser_values = 1.0 * np.less( offset, -self.bandwidth )
		# middle_values = np.ones( offset.shape ) - greater_values - lesser_values

		# middle = derivative_out * ( ( -0.75 / self.bandwidth ) * ( offset / self.bandwidth )**2 + ( 0.75 / self.bandwidth ) )

		#
		# For now, just keep the derivative unchanged and allow points at the boundary to still move
		#
		return derivative_out
		# return ( 0.0 * greater_values ) + ( 0.0 * lesser_values ) + ( middle * middle_values )

		# greater_values = 1.0 * np.greater( variable_in, 1 )
		# lesser_values = 1.0 * np.less( variable_in, 0 )
		# middle_values = np.ones( variable_in.shape ) - greater_values - lesser_values

		# middle = derivative_out

		# return ( 1.0 * greater_values ) + ( 0.0 * lesser_values ) + ( middle * middle_values )


