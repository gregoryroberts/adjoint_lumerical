import filter as filter

import numpy as np

class Scale(filter.Filter):

	#
	# variable_bounds specifies the minimum and maximum value of the scaled variable.
	# We assume we are scaling an input that is between 0 and 1
	#
	def __init__(self, variable_bounds):
		super(Scale, self).__init__(variable_bounds)

		self.min_value = variable_bounds[0]
		self.max_value = variable_bounds[1]
		self.range = self.max_value - self.min_value


	def forward(self, variable_in):
		return np.add(self.min_value, np.multiply(self.range, variable_in))


	def chain_rule(self, derivative_out, variable_out, variable_in):
		return np.multiply(self.range, derivative_out)


	def fabricate(self, variable_in):
		return self.forward(variable_in)
