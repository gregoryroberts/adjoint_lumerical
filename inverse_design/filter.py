import numpy as np

class Filter:

	def __init__(self, variable_bounds):
		self.variable_bounds = variable_bounds

	def verify_bounds(self, variable):
		min_value = np.minimum(variable)
		max_value = np.maximum(variable)

		if ((min_value < self.variable_bounds[0]) or (max_value > self.variable_bounds[1])):
			return False

		return True

	#
	# Propagate the variable_in through the filter and return the result
	#
	def forward(self, variable_in):
		raise NotImplementedError()

	#
	# Apply the chain rule to this filter to propagate the derivative back one step
	def chain_rule(self, derivative_out, variable_out, variable_in):
		raise NotImplementedError()
