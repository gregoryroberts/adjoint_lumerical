import filter as filter

import numpy as np

def sech(input):
	return np.divide(1, np.cosh(input))

class Sigmoid(filter.Filter):

	#
	# beta: strength of sigmoid filter
	# eta: center point of sigmoid filter
	#
	def __init__(self, beta, eta):
		variable_bounds = [0.0, 1.0]
		super(Sigmoid, self).__init__(variable_bounds)

		self.beta = beta
		self.eta = eta
		self.denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))


	def forward(self, variable_in):
		numerator = np.add(np.tanh(self.beta * self.eta), np.tanh(np.multiply(self.beta, np.subtract(variable_in, self.eta))))
		return np.divide(numerator, self.denominator)

	def chain_rule(self, derivative_out, variable_out, variable_in):
		numerator = np.multiply(self.beta, np.power(sech(np.multiply(self.beta, np.subtract(variable_in, self.eta))), 2))
		return np.divide(np.multiply(derivative_out, numerator), self.denominator)

	def fabricate(self, variable_in):
		result = np.add(
			np.multiply(self.variable_bounds[1], np.greater(variable_in, self.eta)),
			np.multiply(self.variable_bounds[0], np.less_equal(variable_in, self.eta))
			)
		return result 



