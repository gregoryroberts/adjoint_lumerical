import numpy as np

class Device:

	def __init__(self, size, permittivity_bounds, init_permittivity):
		self.size = size
		self.init_permittivity = init_permittivity
		self.permittivity_bounds = permittivity_bounds
		self.num_filters = 0
		self.num_variables = self.num_filters + 1

		self.mid_permittivity = 0.5 * (self.permittivity_bounds[0] + self.permittivity_bounds[1])

		self.init_variables()

	def init_variables(self):
		self.w = [np.zeros(self.size, dtype=np.complex) for i in range(0, self.num_variables)]
		self.w[0] = np.multiply(self.init_permittivity, np.ones(self.size, dtype=np.complex))

	def get_design_variable(self):
		return self.w[0]

	def set_design_variable(self, new_design):
		self.w[0] = new_design.copy()
		self.update_permittivity()

	def get_permittivity(self):
		return self.w[-1]

	def update_permittivity(self):
		for filter_idx in range(0, self.num_filters):
			variable_in = self.w[filter_idx]
			variable_out = (self.filters[filter_idx]).forward(variable_in)
			self.w[filter_idx + 1] = variable_out

	def apply_binary_filter_pipeline(self, variable_in):
		# Hmm.. not sure how to make this explicit, but the filters run here do not need
		# to do any padding and only need to produce a partial output of the same size

		# Need a copy or else the original variable in will be modified, which is not what
		# is likely expected
		variable_out = np.copy(variable_in)
		for filter_idx in range(0, self.num_filters):
			variable_out = (self.filters[filter_idx]).fabricate(variable_out)

		return variable_out

	def backpropagate(self, gradient):
		for f_idx in range(0, self.num_filters):
			get_filter = self.filters[self.num_filters - f_idx - 1]
			variable_out = self.w[self.num_filters - f_idx]
			variable_in = self.w[self.num_filters - f_idx - 1]

			gradient = get_filter.chain_rule(gradient, variable_out, variable_in)

		return gradient

	def proposed_design_step(self, gradient, step_size):
		gradient = self.backpropagate(gradient)

		proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)
		proposed_design_variable = np.maximum(
									np.minimum(
										proposed_design_variable,
										self.maximum_design_value),
									self.minimum_design_value)

		return proposed_design_variable

	def fabrication_version(self):
		padded_design = np.pad(
			self.w[0],
			(
				(self.pipeline_half_width[0], self.pipeline_half_width[0]),
				(self.pipeline_half_width[1], self.pipeline_half_width[1]),
				(self.pipeline_half_width[2], self.pipeline_half_width[2])
			),
			'constant'
		)

		# Step 4: We need here the current fabrication target before we start the stepping
		design_shape = (self.w[0]).shape

		current_device = self.convert_to_binary_map(self.apply_binary_filter_pipeline(padded_design))

		return current_device[
				self.pipeline_half_width[0] : (self.pipeline_half_width[0] + design_shape[0]),
				self.pipeline_half_width[1] : (self.pipeline_half_width[1] + design_shape[1]),
				self.pipeline_half_width[2] : (self.pipeline_half_width[2] + design_shape[2])]


	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient, step_size):
		self.w[0] = self.proposed_design_step(gradient, step_size)
		# Update the variable stack including getting the permittivity at the w[-1] position
		self.update_permittivity()






