import filter as filter

import numpy as np
# import matplotlib.pyplot as plt

#
# dim: dimension to layer on
# num_layers: number of layers
#
class Layering(filter.Filter):

	def __init__(self, dim, num_layers, variable_bounds=[0, 1]):
		super(Layering, self).__init__(variable_bounds)

		self.dim = dim
		self.num_layers = num_layers

	def get_layer_idxs(self, variable_shape):
		num_voxels_total = variable_shape[self.dim]

		layer_idxs = []

		num_voxels_per_layer = int(num_voxels_total / self.num_layers)
		num_voxels_last_layer = num_voxels_total - (self.num_layers - 1) * num_voxels_per_layer

		for layer_idx in range(0, self.num_layers - 1):
			layer_start = layer_idx * num_voxels_per_layer
			layer_idxs.append( layer_start )

		last_layer_start = (self.num_layers - 1) * num_voxels_per_layer
		layer_idxs.append( last_layer_start )

		return layer_idxs

	def do_layering(self, data, dim, output):
		data_shape = data.shape
		
		average = np.take(data, 0, axis=dim)

		for idx in range(1, data_shape[dim]):
			average += np.take(data, idx, axis=dim)

		average /= data_shape[dim]

		for idx in range(0, data_shape[dim]):
			idx_bunch = [slice(None)] * data.ndim
			idx_bunch[dim] = np.array(idx)
			output[tuple(idx_bunch)] = average

		return output

	def layer_averaging(self, variable):
		variable_shape = variable.shape
		num_voxels_total = variable_shape[self.dim]

		num_voxels_per_layer = int(num_voxels_total / self.num_layers)
		num_voxels_last_layer = num_voxels_total - (self.num_layers - 1) * num_voxels_per_layer

		variable_out = np.zeros(variable_shape)

		for layer_idx in range(0, self.num_layers - 1):
			layer_start = layer_idx * num_voxels_per_layer

			idx_bunch = [slice(None)] * variable.ndim
			idx_bunch[self.dim] = np.arange(layer_start, layer_start + num_voxels_per_layer)

			variable_out[tuple(idx_bunch)] = self.do_layering(variable[tuple(idx_bunch)], self.dim, variable_out[tuple(idx_bunch)])

		last_layer_start = (self.num_layers - 1) * num_voxels_per_layer
		idx_bunch = [slice(None)] * variable.ndim
		idx_bunch[self.dim] = np.arange(last_layer_start, num_voxels_total)

		variable_out[tuple(idx_bunch)] = self.do_layering(variable[tuple(idx_bunch)], self.dim, variable_out[tuple(idx_bunch)])

		return variable_out

	def forward(self, variable_in):
		return self.layer_averaging(variable_in)

	def chain_rule(self, derivative_out, variable_out_, variable_in_):
		return self.layer_averaging(derivative_out)

	def fabricate(self, variable_in):
		return self.forward(variable_in)



