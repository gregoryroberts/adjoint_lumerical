import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur
import Fourier1D as Fourier1D
import ramp as ramp

import numpy as np

class KSpaceLayeredBayerFilter2D(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_y_layers, feature_size_cutoff_voxels ):
		super(KSpaceLayeredBayerFilter2D, self).__init__(size, permittivity_bounds, init_permittivity)

		self.x_dimension_idx = 0
		self.y_dimension_idx = 1
		self.z_dimension_idx = 2

		#
		# We are assuming for now that everything is already layered, so we will need to do this automatically on the gradient
		#

		self.feature_size_cutoff_voxels = feature_size_cutoff_voxels

		self.num_y_layers = self.size[ 1 ]# num_y_layers
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

		# print(var0)
		var1 = self.fourier_1d_0.forward( var0 ) + self.init_permittivity
		self.w[1] = var1

		# print(var1)
		# import matplotlib.pyplot as plt
		# plt.plot(var1, color='g')

		var2 = self.ramp_1.forward( var1 )
		self.w[2] = var2

		# plt.plot(var2, color='b')
		# plt.show()
		var3 = self.sigmoid_2.forward( var2 )
		self.w[3] = var3

		scale_real_3 = self.scale_3[ 0 ]
		scale_imag_3 = self.scale_3[ 1 ]

		var4 = scale_real_3.forward( var3 ) + 1j * scale_imag_3.forward( var3 )

		for y_idx in range( 0, self.size[ 1 ] ):
			assert self.fourier_1d_0.right_pad > 0, "update_permittivity: expected the 1D Fourier padding to be positive and nonzero"
			self.w[4][ :, y_idx ] = var4[ self.fourier_1d_0.left_pad : ( 0 - self.fourier_1d_0.right_pad ) ]


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient_real, gradient_imag):
		scale_real_3 = self.scale_3[ 0 ]
		scale_imag_3 = self.scale_3[ 1 ]

		gradient = (
			scale_real_3.chain_rule( np.squeeze( np.sum( gradient_real, axis=1 ) ), np.squeeze( np.sum( self.w[4], axis=1 ) ), self.w[3] ) +
			scale_imag_3.chain_rule( np.squeeze( np.sum( gradient_imag, axis=1 ) ), np.squeeze( np.sum( self.w[4], axis=1 ) ), self.w[3] )
		)

		gradient = np.pad( gradient, ((self.fourier_1d_0.left_pad, self.fourier_1d_0.right_pad)), mode='constant' )

		gradient = self.sigmoid_2.chain_rule( gradient, self.w[3], self.w[2] )
		gradient = self.ramp_1.chain_rule( gradient, self.w[2], self.w[1] )
		gradient = self.fourier_1d_0.chain_rule( gradient, self.w[1] - self.init_permittivity, self.w[0] )

		return gradient

	def update_filters(self, epoch):
		self.sigmoid_beta = 0.0625 * (2**epoch)

		self.sigmoid_2 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.filters = [ self.fourier_1d_0, self.ramp_1, self.sigmoid_2, self.scale_3 ]


	def init_variables(self):
		return

	def init_variables_(self, fourier_dim):
		self.w = []
		for idx in range( 0, self.num_variables - 1 ):
			self.w.append( np.zeros( self.fourier_dim, dtype=np.complex ) )

		self.w.append( np.zeros( self.size, dtype=np.complex ) )

		# self.w[0][ self.fourier_dim // 2 ] = self.fourier_dim * self.init_permittivity

	def init_filters_and_variables(self):
		self.num_filters = 4
		self.num_variables = 1 + self.num_filters

		self.fourier_1d_0 = Fourier1D.Fourier1D( self.size[ 0 ], self.feature_size_cutoff_voxels )
		self.fourier_dim = self.fourier_1d_0.fourier_dim

		self.ramp_1 = ramp.Ramp()

		# Start the sigmoid at weak strengths
		self.sigmoid_beta = 0.0625
		self.sigmoid_eta = 0.5
		self.sigmoid_2 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		scale_real_min = np.real( self.permittivity_bounds[0] )
		scale_real_max = np.real( self.permittivity_bounds[1] )
		scale_real_3 = scale.Scale([scale_real_min, scale_real_max])

		scale_imag_min = np.imag( self.permittivity_bounds[0] )
		scale_imag_max = np.imag( self.permittivity_bounds[1] )
		scale_imag_3 = scale.Scale([scale_imag_min, scale_imag_max])

		self.scale_3 = [ scale_real_3, scale_imag_3 ]

		# Initialize the filter chain
		self.filters = [ self.fourier_1d_0, self.ramp_1, self.sigmoid_2, self.scale_3 ]

		self.update_filters( 0 )

		#
		# This will init the variable as complex datatype, which is good for the fourier representation
		#
		self.init_variables_( self.fourier_dim )


	def proposed_design_step(self, gradient_real, gradient_imag, step_size ):
		gradient = self.backpropagate(gradient_real, gradient_imag)

		#
		# Limit the k-vectors you can have in your optimization solution
		#

		middle_point = self.fourier_1d_0.middle_point
		gradient[ 0 : ( middle_point - self.fourier_1d_0.k_limit ) ] = 0
		gradient[ ( middle_point + self.fourier_1d_0.k_limit + 1 ) : ] = 0

		# gradient_norm = np.sqrt( np.sum( np.abs( gradient )**2 ) )
		# eps = 1e-13
		normalized_direction = gradient# / ( gradient_norm + eps )

		proposed_design_variable = self.w[0] - np.multiply( step_size, normalized_direction )
  
		return proposed_design_variable

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient_real, gradient_imag, step_size ):
		save_w0 = self.w[0].copy()
		self.w[0] = self.proposed_design_step( gradient_real, gradient_imag, step_size )
		self.update_permittivity()