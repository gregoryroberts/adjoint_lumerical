import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur
import Fourier1D as Fourier1D
import heaviside as heaviside

import matplotlib.pyplot as plt

import numpy as np

import scipy
from scipy import optimize


class KSpaceLayeredBayerFilter2DLSF(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_y_layers, size_cutoff_k_vectors, feature_size_cutoff_voxels, gap_size_cutoff_voxels ):
		super(KSpaceLayeredBayerFilter2DLSF, self).__init__(size, permittivity_bounds, init_permittivity )

		self.x_dimension_idx = 0
		self.y_dimension_idx = 1
		self.z_dimension_idx = 2

		#
		# We are assuming for now that everything is already layered, so we will need to do this automatically on the gradient
		#

		self.size_cutoff_k_vectors = size_cutoff_k_vectors
		self.feature_size_cutoff_voxels = feature_size_cutoff_voxels
		self.gap_size_cutoff_voxels = gap_size_cutoff_voxels
		self.feature_size_beta = 0.15

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

		var1 = np.real( self.fourier_1d_0.forward( var0 ) )
		self.w[1] = var1

		#
		# I think we should put this after the IFFT step
		#
		var2 = self.heaviside_1.forward( var1 - 0.5 )
		self.w[2] = var2

		scale_real_2 = self.scale_2[ 0 ]
		scale_imag_2 = self.scale_2[ 1 ]

		var3 = scale_real_2.forward( var2 ) + 1j * scale_imag_2.forward( var2 )

		for y_idx in range( 0, self.size[ 1 ] ):
			assert self.fourier_1d_0.right_pad > 0, "update_permittivity: expected the 1D Fourier padding to be positive and nonzero"
			self.w[3][ :, y_idx ] = var3[ self.fourier_1d_0.left_pad : ( 0 - self.fourier_1d_0.right_pad ) ]


	def compute_penalty_map( self ):
		psi = self.w[ 1 ]
		k_prefactor_first_derivative, first_derivative = self.fourier_1d_0.spatial_derivative( self.w[ 0 ] )
		k_prefactor_second_derivative, second_derivative = self.fourier_1d_0.spatial_second_derivative( self.w[ 0 ] )
		k_vectors = np.linspace( -( self.fourier_1d_0.fourier_dim // 2 ), ( self.fourier_1d_0.fourier_dim // 2 ), self.fourier_1d_0.fourier_dim )

		max_alpha = 4

		penalty_map = np.zeros( self.fourier_1d_0.fourier_dim )
		for n_idx in range( 0, self.fourier_1d_0.fourier_dim ):
			if ( psi[ n_idx ] < 0 ) and ( second_derivative[ n_idx ] > 0 ):
				numerator = second_derivative[ n_idx ]
				denominator = ( -np.pi / self.gap_size_cutoff_voxels ) * psi[ n_idx ] + self.feature_size_beta * first_derivative[ n_idx ]

				penalty_value =  np.real( ( numerator / denominator ) - np.pi / self.gap_size_cutoff_voxels )

				# penalty += np.maximum( 0, penalty_value )
				# penalty += ( 1 / max_alpha ) * np.log( np.exp( max_alpha * penalty_value ) + 1 )

				penalty_map[ n_idx ] = np.maximum( 0, penalty_value )

			elif ( psi[ n_idx ] > 0 ) and( second_derivative[ n_idx ] < 0 ):
				numerator = -second_derivative[ n_idx ]
				denominator = ( np.pi / self.feature_size_cutoff_voxels ) * psi[ n_idx ] - self.feature_size_beta * first_derivative[ n_idx ]

				penalty_value = np.real( ( numerator / denominator ) - np.pi / self.feature_size_cutoff_voxels )

				# todo(gdrobert): gradient might be noisier with the hard maximum here.  We can do an approximation, but we might then like to know what the average penalty is so we
				# can weight this appropriately?  How to pick alpha for the approximate maximum correctly?  Maybe we can do another function that goes to 0 at less than
				# or equal to 0? Honestly, it is probably ok to use the hard max here.  Not sure how it affects the convergence.
				# penalty += np.maximum( 0, penalty_value )
				# penalty += ( 1 / max_alpha ) * np.log( np.exp( max_alpha * penalty_value ) + 1 )

				penalty_map[ n_idx ] = np.maximum( 0, penalty_value )


		return penalty_map
			

	def compute_penalty( self ):
		psi = self.w[ 1 ]
		k_prefactor_first_derivative, first_derivative = self.fourier_1d_0.spatial_derivative( self.w[ 0 ] )
		k_prefactor_second_derivative, second_derivative = self.fourier_1d_0.spatial_second_derivative( self.w[ 0 ] )
		k_vectors = np.linspace( -( self.fourier_1d_0.fourier_dim // 2 ), ( self.fourier_1d_0.fourier_dim // 2 ), self.fourier_1d_0.fourier_dim )

		psi = np.real( psi )
		first_derivative = np.real( first_derivative )
		second_derivative = np.real( second_derivative )

		max_alpha = 4

		penalty = 0
		penalty_gradient = np.zeros( self.fourier_1d_0.fourier_dim, dtype=np.complex )
		for n_idx in range( 0, self.fourier_1d_0.fourier_dim ):
			if ( psi[ n_idx ] < 0 ) and ( second_derivative[ n_idx ] > 0 ):
				numerator = second_derivative[ n_idx ]
				# denominator = ( -np.pi / self.gap_size_cutoff_voxels ) * psi[ n_idx ] + self.feature_size_beta * first_derivative[ n_idx ]
				denominator = ( -np.pi / self.gap_size_cutoff_voxels ) * psi[ n_idx ] + self.feature_size_beta * np.abs( first_derivative[ n_idx ] )

				penalty_value =  np.real( ( numerator / denominator ) - np.pi / self.gap_size_cutoff_voxels )

				penalty += np.maximum( 0, penalty_value )
				# penalty += ( 1 / max_alpha ) * np.log( np.exp( max_alpha * penalty_value ) + 1 )

				if penalty_value > 0:
					for k_idx in range( 0, self.fourier_1d_0.fourier_dim ):
						deriv_numerator = denominator * k_prefactor_second_derivative[ k_idx ] - numerator * ( ( -np.pi / self.gap_size_cutoff_voxels ) - self.feature_size_beta * k_prefactor_first_derivative[ k_idx ] )
						# deriv_numerator = denominator * k_prefactor_second_derivative[ k_idx ] - numerator * ( ( -np.pi / self.gap_size_cutoff_voxels ) + self.feature_size_beta * k_prefactor_first_derivative[ k_idx ] )
						deriv_denominator = denominator**2

						if first_derivative[ n_idx ] > 0:
							deriv_numerator = denominator * k_prefactor_second_derivative[ k_idx ] - numerator * ( ( -np.pi / self.gap_size_cutoff_voxels ) + self.feature_size_beta * k_prefactor_first_derivative[ k_idx ] )


						# max_prefactor = np.exp( max_alpha * penalty_value ) / ( 1 + np.exp( max_alpha * penalty_value ) )
						max_prefactor = 1

						penalty_gradient[ k_idx ] += max_prefactor * deriv_numerator * np.exp( 1j * 2 * np.pi * k_vectors[ k_idx ] * n_idx / self.fourier_1d_0.fourier_dim ) / deriv_denominator

			elif ( psi[ n_idx ] > 0 ) and ( second_derivative[ n_idx ] < 0 ):
				numerator = -second_derivative[ n_idx ]
				# denominator = ( np.pi / self.feature_size_cutoff_voxels ) * psi[ n_idx ] - self.feature_size_beta * first_derivative[ n_idx ]
				denominator = ( np.pi / self.feature_size_cutoff_voxels ) * psi[ n_idx ] + self.feature_size_beta * np.abs( first_derivative[ n_idx ] )

				penalty_value = np.real( ( numerator / denominator ) - np.pi / self.feature_size_cutoff_voxels )

				# todo(gdrobert): gradient might be noisier with the hard maximum here.  We can do an approximation, but we might then like to know what the average penalty is so we
				# can weight this appropriately?  How to pick alpha for the approximate maximum correctly?  Maybe we can do another function that goes to 0 at less than
				# or equal to 0? Honestly, it is probably ok to use the hard max here.  Not sure how it affects the convergence.
				penalty += np.maximum( 0, penalty_value )
				# penalty += ( 1 / max_alpha ) * np.log( np.exp( max_alpha * penalty_value ) + 1 )

				if penalty_value > 0:
					for k_idx in range( 0, self.fourier_1d_0.fourier_dim ):
						deriv_numerator = -denominator * k_prefactor_second_derivative[ k_idx ] - numerator * ( ( np.pi / self.feature_size_cutoff_voxels ) - self.feature_size_beta * k_prefactor_first_derivative[ k_idx ] )
						# deriv_numerator = -denominator * k_prefactor_second_derivative[ k_idx ] - numerator * ( ( np.pi / self.feature_size_cutoff_voxels ) - self.feature_size_beta * k_prefactor_first_derivative[ k_idx ] )
						
						if first_derivative[ n_idx ] > 0:
							deriv_numerator = -denominator * k_prefactor_second_derivative[ k_idx ] - numerator * ( ( np.pi / self.feature_size_cutoff_voxels ) + self.feature_size_beta * k_prefactor_first_derivative[ k_idx ] )

						deriv_denominator = denominator**2

						# max_prefactor = np.exp( max_alpha * penalty_value ) / ( 1 + np.exp( max_alpha * penalty_value ) )
						max_prefactor = 1

						penalty_gradient[ k_idx ] += max_prefactor * deriv_numerator * np.exp( 1j * 2 * np.pi * k_vectors[ k_idx ] * n_idx / self.fourier_1d_0.fourier_dim ) / deriv_denominator

		penalty_gradient = ( 1 / self.fourier_1d_0.fourier_dim ) * np.flip( penalty_gradient )

		return penalty_gradient, penalty
			
	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient_real, gradient_imag):
		scale_real_2 = self.scale_2[ 0 ]
		scale_imag_2 = self.scale_2[ 1 ]

		gradient = (
			scale_real_2.chain_rule( np.squeeze( np.sum( gradient_real, axis=1 ) ), np.squeeze( np.sum( self.w[3], axis=1 ) ), self.w[2] ) +
			scale_imag_2.chain_rule( np.squeeze( np.sum( gradient_imag, axis=1 ) ), np.squeeze( np.sum( self.w[3], axis=1 ) ), self.w[2] )
		)

		gradient = np.pad( gradient, ((self.fourier_1d_0.left_pad, self.fourier_1d_0.right_pad)), mode='constant' )

		gradient = self.heaviside_1.chain_rule( gradient, self.w[2], self.w[1] )
		gradient = self.fourier_1d_0.chain_rule( gradient, self.w[1], self.w[0] )

		return gradient

	def update_filters(self, epoch):
		self.heaviside_bandwidth = 0.05 / ( epoch + 1 )
		self.heaviside_1 = heaviside.Heaviside( self.heaviside_bandwidth )
		self.filters = [ self.fourier_1d_0, self.heaviside_1, self.scale_2 ]


	def init_variables(self):
		return

	def init_variables_(self, fourier_dim):
		self.w = []
		for idx in range( 0, self.num_variables - 1 ):
			self.w.append( np.zeros( self.fourier_dim, dtype=np.complex ) )

		self.w.append( np.zeros( self.size, dtype=np.complex ) )

		# self.w[0][ self.fourier_dim // 2 ] = self.fourier_dim * self.init_permittivity

	def init_filters_and_variables(self):
		self.num_filters = 3
		self.num_variables = 1 + self.num_filters

		self.fourier_1d_0 = Fourier1D.Fourier1D( self.size[ 0 ], self.size_cutoff_k_vectors )
		self.fourier_dim = self.fourier_1d_0.fourier_dim

		#
		# We might be able to slowly tune this down
		#
		self.heaviside_bandwidth = 0.05
		self.heaviside_1 = heaviside.Heaviside( self.heaviside_bandwidth )

		scale_real_min = np.real( self.permittivity_bounds[0] )
		scale_real_max = np.real( self.permittivity_bounds[1] )
		scale_real_2 = scale.Scale([scale_real_min, scale_real_max])

		scale_imag_min = np.imag( self.permittivity_bounds[0] )
		scale_imag_max = np.imag( self.permittivity_bounds[1] )
		scale_imag_2 = scale.Scale([scale_imag_min, scale_imag_max])

		self.scale_2 = [ scale_real_2, scale_imag_2 ]

		# Initialize the filter chain
		self.filters = [ self.fourier_1d_0, self.heaviside_1, self.scale_2 ]

		self.update_filters( 0 )

		#
		# This will init the variable as complex datatype, which is good for the fourier representation
		#
		self.init_variables_( self.fourier_dim )



	def proposed_design_step_fabrication( self, step_size ):#enforce_binarization_increase, binarization_increase, max_design_change_point):
		penalty_gradient, fab_penalty = self.compute_penalty()

		print("step size = " + str( step_size ))

		print( "The current fabrication penalty is " + str( fab_penalty ) )

		#
		# Limit the k-vectors you can have in your optimization solution
		#

		middle_point = self.fourier_1d_0.middle_point
		penalty_gradient[ 0 : ( middle_point - self.fourier_1d_0.k_limit ) ] = 0
		penalty_gradient[ ( middle_point + self.fourier_1d_0.k_limit + 1 ) : ] = 0

		# gradient_norm = np.sqrt( np.sum( np.abs( penalty_gradient )**2 ) )
		# eps = 1e-13
		normalized_direction = penalty_gradient# / ( gradient_norm + eps )

		# move_gradient = penalty_gradient * np.max( np.abs( self.w[ 0 ] ) ) / np.max( np.abs( penalty_gradient ) )

		proposed_design_variable = self.w[0] - np.multiply( step_size, normalized_direction )
  
		return proposed_design_variable


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

	def proposed_design_step_raw( self, gradient, step_size ):
		#
		# Limit the k-vectors you can have in your optimization solution
		#

		middle_point = self.fourier_1d_0.middle_point
		gradient[ 0 : ( middle_point - self.fourier_1d_0.k_limit ) ] = 0
		gradient[ ( middle_point + self.fourier_1d_0.k_limit + 1 ) : ] = 0

		proposed_design_variable = self.w[0] - np.multiply( step_size, gradient )
  
		return proposed_design_variable


	# In the step function, we should update the permittivity with update_permittivity
	def step_fabrication( self, step_size ):
		save_w0 = self.w[0].copy()
		self.w[0] = self.proposed_design_step_fabrication( step_size )
		self.update_permittivity()

	# In the step function, we should update the permittivity with update_permittivity
	def step_raw(self, gradient, step_size ):
		save_w0 = self.w[0].copy()
		self.w[0] = self.proposed_design_step_raw( gradient, step_size )
		self.update_permittivity()

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient_real, gradient_imag, step_size ):
		save_w0 = self.w[0].copy()
		self.w[0] = self.proposed_design_step(gradient_real, gradient_imag, step_size )
	
		self.update_permittivity()
