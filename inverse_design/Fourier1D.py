import filter as filter

import numpy as np
import matplotlib.pyplot as plt

class Fourier1D(filter.Filter):

	def __init__(self, dim, feature_size_cutoff, variable_bounds=[0, 1]):
		super(Fourier1D, self).__init__(variable_bounds)

		self.dim = dim
		#
		# We will pad on either side to ensure periodicity and we will make sure it is of odd length
		#
		self.fourier_dim = ( self.dim + 2 ) + ( 1 - ( self.dim % 2 ) )
		self.feature_size_cutoff = feature_size_cutoff
		self.middle_point = self.fourier_dim // 2

		self.fourier_dim_difference = self.fourier_dim - self.dim
		self.left_pad = self.fourier_dim_difference // 2
		self.right_pad = ( 1 + self.fourier_dim_difference ) // 2

		#
		# Choose conservative k limit
		#
		# print( feature_size_cutoff )

		self.k_limit = int( np.floor( self.fourier_dim / ( 2 * feature_size_cutoff ) ) )

		# x_values = np.arange( 0, self.fourier_dim )
		# y_values = np.cos( 2 * np.pi * self.k_limit * x_values / self.fourier_dim )
		# plt.plot( x_values, y_values )
		# plt.show()

	def spatial_derivative( self, variable_in ):
		k_vectors = np.linspace( -( self.fourier_dim // 2 ), ( self.fourier_dim // 2 ), self.fourier_dim )
		k_modulation = 1j * np.sin( 2 * np.pi * k_vectors / self.fourier_dim )
		spatial_derivative_k = variable_in * k_modulation

		return k_modulation, np.fft.ifft( np.fft.ifftshift( spatial_derivative_k ) )

	def spatial_second_derivative( self, variable_in ):
		k_vectors = np.linspace( -( self.fourier_dim // 2 ), ( self.fourier_dim // 2 ), self.fourier_dim )
		k_modulation = 2 * ( np.cos( 2 * np.pi * k_vectors / self.fourier_dim ) - 1 )
		spatial_second_derivative_k = variable_in * k_modulation

		return k_modulation, np.fft.ifft( np.fft.ifftshift( spatial_second_derivative_k ) )

	def forward(self, variable_in):
		#
		# The variable_in in this case is the k-space representation of the density.  And
		# we need to take the 1D IFFT of it to get out the spatial density profile.
		#
		# fourier_dim_difference = self.fourier_dim - self.dim
		# left_pad = fourier_dim_difference // 2
		# right_pad = ( 1 + fourier_dim_difference ) // 2
		# padded_variable = np.pad( variable_in, (( left_pad, right_pad )), 'constant' )


		# rearrange_k_points = np.zeros( self.fourier_dim, dtype=np.complex )
		# rearrange_k_points[ 0 ] = variable_in[ self.middle_point ]
		# rearrange_k_points[ 1 : ( 1 + ( self.fourier_dim // 2 ) ) ] = variable_in[ ( self.middle_point + 1 ) : ]
		# rearrange_k_points[ ( 1 + ( self.fourier_dim // 2 ) ) : ] = variable_in[ 0 : ( self.fourier_dim // 2 ) ]

		# plt.plot( np.real( rearrange_k_points ), color='g', linewidth=2 )
		# plt.plot( np.real( np.fft.ifftshift( variable_in )), color='b', linestyle='--' )
		# plt.show()

		density = np.fft.ifft( np.fft.ifftshift( variable_in ) )

		# density = np.fft.ifft( rearrange_k_points )
		return density

	def chain_rule(self, derivative_out, variable_out_, variable_in_):
		#
		# The variable out is the density and the variable in is the k-space representation.
		#

		derivative_in = np.zeros( derivative_out.shape, dtype=np.complex )
		other_in = np.zeros( derivative_out.shape, dtype=np.complex )

		# rearrange_k_points = np.zeros( self.fourier_dim, dtype=np.complex )

		# rearrange_k_points[ 0 ] = derivative_out[ self.middle_point ]
		# rearrange_k_points[ 1 : ( 1 + ( self.fourier_dim // 2 ) ) ] = derivative_out[ ( self.middle_point + 1 ) : ]
		# rearrange_k_points[ ( 1 + ( self.fourier_dim // 2 ) ) : ] = derivative_out[ 0 : ( self.fourier_dim // 2 ) ]

		# rearrange_k_points[ 0 ] = derivative_out[ 0 ]
		# rearrange_k_points[ 1 : ] = derivative_out[ 1 : ]

		# plt.plot( np.real( rearrange_k_points ) )
		# plt.plot( np.real( derivative_out ), linestyle='--' )
		# plt.show()

		# derivative_in = np.fft.ifft( derivative_out )

		# derivative_in = np.fft.ifftshift( np.fft.ifft( derivative_out ) )
		derivative_in = np.fft.ifft( derivative_out )

		# derivative_in = np.fft.fft( np.fft.fftshift( derivative_in ) )

		# print( derivative_in.shape )

		# for k_idx in range( 0, self.fourier_dim ):
		# 	compute_k = -( self.fourier_dim // 2 ) + k_idx
		# 	accumulate_derivative = 0

		# 	for n_idx in range( 0, self.fourier_dim ):
		# 		accumulate_derivative += ( 1.0 / self.fourier_dim ) * derivative_out[ n_idx ] * np.exp( 1j * 2 * np.pi * compute_k * n_idx / self.fourier_dim )

		# 	other_in[ k_idx ] = accumulate_derivative

		unscramble = np.zeros( self.fourier_dim, dtype=np.complex )
		unscramble[ 0 ] = derivative_in[ self.middle_point + 1 ]
		unscramble[ 1 : ( ( self.fourier_dim // 2 ) ) ] = derivative_in[ ( self.middle_point + 2 ) : ]
		unscramble[ ( ( self.fourier_dim // 2 ) ) : ] = derivative_in[ 0 : 1 + ( self.fourier_dim // 2 ) ]
		unscramble = np.flip( unscramble )

		# plt.subplot( 1, 2, 1 )
		# plt.plot( np.abs( unscramble ) )
		# plt.plot( np.abs( other_in ) )
		# plt.subplot( 1, 2, 2 )
		# plt.plot( np.angle( unscramble ) )
		# plt.plot( np.angle( other_in ) )
		# plt.show()

		return unscramble
		# return derivative_in


