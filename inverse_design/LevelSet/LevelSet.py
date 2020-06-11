import numpy as np

from scipy.ndimage import gaussian_filter

from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def eval_gaussian( x, y, x0, y0, sigma ):
	return np.exp(
		-0.5 * (
			( ( x - x0 )**2 / sigma**2 ) + ( ( y - y0 )**2 / sigma**2 )
		)
	)

def eval_signed_distance_hole( x, y, x0, y0, radius ):
	return ( np.sqrt( ( x - x0 )**2 + ( y - y0 )**2 ) - radius )

def eval_heaviside( x, eta ):
	x_over_eta = x / eta
	middle = 0.5 * ( 1 + ( x_over_eta ) + ( 1 / np.pi ) * np.sin( np.pi * x_over_eta ) )

	return 1.0 * np.greater( x, eta ) + ( 1 - 1.0 * np.less( x, -eta ) ) * np.less_equal( x, eta ) * middle

class LevelSet():

	def __init__( self, dimension, boundary_smoothing_width ):
		self.dimension = dimension
		self.padded_dimension = [ ( d + 4 ) for d in dimension ]
		self.search_bounds = [ [ 2, 2 + dim ] for dim in dimension ]

		self.level_set_function = np.zeros( self.padded_dimension )

		self.boundary_smoothing_width = boundary_smoothing_width

	def init_with_holes( self, init_hole_centers, init_hole_widths ):
		print( init_hole_centers )
		self.init_hole_centers = [ [ ( loc + 2 ) for loc in hole ] for hole in init_hole_centers ]
		self.init_hole_widths = init_hole_widths

		print( self.init_hole_centers )

		num_holes = len( self.init_hole_centers )

		for x_idx in range( 0, self.padded_dimension[ 0 ] ):
			for y_idx in range( 0, self.padded_dimension[ 1 ] ):

				hole_center = self.init_hole_centers[ 0 ]
				hole_width = self.init_hole_widths[ 0 ]

				self.level_set_function[ x_idx, y_idx ] = eval_signed_distance_hole( x_idx, y_idx, hole_center[ 0 ], hole_center[ 1 ], hole_width )

				for hole_idx in range( 1, num_holes ):

					hole_center = self.init_hole_centers[ hole_idx ]
					hole_width = self.init_hole_widths[ hole_idx ]

					self.level_set_function[ x_idx, y_idx ] = np.minimum(
						self.level_set_function[ x_idx, y_idx ],
						eval_signed_distance_hole( x_idx, y_idx, hole_center[ 0 ], hole_center[ 1 ], hole_width )
					)

		self.signed_distance_reinitialization()
		# self.level_set_function -= 0.5

	def init_with_density( self, density ):
		self.level_set_function = np.pad( density - 0.5, ( ( 2, 2 ), ( 2, 2 ) ), mode='edge' )
		self.signed_distance_reinitialization()


	def signed_distance_reinitialization( self ):
		border_points_x, border_points_y = self.find_border_points()

		# import matplotlib.pyplot as plt
		# plt.scatter( border_points_x, border_points_y )
		# plt.show()
		# sdf

		# print( "Number of border points is " + str( len( border_points_x ) ) )
		for x_idx in range( 0, self.padded_dimension[ 0 ] ):
			# print( "Working on x idx = " + str( x_idx ) )
			for y_idx in range( 0, self.padded_dimension[ 1 ] ):

				distance = np.inf
				for border_point_idx in range( 0, len( border_points_x ) ):
					border_point_x_coord = border_points_x[ border_point_idx ]
					border_point_y_coord = border_points_y[ border_point_idx ]
					
					distance_border_point = np.sqrt( ( x_idx - border_point_x_coord )**2 + ( y_idx - border_point_y_coord )**2 )

					distance = np.minimum( distance, distance_border_point )

				self.level_set_function[ x_idx, y_idx ] = np.sign( self.level_set_function[ x_idx, y_idx ] ) * distance


	# def signed_distance_reinitialization( self ):
	# 	delta_ij_plus = np.zeros( self.level_set_function.shape )
	# 	delta_ij_minus = np.zeros( self.level_set_function.shape )

	# 	D_xplus = np.zeros( self.level_set_function.shape )
	# 	D_xminus = np.zeros( self.level_set_function.shape )
	# 	D_yplus = np.zeros( self.level_set_function.shape )
	# 	D_yminus = np.zeros( self.level_set_function.shape )

	# 	delta_t = 0.01
	# 	num_steps = 50

	# 	lsf_sign = np.sign( self.level_set_function )
	# 	smoothed_lsf_sign = gaussian_filter( lsf_sign, sigma=1 )
	# 	smoothed_lsf_sign -= np.min( smoothed_lsf_sign )
	# 	smoothed_lsf_sign /= np.max( smoothed_lsf_sign )
	# 	smoothed_lsf_sign = -1.0 + 2 * smoothed_lsf_sign

	# 	for step_idx in range( 0, num_steps ):
	# 		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
	# 			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):

	# 				D_xplus[ x_idx, y_idx ] = self.level_set_function[ x_idx + 1, y_idx ] - self.level_set_function[ x_idx, y_idx ]
	# 				D_xminus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx ] - self.level_set_function[ x_idx - 1, y_idx ]

	# 				D_yplus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx + 1 ] - self.level_set_function[ x_idx, y_idx ]
	# 				D_yminus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx ] - self.level_set_function[ x_idx, y_idx - 1 ]


	# 		delta_ij_plus = np.sqrt(
	# 			( np.minimum( D_xplus, 0 ) )**2 + ( np.maximum( D_xminus, 0 ) )**2 +
	# 			( np.minimum( D_yplus, 0 ) )**2 + ( np.maximum( D_yminus, 0 ) )**2
	# 		)

	# 		delta_ij_minus = np.sqrt(
	# 			( np.maximum( D_xplus, 0 ) )**2 + ( np.minimum( D_xminus, 0 ) )**2 +
	# 			( np.maximum( D_yplus, 0 ) )**2 + ( np.minimum( D_yminus, 0 ) )**2
	# 		)

	# 		self.level_set_function += delta_t * (
	# 			np.maximum( -smoothed_lsf_sign, 0 ) * delta_ij_plus + np.minimum( -smoothed_lsf_sign, 0 ) * delta_ij_minus + smoothed_lsf_sign
	# 		)


	def binarize( self ):
		padded_lsf = 1.0 * np.greater( self.level_set_function, 0.0 )
		print( padded_lsf[ self.search_bounds[ 0 ][ 0 ] : self.search_bounds[ 0 ][ 1 ], self.search_bounds[ 1 ][ 0 ] : self.search_bounds[ 1 ][ 1 ] ].shape )
		return padded_lsf[ self.search_bounds[ 0 ][ 0 ] : self.search_bounds[ 0 ][ 1 ], self.search_bounds[ 1 ][ 0 ] : self.search_bounds[ 1 ][ 1 ] ]


	def find_border_representation( self ):
		binary_representation = 1.0 * np.greater( self.level_set_function, 0.0 )

		border_representation = np.zeros( binary_representation.shape )

		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):
				center_point = binary_representation[ x_idx, y_idx ]
				surroundings = binary_representation[ ( x_idx - 1 ) : ( x_idx + 2 ), ( y_idx - 1 ) : ( y_idx + 2 ) ]

				if center_point == 1:
					if ( np.sum( surroundings ) < 9 ):
						border_representation[ x_idx, y_idx ] = 1

		return border_representation

	def find_border_points( self ):
		border_points_x = []
		border_points_y = []

		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):
				lsf_center = self.level_set_function[ x_idx, y_idx ]
				surroundings = self.level_set_function[ ( x_idx - 1 ) : ( x_idx + 2 ), ( y_idx - 1 ) : ( y_idx + 2 ) ]

				if ( lsf_center > 0 ):
					for shift_x in range( 0, 3 ):
						for shift_y in range( 0, 3 ):
							lsf_shift = surroundings[ shift_x, shift_y ]
							if ( ( shift_x == 1 ) and ( shift_y == 1 ) ) or ( lsf_shift > 0 ):
								continue

							m_hat_x = ( shift_x - 1 ) / ( lsf_shift - lsf_center )
							b_hat_x = 1 - m_hat_x * lsf_center

							m_hat_y = ( shift_y - 1 ) / ( lsf_shift - lsf_center )
							b_hat_y = 1 - m_hat_y * lsf_center

							border_points_x.append( x_idx + b_hat_x - 1 )
							border_points_y.append( y_idx + b_hat_y - 1 )

		return border_points_x, border_points_y


	def find_border_points_simple( self ):
		border_points_x = []
		border_points_y = []

		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):
				lsf_center = self.level_set_function[ x_idx, y_idx ]
				surroundings = self.level_set_function[ ( x_idx - 1 ) : ( x_idx + 2 ), ( y_idx - 1 ) : ( y_idx + 2 ) ]

				if lsf_center > 0:
					if np.sum( np.greater( surroundings, 0 ) ) < 9:
						border_points_x.append( x_idx )
						border_points_y.append( y_idx )

		return border_points_x, border_points_y


	def device_density_from_level_set( self ):
		device_density = np.zeros( self.level_set_function.shape )

		border_points_x, border_points_y = self.find_border_points()

		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):

				distance = np.inf
				for border_point_idx in range( 0, len( border_points_x ) ):
					border_point_x_coord = border_points_x[ border_point_idx ]
					border_point_y_coord = border_points_y[ border_point_idx ]
					
					distance_border_point = np.sqrt( ( x_idx - border_point_x_coord )**2 + ( y_idx - border_point_y_coord )**2 )

					distance = np.minimum( distance, distance_border_point )

				signed_distance = np.sign( self.level_set_function[ x_idx, y_idx ] ) * distance
				device_density[ x_idx, y_idx ] = eval_heaviside( signed_distance, self.boundary_smoothing_width )

		return device_density

	def extend_velocity_hilbertian( self, velocity_field ):
		border_representation = self.find_border_representation()

		boundary_velocity = border_representation * velocity_field

		vector_len = np.product( self.padded_dimension )

		Dx = lil_matrix( ( vector_len, vector_len ) )
		Dy = lil_matrix( ( vector_len, vector_len ) )
		identity = lil_matrix( ( vector_len, vector_len ) )

		alpha = 1

		for x_idx in range( 0, self.padded_dimension[ 0 ] ):
			for y_idx in range( 0, self.padded_dimension[ 1 ] ):
				loc = x_idx * self.padded_dimension[ 1 ] + y_idx

				identity[ loc, loc ] = 1

		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):
				Dx_loc = x_idx * self.padded_dimension[ 1 ] + y_idx
				Dx_up_loc = ( x_idx + 1 ) * self.padded_dimension[ 1 ] + y_idx
				Dx_down_loc = ( x_idx - 1 ) * self.padded_dimension[ 1 ] + y_idx

				Dx[ Dx_loc, Dx_up_loc ] = 1
				Dx[ Dx_loc, Dx_loc ] = -1

				Dy_loc = x_idx * self.padded_dimension[ 1 ] + y_idx
				Dy_up_loc = x_idx * self.padded_dimension[ 1 ] + y_idx + 1
				Dy_down_loc = x_idx * self.padded_dimension[ 1 ] + y_idx - 1

				Dy[ Dy_loc, Dy_up_loc ] = 1
				Dy[ Dy_loc, Dy_loc ] = -1

		Dx_component = ( Dx.transpose() ).dot( Dx )
		Dy_component = ( Dy.transpose() ).dot( Dy )

		M_to_invert = ( alpha * Dx_component + alpha * Dy_component + identity ).transpose()
		
		sparse_g_omega = csc_matrix( np.transpose( boundary_velocity.flatten() ) )

		V_tilde = spsolve( M_to_invert, sparse_g_omega.transpose() )

		reshape_V_tilde = np.reshape( V_tilde, velocity_field.shape )

		# import matplotlib.pyplot as plt
		# plt.subplot( 1, 4, 1 )
		# plt.imshow( boundary_velocity )
		# plt.colorbar()
		# plt.subplot( 1, 4, 2 )
		# plt.imshow( reshape_V_tilde )
		# plt.colorbar()
		# plt.subplot( 1, 4, 3 )
		# plt.imshow( self.level_set_function )
		# plt.colorbar()
		# plt.subplot( 1, 4, 4 )
		# plt.plot( self.level_set_function[ :, 15 ], linewidth=2, color='b' )
		# plt.show()

		return reshape_V_tilde

	def update( self, velocity_field ):

		padded_velocity_field = np.pad( velocity_field, ( ( 2, 2 ), ( 2, 2 ) ), mode='constant' )

		extended_velocity = self.extend_velocity_hilbertian( padded_velocity_field )

		# import matplotlib.pyplot as plt
		# plt.subplot( 1, 3, 1 )
		# plt.imshow( extended_velocity )
		# plt.subplot( 1, 3, 2 )
		# plt.imshow( velocity_field )
		# plt.subplot( 1, 3, 3 )
		# plt.imshow( self.level_set_function )
		# plt.show()

		delta_ij_plus = np.zeros( self.level_set_function.shape )
		delta_ij_minus = np.zeros( self.level_set_function.shape )

		D_xplus = np.zeros( self.level_set_function.shape )
		D_xminus = np.zeros( self.level_set_function.shape )
		D_yplus = np.zeros( self.level_set_function.shape )
		D_yminus = np.zeros( self.level_set_function.shape )

		D_x2 = np.zeros( self.level_set_function.shape )
		D_y2 = np.zeros( self.level_set_function.shape )

		#delta_t = 20#1.0
		delta_t = 2
		num_steps = 1#5

		c_iso = 0#1e-3

		print( "Max velocity = " + str( np.max( np.abs( extended_velocity ) ) ) )

		for step_idx in range( 0, num_steps ):
			for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
				for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):

					D_xplus[ x_idx, y_idx ] = self.level_set_function[ x_idx + 1, y_idx ] - self.level_set_function[ x_idx, y_idx ]
					D_xminus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx ] - self.level_set_function[ x_idx - 1, y_idx ]

					D_yplus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx + 1 ] - self.level_set_function[ x_idx, y_idx ]
					D_yminus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx ] - self.level_set_function[ x_idx, y_idx - 1 ]

					D_x2[ x_idx, y_idx ] = self.level_set_function[ x_idx + 1, y_idx ] - 2 * self.level_set_function[ x_idx, y_idx ] + self.level_set_function[ x_idx - 1, y_idx ]
					D_y2[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx + 1 ] - 2 * self.level_set_function[ x_idx, y_idx ] + self.level_set_function[ x_idx, y_idx - 1 ]

			delta_ij_plus = np.sqrt(
				( np.minimum( D_xplus, 0 ) )**2 + ( np.maximum( D_xminus, 0 ) )**2 +
				( np.minimum( D_yplus, 0 ) )**2 + ( np.maximum( D_yminus, 0 ) )**2
			)

			delta_ij_minus = np.sqrt(
				( np.maximum( D_xplus, 0 ) )**2 + ( np.minimum( D_xminus, 0 ) )**2 +
				( np.maximum( D_yplus, 0 ) )**2 + ( np.minimum( D_yminus, 0 ) )**2
			)

			self.level_set_function -= delta_t * (
				np.maximum( -extended_velocity, 0 ) * delta_ij_plus + np.minimum( -extended_velocity, 0 ) * delta_ij_minus -
				c_iso * ( D_x2 + D_y2 )
			)
