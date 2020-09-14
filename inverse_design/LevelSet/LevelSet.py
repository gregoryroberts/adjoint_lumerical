import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

#
# A signed distance representation a hold where the inner points have a negative value for the level set function,
# thus placing them in the void domain.  Everywhere else will be in the solid domain.
#
def eval_signed_distance_hole( x, y, x0, y0, radius ):
	return ( np.sqrt( ( x - x0 )**2 + ( y - y0 )**2 ) - radius )

#
# Function used for blurring a density value near boundaries.
# 
def eval_heaviside( x, eta ):
	x_over_eta = x / eta
	middle = 0.5 * ( 1 + ( x_over_eta ) + ( 1 / np.pi ) * np.sin( np.pi * x_over_eta ) )

	return 1.0 * np.greater( x, eta ) + ( 1 - 1.0 * np.less( x, -eta ) ) * np.less_equal( x, eta ) * middle

class LevelSet():

	#
	# Create a level set object.  The dimesion should be the 2D dimension of the design area.  So if you are doing
	# 3D device with distinct layers, then you would create one of these classes per layer.
	# The boundary_smoothing_width controls the blurring over the boundary to use when computing a density from
	# the level set function.  It controls the width of the approximate Heaviside function used across boundaries.
	#
	def __init__( self, dimension, boundary_smoothing_width, connected=8 ):
		self.dimension = dimension
		self.padded_dimension = [ ( d + 4 ) for d in dimension ]
		self.search_bounds = [ [ 2, 2 + dim ] for dim in dimension ]

		self.level_set_function = np.zeros( self.padded_dimension )

		self.boundary_smoothing_width = boundary_smoothing_width

		self.connected = connected

		self.padded_meshgrid_x = np.zeros( ( self.padded_dimension[ 0 ], self.padded_dimension[ 1 ] ) )
		self.padded_meshgrid_y = np.zeros( ( self.padded_dimension[ 0 ], self.padded_dimension[ 1 ] ) )
		for x_idx in range( 0, self.padded_dimension[ 0 ] ):
			for y_idx in range( 0, self.padded_dimension[ 1 ] ):
				self.padded_meshgrid_x[ x_idx, y_idx ] = x_idx
				self.padded_meshgrid_y[ x_idx, y_idx ] = y_idx

		self.alpha_hilbertian = 1.0
		self.setup_hilbertian_velocity_extension_matrices( self.alpha_hilbertian )

	#
	# Initialize the level set function with a bunch of holes as specified by the given arrays.
	# The centers should be specified as a list of [x, y] coordinates.  The widths are an array of
	# radii for the assumed circular holes.  Elliptical holdes requires a change to eval_signed_distance_hole
	# above to allow for two widths.
	#
	def init_with_holes( self, init_hole_centers, init_hole_widths ):
		self.init_hole_centers = [ [ ( loc + 2 ) for loc in hole ] for hole in init_hole_centers ]
		self.init_hole_widths = init_hole_widths

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

	#
	# An initilaization for a structure density from a density-based optimization.  The density is assumed to be
	# bounded below and above by 0 and 1, respectively.
	#
	def init_with_density( self, density ):
		self.level_set_function = np.pad( density - 0.5, ( ( 2, 2 ), ( 2, 2 ) ), mode='edge' )
		self.signed_distance_reinitialization()

	#
	# If you are restarting an optimization and you have saved out the level set function from this class or would like
	# to initialize it directly, you can use this version of initialization.  A signed distance re-initialization will be
	# run after setting the level set function.
	#
	def init_with_level_set_function( self, lsf ):
		self.level_set_function = lsf.copy()
		self.signed_distance_reinitialization()

	#
	# A function for generally applying an operation based on distance from the boundaries of the level set function.
	# operation should take in a distance matrix, which specifies the distance from every point to the boundary of
	# the level set.  It should return whatever you would like to return from this function.  For example, for signed
	# distance reinitialization, the operation returns the sign of the level set function point multiplied by the distance
	# and returns this matrix to reset the level set function.
	#
	def distance_transform( self, operation ):
		border_points_x, border_points_y = self.find_border_points()

		distance_squared = np.inf * np.ones( self.padded_dimension )

		for border_point_idx in range( 0, len( border_points_x ) ):
			border_point_x_coord = border_points_x[ border_point_idx ]
			border_point_y_coord = border_points_y[ border_point_idx ]

			find_distance_squared = (
				( self.padded_meshgrid_x - border_point_x_coord )**2 +
				( self.padded_meshgrid_y - border_point_y_coord )**2 )
			distance_squared = np.minimum( distance_squared, find_distance_squared )

		return operation( np.sqrt( distance_squared ) )

	#
	# Reinitialize the level function to be a signed distance representation.
	#
	def signed_distance_reinitialization( self ):
		signed_distance_operation = lambda distance: np.sign( self.level_set_function ) * distance
		self.level_set_function = self.distance_transform( signed_distance_operation )

	#
	# Ask for the current binary device on a mesh cell basis.  So each voxel is either 0 or 1.  A better representation
	# would likely be to call find_border_points to retrieve the current boundaries of the level set.
	#
	def binarize( self ):
		padded_lsf = 1.0 * np.greater( self.level_set_function, 0.0 )
		return padded_lsf[ self.search_bounds[ 0 ][ 0 ] : self.search_bounds[ 0 ][ 1 ], self.search_bounds[ 1 ][ 0 ] : self.search_bounds[ 1 ][ 1 ] ]

	#
	# A coarse border representation for points that are in the solid domain and have at least one surrounding point in the
	# void domain.  This is used to isolate the input velocity to be restricted onto the border points.
	#
	def find_border_representation( self ):
		binary_representation = 1.0 * np.greater( self.level_set_function, 0.0 )

		border_representation = np.zeros( binary_representation.shape )

		# border_points_x, border_points_y = self.find_border_points()

		# for point_idx in range( 0, len( border_points_x ) ):
		# 	x_coord = border_points_x[ point_idx ]
		# 	y_coord = border_points_y[ point_idx ]

		# 	lower_x = int( x_coord )
		# 	lower_y = int( y_coord )
		# 	upper_x = int( x_coord + 1 )
		# 	upper_y = int( y_coord + 1 )

		# 	diff_lower_x = x_coord - lower_x
		# 	diff_lower_y = y_coord - lower_y
		# 	diff_upper_x = upper_x - x_coord
		# 	diff_upper_y = upper_y - y_coord

		# 	choose_x = upper_x
		# 	choose_y = upper_y
		# 	if diff_lower_x < diff_upper_x:
		# 		choose_x = lower_x
		# 	if diff_lower_y < diff_upper_y:
		# 		choose_y = lower_y

		# 	border_representation[ choose_x, choose_y ] = 1

			# border_representation[ lower_x, lower_y ] = np.sqrt( ( 1 - diff_lower_x )**2 + ( 1 - diff_lower_y )**2 )
			# border_representation[ upper_x, lower_y ] = np.sqrt( ( 1 - diff_upper_x )**2 + ( 1 - diff_lower_y )**2 )
			# border_representation[ lower_x, upper_y ] = np.sqrt( ( 1 - diff_lower_x )**2 + ( 1 - diff_upper_y )**2 )

			# if self.connected == 8:
			# 	border_representation[ upper_x, upper_y ] = np.sqrt( ( 1 - diff_upper_x )**2 + ( 1 - diff_upper_y )**2 )

		# import matplotlib.pyplot as plt
		# plt.plot( border_representation[ :, 3 ] )
		# plt.show()

		for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
			for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):
				center_point = binary_representation[ x_idx, y_idx ]
				surroundings = binary_representation[ ( x_idx - 1 ) : ( x_idx + 2 ), ( y_idx - 1 ) : ( y_idx + 2 ) ]

				if center_point == 1:
					if ( np.sum( surroundings ) < 9 ):
						border_representation[ x_idx, y_idx ] = 1

		return border_representation

	#
	# Find subpixel border points with a linear interpolation for where the level set function crosses 0.
	# This returns two arrays of values, one for the x-coordinates and another for the y-coordinates.
	#
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

							if ( self.connected == 4 ) and ( ( np.abs( shift_y - 1 ) + np.abs( shift_x - 1 ) ) >= 2 ):
								continue

							m_hat_x = ( shift_x - 1 ) / ( lsf_shift - lsf_center )
							b_hat_x = 1 - m_hat_x * lsf_center

							m_hat_y = ( shift_y - 1 ) / ( lsf_shift - lsf_center )
							b_hat_y = 1 - m_hat_y * lsf_center

							border_points_x.append( x_idx + b_hat_x - 1 )
							border_points_y.append( y_idx + b_hat_y - 1 )
							# border_points_x.append( x_idx )
							# border_points_y.append( y_idx )

		return border_points_x, border_points_y

	#
	# Compute a device density with smoothed Heaviside function across the boundaries.
	#
	def device_density_from_level_set( self ):
		density_from_distance = lambda distance: eval_heaviside( np.sign( self.level_set_function ) * distance, self.boundary_smoothing_width )

		padded_density = self.distance_transform( density_from_distance )

		return padded_density[
			self.search_bounds[ 0 ][ 0 ] : self.search_bounds[ 0 ][ 1 ],
			self.search_bounds[ 1 ][ 0 ] : self.search_bounds[ 1 ][ 1 ]
		]

	#
	# Setup function to create the sparse matrices we use for doing a Hilbertian velocity extension.
	# The matrix used is static and so we only create it one time.
	#
	def setup_hilbertian_velocity_extension_matrices( self, alpha ):
		vector_len = np.product( self.padded_dimension )

		Dx = lil_matrix( ( vector_len, vector_len ) )
		Dy = lil_matrix( ( vector_len, vector_len ) )
		identity = lil_matrix( ( vector_len, vector_len ) )

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

		self.M_to_invert = ( alpha * Dx_component + alpha * Dy_component + identity ).transpose()

	#
	# Given a velocity field defined on the whole domain, we restrict it to the boundary and then
	# extend it back to the domain using the Hilbertian extension method.
	#
	def extend_velocity_hilbertian( self, velocity_field ):
		border_representation = self.find_border_representation()

		boundary_velocity = border_representation * velocity_field
		
		sparse_g_omega = csc_matrix( np.transpose( boundary_velocity.flatten() ) )

		V_tilde = spsolve( self.M_to_invert, sparse_g_omega.transpose() )

		reshape_V_tilde = np.reshape( V_tilde, velocity_field.shape )

		return reshape_V_tilde

	#
	# Run a level set update given an input velocity field.  This should just be the usual gradient we get from
	# the density evaluation.  The boundary smoothing eases the requirement on breaking into electric and displacement
	# fields.  It asks for a delta_t which is how far to move the level set forward in time.  Right now, we only take
	# one step in time by default (num_steps) and this delta_t functions as a step size.  Note that the norm of
	# the velocity field should also be considered when providing this parameter.  After a step is taken, the boundary will
	# have moved, so it might be more correct to redo the velocity extension based on the input velocity field, but at this
	# point I am not sure and I have just been moving a single step in time with a certain delta_t value.
	#
	def update( self, velocity_field, delta_t, num_steps=1 ):

		padded_velocity_field = np.pad( velocity_field, ( ( 2, 2 ), ( 2, 2 ) ), mode='constant' )
		# padded_velocity_field = np.pad( velocity_field, ( ( 2, 2 ), ( 2, 2 ) ), mode='edge' )

		extended_velocity = self.extend_velocity_hilbertian( padded_velocity_field )

		# import matplotlib.pyplot as plt
		# plt.plot( extended_velocity[ :, 3 ] )
		# plt.plot( padded_velocity_field[ :, 3 ], linestyle='--' )
		# plt.show()

		delta_ij_plus = np.zeros( self.level_set_function.shape )
		delta_ij_minus = np.zeros( self.level_set_function.shape )

		D_xplus = np.zeros( self.level_set_function.shape )
		D_xminus = np.zeros( self.level_set_function.shape )
		D_yplus = np.zeros( self.level_set_function.shape )
		D_yminus = np.zeros( self.level_set_function.shape )

		for step_idx in range( 0, num_steps ):
			for x_idx in range( self.search_bounds[ 0 ][ 0 ], self.search_bounds[ 0 ][ 1 ] ):
				for y_idx in range( self.search_bounds[ 1 ][ 0 ], self.search_bounds[ 1 ][ 1 ] ):

					D_xplus[ x_idx, y_idx ] = self.level_set_function[ x_idx + 1, y_idx ] - self.level_set_function[ x_idx, y_idx ]
					D_xminus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx ] - self.level_set_function[ x_idx - 1, y_idx ]

					D_yplus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx + 1 ] - self.level_set_function[ x_idx, y_idx ]
					D_yminus[ x_idx, y_idx ] = self.level_set_function[ x_idx, y_idx ] - self.level_set_function[ x_idx, y_idx - 1 ]

			delta_ij_plus = np.sqrt(
				( np.minimum( D_xplus, 0 ) )**2 + ( np.maximum( D_xminus, 0 ) )**2 +
				( np.minimum( D_yplus, 0 ) )**2 + ( np.maximum( D_yminus, 0 ) )**2
			)

			delta_ij_minus = np.sqrt(
				( np.maximum( D_xplus, 0 ) )**2 + ( np.minimum( D_xminus, 0 ) )**2 +
				( np.maximum( D_yplus, 0 ) )**2 + ( np.minimum( D_yminus, 0 ) )**2
			)

			self.level_set_function -= delta_t * (
				np.maximum( -extended_velocity, 0 ) * delta_ij_plus + np.minimum( -extended_velocity, 0 ) * delta_ij_minus
			)

		core_lsf = self.level_set_function[
			self.search_bounds[ 0 ][ 0 ] : self.search_bounds[ 0 ][ 1 ], 
			self.search_bounds[ 1 ][ 0 ] : self.search_bounds[ 1 ][ 1 ] ]

		self.level_set_function = np.pad( core_lsf, ( ( 2, 2 ), ( 2, 2 ) ), mode='edge' )

