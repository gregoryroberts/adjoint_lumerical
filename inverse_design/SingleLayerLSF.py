import numpy as np

def read_density_into_alpha( density ):
	return ( density - 0.5 )

def read_lsf_into_density( lsf ):
	return 1.0 * np.greater( lsf, 0 )

def gaussian_rbf( x0, y0, x, y, sigma ):
	return np.exp( -0.5 * ( ( x - x0 )**2 + ( y - y0 )**2 ) / sigma**2 )

def gaussian_rbf_grad( x0, y0, x, y, sigma ):
	rbf_value = gaussian_rbf( x0, y0, x, y, sigma )

	grad_x = -rbf_value * ( x - x0 ) / sigma**2
	grad_y = -rbf_value * ( y - y0 ) / sigma**2

	return grad_x, grad_y

def compute_lsf( alpha, rbf_sigma, rbf_eval_cutoff ):
	alpha_shape = alpha.shape

	lsf_fn = np.zeros( alpha_shape )

	for x_idx in range( 0, alpha_shape[ 0 ] ):
		for y_idx in range( 0, alpha_shape[ 1 ] ):

			x_left = np.maximum( x_idx - rbf_eval_cutoff, 0 )
			x_right = np.minimum( x_idx + rbf_eval_cutoff + 1, alpha_shape[ 0 ] )

			y_bottom = np.maximum( y_idx - rbf_eval_cutoff, 0 )
			y_top = np.minimum( y_idx + rbf_eval_cutoff + 1, alpha_shape[ 1 ] )

			lsf_fn_component = 0
			for sweep_x in range( x_left, x_right ):
				for sweep_y in range( y_bottom, y_top ):

					lsf_fn_component += alpha[ sweep_x, sweep_y ] * gaussian_rbf( sweep_x, sweep_y, x_idx, y_idx, rbf_sigma )

			lsf_fn[ x_idx, y_idx ] += lsf_fn_component
	
	lsf_fn /= ( ( rbf_eval_cutoff + 1 )**2 )
	return lsf_fn

def compute_lsf_gradient( alpha, x, y, rbf_sigma, rbf_eval_cutoff ):
	alpha_shape = alpha.shape

	x_left = np.maximum( int( x ) - rbf_eval_cutoff, 0 )
	x_right = np.minimum( int( x ) + rbf_eval_cutoff + 1, alpha_shape[ 0 ] )

	y_bottom = np.maximum( int( y ) - rbf_eval_cutoff, 0 )
	y_top = np.minimum( int( y ) + rbf_eval_cutoff + 1, alpha_shape[ 1 ] )

	lsf_grad_x = 0
	lsf_grad_y = 0
	for sweep_x in range( x_left, x_right ):
		for sweep_y in range( y_bottom, y_top ):

			grad_x_component, grad_y_component = gaussian_rbf_grad( sweep_x, sweep_y, x, y, rbf_sigma )

			lsf_grad_x += alpha[ sweep_x, sweep_y ] * grad_x_component
			lsf_grad_y += alpha[ sweep_x, sweep_y ] * grad_y_component

	lsf_grad_x /= ( ( 2 * rbf_eval_cutoff + 1 )**2 )
	lsf_grad_y /= ( ( 2 * rbf_eval_cutoff + 1 )**2 )
	return lsf_grad_x, lsf_grad_y

#
# E-fields should be broken down into [ pol, x, y, z ]
#
def alpha_perturbations( E_field_fwd, E_field_adj, current_lsf, alpha, rbf_sigma, rbf_eval_cutoff, eps_material, eps_void ):
	pad_lsf = np.pad( current_lsf, [ 1, 1 ], mode='constant' )

	lsf_shape = current_lsf.shape

	alpha_gradients = np.zeros( lsf_shape )

	for x_idx in range( 1, lsf_shape[ 0 ] - 1 ):
		for y_idx in range( 1, lsf_shape[ 1 ] - 1 ):
			current_lsf_value = current_lsf[ x_idx, y_idx ]
			# This means we are in the material region
			if current_lsf_value >= 0:

				lsf_region = current_lsf[ ( x_idx - 1 ) : ( x_idx + 2 ), ( y_idx - 1 ) : ( y_idx + 2 ) ]

				directional_checks_x = [ 0, 2, 1, 1 ]
				directional_checks_y = [ 1, 1, 0, 2 ]

				for direction in range( 0, len( directional_checks_x ) ):
					direction_x = directional_checks_x[ direction ]
					direction_y = directional_checks_y[ direction ]
					check_lsf = lsf_region[ direction_x, direction_y ]
					check_lsf_sign = np.sign( check_lsf ) < 0

					if check_lsf_sign:
						# We will consider this a border point.  Let's find the interpolated edge point.

						m_x = ( direction_x - 1 ) / ( check_lsf - current_lsf_value )
						b_x = 1 - m_x * current_lsf_value

						m_y = ( direction_y - 1 ) / ( check_lsf - current_lsf_value )
						b_y = 1 - m_y * current_lsf_value

						normalized_distance_coord_to_direction = (
							np.sqrt( ( b_x - direction_x )**2 + ( b_y - direction_y )**2 ) / np.sqrt( direction_x**2 + direction_y**2 )
						)

						E_field_fwd_interpolate = (
							normalized_distance_coord_to_direction * E_field_fwd[ :, x_idx - 1, y_idx - 1, : ] +
							( 1 - normalized_distance_coord_to_direction ) * E_field_fwd[ :, x_idx - 1 + direction_x, y_idx - 1 + direction_y, : ] )

						E_field_adj_interpolate = (
							normalized_distance_coord_to_direction * E_field_adj[ :, x_idx - 1, y_idx - 1, : ] +
							( 1 - normalized_distance_coord_to_direction ) * E_field_adj[ :, x_idx - 1 + direction_x, y_idx - 1 + direction_y, : ] )

						D_field_fwd_interpolate = (
							normalized_distance_coord_to_direction * eps_material * E_field_fwd[ :, x_idx - 1, y_idx - 1, : ] +
							( 1 - normalized_distance_coord_to_direction ) * eps_void * E_field_fwd[ :, x_idx - 1 + direction_x, y_idx - 1 + direction_y, : ] )

						D_field_adj_interpolate = (
							normalized_distance_coord_to_direction * eps_material * E_field_adj[ :, x_idx - 1, y_idx - 1, : ] +
							( 1 - normalized_distance_coord_to_direction ) * eps_void * E_field_adj[ :, x_idx - 1 + direction_x, y_idx - 1 + direction_y, : ] )

						# Let's get the gradient of the LSF at this point
						grad_x, grad_y = compute_lsf_gradient( alpha, b_x, b_y, rbf_sigma, rbf_eval_cutoff )
						grad_mag = np.sqrt( grad_x**2 + grad_y**2 )

						# This is the normal vector pointing from a material region to void region
						n_hat = -np.array( [ grad_x, grad_y ] ) / np.sqrt( grad_x**2 + grad_y**2 )

						perpendicular_direction = np.array( list( n_hat ) + [ 0 ] )

						gs_input_vector = np.ones( 3 ) / np.sqrt( 3 )
						parallel_direction = gs_input_vector - np.dot( perpendicular_direction, gs_input_vector ) * perpendicular_direction
						parallel_direction /= np.sqrt( np.sum( parallel_direction**2 ) )

						E_field_fwd_project_parallel = np.zeros( E_field_fwd_interpolate[ 1, : ].shape, dtype=np.complex )
						E_field_adj_project_parallel = np.zeros( E_field_adj_interpolate[ 1, : ].shape, dtype=np.complex )

						for pol_idx in range( 0, 3 ):
							E_field_fwd_project_parallel += parallel_direction[ pol_idx ] * E_field_fwd_interpolate[ pol_idx ]
							E_field_adj_project_parallel += parallel_direction[ pol_idx ] * E_field_adj_interpolate[ pol_idx ]

						parallel_gradient_component = ( eps_material - eps_void ) * E_field_fwd_project_parallel * E_field_adj_project_parallel


						D_field_fwd_project_perpendicular = np.zeros( D_field_fwd_interpolate[ 1, : ].shape, dtype=np.complex )
						D_field_adj_project_perpendicular = np.zeros( D_field_adj_interpolate[ 1, : ].shape, dtype=np.complex )

						for pol_idx in range( 0, 3 ):
							D_field_fwd_project_perpendicular += perpendicular_direction[ pol_idx ] * D_field_fwd_interpolate[ pol_idx ]
							D_field_adj_project_perpendicular += perpendicular_direction[ pol_idx ] * D_field_adj_interpolate[ pol_idx ]

						perpendicular_gradient_component = ( ( 1. / eps_void ) - ( 1. / eps_material ) ) * D_field_fwd_project_perpendicular * D_field_adj_project_perpendicular

						sum_gradient_components = parallel_gradient_component + perpendicular_gradient_component

						sum_gradient_components *= 1. / ( grad_mag * ( 2 * rbf_eval_cutoff + 1 )**2 )
						# Sum over the vertical direction
						sum_gradient_components = np.real( np.sum( sum_gradient_components ) )

						#
						# The affected alpha components will be from the basis functions that affect this part of the level set function
						#
						adjust_x = x_idx - 1
						adjust_y = y_idx - 1
						x_left = np.maximum( adjust_x - rbf_eval_cutoff, 0 )
						x_right = np.minimum( adjust_x + rbf_eval_cutoff + 1, lsf_shape[ 0 ] )

						y_bottom = np.maximum( adjust_y - rbf_eval_cutoff, 0 )
						y_top = np.minimum( adjust_y + rbf_eval_cutoff + 1, lsf_shape[ 1 ] )

						for sweep_x in range( x_left, x_right ):
							for sweep_y in range( y_bottom, y_top ):
								# The alpha's are just scales on the basis functions so their gradient is just equal to the value of the basis function
								# at a given point
								alpha_gradients[ sweep_x, sweep_y ] += sum_gradient_components * gaussian_rbf( sweep_x, sweep_y, adjust_x, adjust_y, rbf_sigma )

	return alpha_gradients



