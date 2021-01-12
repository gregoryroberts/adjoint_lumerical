import numpy as np
import matplotlib.pyplot as plt
from LevelSet import LevelSet
import OptimizationState
from scipy import ndimage

import scipy.optimize

import sigmoid

do_sigmoid = False#True#False
sigmoid_start_iter = 50

do_binarization_opt = True

assert ( not( do_sigmoid and do_binarization_opt ) ), "Both sigmoid and forced binarization should not be on simultaneously"	


def compute_binarization( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )

def compute_binarization_gradient( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sign( input_variable - set_point )

def vector_norm( v_in ):
	return np.sqrt( np.sum( np.abs( v_in )**2 ) )


def upsample_nearest( profile, upsampled_length ):
	cur_length = len( profile )

	assert ( upsampled_length % cur_length ) == 0, "Expected an even nearest upsampling"

	upsample_ratio = upsampled_length / cur_length

	upsampled_profile = np.zeros( upsampled_length )

	for idx in range( 0, upsampled_length ):
		down_idx = np.minimum( int( idx / upsample_ratio ), len( profile ) - 1 )

		upsampled_profile[ idx ] = profile[ down_idx ]

	return upsampled_profile

def downsample_nearest( profile, downsampled_length ):
	cur_length = len( profile )

	assert ( cur_length % downsampled_length ) == 0, "Expected an even nearest downsampling"

	downsample_ratio = int( cur_length / downsampled_length )

	downsampled_profile = np.zeros( downsampled_length )

	for idx in range( 0, downsampled_length ):
		up_idx = int( idx * downsample_ratio )

		downsampled_profile[ idx ] = profile[ up_idx ]

	return downsampled_profile


def downsample_average( profile, downsampled_length ):
	cur_length = len( profile )

	assert ( cur_length % downsampled_length ) == 0, "Expected an even average downsampling"

	downsample_ratio = int( cur_length / downsampled_length )

	downsampled_profile = np.zeros( downsampled_length )

	for idx in range( 0, downsampled_length ):
		start_profile_idx = idx * downsample_ratio
		end_profile_idx = start_profile_idx + downsample_ratio

		downsampled_profile[ idx ] = np.mean( profile[ start_profile_idx : end_profile_idx ] )

	return downsampled_profile

def downsample_average_2d( density, downsample_shape ):
	assert ( density.shape[ 0 ] % downsample_shape[ 0 ] ) == 0, "Expected an even average downsampling"
	assert ( density.shape[ 1 ] % downsample_shape[ 1 ] ) == 0, "Expected an even average downsampling"

	downsample_ratio = 1.0 * np.array( density.shape ) / np.array( downsample_shape )

	downsampled = np.zeros( downsample_shape )

	for x in range( 0, downsample_shape[ 0 ] ):
		start_x = int( x * downsample_ratio[ 0 ] )
		end_x = int( start_x + downsample_ratio[ 0 ] )

		for y in range( 0, downsample_shape[ 1 ] ):
			start_y = int( y * downsample_ratio[ 1 ] )
			end_y = int( start_y + downsample_ratio[ 1 ] )

			downsampled[ x, y ] = np.mean(
				density[ start_x : end_x, start_y : end_y ]
			)

	return downsampled

def upsample_nearest_2d( density, upsample_shape ):
	assert ( upsample_shape[ 0 ] % density.shape[ 0 ] ) == 0, "Expected an even nearest upsampling"
	assert ( upsample_shape[ 1 ] % density.shape[ 1 ] ) == 0, "Expected an even nearest upsampling"

	upsample_ratio = 1.0 * np.array( upsample_shape ) / np.array( density.shape )

	upsampled = np.zeros( upsample_shape )

	for x in range( 0, upsample_shape[ 0 ] ):
		down_x = np.minimum( int( x / upsample_ratio[ 0 ] ), density.shape[ 0 ] - 1 )
		for y in range( 0, upsample_shape[ 1 ] ):
			down_y = np.minimum( int( y / upsample_ratio[ 1 ] ), density.shape[ 1 ] - 1 )

			upsampled[ x, y ] = density[ down_x, down_y ]

	return upsampled


class ContinuousCMOS( OptimizationState.OptimizationState ):

	#
	# TODO: Is it bad to have a default value of 1.0 for background density?
	#
	def __init__(
		self,
		permittivity_bounds,
		opt_mesh_size_um, opt_vertical_start_um, opt_width_um,
		minimum_feature_gap_spacing_um, layer_thicknesses_um, layer_spacings_um,
		num_iterations, num_epochs, filename_prefix, device_background_density ):

		super( ContinuousCMOS, self ).__init__( num_iterations, num_epochs, filename_prefix )

		self.permittivity_bounds = permittivity_bounds
		self.opt_mesh_size_um = opt_mesh_size_um

		self.layer_thicknesses_um = layer_thicknesses_um
		self.layer_spacings_um = layer_spacings_um

		self.num_devices = 1

		self.minimum_feature_gap_spacing_um = minimum_feature_gap_spacing_um
		self.minimum_feature_gap_spacing_small_voxels = [ int( minimum_feature_gap_spacing_um[ i ] / opt_mesh_size_um ) for i in range( 0, len( minimum_feature_gap_spacing_um ) ) ]

		self.layer_thicknesses_voxels = np.array( [ int( thickness_um / self.opt_mesh_size_um ) for thickness_um in self.layer_thicknesses_um ] )
		self.spacer_thicknesses_voxels = np.array( [ int( spacing_um / self.opt_mesh_size_um ) for spacing_um in self.layer_spacings_um ] )

		self.opt_width_um = opt_width_um

		self.opt_vertical_start_um = opt_vertical_start_um
		self.opt_vertical_end_um = self.opt_vertical_start_um + np.sum( self.layer_thicknesses_um + self.layer_spacings_um )

		self.opt_vertical_num_voxels = int( ( self.opt_vertical_end_um - self.opt_vertical_start_um ) / self.opt_mesh_size_um )
		self.opt_width_num_voxels = int( self.opt_width_um / self.opt_mesh_size_um )

		self.layer_profiles = []
		for profile_idx in range( 0, len( self.layer_thicknesses_um ) ):
			# profile_width_voxels = self.minimum_feature_gap_spacing_voxels[ profile_idx ]
			profile_width_voxels = int( self.opt_width_um / self.minimum_feature_gap_spacing_um[ profile_idx ] )
			self.layer_profiles.append( np.zeros( profile_width_voxels ) )

		self.level_sets = [ None for idx in range( 0, len( self.layer_thicknesses_um ) ) ]
		self.level_set_boundary_smoothing = 1
		# self.level_set_boundary_smoothing = 7

		self.device_background_density = device_background_density

	def profiles_from_swarm_particle_positions( self, profile_idx, positions ):
		profiles = []

		for particle_idx in range( 0, len( positions ) ):
			profiles.append( self.profile_from_swarm_particle_position( profile_idx, positions[ particle_idx ] ) )

		return profiles


	def profile_from_swarm_particle_position( self, profile_idx, position ):
		vector_dim = len( position )
		erosion_dilation_limit = int( 0.5 * ( self.minimum_feature_gap_spacing_voxels[ profile_idx ] + 1 ) )
		k_cutoff = int( 0.5 * vector_dim / self.minimum_feature_gap_spacing_voxels[ profile_idx ] )
		vector_mid_pt = int( 0.5 * vector_dim )

		assert ( vector_dim % 2 ) == 1, "We are expected positions to be odd-length vectors"

		position[ vector_mid_pt ] = np.real( position[ vector_mid_pt ] )
		position[ 0 : vector_mid_pt ] = np.flip( np.conj( position[ vector_mid_pt + 1 : ] ) )

		position[ vector_mid_pt + k_cutoff + 1 : vector_dim ] = 0
		position[ 0 : vector_mid_pt - k_cutoff ] = 0

		spatial_profile = np.greater_equal( np.real( np.fft.ifft( np.fft.ifftshift( position ) ) ), 0 )

		for idx in range( 0, erosion_dilation_limit ):
			spatial_profile = ndimage.binary_dilation( spatial_profile ).astype( spatial_profile.dtype )

		for idx in range( 0, erosion_dilation_limit ):
			spatial_profile = ndimage.binary_erosion( spatial_profile ).astype( spatial_profile.dtype )

		for idx in range( 0, erosion_dilation_limit ):
			spatial_profile = ndimage.binary_erosion( spatial_profile ).astype( spatial_profile.dtype )

		for idx in range( 0, erosion_dilation_limit ):
			spatial_profile = ndimage.binary_dilation( spatial_profile ).astype( spatial_profile.dtype )
	
		return spatial_profile

	def init_swarm_positions( self, profile_idx, num_particles ):
		vector_dim = len( self.layer_profiles[ profile_idx ] )
		erosion_dilation_limit = int( 0.5 * ( self.minimum_feature_gap_spacing_voxels[ profile_idx ] + 1 ) )
		k_cutoff = int( 0.5 * vector_dim / self.minimum_feature_gap_spacing_voxels[ profile_idx ] )
		vector_mid_pt = int( 0.5 * vector_dim )

		assert ( vector_dim % 2 ) == 1, "We are expected positions to be odd-length vectors"

		swarm_positions = []

		for particle_idx in range( 0, num_particles ):
			position = ( np.random.random( vector_dim ) - 0.5 ) + 1j * ( np.random.random( vector_dim ) - 0.5 )
			position[ vector_mid_pt ] = np.real( position[ vector_mid_pt ] )
			position[ 0 : vector_mid_pt ] = np.flip( np.conj( position[ vector_mid_pt + 1 : ] ) )
			fft_cutoff = int( vector_dim / erosion_dilation_limit )

			position[ vector_mid_pt + k_cutoff + 1 : vector_dim ] = 0
			position[ 0 : vector_mid_pt - k_cutoff ] = 0

			swarm_positions.append( position )

		return swarm_positions

	def run_swarm( self, profile_idx, num_particles, num_iterations, eval_fn, omega=1.0, phi_p=0.5, phi_g=0.5, learning_rate=1.0 ):
		save_level_sets = self.level_sets.copy()
		save_profiles = self.layer_profiles.copy()

		baseline_performance = eval_fn( self.assemble_index() )

		particle_positions = self.init_swarm_positions( profile_idx, num_particles )
		best_positions = particle_positions.copy()

		def run_iteration( particle_positions ):
			foms = []

			profiles = self.profiles_from_swarm_particle_positions( profile_idx, particle_positions )
			for particle_idx in range( 0, num_particles ):
				self.layer_profiles[ profile_idx ] = profiles[ particle_idx ]
				self.assemble_level_sets()

				foms.append( eval_fn( self.assemble_index() ) )

			return foms

		def search_best_fom( foms, positions ):
			best_fom = -np.inf
			best_position = None

			for particle_idx in range( 0, num_particles ):
				get_fom = foms[ particle_idx ]
				if get_fom >= best_fom:
					best_fom = get_fom
					best_position = positions[ particle_idx ].copy()

			return best_fom, best_position

		def random_velocities( particle_length ):
			velocities = []

			for particle_idx in range( 0, num_particles ):
				real_random_velocity = np.random.random( particle_length ) - 0.5
				imag_random_velocity = np.random.random( particle_length ) - 0.5

				norm = np.sqrt( np.sum( real_random_velocity**2 + imag_random_velocity**2 ) )
				real_random_velocity /= norm
				imag_random_velocity /= norm

				velocities.append( real_random_velocity + 1j * real_random_velocity )

			return velocities


		particle_foms = run_iteration( particle_positions )
		best_particle_foms = particle_foms.copy()

		init_particle_foms = particle_foms.copy()

		best_overall_fom, best_overall_position = search_best_fom( particle_foms, particle_positions )

		particle_velocities = random_velocities( len( self.layer_profiles[ profile_idx ] ) )

		for iter_idx in range( 1, num_iterations ):
			for particle_idx in range( 0, num_particles ):

				get_position = particle_positions[ particle_idx ]
				new_position = get_position.copy()
				best_position = best_positions[ particle_idx ]
				get_velocity = particle_velocities[ particle_idx ]
				new_velocity = get_velocity.copy()

				for idx in range( 0, len( get_position ) ):
					rp = np.random.random()
					rg = np.random.random()

					new_velocity[ idx ] = (
						omega * get_velocity[ idx ] +
						phi_p * rp * ( best_position[ idx ] - get_position[ idx ] ) +
						phi_g * rg * ( best_overall_position[ idx ] - get_position[ idx ] ) )

				new_position = get_position + learning_rate * new_velocity

				particle_positions[ particle_idx ] = new_position


			particle_foms = run_iteration( particle_positions )
			for particle_idx in range( 0, num_particles ):
				new_particle_fom = particle_foms[ particle_idx ]

				if new_particle_fom >= best_particle_foms[ particle_idx ]:
					best_particle_foms[ particle_idx ] = new_particle_fom
					best_positions[ particle_idx ] = particle_positions[ particle_idx ].copy()

				if new_particle_fom >= best_overall_fom:
					best_overall_fom = new_particle_fom
					best_overall_position = particle_positions[ particle_idx ].copy()


		successful_swarming = False
		if best_overall_fom >= baseline_performance:

			self.layer_profiles[ profile_idx ] = self.profile_from_swarm_particle_position( profile_idx, best_overall_position )
			successful_swarming = True

		else:
			self.layer_profiles[ profile_idx ] = save_profiles[ profile_idx ].copy()

		self.assemble_level_sets()

		return successful_swarming, init_particle_foms, particle_foms


	# def swarm( self, profile_idx ):
	'''
		1. Initialize random profile and then optimize via level set until we find a local minimum for the device.
			a. For this, likeley need a good way to measure some amount of convergence
		2. Initialize a swarm on a random profile index
			a. Choose number of particles
			b. Choose a termination condition (maybe best device after certain number of iterations or wait until you
			get a certain level better of a device)
		3. Repeat local optimization to get to best spot with respect to the whole device
	'''

	def init_profiles_with_density( self, density ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_mid = int( get_start + 0.5 * self.layer_thicknesses_voxels[ profile_idx ] )

			self.layer_profiles[ profile_idx ] = downsample_nearest( density[ :, get_mid ], len( self.layer_profiles[ profile_idx ] ) )

	def randomize_layer_profiles( self, average_density, density_sigma ):
		self.average_density = average_density
		self.density_sigma = density_sigma

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			self.layer_profiles[ profile_idx ] = self.single_random_layer_profile( profile_idx )

	def uniform_layer_profiles( self, uniform_density ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			self.layer_profiles[ profile_idx ][ : ] = uniform_density

	def single_random_layer_profile( self, profile_idx ):
		return self.single_random_layer_profile_specified( profile_idx, self.average_density, self.density_sigma )

	def single_random_layer_profile_specified( self, profile_idx, average_density, density_sigma ):
		profile = np.random.uniform( average_density, density_sigma, len( self.layer_profiles[ profile_idx ] ) )
		profile = np.minimum( 1.0, np.maximum( profile, 0.0 ) )

		return profile

	def set_layer_profiles( self, profiles ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			self.layer_profiles[ profile_idx ] = profiles[ profile_idx ]

		# self.assemble_level_sets()

	def assemble_level_sets( self ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			layer_density = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				layer_density[ :, internal_idx ] = self.layer_profiles[ profile_idx ]

			self.level_sets[ profile_idx ] = LevelSet.LevelSet( layer_density.shape, self.level_set_boundary_smoothing, 4 )
			self.level_sets[ profile_idx ].init_with_density( layer_density )

	def assemble_index( self, iteration, device_idx=1 ):
		return np.sqrt(
					self.permittivity_bounds[ 0 ] +
					( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * self.assemble_density( iteration ) )

	def assemble_density( self, iteration ):
		# sigmoid_strength = 2**( iteration / 25.0 )

		sigmoid_strength_exp = -4.5 + int( iteration / 15.0 )
		sigmoid_strength = 2**( sigmoid_strength_exp )

		sigmoid_obj = sigmoid.Sigmoid( sigmoid_strength, 0.5 )

		device_density = np.ones( ( self.opt_width_num_voxels, self.opt_vertical_num_voxels ) )

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )

			get_profile = self.layer_profiles[ profile_idx ]
			upsampled_profile = upsample_nearest( get_profile, self.opt_width_num_voxels )

			if ( iteration > sigmoid_start_iter ) and do_sigmoid:
				upsampled_profile = sigmoid_obj.forward( upsampled_profile )

			for internal_idx in range( 0, self.layer_thicknesses_voxels[ profile_idx ] ):
				device_density[ :, get_start + internal_idx ] = upsampled_profile

		return device_density

	def get_layer_profiles( self ):
		profiles = []

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_profile = self.level_sets[ profile_idx ].device_density_from_level_set()
			get_profile = get_profile[ :, 1 ]

			profiles.append( get_profile )

		return profiles

	def check_size_violations( self, profile, profile_size_voxel_limit ):

		def report_sizes( binary_profile ):
			get_sizes = []

			counter = 0

			while counter < len( binary_profile ):
				if not binary_profile[ counter ]:
					counter += 1
					continue
				else:
					sub_counter = counter

					while sub_counter < len( binary_profile ):
						if binary_profile[ sub_counter ]:
							sub_counter += 1
						else:
							break

					get_sizes.append( sub_counter - counter )

					counter = sub_counter

			return np.array( get_sizes )

		# We are implicitly assuming that on the edges, it is ok to have a small feature as long as it is of the high index variety
		pad_profile = np.pad(
			profile,
			( profile_size_voxel_limit, profile_size_voxel_limit ),
			mode='constant',
			constant_values=( self.device_background_density, self.device_background_density ) )

		binarize_profile = np.greater( profile, 0.5 )
		binarize_profile_negative = np.less_equal( profile, 0.5 )
		feature_sizes = report_sizes( binarize_profile )
		gap_sizes = report_sizes( binarize_profile_negative )

		feature_size_violations = np.sum( 1.0 * np.less( 0.5 + feature_sizes - profile_size_voxel_limit, 0 ) )
		gap_size_violations = np.sum( 1.0 * np.less( 0.5 + gap_sizes - profile_size_voxel_limit, 0 ) )
		total_size_violations = int( feature_size_violations + gap_size_violations )

		return ( total_size_violations > 0 )


	def layer_gradients( self, gradient ):
		gradient_real_interpolate = np.squeeze( gradient )

		gradient_real_interpolate = 0.25 * ( 
			gradient_real_interpolate[ 0 : gradient_real_interpolate.shape[ 0 ] - 1, 0 : gradient_real_interpolate.shape[ 1 ] - 1 ] +
			gradient_real_interpolate[ 1 : gradient_real_interpolate.shape[ 0 ], 0 : gradient_real_interpolate.shape[ 1 ] - 1 ] +
			gradient_real_interpolate[ 0 : gradient_real_interpolate.shape[ 0 ] - 1, 1 : gradient_real_interpolate.shape[ 1 ] ] +
			gradient_real_interpolate[ 1 : gradient_real_interpolate.shape[ 0 ], 1 : gradient_real_interpolate.shape[ 1 ] ]
		)

		gradient_real_interpolate = upsample_nearest_2d( gradient_real_interpolate, [ self.opt_width_num_voxels, self.opt_vertical_num_voxels ] )
		gradient_real_interpolate = ( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * gradient_real_interpolate

		profile_gradients = []

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			get_profile = self.layer_profiles[ profile_idx ]

			average_gradient = np.squeeze( np.mean( gradient_real_interpolate[ :, get_start : get_end ], axis=1 ) )
			downsampled_average_grad = downsample_average( average_gradient, len( self.layer_profiles[ profile_idx ] ) )

			profile_gradients.append( downsampled_average_grad.copy() )

		return profile_gradients



	#
	# Binarize_amount is a factor times the maximum movement possible
	#
	def step_binarize( self, gradient, density, binarize_amount_factor, binarize_max_movement, binarization_set_point=0.5, opt_mask=None ):

		density_for_binarizing = density.flatten()
		flatten_gradient = gradient.flatten()

		if opt_mask is None:
			flatten_opt_mask = np.ones( len( flatten_gradient ) )
		else:
			flatten_opt_mask = opt_mask.flatten()


		flatten_design_cuts = []
		flatten_fom_gradients = []
		extract_binarization_gradient = []

		for idx in range( 0, len( flatten_opt_mask ) ):
			if flatten_opt_mask[ idx ] > 0:
				flatten_design_cuts.append( density_for_binarizing[ idx ] )
				flatten_fom_gradients.append( flatten_gradient[ idx ] )

		flatten_design_cuts = np.array( flatten_design_cuts )
		flatten_fom_gradients = np.array( flatten_fom_gradients )
		extract_binarization_gradient = compute_binarization_gradient( flatten_design_cuts, binarization_set_point )

		beta = binarize_max_movement
		projected_binarization_increase = 0

		c = flatten_fom_gradients

		initial_binarization = compute_binarization( flatten_design_cuts, binarization_set_point )

		b = np.real( extract_binarization_gradient )

		lower_bounds = np.zeros( len( c ) )
		upper_bounds = np.zeros( len( c ) )

		for idx in range( 0, len( c ) ):
			upper_bounds[ idx ] = np.maximum( np.minimum( beta, 1 - flatten_design_cuts[ idx ] ), 0 )
			lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -flatten_design_cuts[ idx ] ), 0 )

		max_possible_binarization_change = 0
		for idx in range( 0, len( c ) ):
			if b[ idx ] > 0:
				max_possible_binarization_change += b[ idx ] * upper_bounds[ idx ]
			else:
				max_possible_binarization_change += b[ idx ] * lower_bounds[ idx ]
		
		alpha = binarize_amount_factor * max_possible_binarization_change

		def ramp( x ):
			return np.maximum( x, 0 )

		def opt_function( nu ):
			lambda_1 = ramp( nu * b - c )
			lambda_2 = c + lambda_1 - nu * b

			return -( -np.dot( lambda_1, upper_bounds ) + np.dot( lambda_2, lower_bounds ) + nu * alpha )

		tolerance = 1e-12

		bin_constraint = scipy.optimize.LinearConstraint(A = 1, lb = 0, ub = np.inf)
		optimization_solution_nu = scipy.optimize.minimize(
			opt_function, 0,
			method='trust-constr',
			constraints=[bin_constraint] ,
			options={'xtol': 1E-16, 'gtol': 1E-16, 'barrier_tol': 1E-16})


		nu_star = optimization_solution_nu.x
		lambda_1_star = ramp( nu_star * b - c )
		lambda_2_star = c + lambda_1_star - nu_star * b
		x_star = np.zeros( len( c ) )

		for idx in range( 0, len( c ) ):
			if lambda_1_star[ idx ] > 0:
				x_star[ idx ] = upper_bounds[ idx ]
			else:
				x_star[ idx ] = lower_bounds[ idx ]


		proposed_design_variable = flatten_design_cuts + x_star
		proposed_design_variable = np.minimum( np.maximum( proposed_design_variable, 0 ), 1 )

		refill_idx = 0
		refill_design_variable = density_for_binarizing.copy()
		for idx in range( 0, len( flatten_opt_mask ) ):
			if flatten_opt_mask[ idx ] > 0:
				refill_design_variable[ idx ] = proposed_design_variable[ refill_idx ]
				refill_idx += 1

		return np.reshape( refill_design_variable, density.shape )


	def get_binarization( self, binarization_set_point=0.5 ):
		concatenate_profiles = []
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			for value in self.layer_profiles[ profile_idx ]:
				concatenate_profiles.append( value )

		return compute_binarization( np.array( concatenate_profiles ), binarization_set_point )


	def update( self, gradient_real, graident_imag, gradient_real_lsf, gradient_imag_lsf, epoch, iteration ):
		# gradient_real_interpolate = self.reinterpolate( np.squeeze( gradient_real ), [ self.opt_width_num_voxels, self.opt_vertical_num_voxels ] )
		# gradient_real_interpolate = ( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * gradient_real_interpolate

		gradient_real_interpolate = np.squeeze( gradient_real )

		# import matplotlib.pyplot as plt
		# plt.imshow( np.squeeze( gradient_real_interpolate ) )
		# plt.show()

		gradient_real_interpolate = 0.25 * ( 
			gradient_real_interpolate[ 0 : gradient_real_interpolate.shape[ 0 ] - 1, 0 : gradient_real_interpolate.shape[ 1 ] - 1 ] +
			gradient_real_interpolate[ 1 : gradient_real_interpolate.shape[ 0 ], 0 : gradient_real_interpolate.shape[ 1 ] - 1 ] +
			gradient_real_interpolate[ 0 : gradient_real_interpolate.shape[ 0 ] - 1, 1 : gradient_real_interpolate.shape[ 1 ] ] +
			gradient_real_interpolate[ 1 : gradient_real_interpolate.shape[ 0 ], 1 : gradient_real_interpolate.shape[ 1 ] ]
		)

		gradient_real_interpolate = upsample_nearest_2d( gradient_real_interpolate, [ self.opt_width_num_voxels, self.opt_vertical_num_voxels ] )
		gradient_real_interpolate = ( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * gradient_real_interpolate

		# sigmoid_strength = 2**( iteration / 25.0 )
		sigmoid_strength_exp = -4.5 + int( iteration / 15.0 )
		sigmoid_strength = 2**( sigmoid_strength_exp )

		sigmoid_obj = sigmoid.Sigmoid( sigmoid_strength, 0.5 )

		max_abs_movement = 0
		# avg_movement = 0
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			get_profile = self.layer_profiles[ profile_idx ]

			upsampled_profile = upsample_nearest( get_profile, self.opt_width_num_voxels )
			upsampled_profile_sig = sigmoid_obj.forward( upsampled_profile )

			sigmoid_grad = np.zeros( gradient_real_interpolate[ :, get_start : get_end ].shape )
			for sublayer in range( 0, self.layer_thicknesses_voxels[ profile_idx ] ):
				sigmoid_grad[ :, sublayer ] = sigmoid_obj.chain_rule( gradient_real_interpolate[ :, get_start + sublayer ], upsampled_profile_sig, upsampled_profile )


			average_gradient = np.squeeze( np.mean( gradient_real_interpolate[ :, get_start : get_end ], axis=1 ) )
			if ( iteration > sigmoid_start_iter ) and do_sigmoid:
				average_gradient = np.squeeze( np.mean( sigmoid_grad, axis=1 ) )

			downsampled_average_grad = downsample_average( average_gradient, len( self.layer_profiles[ profile_idx ] ) )

			# fig, ax = plt.subplots(constrained_layout=True)

			# ax.plot( downsampled_average_grad, color='g', linewidth=2 )

			# secax = ax.twiny()

			# secax.plot( average_gradient, color='r', linewidth=2, linestyle='--' )
			# plt.show()


			# plt.plot( average_gradient )
			# plt.show()

			max_abs_movement = np.maximum( max_abs_movement, np.max( np.abs( downsampled_average_grad ) ) )
			# avg_movement += ( 1. / len( self.layer_profiles ) ) * np.mean( np.abs( average_velocity ) )

			# max_abs_movement = np.maximum( max_abs_movement, np.max( np.abs( gradient_real_interpolate[ :, get_start : get_end ] ) ) )

		scaled_gradient = gradient_real_interpolate / max_abs_movement
		# scaled_gradient = gradient_real_interpolate / np.max( np.abs( gradient_real_interpolate ) )
		scaled_step_size = 0.05
		# scaled_step_size = 0.5
		# scaled_step_size = 1.0

		# import matplotlib.pyplot as plt
		# plt.imshow( np.squeeze( scaled_gradient ) )
		# plt.show()

		if not do_binarization_opt:

			for profile_idx in range( 0, len( self.layer_profiles ) ):
				get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
				get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

				upsampled_profile = upsample_nearest( get_profile, self.opt_width_num_voxels )
				upsampled_profile_sig = sigmoid_obj.forward( upsampled_profile )

				sigmoid_grad = np.zeros( scaled_gradient[ :, get_start : get_end ].shape )
				for sublayer in range( 0, self.layer_thicknesses_voxels[ profile_idx ] ):
					sigmoid_grad[ :, sublayer ] = sigmoid_obj.chain_rule( scaled_gradient[ :, get_start + sublayer ], upsampled_profile_sig, upsampled_profile )


				average_gradient = np.squeeze( np.mean( scaled_gradient[ :, get_start : get_end ], axis=1 ) )
				if ( iteration > sigmoid_start_iter ) and do_sigmoid:
					average_gradient = np.squeeze( np.mean( sigmoid_grad, axis=1 ) )

				get_profile = self.layer_profiles[ profile_idx ]
				downsampled_grad = downsample_average( average_gradient, len( self.layer_profiles[ profile_idx ] ) )

				get_profile -= scaled_step_size * downsampled_grad

				get_profile = np.minimum( 1.0, np.maximum( get_profile, 0.0 ) )

				self.layer_profiles[ profile_idx ] = get_profile.copy()

		else:

			concatenate_profiles = []
			for profile_idx in range( 0, len( self.layer_profiles ) ):
				for value in self.layer_profiles[ profile_idx ]:
					concatenate_profiles.append( value )
			concatenate_profiles = np.array( concatenate_profiles )

			concatenate_gradient = []

			for profile_idx in range( 0, len( self.layer_profiles ) ):
				get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
				get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

				upsampled_profile = upsample_nearest( get_profile, self.opt_width_num_voxels )
				average_gradient = np.squeeze( np.mean( scaled_gradient[ :, get_start : get_end ], axis=1 ) )

				get_profile = self.layer_profiles[ profile_idx ]
				downsampled_grad = downsample_average( average_gradient, len( self.layer_profiles[ profile_idx ] ) )

				for value in downsampled_grad:
					concatenate_gradient.append( value )
				concatenate_gradient = np.array( concatenate_gradient )


			permittivity_max_movement = 0.02
			density_max_movement = permittivity_max_movement / ( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] )
			binarize_movement_factor = 0.5
			
			proposed_step = np.minimum( 1.0,
				np.maximum( 0.0,
					self.step_binarize( concatenate_gradient, concatenate_profiles, binarize_movement_factor, density_max_movement )
				)
			)

			cur_counter = 0
			for profile_idx in range( 0, len( self.layer_profiles ) ):
				profile_length = len( self.layer_profiles[ profile_idx ] )
				self.layer_profiles[ profile_idx, : ] = proposed_step[ cur_counter : ( cur_counter + profile_length ) ]
				cur_counter += profile_length


	def save_design( self, filebase, epoch ):
		return





