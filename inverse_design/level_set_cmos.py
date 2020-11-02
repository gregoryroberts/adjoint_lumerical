import numpy as np
import matplotlib.pyplot as plt
from LevelSet import LevelSet
import OptimizationState
from scipy import ndimage


class LevelSetCMOS( OptimizationState.OptimizationState ):

	#
	# TODO: Is it bad to have a default value of 1.0 for background density?
	#
	def __init__(
		self,
		permittivity_bounds,
		opt_mesh_size_um, opt_vertical_start_um, opt_width_um,
		minimum_feature_gap_spacing_voxels, layer_thicknesses_um, layer_spacings_um,
		num_iterations, num_epochs, filename_prefix, device_background_density ):

		super( LevelSetCMOS, self ).__init__( num_iterations, num_epochs, filename_prefix )

		self.permittivity_bounds = permittivity_bounds
		self.opt_mesh_size_um = opt_mesh_size_um

		self.layer_thicknesses_um = layer_thicknesses_um
		self.layer_spacings_um = layer_spacings_um

		self.num_devices = 1

		self.minimum_feature_gap_spacing_voxels = minimum_feature_gap_spacing_voxels

		self.layer_thicknesses_voxels = np.array( [ int( thickness_um / self.opt_mesh_size_um ) for thickness_um in self.layer_thicknesses_um ] )
		self.spacer_thicknesses_voxels = np.array( [ int( spacing_um / self.opt_mesh_size_um ) for spacing_um in self.layer_spacings_um ] )

		self.opt_width_um = opt_width_um

		self.opt_vertical_start_um = opt_vertical_start_um
		self.opt_vertical_end_um = self.opt_vertical_start_um + np.sum( self.layer_thicknesses_um + self.layer_spacings_um )

		self.opt_vertical_num_voxels = int( ( self.opt_vertical_end_um - self.opt_vertical_start_um ) / self.opt_mesh_size_um )
		self.opt_width_num_voxels = int( self.opt_width_um / self.opt_mesh_size_um )

		self.layer_profiles = [ np.zeros( self.opt_width_num_voxels ) for idx in range( 0, len( self.layer_thicknesses_um ) ) ]

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

			self.layer_profiles[ profile_idx ] = density[ :, get_mid ]

		self.assemble_level_sets()

	def randomize_layer_profiles( self, feature_gap_width_sigma_voxels, feature_probability ):
		self.feature_gap_width_sigma_voxels = feature_gap_width_sigma_voxels
		self.feature_probability = feature_probability

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			self.layer_profiles[ profile_idx ] = self.single_random_layer_profile( profile_idx )

		self.assemble_level_sets()

	def single_random_layer_profile( self, profile_idx ):
		return self.single_random_layer_profile_specified( profile_idx, self.feature_gap_width_sigma_voxels, self.feature_probability )

	def single_random_layer_profile_specified( self, profile_idx, input_feature_gap_width_sigma_voxels, input_feature_probability ):
		space_remaining_voxels = self.opt_width_num_voxels

		random_profile = self.layer_profiles[ 0 ].copy()
		random_profile[ : ] = 1.0

		while space_remaining_voxels >= self.minimum_feature_gap_spacing_voxels[ profile_idx ]:
			choose_density_value = 1.0 * ( np.random.uniform( 0, 1 ) < input_feature_probability )
			choose_width_voxels = int(
				np.minimum( space_remaining_voxels,
					self.minimum_feature_gap_spacing_voxels[ profile_idx ] + np.maximum( 0, np.random.normal( 0, input_feature_gap_width_sigma_voxels, 1 ) ) )
			)

			end_pt = space_remaining_voxels
			start_pt = np.maximum( end_pt - choose_width_voxels, 0 )

			# if start_pt < self.minimum_feature_gap_spacing_voxels[ profile_idx ]:
				# Final feature would be too small, which is ok because it is high index which is the same
				# as our background material
				# break

			space_remaining_voxels -= choose_width_voxels

			random_profile[ start_pt : end_pt ] = choose_density_value


		fixup_edge = False
		fixup_edge_value = -1

		if int( np.sum( random_profile ) ) == 0:
			fixup_edge = True
			fixup_edge_value = 1
		elif int( np.sum( random_profile ) ) == len( random_profile ):
			fixup_edge = True
			fixup_edge_value = 0
		
		if fixup_edge:
			edge_start_point = int( np.random.uniform( 0, 1 ) * ( len( random_profile ) - self.minimum_feature_gap_spacing_voxels[ profile_idx ] ) )
			edge_start_point = np.minimum( edge_start_point, len( random_profile ) - self.minimum_feature_gap_spacing_voxels[ profile_idx ] )
			random_profile[ edge_start_point : edge_start_point + self.minimum_feature_gap_spacing_voxels[ profile_idx ] ] = fixup_edge_value

		return random_profile

	def set_layer_profiles( self, profiles ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			self.layer_profiles[ profile_idx ] = profiles[ profile_idx ]

		self.assemble_level_sets()

	def assemble_level_sets( self ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			layer_density = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				layer_density[ :, internal_idx ] = self.layer_profiles[ profile_idx ]

			self.level_sets[ profile_idx ] = LevelSet.LevelSet( layer_density.shape, self.level_set_boundary_smoothing, 4 )
			self.level_sets[ profile_idx ].init_with_density( layer_density )


	def assemble_index( self, device_idx=1 ):
		return np.sqrt(
					self.permittivity_bounds[ 0 ] +
					( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * self.assemble_density() )

	def assemble_density( self ):
		device_density = np.ones( ( self.opt_width_num_voxels, self.opt_vertical_num_voxels ) )

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			get_profile = self.level_sets[ profile_idx ].device_density_from_level_set()
			get_profile = get_profile[ :, 1 ]

			for internal_idx in range( 0, self.layer_thicknesses_voxels[ profile_idx ] ):
				device_density[ :, get_start + internal_idx ] = get_profile

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


	def update( self, gradient_real, graident_imag, gradient_real_lsf, gradient_imag_lsf, epoch, iteration ):

		save_pre_lsfs = self.level_sets.copy()

		pre_update_profiles = self.get_layer_profiles()

		gradient_real_interpolate = self.reinterpolate( np.squeeze( gradient_real ), [ self.opt_width_num_voxels, self.opt_vertical_num_voxels ] )

		pre_density = self.assemble_density()

		pre_profiles = []
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			pre_profiles.append( self.level_sets[ profile_idx ].device_density_from_level_set() )


		max_abs_movement = 0
		avg_movement = 0
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			average_velocity = np.mean( -gradient_real_interpolate[ :, get_start : get_end ], axis=1 )

			max_abs_movement = np.maximum( max_abs_movement, np.max( np.abs( average_velocity ) ) )
			avg_movement += ( 1. / len( self.layer_profiles ) ) * np.mean( np.abs( average_velocity ) )


		# todo: gdroberts - Why is level set function still fragmenting in the y-direction?
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			average_velocity = np.mean( -gradient_real_interpolate[ :, get_start : get_end ], axis=1 ) / max_abs_movement
			expand_velocity = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				expand_velocity[ :, internal_idx ] = average_velocity

			get_lsf_layer = self.level_sets[ profile_idx ]

			# This step size is tricky!
			# get_lsf_layer.update( expand_velocity, 5 )
			get_lsf_layer.update( expand_velocity, 2 )
			# get_lsf_layer.update( expand_velocity, 0.1 )
			get_lsf_layer.signed_distance_reinitialization()

		# return

		post_update_profiles = self.get_layer_profiles()

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			pre_update_profile = pre_update_profiles[ profile_idx ]
			post_update_profile = post_update_profiles[ profile_idx ]

			size_voxel_limit = self.minimum_feature_gap_spacing_voxels[ profile_idx ]

			check_sizes = self.check_size_violations( post_update_profile, size_voxel_limit )
			if not check_sizes:
				continue

			save_pre_profile = pre_update_profile.copy()

			delta_profile = np.greater( np.abs( 1.0 * np.greater( pre_update_profile, 0.5 ) - 1.0 * np.greater( post_update_profile, 0.5 ) ), 0.5 )

			flip = np.random.random( 1 )[ 0 ] > 0.5

			if flip:
				pre_update_profile = np.flip( pre_update_profile )
				post_update_profile = np.flip( post_update_profile )
				delta_profile = np.flip( delta_profile )
				save_pre_profile = np.flip( save_pre_profile )

			while np.sum( 1.0 * delta_profile ) > 0.5:
			
				random_start_point = int( np.random.random( 1 )[ 0 ] * self.opt_width_num_voxels ) % self.opt_width_num_voxels
				while delta_profile[ random_start_point ] == 0:
					random_start_point = ( ( random_start_point + 1 ) % self.opt_width_num_voxels )

				test_update_profile = pre_update_profile.copy()

				feature_type_at_delta = post_update_profile[ random_start_point ]
				
				fill_feature_or_gap_idx = random_start_point

				while ( fill_feature_or_gap_idx < self.opt_width_num_voxels ) and ( delta_profile[ fill_feature_or_gap_idx ] ):
					test_update_profile[ fill_feature_or_gap_idx ] = post_update_profile[ fill_feature_or_gap_idx ]
					delta_profile[ fill_feature_or_gap_idx ] = False
					fill_feature_or_gap_idx += 1

				fill_feature_or_gap_idx = random_start_point

				while ( fill_feature_or_gap_idx >= 0 ) and ( delta_profile[ fill_feature_or_gap_idx ] ):
					test_update_profile[ fill_feature_or_gap_idx ] = feature_type_at_delta
					delta_profile[ fill_feature_or_gap_idx ] = False
					fill_feature_or_gap_idx -= 1

				if not self.check_size_violations( test_update_profile, size_voxel_limit ):
					pre_update_profile = test_update_profile.copy()


			if flip:
				pre_update_profile = np.flip( pre_update_profile )
				post_update_profile = np.flip( post_update_profile )
				delta_profile = np.flip( delta_profile )
				save_pre_profile = np.flip( save_pre_profile )

			# import matplotlib.pyplot as plt
			# plt.plot( self.level_sets[ profile_idx ].device_density_from_level_set()[ :, 1 ], color='g' )
			# plt.plot( 1 + self.level_sets[ profile_idx ].device_density_from_level_set()[ :, 1 ], color='g' )

			new_layer_density = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				new_layer_density[ :, internal_idx ] = pre_update_profile

			self.level_sets[ profile_idx ].init_with_density( new_layer_density )

			# plt.plot( 1 + save_pre_lsfs[ profile_idx ].device_density_from_level_set()[ :, 1 ], color='k', linestyle='--' )
			# plt.plot( self.level_sets[ profile_idx ].device_density_from_level_set()[ :, 1 ], color='r', linestyle='--' )
			# plt.show()

		#
		# After each update, we need to go through and make sure nothing violates the feature size.
		# We may have feature/gap size by layer at some point (i.e. - M1 can have 90nm while the rest have 100nm)
		#

		#
		# We will pick a random starting point each time through the profile and fix it up that way.  This way we aren't
		# always undoing the same changes
		#

		'''

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_profile = self.level_sets[ profile_idx ].device_density_from_level_set()
			get_profile = np.squeeze( get_profile[ :, 1 ] )

			flip = np.random.random( 1 )[ 0 ] > 0.5

			if flip:
				get_profile = np.flip( get_profile )

			cur_loc = int( np.random.random( 1 )[ 0 ] * self.opt_width_num_voxels )
			start_value = get_profile[ cur_loc ] > 0.5

			while ( cur_loc > 0 ):
				cur_loc -= 1

				cur_value = get_profile[ cur_loc ] > 0.5

				if cur_value != start_value:
					cur_loc += 1
					break

			num_voxels_scanned = 0
			while ( num_voxels_scanned < self.opt_width_num_voxels ):

				num_voxels_feature_gap = 0

				cur_value = get_profile[ cur_loc ] > 0.5
				start_loc = cur_loc
				hit_end = False

				while ( cur_value == start_value ) and ( num_voxels_feature_gap < self.opt_width_num_voxels ):
					num_voxels_feature_gap += 1

					cur_loc += 1

					if cur_loc == self.opt_width_num_voxels:
						hit_end = True
						cur_loc = 0
						break

					cur_value = get_profile[ cur_loc ] > 0.5

					num_voxels_scanned += 1


				if ( num_voxels_feature_gap < self.minimum_feature_gap_spacing_voxels[ profile_idx ] ):
					if ( start_loc > 0 ) and ( not hit_end ):
						while ( num_voxels_feature_gap < self.minimum_feature_gap_spacing_voxels[ profile_idx ] ) and ( cur_loc < self.opt_width_num_voxels ):
							get_profile[ cur_loc ] = 1.0 * start_value

							cur_loc += 1
							num_voxels_feature_gap += 1
							num_voxels_scanned += 1

				cur_loc = ( cur_loc % self.opt_width_num_voxels )
				start_value = get_profile[ cur_loc ] > 0.5

			if flip:
				get_profile = np.flip( get_profile )


			new_layer_density = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				new_layer_density[ :, internal_idx ] = get_profile

			self.level_sets[ profile_idx ].init_with_density( new_layer_density )
		'''

	def save_design( self, filebase, epoch ):
		return





