import numpy as np
import matplotlib.pyplot as plt
from LevelSet import LevelSet
import OptimizationState


class LevelSetCMOS( OptimizationState.OptimizationState ):

	def __init__(
		self,
		permittivity_bounds,
		opt_mesh_size_um, opt_vertical_start_um, opt_width_um,
		minimum_feature_gap_spacing_voxels, layer_thicknesses_um, layer_spacings_um,
		num_iterations, num_epochs, filename_prefix ):

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


	def randomize_layer_profiles( self, feature_gap_width_sigma_voxels, feature_probability ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			space_remaining_voxels = self.opt_width_num_voxels

			self.layer_profiles[ profile_idx][ : ] = 1.0

			while space_remaining_voxels >= self.minimum_feature_gap_spacing_voxels[ profile_idx ]:
				choose_density_value = 1.0 * ( np.random.uniform( 0, 1 ) < feature_probability )
				choose_width_voxels = int(
					np.minimum( space_remaining_voxels,
						self.minimum_feature_gap_spacing_voxels[ profile_idx ] + np.maximum( 0, np.random.normal( 0, feature_gap_width_sigma_voxels, 1 ) ) )
				)

				end_pt = space_remaining_voxels
				start_pt = np.maximum( end_pt - choose_width_voxels, 0 )

				space_remaining_voxels -= choose_width_voxels

				self.layer_profiles[ profile_idx ][ start_pt : end_pt ] = choose_density_value

	def assemble_level_sets( self ):
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			layer_density = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				layer_density[ :, internal_idx ] = self.layer_profiles[ profile_idx ]

			self.level_sets[ profile_idx ] = LevelSet.LevelSet( layer_density.shape, self.level_set_boundary_smoothing )
			self.level_sets[ profile_idx ].init_with_density( layer_density )


	def assemble_index( self, device_idx=1 ):
		return np.sqrt(
					self.permittivity_bounds[ 0 ] +
					( self.permittivity_bounds[ 1 ] - self.permittivity_bounds[ 0 ] ) * self.assemble_density() )

	def assemble_density( self ):
		device_density = np.zeros( ( self.opt_width_num_voxels, self.opt_vertical_num_voxels ) )

		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			get_profile = self.level_sets[ profile_idx ].device_density_from_level_set()
			get_profile = get_profile[ :, 1 ]

			for internal_idx in range( 0, self.layer_thicknesses_voxels[ profile_idx ] ):
				device_density[ :, get_start + internal_idx ] = get_profile#self.layer_profiles[ profile_idx ]

		return device_density

	def update( self, gradient_real, graident_imag, gradient_real_lsf, gradient_imag_lsf, epoch, iteration ):

		gradient_real_interpolate = self.reinterpolate( np.squeeze( gradient_real ), [ self.opt_width_num_voxels, self.opt_vertical_num_voxels ] )

		max_abs_movement = 0
		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			average_velocity = np.mean( -gradient_real_interpolate[ :, get_start : get_end ], axis=1 )

			max_abs_movement = np.maximum( max_abs_movement, np.max( np.abs( average_velocity ) ) )


		for profile_idx in range( 0, len( self.layer_profiles ) ):
			get_start = np.sum( self.layer_thicknesses_voxels[ 0 : profile_idx ] ) + np.sum( self.spacer_thicknesses_voxels[ 0 : profile_idx ] )
			get_end = get_start + self.layer_thicknesses_voxels[ profile_idx ]

			average_velocity = np.mean( -gradient_real_interpolate[ :, get_start : get_end ], axis=1 ) / max_abs_movement
			expand_velocity = np.zeros( ( self.opt_width_num_voxels, 3 ) )

			for internal_idx in range( 0, 3 ):
				expand_velocity[ :, internal_idx ] = average_velocity

			get_lsf_layer = self.level_sets[ profile_idx ]
			get_lsf_layer.update( expand_velocity, 5 )

		#
		# After each update, we need to go through and make sure nothing violates the feature size.
		# We may have feature/gap size by layer at some point (i.e. - M1 can have 90nm while the rest have 100nm)
		#

		#
		# We will pick a random starting point each time through the profile and fix it up that way.  This way we aren't
		# always undoing the same changes
		#

		# for profile_idx in range( 0, len( self.layer_profiles ) ):
		# 	get_profile = self.level_sets[ profile_idx ].device_density_from_level_set()
		# 	get_profile = np.squeeze( get_profile[ :, 1 ] )

		# 	flip = np.random.random( 1 )[ 0 ] > 0.5

		# 	if flip:
		# 		get_profile = np.flip( get_profile )

		# 	cur_loc = int( np.random.random( 1 )[ 0 ] * self.opt_width_num_voxels )
		# 	start_value = get_profile[ cur_loc ] > 0.5

		# 	while ( cur_loc > 0 ):
		# 		cur_loc -= 1

		# 		cur_value = get_profile[ cur_loc ] > 0.5

		# 		if cur_value != start_value:
		# 			cur_loc += 1
		# 			break

		# 	num_voxels_scanned = 0
		# 	while ( num_voxels_scanned < self.opt_width_num_voxels ):

		# 		num_voxels_feature_gap = 0

		# 		cur_value = get_profile[ cur_loc ] > 0.5
		# 		start_loc = cur_loc
		# 		hit_end = False

		# 		while ( cur_value == start_value ) and ( num_voxels_feature_gap < self.opt_width_num_voxels ):
		# 			num_voxels_feature_gap += 1

		# 			cur_loc += 1

		# 			if cur_loc == self.opt_width_num_voxels:
		# 				hit_end = True
		# 				cur_loc = 0
		# 				break

		# 			cur_value = get_profile[ cur_loc ] > 0.5

		# 			num_voxels_scanned += 1


		# 		if ( num_voxels_feature_gap < self.minimum_feature_gap_spacing_voxels[ profile_idx ] ):
		# 			if ( start_loc > 0 ) and ( not hit_end ):
		# 				while ( num_voxels_feature_gap < self.minimum_feature_gap_spacing_voxels[ profile_idx ] ) and ( cur_loc < self.opt_width_num_voxels ):
		# 					get_profile[ cur_loc ] = 1.0 * start_value

		# 					cur_loc += 1
		# 					num_voxels_feature_gap += 1
		# 					num_voxels_scanned += 1

		# 		cur_loc = ( cur_loc % self.opt_width_num_voxels )
		# 		start_value = get_profile[ cur_loc ] > 0.5

		# 	if flip:
		# 		get_profile = np.flip( get_profile )


		# 	new_layer_density = np.zeros( ( self.opt_width_num_voxels, 3 ) )

		# 	for internal_idx in range( 0, 3 ):
		# 		new_layer_density[ :, internal_idx ] = get_profile

		# 	self.level_sets[ profile_idx ].init_with_density( new_layer_density )

	def save_design( self, filebase, epoch ):
		return





