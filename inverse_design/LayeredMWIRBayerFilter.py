import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur
import generic_blur_2d as generic_blur_2d

import two_pass_conn_comp

import numpy as np

from LayeredMWIRBayerFilterParameters import *



# Let's apply the topology on the device level!
def step_topo(design, design_change, make_device, pipeline_half_width):
	proposed_design_variable = np.minimum( np.maximum( design + design_change, 0.0 ), 1.0 )

	flip_threshold = 0.5

	# Here, we have already made an assumption that you don't make a difference from your standpoint
	# unless you cross over the threshold
	proposed_changes = np.logical_xor(
		np.greater(proposed_design_variable, flip_threshold),
		np.greater(design, flip_threshold))

	num_proposed_changes = np.sum(np.sum(proposed_changes))

	print("We are trying to change " + str(num_proposed_changes) + " voxels")

	# We have some we think we can safely move, so let's move those!
	# todo: need to make sure this is ok to do without checking because of the inexactness
	# of the blur filter
	design = np.add(
		np.multiply(np.logical_not(proposed_changes), proposed_design_variable),
		np.multiply(proposed_changes, design))

	if num_proposed_changes == 0:
		return design

	# Step 3: We will need ultimately a double padded w[0] w.r.t. the pipeline half width in each dimension.  This will also serve
	# as the stepped design as we slowly figure out which pieces to change
	padded_design = np.pad(
		design,
		(
			(2 * pipeline_half_width[0], 2 * pipeline_half_width[0]),
			(2 * pipeline_half_width[1], 2 * pipeline_half_width[1])
		),
		'constant'
	)

	# Step 4: We need here the current fabrication target before we start the stepping
	design_shape = design.shape

	current_device = make_device( design )

	padded_current_device = np.pad(
		current_device,
		(
			(pipeline_half_width[0] + 1, pipeline_half_width[0] + 1),
			(pipeline_half_width[1] + 1, pipeline_half_width[1] + 1)
		),
		'constant'
	)

	# Step 5: We are going to need to look at each of the changes one at a time to ensure we move safely through the space
	padded_design_shape = padded_design.shape
	# As is usually the case, these are inclusive start positions
	start_positions = np.array([pipeline_half_width[0], pipeline_half_width[1]])
	# May not be as obvious as start positions, but still usual case of exclusive end positions
	end_positions = padded_design_shape - start_positions

	# num = 0

	for x in range( 0, design_shape[ 0 ] ):
		for y in range( 0, design_shape[ 1 ]):

			# Do we want to change this voxel?
			if not proposed_changes[x, y]:
				continue

			# First, we extract the neighborhood we want to test the difference on
			original_value = design[ x, y ]
			padded_design[ x + 2 * pipeline_half_width[0], y + 2 * pipeline_half_width[1] ] = (
				proposed_design_variable[ x, y ] )
			stepped_design_neighborhood = padded_design[
				x : ( x + 4 * pipeline_half_width[0] + 1 ),
				y : ( y + 4 * pipeline_half_width[1] + 1 ) ].copy()

			stepped_device_neighborhood_middle = make_device( stepped_design_neighborhood )
			stepped_device_neighborhood = padded_current_device[
					x : (x + 2 * pipeline_half_width[0] + 3),
					y : (y + 2 * pipeline_half_width[1] + 3) ].copy()


			stepped_device_neighborhood[
					1 : (2 * pipeline_half_width[0] + 2),
					1 : (2 * pipeline_half_width[1] + 2) ] = stepped_device_neighborhood_middle[
					( pipeline_half_width[0] ) : ( 3 * pipeline_half_width[0] + 1 ),
					( pipeline_half_width[1] ) : ( 3 * pipeline_half_width[1] + 1 ) ]


			topo_check = two_pass_conn_comp.check_topology(
				padded_current_device[
					x : (x + 2 * pipeline_half_width[0] + 3),
					y : (y + 2 * pipeline_half_width[1] + 3) ],
				stepped_device_neighborhood)


			def snap_neutral_zone( input ):
				return input

			# todo(groberts): For now, no sigmoid posting, but we will come back to this if things seem to be working
			if not topo_check:
				# Technically the padded device does not need to change in here since we have a very binary
				# fabrication procedure.  But for other procedures, it is probably good to change that because
				# snapping to neutral zone borders could have adjacent neighbor effects
				snapped_value = snap_neutral_zone(original_value)
				padded_design[x + 2 * pipeline_half_width[0], y + 2 * pipeline_half_width[1]] = snapped_value
				design[x, y] = snapped_value
				continue

			# Does this get set above since padded design is padded version of w[0]?
			design[
				x, y] = proposed_design_variable[
					x, y ]

			padded_design[ x + 2 * pipeline_half_width[0], y + 2 * pipeline_half_width[1] ] = proposed_design_variable[
					x, y ]

			padded_current_device[
					x : (x + 2 * pipeline_half_width[0] + 3),
					y : (y + 2 * pipeline_half_width[1] + 3) ] = stepped_device_neighborhood

	return design


class LayeredMWIRBayerFilter(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers):
		super(LayeredMWIRBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

		self.num_z_layers = num_z_layers
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

		var1 = self.layering_z_0.forward(var0)
		self.w[1] = var1

		var2 = self.sigmoid_1.forward(var1)
		self.w[2] = var2

		var3 = np.zeros( self.w[3].shape )
		get_layer_idxs = self.layering_z_0.get_layer_idxs(self.w[0].shape)

		for layer in range( 0, self.layering_z_0.num_layers ):
			get_layer_idx = get_layer_idxs[ layer ]
			next_layer_idx = var3.shape[ 2 ]

			if layer < ( self.layering_z_0.num_layers - 1 ):
				next_layer_idx = get_layer_idxs[ layer + 1 ]

			do_blur = self.max_blur_xy_2.forward(
				var2[ :, :, get_layer_idx ] )

			for sublayer_idx in range( get_layer_idx, next_layer_idx ):
				var3[ :, :, sublayer_idx ] = do_blur

		self.w[3] = var3

		var4 = self.sigmoid_3.forward(var3)
		self.w[4] = var4

		var5 = self.scale_4.forward(var4)
		self.w[5] = var5


	#
	# Need to also override the backpropagation function
	#
	def backpropagate(self, gradient):
		gradient = self.scale_4.chain_rule(gradient, self.w[5], self.w[4])
		gradient = self.sigmoid_3.chain_rule(gradient, self.w[4], self.w[3])

		var3 = np.zeros( self.w[3].shape )
		get_layer_idxs = self.layering_z_0.get_layer_idxs(self.w[0].shape)

		for layer in range( 0, self.layering_z_0.num_layers ):
			get_layer_idx = get_layer_idxs[ layer ]
			next_layer_idx = var3.shape[ 2 ]

			if layer < ( self.layering_z_0.num_layers - 1 ):
				next_layer_idx = get_layer_idxs[ layer + 1 ]

			do_chain_rule = self.max_blur_xy_2.chain_rule(
				gradient[ :, :, get_layer_idx ],
				self.w[3][ :, :, get_layer_idx ],
				self.w[2][ :, :, get_layer_idx ] )

			for sublayer_idx in range( get_layer_idx, next_layer_idx ):
				gradient[ :, :, sublayer_idx ] = do_chain_rule

		gradient = self.sigmoid_1.chain_rule(gradient, self.w[2], self.w[1])
		gradient = self.layering_z_0.chain_rule(gradient, self.w[1], self.w[0])

		return gradient

	def update_filters(self, epoch):
		self.sigmoid_beta = 0.25 * (2**epoch)

		self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.filters = [self.layering_z_0, self.sigmoid_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4]

	def init_variables(self):
		super(LayeredMWIRBayerFilter, self).init_variables()

		self.w[0][ 0, :, : ] = 1
		self.w[0][ :, 0, : ] = 1
		self.w[0][ self.w[0].shape[ 0 ] - 1, :, : ] = 1
		self.w[0][ :, self.w[0].shape[ 1 ] - 1, : ] = 1

	def init_filters_and_variables(self):
		self.num_filters = 5
		self.num_variables = 1 + self.num_filters

		# Start the sigmoids at weak strengths
		self.sigmoid_beta = 0.0625
		self.sigmoid_eta = 0.5
		self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		z_voxel_layers = self.size[2]
		self.layering_z_0 = layering.Layering(z_dimension_idx, self.num_z_layers)

		alpha = 12
		self.blur_half_width = blur_half_width_voxels

		self.max_blur_xy_2 = generic_blur_2d.make_square_blur( alpha, self.blur_half_width )

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		self.scale_4 = scale.Scale([scale_min, scale_max])

		# Initialize the filter chain
		self.filters = [self.layering_z_0, self.sigmoid_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4]

		self.init_variables()
		self.update_permittivity()



	def step( self, gradient, step_size ):
		direction = self.backpropagate( gradient )
		proposed_change = -direction * step_size

		#
		# We would like to set the border so we can maintain different types of
		# connectivity on each layer
		#
		direction[ 0, :, : ] = 0
		direction[ :, 0, : ] = 0
		direction[ direction.shape[ 0 ] - 1, :, : ] = 0
		direction[ :, direction.shape[ 1 ] - 1, : ] = 0

		var0 = self.w[0]
		var1 = self.layering_z_0.forward(var0)

		get_layer_idxs = self.layering_z_0.get_layer_idxs(self.w[0].shape)

		def make_device( input_var ):
			return self.sigmoid_1.fabricate( self.max_blur_xy_2.fabricate( self.sigmoid_3.fabricate( input_var ) ) )

		for layer in range( 0, self.layering_z_0.num_layers ):
			get_layer_idx = get_layer_idxs[ layer ]

			this_design = var1[ :, :, get_layer_idx ]

			this_design = step_topo(
				this_design,
				step_size * direction[ :, :, get_layer_idx ],
				make_device,
				[ self.blur_half_width, self.blur_half_width ]
			)

			next_layer_idx = self.w[ 3 ].shape[ 2 ]

			if layer < ( self.layering_z_0.num_layers - 1 ):
				next_layer_idx = get_layer_idxs[ layer + 1 ]

			for sublayer_idx in range( get_layer_idx, next_layer_idx ):
				var0[ :, :, sublayer_idx ] = this_design

		self.w[0] = var0
		self.update_permittivity()



	def fabricate( self, variable ):
		var0 = self.w[0]
		var1 = self.layering_z_0.fabricate(var0)
		var2 = self.sigmoid_1.fabricate(var1)

		var3 = np.zeros( self.w[3].shape )
		get_layer_idxs = self.layering_z_0.get_layer_idxs(self.w[0].shape)

		for layer in range( 0, self.layering_z_0.num_layers ):
			get_layer_idx = get_layer_idxs[ layer ]
			next_layer_idx = var3.shape[ 2 ]

			if layer < ( self.layering_z_0.num_layers - 1 ):
				next_layer_idx = get_layer_idxs[ layer + 1 ]

			do_blur = self.max_blur_xy_2.fabricate(
				var2[ :, :, get_layer_idx ] )

			for sublayer_idx in range( get_layer_idx, next_layer_idx ):
				var3[ :, :, sublayer_idx ] = do_blur

		var4 = self.sigmoid_3.fabricate(var3)
		var5 = self.scale_4.fabricate(var4)

		return var5

	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)
