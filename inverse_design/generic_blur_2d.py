from filter import Filter
import numpy as np


def make_circular_blur( alpha, blur_size, variable_bounds=[0,1] ):
	return make_ellipsoidal_blur( alpha, blur_size, blur_size, variable_bounds )

def make_square_blur( alpha, blur_size, variable_bounds=[0,1] ):
	return make_rectangular_blur( alpha, blur_size, blur_size, variable_bounds )


def make_ellipsoidal_blur( alpha, blur_size_x, blur_size_y, variable_bounds=[0,1] ):
	# At this point, we can compute which points in a volume will be part of this blurring operation
	ellipse_mask_size_x = 1 + 2 * blur_size_x
	ellipse_mask_size_y = 1 + 2 * blur_size_y

	ellipse_mask = np.zeros((ellipse_mask_size_x, ellipse_mask_size_y))

	for mask_x in range(-blur_size_x, blur_size_x + 1):
		for mask_y in range(-blur_size_y, blur_size_y + 1):
			x_contribution = mask_x**2 / (blur_size_x + 1e-6)**2
			y_contribution = mask_y**2 / (blur_size_y + 1e-6)**2

			if (x_contribution + y_contribution) <= 1:
				ellipse_mask[blur_size_x + mask_x, blur_size_y + mask_y] = 1

	return GenericBlur2D( alpha, blur_size_x, blur_size_y, ellipse_mask, variable_bounds )

def make_rectangular_blur( alpha, blur_size_x, blur_size_y, variable_bounds=[0,1]):
	# At this point, we can compute which points in a volume will be part of this blurring operation
	rectangular_mask_size_x = 1 + 2 * blur_size_x
	rectangular_mask_size_y = 1 + 2 * blur_size_y

	rectangular_mask = np.zeros((rectangular_mask_size_x, rectangular_mask_size_y))

	for mask_x in range(-blur_size_x, blur_size_x + 1):
		for mask_y in range(-blur_size_y, blur_size_y + 1):
			x_contribution = mask_x**2 / (blur_size_x + 1e-6)**2
			y_contribution = mask_y**2 / (blur_size_y + 1e-6)**2

			if ( np.abs( mask_x ) <= blur_size_x ) or ( np.abs( mask_y ) <= blur_size_y ):
				rectangular_mask[blur_size_x + mask_x, blur_size_y + mask_y] = 1

	return GenericBlur2D( alpha, blur_size_x, blur_size_y, rectangular_mask, variable_bounds )


#
# Blurs can be more general.  We just need to specify a mask and maximum approximation function (and its derivative)
#
class GenericBlur2D(Filter):

	#
	# alpha: strength of blur
	# blur: half of blur width at center pixel in x and y directions in units of whole pixels
	#
	def __init__(self, alpha, blur_size_x, blur_size_y, mask, variable_bounds=[0, 1]):
		super(GenericBlur2D, self).__init__(variable_bounds)

		self.alpha = alpha
		self.blur_size_x = blur_size_x
		self.blur_size_y = blur_size_y
		self.mask = mask

		self.number_to_blur = sum((self.mask).flatten())
		print("number to blur = " + str(self.number_to_blur))


	def forward(self, variable_in):
		pad_variable_in = np.pad(
			variable_in,
			((self.blur_size_x, self.blur_size_x), (self.blur_size_y, self.blur_size_y)),
			'constant'
		)

		unpadded_shape = variable_in.shape
		padded_shape = pad_variable_in.shape

		start_x = self.blur_size_x
		start_y = self.blur_size_y

		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]

		blurred_variable = np.zeros((x_length, y_length))
		for mask_x in range(-self.blur_size_x, self.blur_size_x + 1):
			offset_x = start_x + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.blur_size_y, self.blur_size_y + 1):
				offset_y = start_y + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]

				check_mask = self.mask[mask_x + self.blur_size_x, mask_y + self.blur_size_y]

				if check_mask == 1:
					blurred_variable = np.add(
						blurred_variable,
						np.exp(
							np.multiply(
								self.alpha,
								pad_variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]]
								)
							)
						)

		blurred_variable = np.divide(np.log(np.divide(blurred_variable, self.number_to_blur)), self.alpha)

		return blurred_variable


	def chain_rule(self, derivative_out, variable_out, variable_in):
		pad_variable_out = np.pad(
			variable_out,
			((self.blur_size_x, self.blur_size_x), (self.blur_size_y, self.blur_size_y)),
			'constant'
		)

		pad_derivative_out = np.pad(
			derivative_out,
			((self.blur_size_x, self.blur_size_x), (self.blur_size_y, self.blur_size_y)),
			'constant'
		)

		start_x = self.blur_size_x
		start_y = self.blur_size_y

		unpadded_shape = variable_in.shape
		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]

		derivative_in = np.zeros(derivative_out.shape)

		for mask_x in range(-self.blur_size_x, self.blur_size_x + 1):
			offset_x = start_x + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.blur_size_y, self.blur_size_y + 1):
				offset_y = start_y + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]

				check_mask = self.mask[mask_x + self.blur_size_x, mask_y + self.blur_size_y]

				if check_mask == 1:
					derivative_in = np.add(
						derivative_in,
						np.multiply(
							np.exp(
								np.multiply(
									self.alpha,
									np.subtract(
										variable_in,
										pad_variable_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]]
									)
								)
							),
						pad_derivative_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]])
					)

		derivative_in = np.divide(derivative_in, self.number_to_blur)
		return derivative_in

	def fabricate(self, variable_in):
		pad_variable_in = np.pad(
			variable_in,
			((self.blur_size_x, self.blur_size_x), (self.blur_size_y, self.blur_size_y)),
			'constant'
		)

		unpadded_shape = variable_in.shape
		padded_shape = pad_variable_in.shape

		start_x = self.blur_size_x
		start_y = self.blur_size_y

		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]

		blurred_variable = np.zeros((x_length, y_length))
		for mask_x in range(-self.blur_size_x, self.blur_size_x + 1):
			offset_x = start_x + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.blur_size_y, self.blur_size_y + 1):
				offset_y = start_y + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]

				check_mask = self.mask[mask_x + self.blur_size_x, mask_y + self.blur_size_y]

				if check_mask == 1:
					blurred_variable = np.maximum(
						blurred_variable,
						pad_variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]] )


		return blurred_variable



