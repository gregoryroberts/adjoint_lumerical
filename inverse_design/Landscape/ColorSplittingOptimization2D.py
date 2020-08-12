#
# Math
#
import numpy as np
import scipy.optimize

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

#
# System
#
import sys
import os

#
# Electromagnetics
#
run_on_cluster = True

if run_on_cluster:
	sys.path.append( '/central/home/gdrobert/Develompent/ceviche' )
import ceviche

#
# Topology Optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append( os.path.abspath( python_src_directory + "/../LevelSet/" ) )
if run_on_cluster:
	from LevelSet import LevelSet
else:
	import LevelSet


eps_nought = 8.854 * 1e-12
mu_nought = 1.257 * 1e-6 
c = 3.0 * 1e8

# def compute_binarization( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 2 / np.sqrt( total_shape ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
# def compute_binarization_gradient( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 4 / np.sqrt( total_shape ) ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )

def compute_binarization( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )
# def compute_binarization_gradient( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 1. / total_shape ) * ( input_variable - 0.5 ) / np.sum( np.sqrt( ( input_variable - 0.5 )**2 )	)

def compute_binarization_gradient( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sign( input_variable - set_point )


def vector_norm( v_in ):
	return np.sqrt( np.sum( np.abs( v_in )**2 ) )

def upsample( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k * factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		for y_idx in range( 0, output_block_size[ 1 ] ):
			output_block[ x_idx, y_idx ] = input_block[ int( x_idx / factor ), int( y_idx / factor ) ]

	return output_block

def reinterpolate_average( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k / factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		start_x = int( factor * x_idx )
		end_x = start_x + factor
		for y_idx in range( 0, output_block_size[ 1 ] ):
			start_y = int( factor * y_idx )
			end_y = start_y + factor

			average = 0.0

			for sweep_x in range( start_x, end_x ):
				for sweep_y in range( start_y, end_y ):
					average += ( 1. / factor**2 ) * input_block[ sweep_x, sweep_y ]
			
			output_block[ x_idx, y_idx ] = average

	return output_block

class ColorSplittingOptimization2D():

	def __init__( self,
		device_size_voxels, coarsen_factor, mesh_size_nm,
		permittivity_bounds, focal_spots_x_relative, focal_length_y_voxels,
		wavelengths_um, wavelength_idx_to_focal_idx, random_seed,
		num_layers, designable_layer_indicators, non_designable_permittivity,
		save_folder, field_blur=False, field_blur_size_voxels=0.0, density_pairings=None,
		binarization_set_point=0.5 ):
		
		self.device_width_voxels = device_size_voxels[ 0 ]
		self.device_height_voxels = device_size_voxels[ 1 ]

		self.coarsen_factor = coarsen_factor
		assert ( self.device_width_voxels % coarsen_factor ) == 0, "The specified coarsening factor does not evenly divide the device width in voxels!"
		assert ( self.device_height_voxels % coarsen_factor ) == 0, "The specified coarsening factor does not evenly divide the device height in voxels!"

		self.design_width_voxels = int( device_size_voxels[ 0 ] / coarsen_factor )
		self.design_height_voxels = int( device_size_voxels[ 1 ] / coarsen_factor )

		self.design_density = None

		self.mesh_size_nm = mesh_size_nm
		self.mesh_size_um = 1e-3 * mesh_size_nm
		self.mesh_size_m = 1e-9 * mesh_size_nm

		self.device_size_um = [ self.mesh_size_um * device_size_voxels[ idx ] for idx in range( 0, len( device_size_voxels ) ) ]

		self.permittivity_bounds = permittivity_bounds
		self.min_relative_permittivity = permittivity_bounds[ 0 ]
		self.max_relative_permittivity = permittivity_bounds[ 1 ]

		self.focal_spots_x_relative = focal_spots_x_relative
		self.focal_length_y_voxels = focal_length_y_voxels
		self.wavelengths_um = wavelengths_um
		self.wavelength_intensity_scaling_factor = 1. / ( eps_nought * np.max( self.wavelengths_um )**2 )
		self.wavelength_intensity_scaling = self.wavelengths_um**2 * self.wavelength_intensity_scaling_factor

		self.num_wavelengths = len( wavelengths_um )

		self.omega_values = 2 * np.pi * c / ( 1e-6 * wavelengths_um )

		self.wavelength_idx_to_focal_idx = wavelength_idx_to_focal_idx

		self.random_seed = random_seed
		np.random.seed( self.random_seed )

		self.density_pairings = density_pairings
		self.do_density_pairings = not ( self.density_pairings is None )

		assert( self.design_height_voxels % num_layers ) == 0, "Expected the number of layers to evenly divide the design region"

		self.num_layers = num_layers
		self.design_voxels_per_layer = int( self.design_height_voxels / num_layers )

		assert ( len( designable_layer_indicators ) == self.num_layers ), "The layer designability indicator length does not make sense!"
		assert ( len( non_designable_permittivity ) == len( designable_layer_indicators ) ), "Expected a different length for the non designable permittivity "

		self.designable_layer_indicators = np.array( designable_layer_indicators )
		self.non_designable_permittivity = np.array( non_designable_permittivity )
		self.non_designable_density = ( self.non_designable_permittivity - self.min_relative_permittivity ) / ( self.max_relative_permittivity - self.min_relative_permittivity )

		self.save_folder = save_folder

		self.field_blur = field_blur
		self.field_blur_size_voxels = field_blur_size_voxels

		self.binarization_set_point = binarization_set_point

		self.setup_simulation()

	def init_density_with_random( self, mean_density, sigma_density ):
		num_random_values = self.design_width_voxels * self.num_layers

		random_array_normal_distribution = np.random.normal(
			loc=mean_density,
			scale=sigma_density, size=[ num_random_values ] )

		self.design_density = np.ones( [ self.design_width_voxels, self.design_height_voxels ] )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			random_values_start = layer_idx * self.design_width_voxels
			random_values_end = random_values_start + self.design_width_voxels

			fill_data = self.non_designable_density[ layer_idx ] * np.ones( self.design_width_voxels )

			if self.designable_layer_indicators[ layer_idx ]:
				fill_data = random_array_normal_distribution[ random_values_start : random_values_end ]

			for internal_layer_idx in range( layer_start, layer_end ):
				self.design_density[ :, internal_layer_idx ] = fill_data

		self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

		if self.do_density_pairings:
			self.design_density = self.pair_array( self.design_density )

	def init_density_with_uniform( self, density_value ):
		assert ( ( density_value <= 1.0 ) and ( density_value >= 0.0 ) ), "Invalid density value specified!"

		self.design_density = np.ones( [ self.design_width_voxels, self.design_height_voxels ] )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			random_values_start = layer_idx * self.design_width_voxels
			random_values_end = random_values_start + self.design_width_voxels

			choose_density = self.non_designable_density[ layer_idx ]

			if self.designable_layer_indicators[ layer_idx ]:
				choose_density = density_value

			for internal_layer_idx in range( layer_start, layer_end ):
				self.design_density[ :, internal_layer_idx ] = choose_density

	def init_density_directly( self, input_density ):
		assert ( ( input_density.shape[ 0 ] == self.design_width_voxels ) and ( input_density.shape[ 1 ] == self.design_height_voxels ) ), "Specified design has the wrong shape"

		self.design_density = input_density.copy()

	def init_density_with_this_class( self, this_class ):
		self.init_density_directly( this_class.design_density )

	def setup_simulation( self ):
		self.width_gap_voxels = int( 1.0 * np.max( self.wavelengths_um ) / self.mesh_size_um )
		self.height_gap_voxels_top = int( 1.5 * np.max( self.wavelengths_um ) / self.mesh_size_um )
		self.height_gap_voxels_bottom = self.width_gap_voxels
		self.pml_voxels = int( 1.0 * np.max( self.wavelengths_um ) / self.mesh_size_um )

		self.simulation_width_voxels = self.device_width_voxels + 2 * self.width_gap_voxels + 2 * self.pml_voxels
		self.simulation_height_voxels = self.device_height_voxels + self.focal_length_y_voxels + self.height_gap_voxels_bottom + self.height_gap_voxels_top + 2 * self.pml_voxels

		self.device_width_start = int( 0.5 * ( self.simulation_width_voxels - self.device_width_voxels ) )
		self.device_width_end = self.device_width_start + self.device_width_voxels
		self.device_height_start = int( self.pml_voxels + self.height_gap_voxels_bottom + self.focal_length_y_voxels )
		self.device_height_end = self.device_height_start + self.device_height_voxels

		self.focal_spots_x_voxels = [
			int( self.device_width_start + self.focal_spots_x_relative[ idx ] * self.device_width_voxels ) for idx in range( 0, len( self.focal_spots_x_relative ) )
		]

		self.fwd_src_y = int( self.pml_voxels + self.height_gap_voxels_bottom + self.focal_length_y_voxels + self.device_height_voxels + 0.75 * self.height_gap_voxels_top )
		self.focal_point_y = int( self.pml_voxels + self.height_gap_voxels_bottom )

		self.rel_eps_simulation = np.ones( ( self.simulation_width_voxels, self.simulation_height_voxels ) )

		fwd_src_x_range = np.arange( 0, self.simulation_width_voxels )
		fwd_src_y_range = self.fwd_src_y * np.ones( fwd_src_x_range.shape, dtype=int )

		self.fwd_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		self.fwd_source[ fwd_src_x_range, fwd_src_y_range ] = 1

	def plot_subcell_gradient_variations( self, omega_idx, factor ):
		import matplotlib.pyplot as plt
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		device_width_array = np.linspace( 0, 1, self.device_width_voxels )
		device_height_array = np.linspace( 0, 1, self.device_height_voxels )

		interp_width_array = np.linspace( 0, 1, factor * self.device_width_voxels )
		interp_height_array = np.linspace( 0, 1, factor * self.device_height_voxels )

		omega = self.omega_values[ omega_idx ]

		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		
		focal_point_x_loc = self.focal_spots_x_voxels[ 0 ]

		interp_spline_fwd_real = RectBivariateSpline(
			device_width_array, device_height_array, np.real( fwd_Ez[
				self.device_width_start : self.device_width_end,
				self.device_height_start : self.device_height_end
		] ) )

		interp_spline_fwd_imag = RectBivariateSpline(
			device_width_array, device_height_array, np.imag( fwd_Ez[
				self.device_width_start : self.device_width_end,
				self.device_height_start : self.device_height_end
		] ) )


		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		interp_spline_adj_real = RectBivariateSpline(
			device_width_array, device_height_array, np.real( adj_Ez[
				self.device_width_start : self.device_width_end,
				self.device_height_start : self.device_height_end
		] ) )

		interp_spline_adj_imag = RectBivariateSpline(
			device_width_array, device_height_array, np.imag( adj_Ez[
				self.device_width_start : self.device_width_end,
				self.device_height_start : self.device_height_end
		] ) )


		interpolated_fwd_real = interp_spline_fwd_real( interp_width_array, interp_height_array )
		interpolated_adj_real = interp_spline_adj_real( interp_width_array, interp_height_array )
		interpolated_fwd_imag = interp_spline_fwd_imag( interp_width_array, interp_height_array )
		interpolated_adj_imag = interp_spline_adj_imag( interp_width_array, interp_height_array )

		interpolated_fwd = interpolated_fwd_real + 1j * interpolated_fwd_imag
		interpolated_adj = interpolated_adj_real + 1j * interpolated_adj_imag

		interp_grad = 2 * np.real( interpolated_fwd * interpolated_adj )

		averaged_grad = np.zeros( ( self.device_width_voxels, self.device_height_voxels ) )
		middle_grad = 2 * np.real(
			fwd_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] *
			adj_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] )

		for x_idx in range( 0, self.device_width_voxels ):
			for y_idx in range( 0, self.device_height_voxels ):
				for off_x in range( 0, factor ):
					start_x = x_idx * factor
					for off_y in range( 0, factor ):
						start_y = y_idx * factor

						averaged_grad[ x_idx, y_idx ] += (
							( 1. / ( factor * factor ) ) *
							interp_grad[ start_x + off_x, start_y + off_y ]
						)

		plt.subplot( 1, 3, 1 )
		plt.imshow( middle_grad, cmap='Blues' )
		plt.colorbar()
		plt.subplot( 1, 3, 2 )
		plt.imshow( averaged_grad, cmap='Blues' )
		plt.colorbar()
		plt.subplot( 1, 3, 3 )
		plt.imshow( ( middle_grad - averaged_grad ), cmap='Greens' )
		plt.colorbar()
		plt.show()





		half_width = int( factor * 0.5 * self.device_width_voxels )
		half_height = int( factor * 0.5 * self.device_height_voxels )

		plt.subplot( 1, 3, 1 )
		plt.imshow( np.abs( interpolated_fwd[ half_width : ( half_width + self.coarsen_factor * 2 * factor ), half_height : ( half_height + self.coarsen_factor * 2 * factor) ] ), cmap='Reds' )
		plt.colorbar()
		plt.subplot( 1, 3, 2 )
		plt.imshow( np.abs( interpolated_adj[ half_width : ( half_width + self.coarsen_factor * 2 * factor ), half_height : ( half_height + self.coarsen_factor * 2 * factor) ] ), cmap='Reds' )
		plt.colorbar()
		plt.subplot( 1, 3, 3 )
		plt.imshow(
			np.real(
				interpolated_fwd[ half_width : ( half_width + self.coarsen_factor * 2 * factor ), half_height : ( half_height + self.coarsen_factor * 2 * factor ) ] *
				interpolated_adj[ half_width : ( half_width + self.coarsen_factor * 2 * factor ), half_height : ( half_height + self.coarsen_factor * 2 * factor ) ] ),
			cmap='Reds' )
		plt.colorbar()
		plt.show()


	def get_device_efields( self, omega_idx ):
		import matplotlib.pyplot as plt
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		Ez = self.compute_forward_fields( self.omega_values[ omega_idx ], device_permittivity )

		return Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ], device_permittivity

	def plot_fields( self, omega_idx ):
		import matplotlib.pyplot as plt
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		Ez = self.compute_forward_fields( self.omega_values[ omega_idx ], device_permittivity )

		plt.subplot( 1, 2, 1 )
		plt.imshow( np.abs( Ez ), cmap='Blues' )
		plt.subplot( 1, 2, 2 )
		plt.imshow( np.real( Ez ), cmap='Greens' )
		plt.show()

		plt.subplot( 1, 2, 1 )
		ceviche.viz.abs(Ez, outline=self.rel_eps_simulation, ax=plt.gca(), cbar=False)
		plt.subplot( 1, 2, 2 )
		plt.imshow( self.rel_eps_simulation, cmap='Greens' )
		plt.show()

	def compute_forward_fields( self, omega, device_permittivity ):
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( self.fwd_source )

		return fwd_Ez

	def compute_fom( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		return fom

	def compute_fom_and_gradient( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		if self.field_blur:
			blur_fwd_Ez_real = gaussian_filter( np.real( fwd_Ez ), sigma=self.field_blur_size_voxels )
			blur_fwd_Ez_imag = gaussian_filter( np.imag( fwd_Ez ), sigma=self.field_blur_size_voxels )

			blur_adj_Ez_real = gaussian_filter( np.real( adj_Ez ), sigma=self.field_blur_size_voxels )
			blur_adj_Ez_imag = gaussian_filter( np.imag( adj_Ez ), sigma=self.field_blur_size_voxels )

			blur_fwd_Ez = blur_fwd_Ez_real + 1j * blur_fwd_Ez_imag
			blur_adj_Ez = blur_adj_Ez_real + 1j * blur_adj_Ez_imag

			gradient = fom_scaling * 2 * np.real( omega * eps_nought * blur_fwd_Ez * blur_adj_Ez / 1j )

		return fom, gradient

	# CODE DUPLICATION! FIX
	def compute_net_fom( self ):
		fom_by_wl = []

		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		for wl_idx in range( 0, self.num_wavelengths ):
			get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]
			get_fom = self.compute_fom(
				self.omega_values[ wl_idx ], device_permittivity,
				self.focal_spots_x_voxels[ get_focal_point_idx ], self.wavelength_intensity_scaling[ wl_idx ] )
			fom_by_wl.append( get_fom )

		net_fom = np.product( fom_by_wl )

		return net_fom

	def verify_adjoint_against_finite_difference_lambda( self ):
		fd_x = int( 0.5 * self.device_width_voxels )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd = np.zeros( len( fd_y ) )
		omega_idx = 0#int( 0.5 * len( self.omega_values ) )
		fd_omega = self.omega_values[ omega_idx ]
		lambda_value_um = 1e6 * c / fd_omega

		h = 1e-5

		lambda_low_um = lambda_value_um - h
		lambda_high_um = lambda_value_um + h

		omega_low = c / ( 1e-6 * lambda_low_um )
		omega_high = c / ( 1e-6 * lambda_high_um )

		# fd_init_device = 1.5 * np.ones( ( self.device_width_voxels, self.device_height_voxels ) )
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		fd_init_device = device_permittivity

		focal_point_x = self.focal_spots_x_voxels[ 0 ]

		get_fom, get_grad = self.compute_fom_and_gradient(
			fd_omega, fd_init_device, focal_point_x )

		# import matplotlib.pyplot as plt
		# plt.imshow( get_grad )
		# plt.colorbar()
		# plt.show()

		print( "get fom = " + str( get_fom ) )
		# get_grad = get_grad[
		# 	self.device_width_start : self.device_width_end,
		# 	self.device_height_start : self.device_height_end ]

		get_fom_low = self.compute_fom(
			omega_low, fd_init_device, focal_point_x )

		print( "fom low = " + str( get_fom_low ) )

		get_fom_high = self.compute_fom(
			omega_high, fd_init_device, focal_point_x )

		print( "fom high = " + str( get_fom_high ) )

		fd_grad = ( get_fom_high - get_fom_low ) / ( 2 * h )

		full_eps = np.ones( self.rel_eps_simulation.shape )
		full_eps[
			self.device_width_start : self.device_width_end,
			self.device_height_start : self.device_height_end ] = device_permittivity

		adj_grad = -np.sum( get_grad * fd_omega * eps_nought * full_eps * 2 * ( 2 * np.pi )**2  / ( ( 1e-6 * lambda_value_um )**2 ) )
		# adj_grad = -get_grad[ focal_point_x, self.focal_point_y ] * 2 * ( 2 * np.pi )**2 / ( lambda_value_um )**3

		print( "fd grad = " + str( fd_grad ) )
		print( "adj grad = " + str( adj_grad ) )



	def verify_adjoint_against_finite_difference( self ):
		fd_x = int( 0.5 * self.device_width_voxels )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd = np.zeros( len( fd_y ) )
		omega_idx = int( 0.5 * len( self.omega_values ) )
		fd_omega = self.omega_values[ omega_idx ]

		# fd_init_device = 1.5 * np.ones( ( self.device_width_voxels, self.device_height_voxels ) )
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		fd_init_device = device_permittivity

		focal_point_x = self.focal_spots_x_voxels[ 0 ]

		get_fom, get_grad = self.compute_fom_and_gradient(
			fd_omega, fd_init_device, focal_point_x )
		get_grad = get_grad[
			self.device_width_start : self.device_width_end,
			self.device_height_start : self.device_height_end ]

		fd_step_eps = 1e-4

		num = 10

		for fd_y_idx in range( 0, num ):#20 ):#len( fd_y ) ):
			print( "working on " + str( fd_y_idx ) )
			fd_device_permittivity = fd_init_device.copy()
			fd_device_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps

			get_fom_step = self.compute_fom( fd_omega, fd_device_permittivity, focal_point_x )

			compute_fd[ fd_y_idx ] = ( get_fom_step - get_fom ) / fd_step_eps

		import matplotlib.pyplot as plt
		plt.plot( get_grad[ fd_x, 0 : num ], color='g', linewidth=2 )
		plt.plot( compute_fd[ 0 : num ], color='r', linewidth=2, linestyle='--' )
		plt.show()

		self.average_adjoint_finite_difference_error = np.sqrt( np.mean( np.abs( get_grad[ fd_x, self.device_height_start : self.device_height_end ] - compute_fd )**2 ) )

	def density_to_permittivity( self, density ):
		return ( self.min_relative_permittivity + ( self.max_relative_permittivity - self.min_relative_permittivity ) * density )

	def layer_spacer_averaging( self, gradient_input ):
		gradient_output = np.zeros( gradient_input.shape )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			fill_gradient = np.zeros( self.design_width_voxels )

			if self.designable_layer_indicators[ layer_idx ]:
				fill_gradient = np.mean( gradient_input[ :, layer_start : layer_end ], axis=1 )

			for internal_layer_idx in range( layer_start, layer_end ):
				gradient_output[ :, internal_layer_idx ] = fill_gradient

		return gradient_output

	def step_binarize( self, gradient, binarize_amount, binarize_max_movement ):

		density_for_binarizing = self.design_density.flatten()
		flatten_gradient = gradient.flatten()

		flatten_design_cuts = density_for_binarizing.copy()
		extract_binarization_gradient = compute_binarization_gradient( flatten_design_cuts, self.binarization_set_point )
		flatten_fom_gradients = flatten_gradient.copy()

		beta = binarize_max_movement
		projected_binarization_increase = 0

		c = flatten_fom_gradients

		initial_binarization = compute_binarization( flatten_design_cuts, self.binarization_set_point )

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
		
		# Try this! Not sure how well it will work
		alpha = np.minimum( initial_binarization * max_possible_binarization_change, binarize_amount )

		def ramp( x ):
			return np.maximum( x, 0 )

		def opt_function( nu ):
			lambda_1 = ramp( nu * b - c )
			lambda_2 = c + lambda_1 - nu * b

			return -( -np.dot( lambda_1, upper_bounds ) + np.dot( lambda_2, lower_bounds ) + nu * alpha )

		tolerance = 1e-12
		optimization_solution_nu = scipy.optimize.minimize( opt_function, 0, tol=tolerance )

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

		# print( initial_binarization )
		# print( compute_binarization( proposed_design_variable.flatten() ) )

		return np.reshape( proposed_design_variable, self.design_density.shape )

	def get_device_permittivity( self ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		return device_permittivity

	def pair_array( self, input_array ):
		output_array = np.zeros( input_array.shape, dtype=input_array.dtype )
		for pair_idx in range( 0, len( self.density_pairings ) ):
			get_pair = self.density_pairings[ pair_idx ]
			density0 = input_array[ get_pair[ 0 ], get_pair[ 1 ] ]
			density1 = input_array[ get_pair[ 2 ], get_pair[ 3 ] ]

			density_average = 0.5 * ( density0 + density1 )

			output_array[ get_pair[ 0 ], get_pair[ 1 ] ] = density_average
			output_array[ get_pair[ 2 ], get_pair[ 3 ] ] = density_average

		return output_array


	def optimize(
		self, num_iterations,
		random_globals=False, random_global_iteration_frequency=10, random_global_scan_points=10,
		opt_mask=None,
		use_log_fom=False,
		wavelength_adversary=False, adversary_update_iters=-1, bottom_wls_um=None, top_wls_um=None,
		binarize=False, binarize_movement_per_step=0.01, binarize_max_movement_per_voxel=0.025,
		dropout_start=0, dropout_end=0, dropout_p=0.5,
		dense_plot_iters=-1, dense_plot_lambda=None, focal_assignments=None ):

		if dense_plot_iters == -1:
			dense_plot_iters = num_iterations
			dense_plot_lambda = self.wavelengths_um
			focal_assignments = self.focal_spots_x_voxels

		if opt_mask is None:
			opt_mask = np.ones( self.design_density.shape )

		if wavelength_adversary:
			adversary_scan_density = 8#4

			num_bottom_wls = len( bottom_wls_um )
			num_top_wls = len( top_wls_um )

			bandwidth_bottom_um = bottom_wls_um[ num_bottom_wls - 1 ] - bottom_wls_um[ 0 ]
			bandwidth_top_um = top_wls_um[ num_top_wls - 1 ] - top_wls_um[ 0 ]

			divider_bottom_um = bandwidth_bottom_um / ( 2 * ( num_bottom_wls - 1 ) )
			divider_top_um = bandwidth_top_um / ( 2 * ( num_top_wls - 1 ) )

			bottom_bins = np.zeros( ( num_bottom_wls, 2 ) )
			top_bins = np.zeros( ( num_top_wls, 2 ) )

			cur_bottom_um = np.min( bottom_wls_um )
			for bottom_idx in range( 0, num_bottom_wls ):
				bottom_bins[ bottom_idx ] = np.array( [ cur_bottom_um, cur_bottom_um + 2 * divider_bottom_um ] )
				cur_bottom_um += 2 * divider_bottom_um

			cur_top_um = np.min( top_wls_um )
			for top_idx in range( 0, num_top_wls ):
				top_bins[ top_idx ] = np.array( [ cur_top_um, cur_top_um + 2 * divider_top_um ] )
				cur_top_um += 2 * divider_top_um

			optimization_wavelengths_um = self.wavelengths_um.copy()
		self.track_optimization_wavelengths_um = np.zeros( ( num_iterations, len( self.wavelengths_um ) ) )


		dense_plot_omega = 2 * np.pi * c / ( 1e-6 * dense_plot_lambda )

		self.max_density_change_per_iteration_start = 0.03#0.05
		self.max_density_change_per_iteration_end = 0.01#0.005

		self.gradient_norm_evolution = np.zeros( num_iterations )
		self.fom_evolution = np.zeros( num_iterations )
		self.binarization_evolution = np.zeros( num_iterations )
		self.fom_by_wl_evolution = np.zeros( ( num_iterations, self.num_wavelengths ) )
		self.gradient_directions = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		self.dense_plot_idxs = []
		self.dense_plots = []

		self.design_density *= opt_mask

		for iter_idx in range( 0, num_iterations ):
			if ( iter_idx % 10 ) == 0:
				log_file = open( self.save_folder + "/log.txt", 'a' )
				log_file.write( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) + "\n")
				log_file.close()


			mask_density = opt_mask * self.design_density
			import_density = upsample( mask_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )


			if random_globals and ( ( iter_idx % random_global_iteration_frequency ) == 0 ):
				random_direction = opt_mask * np.random.normal( 0, 1 )
				random_direction /= np.sqrt( np.sum( random_direction**2 ) )

				alpha_0 = np.sum( random_direction * mask_density )
				rho_0 = mask_density - alpha_0 * random_direction

				lower_alpha_bound = np.inf
				upper_alpha_bound = -np.inf

				flatten_rho_0 = rho_0.flatten()
				flatten_random_direction = random_direction.flatten()

				critical_low_alpha = ( 0.0 - flatten_rho_0 ) / ( flatten_random_direction + 1e-8 )
				critical_high_alpha = ( 1.0 - flatten_rho_0 ) / ( flatten_random_direction + 1e-8 )

				for idx in range( 0, len( flatten_rho_0 ) ):
					if ( flatten_random_direction[ idx ] > 0 ):
						upper_alpha_bound = np.minimum( upper_alpha_bound, critical_high_alpha[ idx ] )
						lower_alpha_bound = np.maximum( lower_alpha_bound, critical_low_alpha[ idx ] )
					else:
						lower_alpha_bound = np.maximum( lower_alpha_bound, critical_high_alpha[ idx ] )
						upper_alpha_bound = np.minimum( upper_alpha_bound, critical_low_alpha[ idx ] )

				alpha_sweep = np.linspace( lower_alpha_bound, upper_alpha_bound, random_global_scan_points )


				def sweep_fom( test_rho ):
					mask_density = opt_mask * test_rho
					import_density = upsample( mask_density, self.coarsen_factor )
					test_permittivity = self.density_to_permittivity( import_density )	

					total_product_fom = 1.0
					for wl_idx in range( 0, self.num_wavelengths ):
						get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

						get_fom = self.compute_fom(
							self.omega_values[ wl_idx ], test_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
							self.wavelength_intensity_scaling[ wl_idx ] )

						total_product_fom *= get_fom

					if use_log_fom:
						total_product_fom = np.log( total_product_fom )

					return total_product_fom

				fom_to_beat = sweep_fom( self.design_density )
				alpha_to_beat = 0

				for alpha_idx in range( 0, random_global_scan_points ):
					sweep_density = alpha_sweep[ alpha_idx ] * random_direction + rho_0

					cur_fom = sweep_fom( sweep_density )
					if cur_fom > fom_to_beat:
						fom_to_beat = cur_fom
						alpha_to_beat = alpha_sweep[ alpha_idx ]

				self.design_density = alpha_to_beat * random_direction + rho_0
				mask_density = opt_mask * self.design_density
				import_density = upsample( mask_density, self.coarsen_factor )
				device_permittivity = self.density_to_permittivity( import_density )


			gradient_by_wl = []
			fom_by_wl = []
			dense_plot = []

			if ( iter_idx % dense_plot_iters ) == 0:
				dense_wavelength_intensity_scaling = dense_plot_lambda**2 / ( eps_nought * np.max( self.wavelengths_um )**2 )

				for wl_idx in range( 0, len( dense_plot_lambda ) ):
					omega_value = dense_plot_omega[ wl_idx ]

					get_focal_point_idx = focal_assignments[ wl_idx ]

					dense_plot.append( self.compute_fom(
						omega_value, device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
						dense_wavelength_intensity_scaling[ wl_idx ] ) )

				self.dense_plots.append( dense_plot )
				self.dense_plot_idxs.append( iter_idx )

			if wavelength_adversary:
				if ( iter_idx % adversary_update_iters ) == 0:

					new_bottom_wls_um = np.zeros( num_bottom_wls )
					new_top_wls_um = np.zeros( num_top_wls )

					for bottom_idx in range( 0, num_bottom_wls ):
						worst_fom = np.inf
						worst_wl_um = 0

						scan_wls_um = np.linspace( bottom_bins[ bottom_idx ][ 0 ], bottom_bins[ bottom_idx ][ 1 ], adversary_scan_density )

						for scan_wl_idx in range( 0, adversary_scan_density ):
							scan_wl_um = scan_wls_um[ scan_wl_idx ]
							scan_omega = 2 * np.pi * c / ( scan_wl_um * 1e-6 )

							wl_intensity_scaling = scan_wl_um**2 * self.wavelength_intensity_scaling_factor

							get_fom = self.compute_fom(
								scan_omega, device_permittivity, self.focal_spots_x_voxels[ 0 ],
								wl_intensity_scaling )

							if get_fom < worst_fom:
								worst_fom = get_fom
								worst_wl_um = scan_wl_um

						new_bottom_wls_um[ bottom_idx ] = worst_wl_um

					for top_idx in range( 0, num_top_wls ):
						worst_fom = np.inf
						worst_wl_um = 0

						scan_wls_um = np.linspace( top_bins[ top_idx ][ 0 ], top_bins[ top_idx ][ 1 ], adversary_scan_density )

						for scan_wl_idx in range( 0, adversary_scan_density ):
							scan_wl_um = scan_wls_um[ scan_wl_idx ]
							scan_omega = 2 * np.pi * c / ( scan_wl_um * 1e-6 )

							wl_intensity_scaling = scan_wl_um**2 * self.wavelength_intensity_scaling_factor

							get_fom = self.compute_fom(
								scan_omega, device_permittivity, self.focal_spots_x_voxels[ 1 ],
								wl_intensity_scaling )

							if get_fom < worst_fom:
								worst_fom = get_fom
								worst_wl_um = scan_wl_um

						new_top_wls_um[ top_idx ] = worst_wl_um

					optimization_wavelengths_um = np.array( list( new_bottom_wls_um ) + list( new_top_wls_um ) )

			if wavelength_adversary:
				self.track_optimization_wavelengths_um[ iter_idx ] = optimization_wavelengths_um
			else:
				self.track_optimization_wavelengths_um[ iter_idx ] = self.wavelengths_um

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				if wavelength_adversary:

					opt_omega_value = 2 * np.pi * c / ( 1e-6 * optimization_wavelengths_um[ wl_idx ] )
					wl_intensity_scaling = optimization_wavelengths_um[ wl_idx ]**2 * self.wavelength_intensity_scaling_factor

					get_fom = self.compute_fom(
						self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
						self.wavelength_intensity_scaling[ wl_idx ] )

					get_fom_, get_grad = self.compute_fom_and_gradient(
						opt_omega_value, device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
						wl_intensity_scaling )
				else:
					get_fom, get_grad = self.compute_fom_and_gradient(
						self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
						self.wavelength_intensity_scaling[ wl_idx ] )

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

				scale_fom_for_wl = get_fom
				scale_gradient_for_wl = upsampled_device_grad

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )

			net_fom = np.product( fom_by_wl )

			if use_log_fom:
				net_fom = np.log( net_fom )

			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

			# We are currently not doing a performance based weighting here, but we can add it in
			for wl_idx in range( 0, self.num_wavelengths ):
				wl_gradient = ( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				if use_log_fom:
					weighting = 1. / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			net_gradient = reinterpolate_average( net_gradient, self.coarsen_factor )

			#
			# Now, we should zero out non-designable regions and average over designable layers
			#
			net_gradient = self.layer_spacer_averaging( net_gradient )

			if ( iter_idx >= dropout_start ) and ( iter_idx < dropout_end ):
				net_gradient *= 1.0 * np.greater( np.random.random( net_gradient.shape ), dropout_p )

			net_gradient *= opt_mask
			gradient_norm = vector_norm( net_gradient )

			self.fom_evolution[ iter_idx ] = net_fom
			self.fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
			self.gradient_norm_evolution[ iter_idx ] = gradient_norm

			norm_scaled_gradient = net_gradient / gradient_norm

			if self.do_density_pairings:
				norm_scaled_gradient = self.pair_array( norm_scaled_gradient )

			self.gradient_directions[ iter_idx ] = norm_scaled_gradient

			max_density_change = (
				self.max_density_change_per_iteration_start +
				( iter_idx / ( num_iterations - 1 ) ) * ( self.max_density_change_per_iteration_end - self.max_density_change_per_iteration_start )
			)

			self.binarization_evolution[ iter_idx ] = compute_binarization( self.design_density.flatten() )

			if binarize:
				self.design_density = self.step_binarize( -norm_scaled_gradient, binarize_movement_per_step, binarize_max_movement_per_voxel )
			else:
				self.design_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
				self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

			if self.do_density_pairings:
				self.design_density = self.pair_array( self.design_density )

	def optimize_with_level_set( self, num_iterations ):
		self.lsf_gradient_norm_evolution = np.zeros( num_iterations )
		self.lsf_fom_evolution = np.zeros( num_iterations )
		self.lsf_fom_by_wl_evolution = np.zeros( ( num_iterations, self.num_wavelengths ) )
		self.lsf_gradient_directions = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )
		self.level_set_device_evolution = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		level_set = LevelSet.LevelSet( [ self.design_width_voxels, self.design_height_voxels ], 1. / self.coarsen_factor )
		level_set.init_with_density( self.design_density )

		# import matplotlib.pyplot as plt
		# plt.subplot( 1, 2, 1 )
		# plt.imshow( self.design_density, cmap='Greens' )
		# plt.subplot( 1, 2, 2 )
		# plt.imshow( level_set.device_density_from_level_set(), cmap='Greens' )
		# plt.show()
		# # sys.exit(0)

		# test_grad = np.random.random( ( self.design_width_voxels, self.design_height_voxels ) )
		# test_grad /= np.sqrt( np.sum( test_grad**2 ) )

		# start = level_set.device_density_from_level_set().copy()
		# level_set.update( test_grad, 1 )

		# import matplotlib.pyplot as plt
		# plt.subplot( 1, 3, 1 )
		# plt.imshow( start, cmap='Greens' )
		# plt.subplot( 1, 3, 2 )
		# plt.imshow( level_set.device_density_from_level_set(), cmap='Greens' )
		# plt.subplot( 1, 3, 3 )
		# plt.imshow( start - level_set.device_density_from_level_set(), cmap='Blues' )
		# plt.show()
		# sys.exit(0)


		for iter_idx in range( 0, num_iterations ):
			if ( iter_idx % 1 ) == 0:
				log_file = open( self.save_folder + "/log_level_set.txt", 'a' )
				log_file.write( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) + "\n")
				log_file.close()

				print( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) )
 
			self.level_set_device_evolution[ iter_idx ] = level_set.device_density_from_level_set()

			import_density = upsample( level_set.device_density_from_level_set(), self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )

			gradient_by_wl = []
			fom_by_wl = []

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				get_fom, get_grad = self.compute_fom_and_gradient(
					self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
					self.wavelength_intensity_scaling[ wl_idx ] )

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

				scale_fom_for_wl = get_fom
				scale_gradient_for_wl = upsampled_device_grad

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )

			net_fom = np.product( fom_by_wl )
			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

			# print( net_fom )

			# We are currently not doing a performance based weighting here, but we can add it in
			for wl_idx in range( 0, self.num_wavelengths ):
				wl_gradient = ( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			net_gradient = reinterpolate_average( net_gradient, self.coarsen_factor )

			#
			# Now, we should zero out non-designable regions and average over designable layers
			#
			net_gradient = self.layer_spacer_averaging( net_gradient )

			gradient_norm = vector_norm( net_gradient )

			self.lsf_fom_evolution[ iter_idx ] = net_fom
			self.lsf_fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
			self.lsf_gradient_norm_evolution[ iter_idx ] = gradient_norm

			norm_scaled_gradient = net_gradient / gradient_norm

			self.lsf_gradient_directions[ iter_idx ] = norm_scaled_gradient

			step_size = 1.0 + ( 0.1 - 1.0 ) * iter_idx / ( num_iterations - 1 )

			level_set.update( norm_scaled_gradient, step_size )

			# max_density_change = (
			# 	self.max_density_change_per_iteration_start +
			# 	( iter_idx / ( num_iterations - 1 ) ) * ( self.max_density_change_per_iteration_end - self.max_density_change_per_iteration_start )
			# )

			# self.binarization_evolution[ iter_idx ] = compute_binarization( self.design_density.flatten() )

			# if binarize:
			# 	self.design_density = self.step_binarize( -norm_scaled_gradient, binarize_movement_per_step, binarize_max_movement_per_voxel )
			# else:
			# 	self.design_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
			# 	self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

			# if self.do_density_pairings:
			# 	self.design_density = self.pair_array( self.design_density )

	def save_optimization_data( self, file_base ):
		np.save( file_base + "_gradient_norm_evolution.npy", self.gradient_norm_evolution )
		np.save( file_base + "_fom_evolution.npy", self.fom_evolution )
		np.save( file_base + "_binarization_evolution.npy", self.binarization_evolution )
		np.save( file_base + "_fom_by_wl_evolution.npy", self.fom_by_wl_evolution )
		np.save( file_base + "_gradient_directions.npy", self.gradient_directions )
		np.save( file_base + "_optimized_density.npy", self.design_density )
		np.save( file_base + "_random_seed.npy", self.random_seed )
		np.save( file_base + "_dense_plots.npy", np.array( self.dense_plots ) )
		np.save( file_base + "_dense_plot_idxs.npy", np.array( self.dense_plot_idxs ) )
		np.save( file_base + "_optimization_wavelengths.npy", self.track_optimization_wavelengths_um )



