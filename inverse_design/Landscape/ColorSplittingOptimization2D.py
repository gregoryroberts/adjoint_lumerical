#
# System
#
import sys
import os

#
# Math
#
import numpy as np
import scipy.optimize

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

sys.path.insert(1, '../')
import heaviside

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
small = 1e-10

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
		self.simulation_height_voxels = self.device_height_voxels + np.maximum( self.focal_length_y_voxels, 0 ) + self.height_gap_voxels_bottom + self.height_gap_voxels_top + 2 * self.pml_voxels

		self.device_width_start = int( 0.5 * ( self.simulation_width_voxels - self.device_width_voxels ) )
		self.device_width_end = self.device_width_start + self.device_width_voxels
		self.device_height_start = int( self.pml_voxels + self.height_gap_voxels_bottom + np.maximum( self.focal_length_y_voxels, 0 ) )
		self.device_height_end = self.device_height_start + self.device_height_voxels

		self.focal_spots_x_voxels = [
			int( self.device_width_start + self.focal_spots_x_relative[ idx ] * self.device_width_voxels ) for idx in range( 0, len( self.focal_spots_x_relative ) )
		]

		self.fwd_src_y = int( self.pml_voxels + self.height_gap_voxels_bottom + np.maximum( self.focal_length_y_voxels, 0 ) + self.device_height_voxels + 0.75 * self.height_gap_voxels_top )
		self.focal_point_y = int( self.pml_voxels + self.height_gap_voxels_bottom - np.minimum( self.focal_length_y_voxels, 0 ) )

		self.rel_eps_simulation = np.ones( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )

		fwd_src_x_range = np.arange( 0, self.simulation_width_voxels )
		fwd_src_y_range = self.fwd_src_y * np.ones( fwd_src_x_range.shape, dtype=int )

		self.fwd_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		self.fwd_source[ fwd_src_x_range, fwd_src_y_range ] = 1

	def plot_geometry( self, opt_mask=None ):
		import matplotlib.pyplot as plt

		focal_y = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ) )
		for spot in range( 0, len( self.focal_spots_x_voxels ) ):
			focal_y[
				self.focal_spots_x_voxels[ spot ] - 5 : self.focal_spots_x_voxels[ spot ] + 5,
				self.focal_point_y - 5 : self.focal_point_y + 5 ] = 1

		device_region = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ) )
		device_region[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = 1

		for spot in range( 0, len( self.focal_spots_x_voxels ) ):
			device_region[
				self.focal_spots_x_voxels[ spot ] - 5 : self.focal_spots_x_voxels[ spot ] + 5,
				self.focal_point_y - 5 : self.focal_point_y + 5 ] = 2


		plt.subplot( 2, 2, 1 )
		plt.imshow( np.real( self.fwd_source ) )
		plt.title( 'Forward Source' )
		plt.subplot( 2, 2, 2 )
		plt.imshow( focal_y )
		plt.title( 'Focal Y' )
		plt.subplot( 2, 2, 3 )
		plt.imshow( device_region )
		plt.title( 'Device Region' )
		if opt_mask is not None:
			opt_mask_region = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ) )
			upsampled_mask = upsample( opt_mask, self.coarsen_factor )

			for row in range( 0, upsampled_mask.shape[ 0 ] ):
				for col in range( 0, upsampled_mask.shape[ 1 ] ):
					opt_mask_region[ self.device_width_start + row, self.device_height_start + col ] = upsampled_mask[ row, col ]

			for spot in range( 0, len( self.focal_spots_x_voxels ) ):
				opt_mask_region[
					self.focal_spots_x_voxels[ spot ] - 5 : self.focal_spots_x_voxels[ spot ] + 5,
					self.focal_point_y - 5 : self.focal_point_y + 5 ] = 3


			plt.subplot( 2, 2, 4 )
			plt.imshow( opt_mask_region )
			plt.title( 'Masked optimization region' )
		plt.show()



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

	def eval_loss( self, omega ):
		print( self.max_relative_permittivity )
		self.rel_eps_simulation[ :, : ] = self.max_relative_permittivity

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( self.fwd_source )

		return fwd_Ez

	def compute_fom( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		return fom

	def compute_fom_and_gradient_with_polarizability__( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		assert not self.field_blur, "Field blur not supported with polarizability"

		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		gradient_design = np.zeros( ( self.design_width_voxels, self.design_height_voxels ) )
		gradient_design_orig = np.zeros( ( self.design_width_voxels, self.design_height_voxels ) )
		save_p_ind = np.zeros( fwd_Ez.shape, dtype=np.complex )
		save_p_ind2 = np.zeros( fwd_Ez.shape, dtype=np.complex )
		save_p_ind3 = np.zeros( fwd_Ez.shape, dtype=np.complex )


		for design_row in range( 0, self.design_width_voxels ):
			for design_col in range( 0, self.design_height_voxels ):

				device_start_row = self.device_width_start + self.coarsen_factor * design_row
				device_end_row = device_start_row + self.coarsen_factor

				device_start_col = self.device_height_start + self.coarsen_factor * design_col
				device_end_col = device_start_col + self.coarsen_factor


				local_adj_Ez = adj_Ez[
					device_start_row : device_end_row,
					device_start_col : device_end_col
				]
				guess_pind = fwd_Ez

				local_p_ind = guess_pind[ device_start_row : device_end_row,
					device_start_col : device_end_col ] * local_adj_Ez

				for design_row_other in range( 0, self.design_width_voxels ):
					for design_col_other in range( 0, self.design_height_voxels ):
						if ( design_row_other == design_row ) and ( design_col_other == design_col ):
							continue

						device_start_row_other = self.device_width_start + self.coarsen_factor * design_row_other
						device_end_row_other = device_start_row_other + self.coarsen_factor

						device_start_col_other = self.device_height_start + self.coarsen_factor * design_col_other
						device_end_col_other = device_start_col_other + self.coarsen_factor

						polarizability_src = np.zeros( self.fwd_source.shape, dtype=self.fwd_source.dtype )

						#
						# Oh wait, you should only have to do this once per forward source! Wasteful, but ok for now
						#
						polarizability_src[
							device_start_row : device_end_row,
							device_start_col : device_end_col ] = eps_nought * omega * ( 1 / 1j ) * fwd_Ez[
							device_start_row : device_end_row,
							device_start_col : device_end_col ]

						pol_Hx, pol_Hy, pol_Ez = simulation.solve( polarizability_src )

						local_adj_Ez_other = adj_Ez[
							device_start_row_other : device_end_row_other,
							device_start_col_other : device_end_col_other
						]
						guess_pind_other = self.rel_eps_simulation * pol_Ez

						local_p_ind += guess_pind_other[ device_start_row : device_end_row,
							device_start_col : device_end_col ] * local_adj_Ez_other


						local_gradient_device = fom_scaling * 2 * omega * eps_nought * np.real( local_p_ind / 1j )
						gradient_design[ design_row, design_col ] = np.mean( local_gradient_device )

						local_gradient_device_orig = fom_scaling * 2 * omega * eps_nought * np.real( fwd_Ez[ device_start_row : device_end_row,
							device_start_col : device_end_col ] * local_adj_Ez / 1j )

						gradient_design_orig[ design_row, design_col ] = np.mean( local_gradient_device_orig )


		# return fom, gradient_design, gradient_design_orig#, save_p_ind, save_p_ind2, save_p_ind3
		return fom, gradient_design_orig, gradient_design#, save_p_ind, save_p_ind2, save_p_ind3


	def compute_fom_and_gradient_with_polarizability( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		assert not self.field_blur, "Field blur not supported with polarizability"

		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		# bare_simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, np.ones( self.rel_eps_simulation.shape ), [ self.pml_voxels, self.pml_voxels ] )
		# bare_adj_Hx, bare_adj_Hy, bare_adj_Ez = bare_simulation.solve( adj_source )

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		gradient_design = np.zeros( ( self.design_width_voxels, self.design_height_voxels ) )
		gradient_design_orig = np.zeros( ( self.design_width_voxels, self.design_height_voxels ) )
		save_p_ind = np.zeros( fwd_Ez.shape, dtype=np.complex )
		save_p_ind2 = np.zeros( fwd_Ez.shape, dtype=np.complex )
		save_p_ind3 = np.zeros( fwd_Ez.shape, dtype=np.complex )

		# polarizability_sim = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )

		for design_row in range( 0, self.design_width_voxels ):
			for design_col in range( 0, self.design_height_voxels ):
				device_start_row = self.device_width_start + self.coarsen_factor * design_row
				device_end_row = device_start_row + self.coarsen_factor

				device_start_col = self.device_height_start + self.coarsen_factor * design_col
				device_end_col = device_start_col + self.coarsen_factor

				polarizability_src = np.zeros( self.fwd_source.shape, dtype=self.fwd_source.dtype )
				# polarizability_src_conj = np.zeros( self.fwd_source.shape, dtype=self.fwd_source.dtype )

				#
				# Oh wait, you should only have to do this once per forward source! Wasteful, but ok for now
				#
				polarizability_src[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] = eps_nought * omega * ( 1 / 1j ) * fwd_Ez[
					device_start_row : device_end_row,
					device_start_col : device_end_col ]

				pol_Hx, pol_Hy, pol_Ez = simulation.solve( polarizability_src )

				guess_pind = fwd_Ez + self.rel_eps_simulation * pol_Ez
				guess_pind2 = fwd_Ez
				guess_pind3 = self.rel_eps_simulation * pol_Ez
				# guess_pind = fwd_Ez + ( self.rel_eps_simulation - 1 ) * pol_Ez
				local_p_ind = guess_pind[ device_start_row : device_end_row,
					device_start_col : device_end_col ]
				local_p_ind2 = guess_pind2[ device_start_row : device_end_row,
					device_start_col : device_end_col ]
				local_p_ind3 = guess_pind3[ device_start_row : device_end_row,
					device_start_col : device_end_col ]


				# make_current = np.zeros( self.fwd_source.shape, dtype=self.fwd_source.dtype )
				make_current = omega**2 * eps_nought * local_p_ind

				local_adj_Ez = adj_Ez[
					device_start_row : device_end_row,
					device_start_col : device_end_col
				]
				# bare_local_adj_Ez = bare_adj_Ez[
				# 	device_start_row : device_end_row,
				# 	device_start_col : device_end_col
				# ]

				# local_gradient_device = fom_scaling * 2 * omega * eps_nought * np.real( make_current * bare_local_adj_Ez / 1j )
				# local_gradient_device = fom_scaling * 2 * ( 1. / omega ) * np.real( make_current * bare_local_adj_Ez / 1j )
				# local_gradient_device = fom_scaling * 2 * ( 1. / omega ) * np.real( make_current * local_adj_Ez / 1j )

				# test_mask = np.zeros( ( self.coarsen_factor, self.coarsen_factor ) )
				# quarter_width = int( 0.25 * self.coarsen_factor )
				# test_mask[ quarter_width : ( self.coarsen_factor - quarter_width ), quarter_width : ( self.coarsen_factor - quarter_width ) ] = 1
				# test_mask[ int( 0.5 * self.coarsen_factor ), int( 0.5 * self.coarsen_factor ) ] = 1

				local_gradient_device = fom_scaling * 2 * omega * eps_nought * np.real( local_p_ind * local_adj_Ez / 1j )
				# local_gradient_device = fom_scaling * 2 * omega * eps_nought * np.real( local_p_ind * bare_local_adj_Ez / 1j )
				gradient_design[ design_row, design_col ] = np.mean( local_gradient_device )

				local_gradient_device_orig = fom_scaling * 2 * omega * eps_nought * np.real( fwd_Ez[ device_start_row : device_end_row,
					device_start_col : device_end_col ] * local_adj_Ez / 1j )


				gradient_design_orig[ design_row, design_col ] = np.mean( local_gradient_device_orig )

				save_p_ind[ device_start_row : device_end_row, device_start_col : device_end_col ] = local_p_ind
				save_p_ind2[ device_start_row : device_end_row, device_start_col : device_end_col ] = local_p_ind2
				save_p_ind3[ device_start_row : device_end_row, device_start_col : device_end_col ] = local_p_ind3



		# gradient_design = gradient_design_orig.copy()
		# save00 = gradient_design[ 0, 0 ]
		# savem1m1 = gradient_design[ self.design_width_voxels - 1, self.design_height_voxels - 1 ]
		# gradient_design = np.zeros( gradient_design.shape )
		# gradient_design[ 0, 0 ] = save00
		# gradient_design[ self.design_width_voxels - 1, self.design_height_voxels - 1 ] = savem1m1
		return fom, gradient_design, gradient_design_orig, save_p_ind, save_p_ind2, save_p_ind3
		# return fom, gradient_design, gradient_design_orig, save_p_ind, save_p_ind2, save_p_ind3
		# return fom, gradient_design_orig, gradient_design#, save_p_ind, save_p_ind2, save_p_ind3



	def compute_fom_and_gradient_with_polarizability_( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		assert not self.field_blur, "Field blur not supported with polarizability"

		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )
		adj_source[ focal_point_x_loc + 4, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc + 4, self.focal_point_y ] )

		# adj_source_2 = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		# adj_source_2[ focal_point_x_loc, self.focal_point_y ] = ( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )
		# adj_source_2[ focal_point_x_loc + 4, self.focal_point_y ] = ( fwd_Ez[ focal_point_x_loc + 4, self.focal_point_y ] )


		# adj_source[
		# 	self.device_width_start : self.device_width_end,
		# 	self.device_height_start : self.device_height_end ] = (
		# 		np.random.random( ( self.device_width_voxels, self.device_height_voxels ) ) + 1j * 
		# 		np.random.random( ( self.device_width_voxels, self.device_height_voxels ) ) )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )
		# adj_Hx_2, adj_Hy_2, adj_Ez_2 = simulation.solve( adj_source_2 )

		# adj_Hx_, adj_Hy_, adj_Ez_ = simulation.solve( np.conj( adj_source ) )

		local_adj_Ez = adj_Ez[
			self.device_width_start : self.device_width_end,
			self.device_height_start : self.device_height_end
		]

		# local_adj_Ez_2 = adj_Ez_2[
		# 	self.device_width_start : self.device_width_end,
		# 	self.device_height_start : self.device_height_end
		# ]

		# device_start_row = self.coarsen_factor * 3
		# device_end_row = device_start_row + self.coarsen_factor
		# device_start_col = self.coarsen_factor * 2
		# device_end_col = device_start_col + self.coarsen_factor

		# import matplotlib.pyplot as plt
		# plt.subplot( 1, 2, 1 )
		# plt.imshow( np.abs( local_adj_Ez[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
		# plt.subplot( 1, 2, 2 )
		# plt.imshow( np.abs( local_adj_Ez_2[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
		# plt.show()
		# asdfasd

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		gradient_design = np.zeros( ( self.design_width_voxels, self.design_height_voxels ) )

		# polarizability_sim = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		polarizability_sim = ceviche.fdfd_ez( omega, self.mesh_size_m, np.ones( self.rel_eps_simulation.shape ), [ self.pml_voxels, self.pml_voxels ] )

		for design_row in range( 2, self.design_width_voxels ):
			for design_col in range( 1, self.design_height_voxels ):
				device_start_row = self.device_width_start + self.coarsen_factor * design_row
				device_end_row = device_start_row + self.coarsen_factor

				device_start_col = self.device_height_start + self.coarsen_factor * design_col
				device_end_col = device_start_col + self.coarsen_factor

				polarizability_src = np.zeros( self.fwd_source.shape, dtype=np.complex )#dtype=self.fwd_source.dtype )
				polarizability_src_conj = np.zeros( self.fwd_source.shape, dtype=np.complex )#dtype=self.fwd_source.dtype )

				# testsrc = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )

				# testsrc[
				# 	device_start_row + self.coarsen_factor : device_end_row + self.coarsen_factor,
				# 	device_start_col + self.coarsen_factor : device_end_col + self.coarsen_factor ] = (
				# 		np.random.random( ( self.coarsen_factor, self.coarsen_factor ) ) +
				# 		1j * np.random.random( ( self.coarsen_factor, self.coarsen_factor ) ) )


				#
				# Oh wait, you should only have to do this once per forward source! Wasteful, but ok for now
				#
				# polarizability_src[
				# 	device_start_row + self.coarsen_factor : device_end_row + self.coarsen_factor,
				# 	device_start_col + self.coarsen_factor : device_end_col + self.coarsen_factor ] = testsrc[
				# 	device_start_row + self.coarsen_factor : device_end_row + self.coarsen_factor,
				# 	device_start_col + self.coarsen_factor : device_end_col + self.coarsen_factor ]
				# polarizability_src = np.imag( polarizability_src )

				# Pnew = (eps + del_eps) * (Eold + delE)
				# Pold = eps * Eold
				# del_P = del_eps * Eold + eps * delE

				# delE = 

				polarizability_src[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] = ( eps_nought / 1j ) * omega * fwd_Ez[
					device_start_row : device_end_row,
					device_start_col : device_end_col ]
				# polarizability_src[
				# device_start_row + 4,
				# 	device_start_col + 8 ] = ( eps_nought / 1j ) * omega * fwd_Ez[ device_start_row + 4, device_start_col + 8 ]

				pol_Hx, pol_Hy, pol_Ez = simulation.solve( polarizability_src )

				# polarizability_src_conj[
				# 	device_start_row : device_end_row,
				# 	device_start_col : device_end_col ] = ( eps_nought / 1j ) * omega * np.conj( fwd_Ez[
				# 	device_start_row : device_end_row,
				# 	device_start_col : device_end_col ] )
				# polarizability_src_conj[
				# device_start_row + 4,
				# 	device_start_col + 8 ] = ( eps_nought / 1j ) * omega * np.conj( fwd_Ez[ device_start_row + 4, device_start_col + 8 ] )

				# pol_Hx_conj, pol_Hy_conj, pol_Ez_conj = simulation.solve( polarizability_src_conj )

				# import matplotlib.pyplot as plt
				# plt.subplot( 5, 2, 1 )
				# plt.imshow( np.real( pol_Ez[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 2 )
				# plt.imshow( np.imag( pol_Ez[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 3 )
				# plt.imshow( np.real( pol_Ez_conj[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 4 )
				# plt.imshow( np.imag( pol_Ez_conj[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 5 )
				# plt.imshow( np.abs( pol_Ez[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 6 )
				# plt.imshow( np.abs( pol_Ez_conj[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 7 )
				# plt.imshow( np.real( polarizability_src[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 8 )
				# plt.imshow( np.real( polarizability_src_conj[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 9 )
				# plt.imshow( np.imag( polarizability_src ) )
				# plt.colorbar()
				# plt.subplot( 5, 2, 10 )
				# plt.imshow( np.imag( polarizability_src_conj ) )
				# plt.colorbar()
				# plt.show()




				new_rel_eps = self.rel_eps_simulation.copy()
				test_delta = 0.001
				new_rel_eps[ device_start_row : device_end_row, device_start_col : device_end_col ] += test_delta

				next_row_over_start = device_start_row + self.coarsen_factor
				next_row_over_end = next_row_over_start + self.coarsen_factor
				next_col_over_start = device_start_row + self.coarsen_factor
				next_col_over_end = next_row_over_start + self.coarsen_factor

				new_rel_eps[ next_row_over_start : next_row_over_end, next_col_over_start : next_col_over_end ] -= test_delta


				new_polarizability_sim = ceviche.fdfd_ez( omega, self.mesh_size_m, new_rel_eps, [ self.pml_voxels, self.pml_voxels ] )
				check_Hx, check_Hy, check_Ez = new_polarizability_sim.solve( self.fwd_source )

				check_pind = ( new_rel_eps * check_Ez - self.rel_eps_simulation * fwd_Ez )
				guess_pind = test_delta * fwd_Ez
				guess_pind2 = test_delta * ( fwd_Ez + self.rel_eps_simulation * pol_Ez )
				# check_pind_conj = ( new_rel_eps * np.conj( check_Ez ) - self.rel_eps_simulation * np.conj( fwd_Ez ) ) / test_delta

				import matplotlib.pyplot as plt
				plt.subplot( 3, 2, 1 )
				plt.imshow( np.real( check_pind[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 2 )
				plt.imshow( np.imag( check_pind[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 3 )
				plt.imshow( np.real( guess_pind[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 4 )
				plt.imshow( np.imag( guess_pind[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 5 )
				plt.imshow( np.real( guess_pind2[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 6 )
				plt.imshow( np.imag( guess_pind2[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()

				plt.show()
				continue


				# beta_piece = self.rel_eps_simulation[
				# 	device_start_row : device_end_row,
				# 	device_start_col : device_end_col ] * pol_Ez[ 
				# 		device_start_row : device_end_row,
				# 		device_start_col : device_end_col
				# 	]
				e_piece = fwd_Ez[ device_start_row : device_end_row,
								device_start_col : device_end_col ]
				e_piece_conj = np.conj( fwd_Ez[ device_start_row : device_end_row,
								device_start_col : device_end_col ] )

				rel_eps_piece = self.rel_eps_simulation[
					device_start_row : device_end_row,
					device_start_col : device_end_col ]
				pol_piece = pol_Ez[ 
						device_start_row : device_end_row,
						device_start_col : device_end_col ]
				# pol_piece_conj = pol_Ez_conj[ 
				# 		device_start_row : device_end_row,
				# 		device_start_col : device_end_col ]


				# new_E = ( rel_eps_piece * pol_piece / ( eps_nought * omega ) ) + e_piece
				# new_E = ( test_delta * rel_eps_piece * pol_piece ) + e_piece
				# new_E_conj = ( test_delta * rel_eps_piece * pol_piece_conj ) + np.conj( e_piece )
				# new_E = ( test_delta * pol_piece ) + e_piece
				# new_delta_E_conj = np.conj( new_E ) - np.conj( e_piece )
				# new_delta_E = test_delta * rel_eps_piece * pol_piece
				
				import matplotlib.pyplot as plt
				plt.subplot( 3, 2, 1 )
				plt.imshow( np.real( pol_piece ) )
				plt.title('New E')
				plt.colorbar()
				plt.subplot( 3, 2, 2 )
				plt.imshow( np.real( e_piece ) )
				plt.title('New E conj')
				plt.colorbar()
				plt.subplot( 3, 2, 3 )
				plt.imshow( np.imag( pol_piece ) )
				plt.title('New E')
				plt.colorbar()
				plt.subplot( 3, 2, 4 )
				plt.imshow( np.imag( e_piece ) )
				plt.title('New E conj')
				plt.colorbar()
				plt.subplot( 3, 2, 5 )
				plt.imshow( np.abs( pol_piece / e_piece ) )
				plt.title('New E')
				plt.colorbar()
				plt.subplot( 3, 2, 6 )
				plt.imshow( np.angle( pol_piece / e_piece ) )
				plt.title('New E conj')
				plt.colorbar()
				plt.show()	
				print( np.angle( e_piece ) )
				print( np.abs( pol_piece / e_piece ) )
				print( np.angle( pol_piece / e_piece ) )
				print( eps_nought * omega )
				print( 1. / ( eps_nought * omega ) )
				print()
				continue


				# import matplotlib.pyplot as plt
				# plt.subplot( 3, 2, 1 )
				# plt.imshow( np.real( new_E - e_piece ) )
				# plt.colorbar()
				# plt.subplot( 3, 2, 2 )
				# plt.imshow( np.imag( new_E - e_piece ) )
				# plt.colorbar()
				# plt.subplot( 3, 2, 3 )
				# plt.imshow( np.real( check_Ez[ 
				# 		device_start_row : device_end_row,
				# 		device_start_col : device_end_col ] - e_piece ) )
				# plt.colorbar()
				# plt.subplot( 3, 2, 4 )
				# plt.imshow( np.imag( check_Ez[ 
				# 		device_start_row : device_end_row,
				# 		device_start_col : device_end_col ] - e_piece ) )
				# plt.colorbar()
				# plt.subplot( 3, 2, 5 )
				# plt.imshow( np.abs( new_E - e_piece ) / ( np.abs( e_piece ) + 1e-14 ) )
				# plt.colorbar()
				# plt.subplot( 3, 2, 6 )
				# plt.imshow( np.abs( check_Ez[ 
				# 		device_start_row : device_end_row,
				# 		device_start_col : device_end_col ] - e_piece ) / ( np.abs( e_piece ) + 1e-14 ) )
				# plt.colorbar()
				# plt.show()


				local_p_ind = self.rel_eps_simulation[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] * pol_Ez[ 
						device_start_row : device_end_row,
						device_start_col : device_end_col
					] + e_piece

				local_p_ind_conj = self.rel_eps_simulation[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] * np.conj( pol_Ez[ 
						device_start_row : device_end_row,
						device_start_col : device_end_col
					] ) + e_piece_conj

				plt.subplot( 4, 2, 1 )
				plt.imshow( np.real( check_pind[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] ) )
				plt.title('Check pind')
				plt.colorbar()
				plt.subplot( 4, 2, 2 )
				plt.imshow( np.real( check_pind_conj[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] ) )
				plt.title('Check pind conj')
				plt.colorbar()
				plt.subplot( 4, 2, 3 )
				plt.imshow( np.imag( check_pind[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] ) )
				plt.title('Check pind')
				plt.colorbar()
				plt.subplot( 4, 2, 4 )
				plt.imshow( np.imag( check_pind_conj[
					device_start_row : device_end_row,
					device_start_col : device_end_col ] ) )
				plt.title('Check pind conj')
				plt.colorbar()

				plt.subplot( 4, 2, 5 )
				plt.imshow( np.real( local_p_ind ) )
				plt.title('New P')
				plt.colorbar()
				plt.subplot( 4, 2, 6 )
				plt.imshow( np.real( local_p_ind_conj ) )
				plt.title('New P conj')
				plt.colorbar()
				plt.subplot( 4, 2, 7 )
				plt.imshow( np.imag( local_p_ind ) )
				plt.title('New P')
				plt.colorbar()
				plt.subplot( 4, 2, 8 )
				plt.imshow( np.imag( local_p_ind_conj ) )
				plt.title('New P conj')
				plt.colorbar()
				plt.show()	

				# local_p_ind =  e_piece

				# local_p_ind_conj = e_piece_conj


				local_adj_Ez = adj_Ez[
					device_start_row : device_end_row,
					device_start_col : device_end_col
				]

				local_adj_Ez_conj = adj_Ez_2[
					device_start_row : device_end_row,
					device_start_col : device_end_col
				]

				calc_0 = omega * eps_nought * ( local_p_ind * local_adj_Ez / 1j )
				calc_1 = omega * eps_nought * ( local_p_ind_conj * np.conj( local_adj_Ez / 1j ) )


				plt.subplot( 2, 2, 1 )
				plt.imshow( np.real( calc_0 ) )
				plt.colorbar()
				plt.subplot( 2, 2, 2 )
				plt.imshow( np.imag( calc_0 ) )
				plt.colorbar()
				plt.subplot( 2, 2, 3 )
				plt.imshow( np.real( calc_1 ) )
				plt.colorbar()
				plt.subplot( 2, 2, 4 )
				plt.imshow( np.imag( calc_1 ) )
				plt.colorbar()
				plt.show()

				print( calc_0 + calc_1 )
				print('\n\n\n---------\n\n\n')
				print(2 * omega * eps_nought * np.real( local_p_ind * local_adj_Ez / 1j ))


				import matplotlib.pyplot as plt
				# plt.subplot( 2, 2, 1 )
				# plt.imshow( np.abs( self.rel_eps_simulation[
				# 	device_start_row : device_end_row,
				# 	device_start_col : device_end_col ] * pol_Ez[ 
				# 		device_start_row : device_end_row,
				# 		device_start_col : device_end_col
				# 	] ) )
				# plt.colorbar()
				# plt.subplot( 2, 2, 2 )
				# plt.imshow( np.abs( fwd_Ez[ device_start_row : device_end_row,
				# 				device_start_col : device_end_col ] ) )
				# plt.colorbar()

				plt.subplot( 3, 2, 1 )
				plt.imshow( np.real( local_p_ind ) )
				plt.colorbar()
				plt.subplot( 3, 2, 2 )
				plt.imshow( np.imag( local_p_ind ) )
				plt.colorbar()

				plt.subplot( 3, 2, 3 )
				plt.imshow( np.real( check_pind[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 4 )
				plt.imshow( np.imag( check_pind[ device_start_row : device_end_row, device_start_col : device_end_col ] ) )
				plt.colorbar()
				# plt.show()

				plt.subplot( 3, 2, 5 )
				plt.imshow( np.real( fwd_Ez[ device_start_row : device_end_row,
								device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.subplot( 3, 2, 6 )
				plt.imshow( np.imag( fwd_Ez[ device_start_row : device_end_row,
								device_start_col : device_end_col ] ) )
				plt.colorbar()
				plt.show()

				# sum_pind_contribution = np.mean( 2 * omega * eps_nought * np.real( local_p_ind * local_adj_Ez / 1j ) )
				# sum_e_contribution = np.mean( 2 * omega * eps_nought * np.real( e_piece * local_adj_Ez / 1j ) )

				# if sum_pind_contribution >= sum_e_contribution:
				# 	print('a')
				# 	gradient_design[ design_row, design_col ] = fom_scaling * sum_e_contribution
				# else:
				# 	print('b')
				# 	gradient_design[ design_row, design_col ] = 2 * eps_nought * fom_scaling * sum_pind_contribution / sum_e_contribution

				# 	print( fom_scaling * sum_e_contribution )
				# 	print( gradient_design[ design_row, design_col ] )
				# 	print()

				local_gradient_device = fom_scaling * 2 * omega * eps_nought * np.real( local_p_ind * local_adj_Ez / 1j )
				gradient_design[ design_row, design_col ] = np.mean( local_gradient_device )

		return fom, gradient_design


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


	def compute_net_fom_from_density( self, input_density ):
		fom_by_wl = []

		import_density = upsample( input_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		for wl_idx in range( 0, self.num_wavelengths ):
			get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]
			get_fom = self.compute_fom(
				self.omega_values[ wl_idx ], device_permittivity,
				self.focal_spots_x_voxels[ get_focal_point_idx ], self.wavelength_intensity_scaling[ wl_idx ] )
			fom_by_wl.append( get_fom )

		net_fom = np.product( fom_by_wl )

		return net_fom

	# CODE DUPLICATION! FIX
	def compute_net_fom( self ):
		return self.compute_net_fom_from_density( self.design_density )


	def verify_adjoint_against_finite_difference_lambda_design_line( self, save_loc ):
		# get_density = upsample( self.design_density, self.coarsen_factor )
		# get_permittivity = self.density_to_permittivity( get_density )
		np.random.seed( 23123 )

		# random_density = upsample( np.random.random( ( int( self.device_width_voxels / 4 ), int( self.device_height_voxels / 4 ) ) ), 4 )
		# random_perm = self.density_to_permittivity( random_density )

		# random_density = np.random.random( ( self.design_width_voxels, self.design_height_voxels ) )
		random_density = 0.5 * np.ones( ( self.design_width_voxels, self.design_height_voxels ) )
		random_density = upsample( random_density, self.coarsen_factor )
		random_perm = self.density_to_permittivity( random_density )


		fd_focal_x_loc = self.focal_spots_x_voxels[ 0 ]
		fd_grad = np.zeros( self.design_density.shape )
		fd_grad_second = np.zeros( self.design_density.shape )
		# fom_init, adj_grad, adj_grad_orig, save_p_ind, save_p_ind2, save_p_ind3 = self.compute_fom_and_gradient_with_polarizability(
		# 	self.omega_values[ 0 ], random_perm, fd_focal_x_loc )
		fom_init, adj_grad = self.compute_fom_and_gradient(
			self.omega_values[ 0 ], random_perm, fd_focal_x_loc )

		adj_grad = adj_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
		adj_grad = ( self.coarsen_factor )**2 * reinterpolate_average( adj_grad, self.coarsen_factor )

		choose_row = int( 0.5 * self.design_width_voxels )
		choose_col = int( 0.5 * self.design_height_voxels )

		h_min = -0.05
		h_max = 0.05
		num_h = 201

		h_values = np.linspace( h_min, h_max, num_h )

		fom_line = np.zeros( num_h )

		for h_idx in range( 0, num_h ):
			copy_perm = random_perm.copy()
			copy_perm[
				choose_row * self.coarsen_factor : ( choose_row + 1 ) * self.coarsen_factor,
				choose_col * self.coarsen_factor : ( choose_col + 1 ) * self.coarsen_factor ] += h_values[ h_idx ]
			# copy_perm[
			# 	choose_row * self.coarsen_factor,
			# 	choose_col * self.coarsen_factor ] += h_values[ h_idx ]

			fd_permittivity = copy_perm.copy()

			fom_line[ h_idx ] = self.compute_fom( self.omega_values[ 0 ], fd_permittivity, fd_focal_x_loc )			

		np.save( save_loc + "_h_values.npy", h_values )
		np.save( save_loc + "_fd_line.npy", fom_line )
		np.save( save_loc + "_adj_grad.npy", adj_grad )
		# np.save( save_loc + "_adj_grad_orig.npy", adj_grad_orig )


	def verify_adjoint_against_finite_difference_lambda_design_anisotropic( self, save_loc ):
		# get_density = upsample( self.design_density, self.coarsen_factor )
		# get_permittivity = self.density_to_permittivity( get_density )
		np.random.seed( 23123 )

		random_density = upsample( np.random.random( ( int( self.device_width_voxels / 4 ), int( self.device_height_voxels / 4 ) ) ), 4 )
		random_perm = self.density_to_permittivity( random_density )

		# random_density = np.random.random( ( self.design_width_voxels, self.design_height_voxels ) )
		# random_density = 0.5 * np.ones( ( self.design_width_voxels, self.design_height_voxels ) )
		# random_density = upsample( random_density, self.coarsen_factor )
		# random_perm = self.density_to_permittivity( random_density )
	
		fd_focal_x_loc = self.focal_spots_x_voxels[ 0 ]
		fd_grad = np.zeros( self.design_density.shape )
		fd_grad_second = np.zeros( self.design_density.shape )
		fom_init, adj_grad, adj_grad_orig, save_p_ind, save_p_ind2, save_p_ind3 = self.compute_fom_and_gradient_with_polarizability(
			self.omega_values[ 0 ], random_perm, fd_focal_x_loc )

		h = 1e-3 / ( self.coarsen_factor**2 )

		for row in range( 0, self.design_width_voxels ):
			for col in range( 0, self.design_height_voxels ):
				# copy_density = random_density.copy()
				copy_perm = random_perm.copy()

				copy_perm[ row * self.coarsen_factor : (row + 1) * self.coarsen_factor, col * self.coarsen_factor : ( col + 1 ) * self.coarsen_factor ] += h

				# copy_density[ row, col ] += ( h / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
				# fd_density = upsample( copy_density, self.coarsen_factor )
				fd_permittivity = copy_perm.copy()#self.density_to_permittivity( fd_density )

				fom_up = self.compute_fom( self.omega_values[ 0 ], fd_permittivity, fd_focal_x_loc )			




				copy_perm = random_perm.copy()

				copy_perm[ row * self.coarsen_factor : (row + 1) * self.coarsen_factor, col * self.coarsen_factor : ( col + 1 ) * self.coarsen_factor ] -= h

				# copy_density[ row, col ] += ( h / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
				# fd_density = upsample( copy_density, self.coarsen_factor )
				fd_permittivity = copy_perm#self.density_to_permittivity( fd_density )


				# copy_density = random_density.copy()
				# copy_density[ row, col ] -= ( h / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
				# fd_density = upsample( copy_density, self.coarsen_factor )
				fd_permittivity = copy_perm.copy()#self.density_to_permittivity( fd_density )

				fom_down = self.compute_fom( self.omega_values[ 0 ], fd_permittivity, fd_focal_x_loc )

				fd_grad[ row, col ] = ( fom_up - fom_down ) / ( 2 * h )

				fd_grad_second[ row, col ] = ( fom_up + fom_down - 2 * fom_init ) / ( h**2 )


		np.save( save_loc + "_fd_grad.npy", fd_grad )
		np.save( save_loc + "_fd_grad_second.npy", fd_grad_second )
		np.save( save_loc + "_adj_grad.npy", adj_grad )
		np.save( save_loc + "_adj_grad_orig.npy", adj_grad_orig )


	def verify_adjoint_against_finite_difference_lambda_design( self, save_loc ):
		get_density = upsample( self.design_density, self.coarsen_factor )
		get_permittivity = self.density_to_permittivity( get_density )

		fd_focal_x_loc = self.focal_spots_x_voxels[ 0 ]
		fd_grad = np.zeros( self.design_density.shape )
		fd_grad_second = np.zeros( self.design_density.shape )
		fom_init, adj_grad, adj_grad_orig, save_p_ind, save_p_ind2, save_p_ind3 = self.compute_fom_and_gradient_with_polarizability__(
			self.omega_values[ 0 ], get_permittivity, fd_focal_x_loc )

		# first_ez = self.compute_forward_fields( self.omega_values[ 0 ], get_permittivity )


		# make_current = np.zeros( self.fwd_source.shape, dtype=self.fwd_source.dtype )
		# make_current_ = np.zeros( self.fwd_source.shape, dtype=self.fwd_source.dtype )

		diff = adj_grad - adj_grad_orig
		import matplotlib.pyplot as plt
		# plt.subplot( 1, 2, 1 )
		# plt.imshow( np.abs( diff ) )
		# plt.colorbar()
		# plt.subplot( 1, 2, 2 )
		# plt.imshow( np.abs( adj_grad_orig ) )
		# plt.colorbar()
		# plt.show()

		np.random.seed( 23123 )
		test_delta = 1e-3 * 2 * ( np.random.random( self.design_density.shape ) - 0.5 )#+ 0.5 )
		# test_delta = np.zeros( test_delta_.shape )
		# test_delta[ 0, 0 ] = test_delta_[ 0, 0 ]

		# for row in range( 0, self.design_density.shape[ 0 ] ):
		# 	for col in range( 0, self.design_density.shape[ 1 ] ):
		# 		make_current[
		# 			self.device_width_start + row * self.coarsen_factor : self.device_width_start + ( row + 1 ) * self.coarsen_factor,
		# 			self.device_height_start + col * self.coarsen_factor : self.device_height_start + ( col + 1 ) * self.coarsen_factor ] = (
		# 			# ( get_permittivity - 1 ) *
		# 			self.omega_values[ 0 ] * test_delta[ row, col ] * eps_nought *
		# 			save_p_ind[
		# 				self.device_width_start + row * self.coarsen_factor : self.device_width_start + ( row + 1 ) * self.coarsen_factor,
		# 				self.device_height_start + col * self.coarsen_factor : self.device_height_start + ( col + 1 ) * self.coarsen_factor ] / 1j
		# 		)

		# 		# make_current[
		# 		# 	self.device_width_start + row * self.coarsen_factor : self.device_width_start + ( row + 1 ) * self.coarsen_factor,
		# 		# 	self.device_height_start + col * self.coarsen_factor : self.device_height_start + ( col + 1 ) * self.coarsen_factor ] += (
		# 		# 	( get_permittivity[ row * self.coarsen_factor : ( row + 1 ) * self.coarsen_factor, col * self.coarsen_factor : ( col + 1 ) * self.coarsen_factor ] - 1 ) *
		# 		# 	self.omega_values[ 0 ] * eps_nought *
		# 		# 	first_ez[
		# 		# 		self.device_width_start + row * self.coarsen_factor : self.device_width_start + ( row + 1 ) * self.coarsen_factor,
		# 		# 		self.device_height_start + col * self.coarsen_factor : self.device_height_start + ( col + 1 ) * self.coarsen_factor ] / 1j
		# 		# )

		# 		make_current_[
		# 			self.device_width_start + row * self.coarsen_factor : self.device_width_start + ( row + 1 ) * self.coarsen_factor,
		# 			self.device_height_start + col * self.coarsen_factor : self.device_height_start + ( col + 1 ) * self.coarsen_factor ] = (
		# 			# ( get_permittivity - 1 ) *
		# 			self.omega_values[ 0 ] * (
		# 				( get_permittivity[
		# 					row * self.coarsen_factor : ( row + 1 ) * self.coarsen_factor,
		# 					col * self.coarsen_factor : ( col + 1 ) * self.coarsen_factor ] - 1 ) + test_delta[ row, col ] ) * eps_nought *
		# 			first_ez[
		# 				self.device_width_start + row * self.coarsen_factor : self.device_width_start + ( row + 1 ) * self.coarsen_factor,
		# 				self.device_height_start + col * self.coarsen_factor : self.device_height_start + ( col + 1 ) * self.coarsen_factor ] / 1j
		# 		)


		# # print( make_current )
		# # make_current += self.fwd_source
		# make_current_ += self.fwd_source

		# bare_simulation = ceviche.fdfd_ez( self.omega_values[ 0 ], self.mesh_size_m, np.ones( self.rel_eps_simulation.shape ), [ self.pml_voxels, self.pml_voxels ] )
		# bare_Hx, bare_Hy, bare_Ez = bare_simulation.solve( make_current )
		# bare_Hx_, bare_Hy_, bare_Ez_ = bare_simulation.solve( make_current_ )

		# test_density = self.design_density.copy()
		# test_density += ( test_delta / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
		# get_density = upsample( test_density, self.coarsen_factor )
		# get_permittivity = self.density_to_permittivity( get_density )

		# fom_up = self.compute_fom( self.omega_values[ 0 ], get_permittivity, fd_focal_x_loc )

		# other_ez = self.compute_forward_fields( self.omega_values[ 0 ], get_permittivity )

		# plt.subplot( 1, 3, 1 )
		# plt.imshow( np.real( bare_Ez_ ) )
		# plt.colorbar()
		# plt.subplot( 1, 3, 2 )
		# plt.imshow( np.real( bare_Ez ) )
		# plt.colorbar()
		# plt.subplot( 1, 3, 3 )
		# plt.imshow( np.real( other_ez ) )
		# plt.colorbar()
		# plt.show()

		# # plt.plot( np.real( bare_Ez_[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='g', linestyle='-' )
		# # plt.plot( np.real( first_ez[ :, self.focal_point_y ] ), color='b', linestyle='--' )
		# # plt.plot( np.real( other_ez[ :, self.focal_point_y ] ), color='r', linestyle=':' )
		# # plt.show()

		# plt.plot( np.real( other_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] - first_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='b' )
		# plt.plot( np.real( bare_Ez_[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] - first_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='g', linestyle='--' )
		# plt.plot( np.real( bare_Ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] - 0 * first_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='r', linestyle='--' )

		# plt.plot( 3e-10 + np.imag( other_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] - first_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='k' )
		# plt.plot( 3e-10 + np.imag( bare_Ez_[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] - first_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='c', linestyle='--' )
		# plt.plot( 3e-10 + np.imag( bare_Ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] - 0 * first_ez[ self.pml_voxels : self.simulation_width_voxels - self.pml_voxels, self.focal_point_y ] ), color='m', linestyle='--' )

		# plt.show()

		# asdfas


		# print( adj_grad.shape )

		# predicted_grad_delta = self.coarsen_factor**2 * ( np.sum( test_delta * adj_grad_orig ) + np.sum( test_delta**2 * diff ) )
		# predicted_grad_delta_orig = self.coarsen_factor**2 * np.sum( test_delta * adj_grad_orig )
		# print( predicted_grad_delta )
		# print( predicted_grad_delta_orig )


		# print( ( fom_up - fom_init ) )

		# asdfads



		save_p_ind = save_p_ind[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
		save_p_ind2 = save_p_ind2[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
		save_p_ind3 = save_p_ind3[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

		num = 9
		mid_num = int( 0.5 * num )
		h = np.linspace( -5e-2, 5e-2, num )

		import matplotlib.pyplot as plt
		# choose_row = int( 0.5 * 0.25 * self.device_width_voxels )
		# choose_col = int( 0.85 * 0.25 * self.device_height_voxels )

		choose_row = int( 0.25 * self.coarsen_factor )
		choose_col = int( 0.85 * self.coarsen_factor )

		# choose_row = self.coarsen_factor - 1
		# choose_col = int( 0.5 * self.coarsen_factor )# - 1

		ez_vals = np.zeros( ( num, self.device_width_voxels, self.device_height_voxels ), dtype=np.complex )
		ez_diff = np.zeros( ( num, self.device_width_voxels, self.device_height_voxels ), dtype=np.complex )
		del_p = np.zeros( ( num, self.device_width_voxels, self.device_height_voxels ), dtype=np.complex )

		movement = adj_grad_orig / np.max( np.abs( adj_grad_orig ) )
		movement *= 0.01

		for idx in range( 0, num ):
			copy_density = self.design_density.copy()
			# copy_density[ 0, 0 ] += ( 0.5 * h[ idx ] / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			copy_density[ 0, 0 ] += ( h[ idx ] / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			
			# copy_density[ 1, 0 ] += movement[ 1, 0 ]# ( 0.02 / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			# copy_density[ 1, 1 ] += movement[ 1, 1 ]# ( 0.1 / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			# copy_density[ 0, 1 ] += movement[ 0, 1 ]# ( -0.05 / ( self.max_relative_permittivity - self.min_relative_permittivity ) )

			# copy_density[ 1, 0 ] += ( 0.02 / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			# copy_density[ 1, 1 ] += ( 0.1 / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			# copy_density[ 0, 1 ] += ( -0.05 / ( self.max_relative_permittivity - self.min_relative_permittivity ) )

			# if not ( idx == mid_num ):

			# for row in range( 0, self.design_density.shape[ 0 ] ):
			# 	for col in range( 0, self.design_density.shape[ 1 ] ):
			# 		if ( row == 0 ) and ( col == 0 ):
			# 			continue
			# 		copy_density[ row, col ] += movement[ row, col ]



			fd_density = upsample( copy_density, self.coarsen_factor )
			fd_permittivity = self.density_to_permittivity( fd_density )
			# fd_permittivity[ choose_row, choose_col ] += ( 0.5 * h[ idx ] / ( self.max_relative_permittivity - self.min_relative_permittivity ) )

			ez = self.compute_forward_fields( self.omega_values[ 0 ], fd_permittivity )
			ez_vals[ idx ] = ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

			# del_p[ idx ] = (
			# 	fd_permittivity[ 0 : self.device_width_voxels, 0 : self.device_height_voxels ] * ez_vals[ idx ] -
			# 	get_permittivity[ 0 : self.device_width_voxels, 0 : self.device_height_voxels ] * ez_vals[ mid_num ] )
			# del_p[ idx ] = (
			# 	# fd_permittivity[ 0 : self.device_width_voxels, 0 : self.device_height_voxels ] * ez_vals[ idx ] -
			# 	h[ idx ] * ez_vals[ 0 ] )

		for idx in range( 0, num ):
			copy_density = self.design_density.copy()
			copy_density[ 0, 0 ] += ( h[ idx ] / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
			fd_density = upsample( copy_density, self.coarsen_factor )
			fd_permittivity = self.density_to_permittivity( fd_density )

			del_p[ idx ] = (
				fd_permittivity[ 0 : self.device_width_voxels, 0 : self.device_height_voxels ] * ez_vals[ idx ] -
				get_permittivity[ 0 : self.device_width_voxels, 0 : self.device_height_voxels ] * ez_vals[ mid_num ] )


		print( np.real( ez_vals[ mid_num, choose_row, choose_col ] ) )
		# print( save_p_ind2[ choose_row, choose_col ] )
		# print( save_p_ind2[ choose_row + 3, choose_col - 6 ] )
		# print( save_p_ind3[ choose_row, choose_col ] )
		# plt.subplot( 1, 2, 1 )
		plt.plot( h, np.real( del_p[ :, choose_row, choose_col ] ), color='b', linewidth=2 )
		plt.plot( h, np.imag( del_p[ :, choose_row, choose_col ] ), color='r', linewidth=2 )
		# plt.plot( h, h * np.real( ez_vals[ mid_num, choose_row, choose_col ] ), color='k', linestyle='--', linewidth=2 )
		# plt.plot( h, h * np.imag( ez_vals[ mid_num, choose_row, choose_col ] ), color='orange', linestyle='--', linewidth=2 )
		plt.plot( h, h * np.real( save_p_ind[ choose_row, choose_col ] ), color='orange', linestyle=':' )
		plt.plot( h, h * np.imag( save_p_ind[ choose_row, choose_col ] ), color='k', linestyle=':' )
		# plt.plot( h, h * np.real( save_p_ind2[ choose_row, choose_col ] + save_p_ind3[ choose_row, choose_col ] ), color='c', linestyle=':' )
		# plt.plot( h, h * np.imag( save_p_ind2[ choose_row, choose_col ] + save_p_ind3[ choose_row, choose_col ] ), color='m', linestyle=':' )
		plt.plot( h, h * np.real( save_p_ind2[ choose_row, choose_col ] ), color='g', linestyle='--' )
		plt.plot( h, h * np.imag( save_p_ind2[ choose_row, choose_col ] ), color='purple', linestyle='--' )
		plt.show()

		adsf


		h = 1e-3
	
		for row in range( 0, self.design_width_voxels ):
			for col in range( 0, self.design_height_voxels ):
				copy_density = self.design_density.copy()
				copy_density[ row, col ] += ( h / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
				fd_density = upsample( copy_density, self.coarsen_factor )
				fd_permittivity = self.density_to_permittivity( fd_density )

				fom_up = self.compute_fom( self.omega_values[ 0 ], fd_permittivity, fd_focal_x_loc )			

				copy_density = self.design_density.copy()
				copy_density[ row, col ] -= ( h / ( self.max_relative_permittivity - self.min_relative_permittivity ) )
				fd_density = upsample( copy_density, self.coarsen_factor )
				fd_permittivity = self.density_to_permittivity( fd_density )

				fom_down = self.compute_fom( self.omega_values[ 0 ], fd_permittivity, fd_focal_x_loc )

				fd_grad[ row, col ] = ( fom_up - fom_down ) / ( 2 * h )

				fd_grad_second[ row, col ] = ( fom_up + fom_down - 2 * fom_init ) / ( h**2 )



		np.save( save_loc + "_fd_grad.npy", fd_grad )
		np.save( save_loc + "_fd_grad_second.npy", fd_grad_second )
		np.save( save_loc + "_adj_grad.npy", adj_grad )
		np.save( save_loc + "_adj_grad_orig.npy", adj_grad_orig )




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

		omega_low = 2 * np.pi * c / ( 1e-6 * lambda_low_um )
		omega_high = 2 * np.pi * c / ( 1e-6 * lambda_high_um )

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


	def polarizability_test( self ):
		import matplotlib.pyplot as plt

		self.init_density_with_uniform( 0.0 )
		select_omega = self.omega_values[ 0 ]#int( 0.5 * len( self.omega_values ) ) ]

		print( 'curiosity = ' + str( c**2 / select_omega ) )

		density_middle = [ int( 0.5 * dim ) for dim in self.design_density.shape ]

		mask = np.zeros( self.design_density.shape )

		self.design_density[ density_middle[ 0 ], density_middle[ 1 ] ] = 0.5
		rel_permittivity_delta = 1e-2

		offset = 1

		# self.design_density[ density_middle[ 0 ], density_middle[ 1 ] + offset ] = 1
		# self.design_density[ density_middle[ 0 ], density_middle[ 1 ] - offset ] = 0.5

		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] ] = 0.5
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] ] = 0.5
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] + offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] - offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] - offset ] = 0.5

		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] + offset ] = 0.5

		# offset = 2
		# self.design_density[ density_middle[ 0 ], density_middle[ 1 ] + offset ] = 0.5
		# self.design_density[ density_middle[ 0 ], density_middle[ 1 ] - offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] ] = 1.0
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] ] = 0.5
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] + offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] - offset ] = 1.0
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] - offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] + offset ] = 0.5

		# offset = 3
		# self.design_density[ density_middle[ 0 ], density_middle[ 1 ] + offset ] = 0.5
		# self.design_density[ density_middle[ 0 ], density_middle[ 1 ] - offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] ] = 1.0
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] ] = 0.5
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] + offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] - offset ] = 1.0
		# self.design_density[ density_middle[ 0 ] + offset, density_middle[ 1 ] - offset ] = 0.5
		# self.design_density[ density_middle[ 0 ] - offset, density_middle[ 1 ] + offset ] = 0.5



		mask[ density_middle[ 0 ], density_middle[ 1 ] ] = 1
		# mask_sigma = 0.1
		# mask = gaussian_filter( mask, sigma=mask_sigma )
		print('mask!')
		print( mask[ density_middle[ 0 ], density_middle[ 1 ] ] )

		import_density = upsample( self.design_density, self.coarsen_factor )
		field_mask = upsample( mask, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		fwd_Ez = self.compute_forward_fields( select_omega, device_permittivity )
		fwd_Ez = fwd_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

		device_permittivity_2 = device_permittivity.copy()
		device_permittivity_2[
			density_middle[ 0 ] * self.coarsen_factor : ( density_middle[ 0 ] * self.coarsen_factor + self.coarsen_factor ),
			density_middle[ 1 ] * self.coarsen_factor : ( density_middle[ 1 ] * self.coarsen_factor + self.coarsen_factor )
		] += rel_permittivity_delta

		fwd_Ez_2 = self.compute_forward_fields( select_omega, device_permittivity_2 )
		fwd_Ez_2 = fwd_Ez_2[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]


		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity
		delta_src = np.zeros( self.rel_eps_simulation.shape, dtype=np.complex )

		size_region = 3
		half_size_region = 1

		small_simulation_width_voxels = size_region * self.coarsen_factor + 2 * self.pml_voxels
		small_simulation_height_voxels = size_region * self.coarsen_factor + 2 * self.pml_voxels

		small_sim_start_width = self.pml_voxels
		small_sim_start_height = self.pml_voxels
		small_sim_end_width = small_sim_start_width + size_region * self.coarsen_factor
		small_sim_end_height = small_sim_start_height + size_region * self.coarsen_factor

		extract_width_start = density_middle[ 0 ] * self.coarsen_factor - half_size_region * self.coarsen_factor
		extract_width_end = extract_width_start + size_region * self.coarsen_factor
		extract_height_start = density_middle[ 1 ] * self.coarsen_factor - half_size_region * self.coarsen_factor
		extract_height_end = extract_height_start + size_region * self.coarsen_factor

		extract_device_permittivity = device_permittivity[ extract_width_start : extract_width_end, extract_height_start : extract_height_end ]

		small_rel_eps_sim = np.zeros( ( small_simulation_width_voxels, small_simulation_height_voxels ) )
		small_rel_eps_sim[ small_sim_start_width : small_sim_end_width, small_sim_start_height : small_sim_end_height ] = extract_device_permittivity

		masked_ez = field_mask * fwd_Ez
		extract_fwd_Ez = masked_ez[
			extract_width_start : extract_width_end,
			extract_height_start : extract_height_end ]

		make_small_src = np.zeros( small_rel_eps_sim.shape, dtype=np.complex )
		make_small_src[
			self.pml_voxels : ( self.pml_voxels + extract_fwd_Ez.shape[ 0 ] ),
			self.pml_voxels : ( self.pml_voxels + extract_fwd_Ez.shape[ 1 ] ) ] = (
				# select_omega**2 * rel_permittivity_delta * fwd_Ez[ extract_width_start : extract_width_end, extract_height_start : extract_height_end ] * eps_nought / ( c**2 ) )

				( eps_nought / 1j ) * select_omega * rel_permittivity_delta * extract_fwd_Ez )


				# ( 1j * select_omega / c**2 ) * extract_device_permittivity * extract_fwd_Ez

		delta_src[
			self.device_width_start + density_middle[ 0 ] * self.coarsen_factor : self.device_width_start + density_middle[ 0 ] * self.coarsen_factor + self.coarsen_factor,
			self.device_height_start + density_middle[ 1 ] * self.coarsen_factor : self.device_height_start + density_middle[ 1 ] * self.coarsen_factor + self.coarsen_factor ] = (

			( eps_nought / 1j ) * select_omega * rel_permittivity_delta *
			fwd_Ez[ density_middle[ 0 ] * self.coarsen_factor : density_middle[ 0 ] * self.coarsen_factor + self.coarsen_factor,
		 	density_middle[ 1 ] * self.coarsen_factor : + density_middle[ 1 ] * self.coarsen_factor + self.coarsen_factor ]

		)

		filter_delta_src = delta_src.copy()
		max_abs = np.max( np.abs( filter_delta_src ) )
		num = 0
		for row in range( 0, filter_delta_src.shape[ 0 ] ):
			for col in range( 0, filter_delta_src.shape[ 1 ] ):
				if np.abs( filter_delta_src[ row, col ] ) > 0.75 * max_abs:
					num += 1
				else:
					filter_delta_src[ row, col ] = 0

		print( 'num = ' + str( num ) )

		# plt.imshow( np.real( delta_src ) )
		# plt.show()


		use_test_perm = False
		use_filtered_source = False

		if use_test_perm:

			# test_density = np.zeros( self.design_density.shape )
			# test_density[ density_middle[ 0 ], density_middle[ 1 ] ] = 1
			# blank_perm = self.density_to_permittivity( upsample( test_density, self.coarsen_factor ) )
			# test_rel_eps_permittivity = self.rel_eps_simulation.copy()
			# test_rel_eps_permittivity[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = blank_perm

			perm_mask = np.zeros( self.rel_eps_simulation.shape )
			perm_mask[
				self.device_width_start + extract_width_start : self.device_width_start + extract_width_end,
				self.device_height_start + extract_height_start : self.device_height_start + extract_height_end ] = 1

			test_perm = self.rel_eps_simulation.copy() * perm_mask + ( 1 - perm_mask )

			plt.subplot( 1, 2, 1 )
			plt.imshow( test_perm[ self.pml_voxels : test_perm.shape[ 0 ] - self.pml_voxels, self.pml_voxels : test_perm.shape[ 1 ] - self.pml_voxels ] )
			plt.colorbar()
			plt.subplot( 1, 2, 2 )
			plt.imshow( self.rel_eps_simulation[ self.pml_voxels : test_perm.shape[ 0 ] - self.pml_voxels, self.pml_voxels : test_perm.shape[ 1 ] - self.pml_voxels ] )
			plt.colorbar()
			plt.show()
			# test_perm[
			# 	self.device_width_start + extract_width_start : self.device_width_start + extract_width_end,
			# 	self.device_height_start + extract_height_start : self.device_height_start + extract_height_end ]

			test_sim = ceviche.fdfd_ez( select_omega, self.mesh_size_m, test_perm, [ self.pml_voxels, self.pml_voxels ] )
			if use_filtered_source:
				test_fwd_Hx, test_fwd_Hy, test_fwd_Ez = test_sim.solve( filter_delta_src )
			else:
				test_fwd_Hx, test_fwd_Hy, test_fwd_Ez = test_sim.solve( delta_src )

			test_fwd_Ez = test_fwd_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]


			# test_sim = ceviche.fdfd_ez( select_omega, self.mesh_size_m, test_rel_eps_permittivity, [ self.pml_voxels, self.pml_voxels ] )
			# test_fwd_Hx, test_fwd_Hy, test_fwd_Ez = test_sim.solve( delta_src )
			# test_fwd_Ez = test_fwd_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
		else:
			test_sim = ceviche.fdfd_ez( select_omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )

			if use_filtered_source:
				test_fwd_Hx, test_fwd_Hy, test_fwd_Ez = test_sim.solve( filter_delta_src )
			else:
				test_fwd_Hx, test_fwd_Hy, test_fwd_Ez = test_sim.solve( delta_src )

			test_fwd_Ez = test_fwd_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

			plt.subplot( 1, 3, 1 )
			plt.imshow( np.real( delta_src[ self.device_width_start + density_middle[ 0 ] * self.coarsen_factor : self.device_width_start + density_middle[ 0 ] * self.coarsen_factor + self.coarsen_factor,
			self.device_height_start + density_middle[ 1 ] * self.coarsen_factor : self.device_height_start + density_middle[ 1 ] * self.coarsen_factor + self.coarsen_factor ] ) )
			plt.subplot( 1, 3, 2 )
			plt.imshow( np.real( filter_delta_src[ self.device_width_start + density_middle[ 0 ] * self.coarsen_factor : self.device_width_start + density_middle[ 0 ] * self.coarsen_factor + self.coarsen_factor,
			self.device_height_start + density_middle[ 1 ] * self.coarsen_factor : self.device_height_start + density_middle[ 1 ] * self.coarsen_factor + self.coarsen_factor ] ) )
			plt.subplot( 1, 3, 3 )
			plt.imshow( np.log10( np.abs( test_fwd_Ez ) ), cmap='Reds' )
			plt.colorbar()
			plt.show()

		# plt.subplot( 1, 2, 1 )
		# plt.imshow( np.real( test_fwd_Ez[
		# 	extract_width_start : extract_width_end,
		# 	extract_height_start : extract_height_end,
		# 	] ) )
		# plt.colorbar()
		# plt.subplot( 1, 2, 2 )
		# plt.imshow( np.real( fwd_Ez[
		# 	extract_width_start : extract_width_end,
		# 	extract_height_start : extract_height_end,
		# 	] ) )
		# plt.colorbar()
		# plt.show()

		print( small_rel_eps_sim.shape )
		print( make_small_src.shape )

		small_simulation = ceviche.fdfd_ez( select_omega, self.mesh_size_m, small_rel_eps_sim, [ self.pml_voxels, self.pml_voxels ] )
		small_fwd_Hx, small_fwd_Hy, small_fwd_Ez = small_simulation.solve( make_small_src )

		small_fwd_Ez = small_fwd_Ez[ small_sim_start_width : small_sim_end_width, small_sim_start_height : small_sim_end_height ]
		final_permittivity_extract = extract_device_permittivity[ half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor, half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor ]

		# diff_ez = fwd_Ez_2 - fwd_Ez
		# small_fwd_Ez = diff_ez

		extract_small_fwd_Ez = small_fwd_Ez[ half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor, half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor ]
		extract_orig_fwd_Ez = extract_fwd_Ez[ half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor, half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor ]

		scaling = 1.0#24.5#( self.design_width_voxels / size_region ) * ( self.design_height_voxels / size_region )

		print('scaling = ' + str( scaling ) )

		# predicted_ez2 = fwd_Ez + small_fwd_Ez
		plt_fwd_ez2 = fwd_Ez_2[ extract_width_start : extract_width_end, extract_height_start : extract_height_end ]
		plt_fwd_ez = fwd_Ez[ extract_width_start : extract_width_end, extract_height_start : extract_height_end ]
		test_fwd_ez_extract = test_fwd_Ez[ extract_width_start : extract_width_end, extract_height_start : extract_height_end ]

		plt.subplot( 1, 3, 1 )
		plt.imshow( np.real( plt_fwd_ez2 - plt_fwd_ez ) )
		plt.colorbar()
		plt.subplot( 1, 3, 2 )
		# plt.imshow( np.real( scaling * small_fwd_Ez ) )
		# plt.imshow( np.real( plt_fwd_ez2 - plt_fwd_ez ) / np.real( 1e-6 + small_fwd_Ez ) )
		plt.imshow( np.maximum( np.minimum( np.real( plt_fwd_ez2 - plt_fwd_ez ) / np.real( test_fwd_ez_extract ), 50 ), -50 ) )
		plt.colorbar()
		plt.subplot( 1, 3, 3 )
		plt.imshow( np.real( test_fwd_ez_extract ) )
		plt.colorbar()
		plt.show()

		small_fwd_Ez = scaling * test_fwd_ez_extract
		extract_small_fwd_Ez = small_fwd_Ez[ half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor, half_size_region * self.coarsen_factor : ( 1 + half_size_region ) * self.coarsen_factor ]

		total_p_ind = final_permittivity_extract * extract_small_fwd_Ez + rel_permittivity_delta * extract_orig_fwd_Ez
		# total_p_ind = scaling * final_permittivity_extract * extract_small_fwd_Ez + rel_permittivity_delta * extract_orig_fwd_Ez
		# total_p_ind = rel_permittivity_delta * extract_orig_fwd_Ez
		# total_p_ind = final_permittivity_extract * extract_small_fwd_Ez

		plt.subplot( 1, 2, 1 )
		plt.imshow( np.real( scaling * final_permittivity_extract * extract_small_fwd_Ez ) )
		plt.colorbar()
		plt.subplot( 1, 2, 2 )
		plt.imshow( np.real( rel_permittivity_delta * extract_orig_fwd_Ez ) )
		plt.colorbar()
		plt.show()

		# plt.imshow( device_permittivity )
		# plt.show()

		# plt.imshow( np.real( fwd_Ez ) )
		# plt.show()

		p_ind = device_permittivity * fwd_Ez * field_mask

		permittivity_range = self.max_relative_permittivity - self.min_relative_permittivity


		self.design_density[ density_middle[ 0 ], density_middle[ 1 ] ] += ( rel_permittivity_delta / permittivity_range )
		import_density_prime = upsample( self.design_density, self.coarsen_factor )
		device_permittivity_prime = self.density_to_permittivity( import_density_prime )

		fwd_Ez_prime = self.compute_forward_fields( select_omega, device_permittivity_prime )
		fwd_Ez_prime = fwd_Ez_prime[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
		p_ind_prime = device_permittivity_prime * fwd_Ez_prime * field_mask

		expected_p_ind_prime = fwd_Ez * field_mask

		delta_p = p_ind_prime - p_ind

		extract_delta_p = np.zeros( ( self.coarsen_factor, self.coarsen_factor ), dtype=delta_p.dtype )
		extract_expected_delta_p = np.zeros( ( self.coarsen_factor, self.coarsen_factor ), dtype=delta_p.dtype )

		for row in range( density_middle[ 0 ] * self.coarsen_factor, density_middle[ 0 ] * self.coarsen_factor + self.coarsen_factor ):
			for col in range( density_middle[ 1 ] * self.coarsen_factor, density_middle[ 1 ] * self.coarsen_factor + self.coarsen_factor ):
				extract_delta_p[ row - density_middle[ 0 ] * self.coarsen_factor, col - density_middle[ 1 ] * self.coarsen_factor ] = (
					delta_p[ row, col ] )

				extract_expected_delta_p[ row - density_middle[ 0 ] * self.coarsen_factor, col - density_middle[ 1 ] * self.coarsen_factor ] = (
					expected_p_ind_prime[ row, col ] )


		plt.subplot( 1, 2, 1 )
		plt.imshow( np.real( extract_expected_delta_p ) )
		plt.subplot( 1, 2, 2 )
		plt.imshow( np.real( extract_orig_fwd_Ez ) )
		plt.show()


		plt.subplot( 2, 3, 1 )
		plt.imshow( np.real( extract_delta_p ) )
		plt.colorbar()
		plt.subplot( 2, 3, 2 )
		plt.imshow( np.real( extract_expected_delta_p ) )
		plt.colorbar()
		plt.subplot( 2, 3, 3 )
		plt.imshow( np.real( total_p_ind ) )
		plt.colorbar()


		plt.subplot( 2, 3, 5 )
		print( np.max( np.abs( np.real( extract_delta_p ))))
		print( np.max( np.abs( np.real( extract_expected_delta_p ))))
		print( np.max( np.abs( np.real( total_p_ind ))))

		print( np.max( np.abs( np.imag( extract_delta_p ))))
		print( np.max( np.abs( np.imag( extract_expected_delta_p ))))
		print( np.max( np.abs( np.imag( total_p_ind ))))


		real_delta_p = np.real( extract_delta_p[ :, int( 0.5 * self.coarsen_factor ) ] )
		real_expected_delta_p = np.real( extract_expected_delta_p[ :, int( 0.5 * self.coarsen_factor ) ] )
		real_delta_p_prime = np.real( total_p_ind[ :, int( 0.5 * self.coarsen_factor ) ] )

		imag_delta_p = np.imag( extract_delta_p[ :, int( 0.5 * self.coarsen_factor ) ] )
		imag_expected_delta_p = np.imag( extract_expected_delta_p[ :, int( 0.5 * self.coarsen_factor ) ] )
		imag_delta_p_prime = np.imag( total_p_ind[ :, int( 0.5 * self.coarsen_factor ) ] )

		plt.plot( real_delta_p / np.max( np.abs( real_delta_p ) ), linewidth=2, color='r' )
		plt.plot( real_expected_delta_p / np.max( np.abs( real_expected_delta_p ) ), linewidth=2, color='b' )
		plt.plot( real_delta_p_prime / np.max( np.abs( real_delta_p_prime ) ), linewidth=2, color='k', linestyle='--' )

		plt.plot( 2.5 + imag_delta_p / np.max( np.abs( imag_delta_p ) ), linewidth=2, color='r' )
		plt.plot( 2.5 + imag_expected_delta_p / np.max( np.abs( imag_expected_delta_p ) ), linewidth=2, color='b' )
		plt.plot( 2.5 + imag_delta_p_prime / np.max( np.abs( imag_delta_p_prime ) ), linewidth=2, color='k', linestyle='--' )


		plt.subplot( 2, 3, 6 )
		real_delta_p = np.real( extract_delta_p[ int( 0.5 * self.coarsen_factor ), : ] )
		real_expected_delta_p = np.real( extract_expected_delta_p[ int( 0.5 * self.coarsen_factor ), : ] )
		real_delta_p_prime = np.real( total_p_ind[ int( 0.5 * self.coarsen_factor ), : ] )

		imag_delta_p = np.imag( extract_delta_p[ int( 0.5 * self.coarsen_factor ), : ] )
		imag_expected_delta_p = np.imag( extract_expected_delta_p[ int( 0.5 * self.coarsen_factor ), : ] )
		imag_delta_p_prime = np.imag( total_p_ind[ int( 0.5 * self.coarsen_factor ), : ] )


		plt.plot( real_delta_p / np.max( np.abs( real_delta_p ) ), linewidth=2, color='m' )
		plt.plot( real_expected_delta_p / np.max( np.abs( real_expected_delta_p ) ), linewidth=2, color='c' )
		plt.plot( real_delta_p_prime / np.max( np.abs( real_delta_p_prime ) ), linewidth=2, color='k', linestyle='--' )

		plt.plot( 2.5 + imag_delta_p / np.max( np.abs( imag_delta_p ) ), linewidth=2, color='m' )
		plt.plot( 2.5 + imag_expected_delta_p / np.max( np.abs( imag_expected_delta_p ) ), linewidth=2, color='c' )
		plt.plot( 2.5 + imag_delta_p_prime / np.max( np.abs( imag_delta_p_prime ) ), linewidth=2, color='k', linestyle='--' )


		plt.show()




	def verify_adjoint_against_finite_difference( self ):
		fd_x = int( 0.5 * self.device_width_voxels )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd = np.zeros( len( fd_y ) )
		omega_idx = int( 0.5 * len( self.omega_values ) )
		fd_omega = self.omega_values[ fomega_idx ]

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

	def step_binarize( self, gradient, binarize_amount, binarize_max_movement, opt_mask ):

		density_for_binarizing = self.design_density.flatten()
		flatten_gradient = gradient.flatten()

		# flatten_design_cuts = density_for_binarizing.copy()
		extract_binarization_gradient_full = compute_binarization_gradient( density_for_binarizing, self.binarization_set_point )
		# flatten_fom_gradients = flatten_gradient.copy()
		flatten_opt_mask = opt_mask.flatten()


		flatten_design_cuts = []
		flatten_fom_gradients = []
		extract_binarization_gradient = []

		for idx in range( 0, len( flatten_opt_mask ) ):
			if flatten_opt_mask[ idx ] > 0:
				flatten_design_cuts.append( density_for_binarizing[ idx ] )
				flatten_fom_gradients.append( flatten_gradient[ idx ] )
				extract_binarization_gradient.append( extract_binarization_gradient_full[ idx ] )

		flatten_design_cuts = np.array( flatten_design_cuts )
		flatten_fom_gradients = np.array( flatten_fom_gradients )
		extract_binarization_gradient = np.array( extract_binarization_gradient )

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
		# if initial_binarization < 0.1:
		# 	alpha = binarize_amount
		# else:
			# alpha = np.minimum( initial_binarization * max_possible_binarization_change, binarize_amount )
		alpha = np.minimum( max_possible_binarization_change, binarize_amount )

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

		refill_idx = 0
		refill_design_variable = density_for_binarizing.copy()
		for idx in range( 0, len( flatten_opt_mask ) ):
			if flatten_opt_mask[ idx ] > 0:
				refill_design_variable[ idx ] = proposed_design_variable[ refill_idx ]
				refill_idx += 1

		return np.reshape( refill_design_variable, self.design_density.shape )

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
		folder_for_saving,
		random_globals=False, random_global_iteration_frequency=10, random_global_scan_points=10, bounds_cutoff=0.9,
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
			adversary_scan_density = 8

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
		self.fom_evolution_no_loss = np.zeros( num_iterations )
		self.binarization_evolution = np.zeros( num_iterations )
		self.fom_by_wl_evolution = np.zeros( ( num_iterations, self.num_wavelengths ) )
		self.gradient_directions = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		self.dense_plot_idxs = []
		self.dense_plots = []

		function_for_fom_and_gradient = self.compute_fom_and_gradient

		for iter_idx in range( 0, num_iterations ):
			if ( iter_idx % 10 ) == 0:
				log_file = open( self.save_folder + "/log.txt", 'a' )
				log_file.write( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) + "\n")
				log_file.close()


			# mask_density = opt_mask * self.design_density
			import_density = upsample( self.design_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )


			if random_globals and ( ( iter_idx % random_global_iteration_frequency ) == 0 ):
				random_direction = opt_mask * np.random.normal( 0, 1, self.design_density.shape )

				for row in range( 0, random_direction.shape[ 0 ] ):
					for col in range( 0, random_direction.shape[ 1 ] ):
						if ( mask_density[ row, col ] > bounds_cutoff ) or ( mask_density[ row, col ] < ( 1 - bounds_cutoff ) ):
							random_direction[ row, col ] = 0

				if np.sqrt( np.sum( random_direction**2 ) ) > small:

					random_direction /= np.sqrt( np.sum( random_direction**2 ) )

					alpha_0 = np.sum( random_direction * mask_density )
					rho_0 = mask_density - alpha_0 * random_direction
					lower_alpha_bound = -np.inf
					upper_alpha_bound = np.inf

					flatten_rho_0 = rho_0.flatten()
					flatten_random_direction = random_direction.flatten()

					critical_low_alpha = ( 0.0 - flatten_rho_0 ) / ( flatten_random_direction + small )
					critical_high_alpha = ( 1.0 - flatten_rho_0 ) / ( flatten_random_direction + small )

					for idx in range( 0, len( flatten_rho_0 ) ):
						if ( flatten_random_direction[ idx ] > 0 ):
							upper_alpha_bound = np.minimum( upper_alpha_bound, critical_high_alpha[ idx ] )
						elif ( flatten_random_direction[ idx ] < 0  ):
							lower_alpha_bound = np.maximum( lower_alpha_bound, critical_low_alpha[ idx ] )

					alpha_sweep = np.linspace( lower_alpha_bound, upper_alpha_bound, random_global_scan_points )

					def sweep_fom( test_rho ):
						import_density = upsample( self.design_density, self.coarsen_factor )
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
					alpha_to_beat = alpha_0

					for alpha_idx in range( 0, random_global_scan_points ):
						sweep_density = alpha_sweep[ alpha_idx ] * random_direction + rho_0
						sweep_density = np.maximum( 0.0, np.minimum( sweep_density, 1.0 ) )

						cur_fom = sweep_fom( sweep_density )
						if cur_fom > fom_to_beat:
							fom_to_beat = cur_fom
							alpha_to_beat = alpha_sweep[ alpha_idx ]

					log_file = open( self.save_folder + "/log.txt", 'a' )
					log_file.write( "Initial alpha: " + str( alpha_0 ) + "\n" )
					log_file.write( "Final alpha: " + str( alpha_to_beat ) + "\n\n" )
					log_file.close()

					self.design_density = alpha_to_beat * random_direction + rho_0
					self.design_density = np.maximum( 0.0, np.minimum( self.design_density, 1.0 ) )
					import_density = upsample( self.design_density, self.coarsen_factor )
					device_permittivity = self.density_to_permittivity( import_density )


			gradient_by_wl = []
			fom_by_wl = []
			fom_no_loss_by_wl = []
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

					get_fom_, get_grad = function_for_fom_and_gradient(
						opt_omega_value, device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
						wl_intensity_scaling )
				else:
					get_fom, get_grad = function_for_fom_and_gradient(
						self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
						self.wavelength_intensity_scaling[ wl_idx ] )

					if np.max( np.abs( np.imag( device_permittivity ) ) ) > 0:
						get_fom_no_loss = self.compute_fom(
							self.omega_values[ wl_idx ], np.real( device_permittivity ), self.focal_spots_x_voxels[ get_focal_point_idx ],
							self.wavelength_intensity_scaling[ wl_idx ] )
					else:
						get_fom_no_loss = get_fom

				scale_fom_for_wl = get_fom

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
				scale_gradient_for_wl = upsampled_device_grad

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )
				fom_no_loss_by_wl.append( get_fom_no_loss )

			net_fom = np.product( fom_by_wl )
			net_fom_no_loss = np.product( fom_no_loss_by_wl )

			if use_log_fom:
				net_fom = np.log( net_fom )


			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

			# We are currently not doing a performance based weighting here, but we can add it in
			for wl_idx in range( 0, self.num_wavelengths ):
				wl_gradient = np.real( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				if use_log_fom:
					weighting = 1. / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			#
			# Otherwise we are already in design space! Sloppy here, but want to try it out
			#
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
			self.fom_evolution_no_loss[ iter_idx ] = net_fom_no_loss
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

			# print('pre binarize')

			if binarize:
				min_binarize_step = 0.1 * binarize_max_movement_per_voxel
				max_binarize_step = 1.2 * binarize_max_movement_per_voxel
				num_steps = 30

				binarization_steps = np.linspace( min_binarize_step, max_binarize_step, num_steps )

				pre_binarization = compute_binarization( self.design_density.flatten() )

				best_choice = None
				best_fom_increase = -np.inf

				for step_idx in range( 0, num_steps ):
					scan_movement_per_voxel = binarization_steps[ step_idx ]
					proposed_step = self.step_binarize( -norm_scaled_gradient, binarize_movement_per_step, scan_movement_per_voxel, opt_mask )
					proposed_step = opt_mask * proposed_step + ( 1 - opt_mask ) * self.design_density

					achieved_binarization = compute_binarization( proposed_step.flatten() ) - pre_binarization

					if achieved_binarization >= 0.9 * binarize_movement_per_step:
						fom_increase = np.sum( ( proposed_step - self.design_density ) * norm_scaled_gradient )

						if fom_increase > best_fom_increase:
							best_fom_increase = fom_increase
							best_choice = proposed_step.copy()

				if best_choice is None:
					proposed_step = self.step_binarize( -norm_scaled_gradient, binarize_movement_per_step, binarize_max_movement_per_voxel, opt_mask )
					proposed_step = opt_mask * proposed_step + ( 1 - opt_mask ) * self.design_density
					best_choice = proposed_step.copy()

				# proposed_step = self.step_binarize( -norm_scaled_gradient, binarize_movement_per_step, binarize_max_movement_per_voxel, opt_mask )
				# self.design_density = opt_mask * proposed_step + ( 1 - opt_mask ) * self.design_density
				self.design_density = best_choice
			else:
				self.design_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
				self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

			if self.do_density_pairings:
				self.design_density = self.pair_array( self.design_density )

			np.save( folder_for_saving + "_fom_evolution.npy", self.fom_evolution )
			np.save( folder_for_saving + "_binarization_evolution.npy", self.binarization_evolution )
			np.save( folder_for_saving + "_fom_by_wl_evolution.npy", self.fom_by_wl_evolution )

	def optimize_sciopt( self, folder_for_saving, max_iterations, second_order=False ):

		heaviside_bandwidth = 0.5
		make_heaviside = heaviside.Heaviside( heaviside_bandwidth )
		init_guess = 0.5 * np.ones( self.design_width_voxels * self.design_height_voxels )

		apply_init_heaviside = make_heaviside.forward( init_guess - 0.5 )
		init_fom_abs = np.abs( self.compute_net_fom_from_density( np.reshape( apply_init_heaviside, self.design_density.shape ) ) )

		np.save( folder_for_saving + "_init_density_heaviside.npy", np.reshape( apply_init_heaviside, self.design_density.shape ) )


		def min_func( x_density ):
			apply_heaviside = make_heaviside.forward( x_density - 0.5 )

			shape_density = np.reshape( apply_heaviside, self.design_density.shape )
			return ( -self.compute_net_fom_from_density( shape_density ) / init_fom_abs )


		def jac_func( x_density ):
			apply_heaviside = make_heaviside.forward( x_density - 0.5 )

			shape_density = np.reshape( apply_heaviside, self.design_density.shape )
			upsample_density = upsample( shape_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( upsample_density )

			gradient_by_wl = []
			fom_by_wl = []

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				get_fom, get_grad = self.compute_fom_and_gradient(
					self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
					self.wavelength_intensity_scaling[ wl_idx ] )

				scale_fom_for_wl = get_fom

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
				scale_gradient_for_wl = upsampled_device_grad
				scale_gradient_for_wl = reinterpolate_average( scale_gradient_for_wl, self.coarsen_factor )
				scale_gradient_for_wl = make_heaviside.chain_rule( scale_gradient_for_wl.flatten(), apply_heaviside, x_density - 0.5 )

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )

			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )
			net_fom = np.product( fom_by_wl )

			# We are currently not doing a performance based weighting here, but we can add it in
			for wl_idx in range( 0, self.num_wavelengths ):
				# wl_gradient = np.real( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				wl_gradient = gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			return ( -net_gradient ) / init_fom_abs


		def hess_func( x_density ):
			hessian = np.zeros( ( len( x_density ), len( x_density ) ) )

			fom_by_wl_for_weighting = np.zeros( self.num_wavelengths )

			shape_density = np.reshape( x_density, self.design_density.shape )
			upsample_density = upsample( shape_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( upsample_density )

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				fom_by_wl_for_weighting[ wl_idx ] = self.compute_fom(
					self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
					self.wavelength_intensity_scaling[ wl_idx ] )


			h = 1e-4
			for hess_idx in range( 0, len( x_density ) ):
				copy_density = x_density.copy()
				copy_density[ hess_idx ] += h

				jac_up = jac_func_with_weights( copy_density, fom_by_wl_for_weighting )

				copy_density = x_density.copy()
				copy_density[ hess_idx ] -= h

				jac_down = jac_func_with_weights( copy_density, fom_by_wl_for_weighting )

				hessian[ hess_idx, : ] = ( jac_up - jac_down ) / ( 2 * h )

			return hessian


		def optimize_callback( xk ):
			global iter_idx
			global fom_evolution
			global binarization_evolution

			fom_evolution.append( min_func( xk ) )
			binarization_evolution.append( compute_binarization( xk ) )

			np.save( folder_for_saving + "_init_fom_abs.npy", init_fom_abs )
			np.save( folder_for_saving + "_fom_evolution.npy", fom_evolution )
			np.save( folder_for_saving + "_binarization_evolution.npy", binarization_evolution )

			iter_idx += 1
			if iter_idx >= max_iterations:
				return True
			else:
				return False

		solution = None
		if second_order:
			solution = scipy.optimize.minimize(
				min_func,
				init_guess,
				method='Newton-CG',
				jac=jac_func,
				hess=hess_func,
				callback=optimize_callback )
		else:
			solution = scipy.optimize.minimize(
				min_func,
				init_guess,
				method='CG',
				jac=jac_func,
				callback=optimize_callback )

		final_density = solution.x
		fom_evolution.append( solution.fun )
		binarization_evolution.append( compute_binarization( solution.x ) )

		np.save( folder_for_saving + "_init_fom_abs.npy", init_fom_abs )
		np.save( folder_for_saving + "_fom_evolution.npy", fom_evolution )
		np.save( folder_for_saving + "_binarization_evolution.npy", binarization_evolution )
		np.save( folder_for_saving + "_optimized_density.npy", np.reshape( solution.x, self.design_density.shape ) )
		np.save( folder_for_saving + "_optimized_density_heaviside.npy", np.reshape( make_heaviside.forward( solution.x - 0.5 ), self.design_density.shape ) )


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
		np.save( file_base + "_fom_evolution_no_loss.npy", self.fom_evolution_no_loss )
		np.save( file_base + "_binarization_evolution.npy", self.binarization_evolution )
		np.save( file_base + "_fom_by_wl_evolution.npy", self.fom_by_wl_evolution )
		np.save( file_base + "_gradient_directions.npy", self.gradient_directions )
		np.save( file_base + "_optimized_density.npy", self.design_density )
		np.save( file_base + "_random_seed.npy", self.random_seed )
		np.save( file_base + "_dense_plots.npy", np.array( self.dense_plots ) )
		np.save( file_base + "_dense_plot_idxs.npy", np.array( self.dense_plot_idxs ) )
		np.save( file_base + "_optimization_wavelengths.npy", self.track_optimization_wavelengths_um )



