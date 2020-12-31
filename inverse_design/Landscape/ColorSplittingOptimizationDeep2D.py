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
# Learning
#
import torch

import torch.nn as nn
import torch.nn.functional as F

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

sys.path.append( os.path.abspath( python_src_directory + "/../" ) )
import sigmoid



eps_nought = 8.854 * 1e-12
mu_nought = 1.257 * 1e-6 
c = 3.0 * 1e8
small = 1e-10



class PermittivityPredictor(nn.Module):
	def __init__(self, num_lambda, kernel_size):
		super(PermittivityPredictor, self).__init__()

		num_fields = 2 * num_lambda
		num_channels = 1 + num_fields
		self.kernel_size = kernel_size

		self.conv1 = nn.Conv2d( num_channels, num_channels, self.kernel_size, padding=1 )
		self.conv2 = nn.Conv2d( num_channels, 2 * num_channels, self.kernel_size, padding=1 )

		self.conv3 = nn.Conv2d( 2 * num_channels, 2 * num_channels, self.kernel_size, padding=1 )
		self.conv4 = nn.Conv2d( 2 * num_channels, 4 * num_channels, self.kernel_size, padding=1 )

		self.conv5 = nn.Conv2d( 4 * num_channels, 4 * num_channels, self.kernel_size, padding=1 )
		self.conv6 = nn.Conv2d( 4 * num_channels, 2 * num_channels, self.kernel_size, padding=1 )

		self.conv7 = nn.Conv2d( 2 * num_channels, 2 * num_channels, self.kernel_size, padding=1 )
		self.conv8 = nn.Conv2d( 2 * num_channels, num_channels, self.kernel_size, padding=1 )

		self.conv9 = nn.Conv2d( num_channels, 2, self.kernel_size, padding=1 )

		self.pool = nn.MaxPool2d( 2, 2 )
		self.upsample = nn.Upsample( scale_factor=2, mode='bilinear' )

		self.softmax = nn.Softmax( 1 )

	def forward(self, x):
		x = F.relu( self.conv1( x ) )
		x = self.pool( F.relu( self.conv2( x ) ) )

		x = F.relu( self.conv3( x ) )
		x = self.pool( F.relu( self.conv4( x ) ) )

		x = F.relu( self.conv5( x ) )
		x = self.upsample( F.relu( self.conv6( x ) ) )

		x = F.relu( self.conv7( x ) )
		x = self.upsample( F.relu( self.conv8( x ) ) )

		x = self.softmax( self.conv9( x ) )

		return x


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


def reinterpolate_abs_max( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k / factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		start_x = int( factor * x_idx )
		end_x = start_x + factor
		for y_idx in range( 0, output_block_size[ 1 ] ):
			start_y = int( factor * y_idx )
			end_y = start_y + factor

			abs_max = 0.0
			best_x = 0
			best_y = 0

			for sweep_x in range( start_x, end_x ):
				for sweep_y in range( start_y, end_y ):
					get_abs = np.abs( input_block[ sweep_x, sweep_y ] )

					if get_abs > abs_max:
						abs_max = get_abs
						best_x = sweep_x
						best_y = sweep_y

					# abs_max = np.maximum( abs_max, get_abs )

					# average += ( 1. / factor**2 ) * input_block[ sweep_x, sweep_y ]
			
			output_block[ x_idx, y_idx ] = input_block[ best_x, best_y ]

	return output_block

class ColorSplittingOptimizationDeep2D():

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
		plt.imshow( np.flip( np.swapaxes( np.real( self.rel_eps_simulation ), 0, 1 ), axis=0 ), cmap='Greens' )
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
	
	def compute_fom_and_gradient( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		return fom, gradient

	def compute_fom_and_gradient_and_fields( self, omega, device_permittivity, focal_point_x_loc, fom_scaling=1.0 ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = fom_scaling * np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = fom_scaling * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		return fom, gradient, fwd_Ez

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



	def verify_adjoint_against_finite_difference( self ):
		fd_x = int( 0.5 * self.device_width_voxels )
		fd_x = int( self.coarsen_factor * 3 )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd_density = np.zeros( len( fd_y ) )
		compute_fd_real = np.zeros( len( fd_y ) )
		compute_fd_imag = np.zeros( len( fd_y ) )
		omega_idx = int( 0.5 * len( self.omega_values ) )
		fd_omega = self.omega_values[ omega_idx ]

		# fd_init_device = 1.5 * np.ones( ( self.device_width_voxels, self.device_height_voxels ) )
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		fd_init_device = device_permittivity

		focal_point_x = self.focal_spots_x_voxels[ 0 ]

		get_fom, get_grad_real, get_grad_imag = self.compute_fom_and_gradient_real_imag(
			fd_omega, fd_init_device, focal_point_x )

		get_grad_real = get_grad_real[
			self.device_width_start : self.device_width_end,
			self.device_height_start : self.device_height_end ]
		get_grad_imag = get_grad_imag[
			self.device_width_start : self.device_width_end,
			self.device_height_start : self.device_height_end ]

		interpolate_grad_real = reinterpolate_average( get_grad_real, self.coarsen_factor )
		interpolate_grad_imag = reinterpolate_average( get_grad_imag, self.coarsen_factor )

		get_grad_density = (
			np.real( self.max_relative_permittivity - self.min_relative_permittivity ) * interpolate_grad_real +
			np.imag( self.max_relative_permittivity - self.min_relative_permittivity ) * interpolate_grad_imag );

		fd_x_density = int( fd_x / self.coarsen_factor )

		fd_step_eps = 1e-6
		fd_step_rho = 1e-6

		num = 10

		for fd_y_idx in range( 0, num ):
			print( "working on " + str( fd_y_idx ) )
			fd_design = self.design_density.copy()
			fd_design[ fd_x_density, fd_y[ fd_y_idx ] ] += fd_step_rho

			import_fd_density = upsample( fd_design, self.coarsen_factor )
			device_fd_permittivity = self.density_to_permittivity( import_fd_density )

			get_fom_step_density = self.compute_fom( fd_omega, device_fd_permittivity, focal_point_x )


			compute_fd_density[ fd_y_idx ] = ( get_fom_step_density - get_fom ) / fd_step_eps

			fd_device_permittivity = fd_init_device.copy()
			fd_device_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps
			get_fom_step_real = self.compute_fom( fd_omega, fd_device_permittivity, focal_point_x )

			fd_device_permittivity = fd_init_device.copy()
			fd_device_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += 1j * fd_step_eps

			get_fom_step_imag = self.compute_fom( fd_omega, fd_device_permittivity, focal_point_x )

			compute_fd_real[ fd_y_idx ] = ( get_fom_step_real - get_fom ) / fd_step_eps
			compute_fd_imag[ fd_y_idx ] = ( get_fom_step_imag - get_fom ) / fd_step_eps

		import matplotlib.pyplot as plt
		plt.subplot( 1, 3, 1 )
		plt.plot( get_grad_real[ fd_x, 0 : num ], color='g', linewidth=2 )
		plt.plot( compute_fd_real[ 0 : num ], color='r', linewidth=2, linestyle='--' )
		plt.subplot( 1, 3, 2 )
		plt.plot( get_grad_imag[ fd_x, 0 : num ], color='g', linewidth=2 )
		plt.plot( compute_fd_imag[ 0 : num ], color='r', linewidth=2, linestyle='--' )		
		plt.subplot( 1, 3, 3 )
		plt.plot( ( self.coarsen_factor**2 ) * get_grad_density[ fd_x_density, 0 : num ], color='g', linewidth=2 )
		plt.plot( compute_fd_density[ 0 : num ], color='r', linewidth=2, linestyle='--' )		
		plt.show()

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
		folder_for_saving ):

		self.fom_evolution = np.zeros( num_iterations )
		self.density_prediction_evolution = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		make_sigmoid = sigmoid.Sigmoid( 0.5, 2.0 )

		def density_to_fields_fom_and_grad( test_density ):
			import_density = upsample( test_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )

			gradient_by_wl = []
			fom_by_wl = []
			real_fields_by_wl = []
			imag_fields_by_wl = []

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				get_fom, get_grad, get_Ez = self.compute_fom_and_gradient_and_fields(
					self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ],
					self.wavelength_intensity_scaling[ wl_idx ] )

				scale_fom_for_wl = get_fom

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]
				scale_gradient_for_wl = upsampled_device_grad

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )

				real_fields_by_wl.append(
					reinterpolate_average( 
						np.real( get_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] ),
						self.coarsen_factor ) )
				imag_fields_by_wl.append(
					reinterpolate_average( 
						np.imag( get_Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] ),
						self.coarsen_factor ) )

			net_fom = np.product( fom_by_wl )
			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

			# We are currently not doing a performance based weighting here, but we can add it in
			for wl_idx in range( 0, self.num_wavelengths ):
				wl_gradient = np.real( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			net_gradient = reinterpolate_average( net_gradient, self.coarsen_factor )


			#
			# Now, we should zero out non-designable regions and average over designable layers
			#
			net_gradient = self.layer_spacer_averaging( net_gradient )
			gradient_norm = vector_norm( net_gradient )

			# Using a scaled gradient might mess up the comparison between different iterations in terms of gradient
			# magnitude
			norm_scaled_gradient = net_gradient / gradient_norm

			return net_fom, net_gradient, real_fields_by_wl, imag_fields_by_wl


		np.random.seed( 2143123 )
		network_input_np = np.random.random( ( 1, 2 * self.num_wavelengths + 1, self.design_width_voxels, self.design_height_voxels ) ) - 0.5
		network_input_np[ 0, 0 ] += 0.5
		preinput_sigmoid = network_input_np[ 0, 0 ]
		network_input_np[ 0, 0 ] = make_sigmoid.forward( preinput_sigmoid )# 1.0 * np.greater_equal( network_input_np[ 0, 0 ], 0.5 )

		kernel_size = 3
		make_net = PermittivityPredictor( self.num_wavelengths, kernel_size )

		get_density_predictions = make_net.forward( torch.tensor( network_input_np, requires_grad=True ).float() )[ 0, 0 ]

		preinput_sigmoid = get_density_predictions.detach().numpy()
		network_input_np[ 0, 0 ] = make_sigmoid.forward( preinput_sigmoid )# 1.0 * np.greater_equal( network_input_np[ 0, 0 ], 0.5 )

		eval_fom, eval_grad, eval_real_fields, eval_imag_fields = density_to_fields_fom_and_grad( network_input_np[ 0, 0 ] )

		for wl_idx in range( 0, self.num_wavelengths ):
			network_input_np[ 0, 1 + 2 * wl_idx ] = eval_real_fields[ wl_idx ]
			network_input_np[ 0, 1 + 2 * wl_idx + 1 ] = eval_imag_fields[ wl_idx ]



		optimizer = torch.optim.SGD( make_net.parameters(), lr=100.0 )

		for iter_idx in range( 0, num_iterations ):
			if ( iter_idx % 10 ) == 0:
				log_file = open( self.save_folder + "/log.txt", 'a' )
				log_file.write( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) + "\n")
				log_file.close()

			sigmoid_backprop = make_sigmoid.chain_rule( -eval_grad, network_input_np[ 0, 0 ], preinput_sigmoid )

			optimizer.zero_grad()
			loss = ( torch.tensor( sigmoid_backprop ).float() * get_density_predictions ).sum()
			loss.backward()
			optimizer.step()

			print( eval_fom )

			# network_input_np[ 0, 0 ] = np.greater_equal( get_density_predictions.detach().numpy(), 0.5 )
			# preinput_sigmoid = get_density_predictions.detach().numpy()
			# network_input_np[ 0, 0 ] = make_sigmoid.forward( preinput_sigmoid )# 1.0 * np.greater_equal( network_input_np[ 0, 0 ], 0.5 )


			get_density_predictions = make_net.forward( torch.tensor( network_input_np, requires_grad=True ).float() )[ 0, 0 ]

			preinput_sigmoid = get_density_predictions.detach().numpy()
			network_input_np[ 0, 0 ] = make_sigmoid.forward( preinput_sigmoid )# 1.0 * np.greater_equal( network_input_np[ 0, 0 ], 0.5 )

			# binarize_predictions = np.greater_equal( np.squeeze( get_density_predictions.detach().numpy() ), 0.5 )
			eval_fom, eval_grad, eval_real_fields, eval_imag_fields = density_to_fields_fom_and_grad( network_input_np[ 0, 0 ] )


			for wl_idx in range( 0, self.num_wavelengths ):
				network_input_np[ 0, 1 + 2 * wl_idx ] = eval_real_fields[ wl_idx ]
				network_input_np[ 0, 1 + 2 * wl_idx + 1 ] = eval_imag_fields[ wl_idx ]


			self.fom_evolution[ iter_idx ] = eval_fom
			self.density_prediction_evolution[ iter_idx ] = get_density_predictions.detach().numpy()

			np.save( folder_for_saving + "_fom_evolution.npy", self.fom_evolution )
			np.save( folder_for_saving + "_density_prediction_evolution.npy", self.density_prediction_evolution )


	def save_optimization_data( self, file_base ):
		np.save( file_base + "_fom_evolution.npy", self.fom_evolution )
		np.save( file_base + "_density_prediction_evolution.npy", self.density_prediction_evolution )



