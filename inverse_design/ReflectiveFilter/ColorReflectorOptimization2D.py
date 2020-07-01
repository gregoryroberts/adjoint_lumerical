#
# Math
#
import numpy as np
import scipy.optimize

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

#
# System
#
import sys

#
# Electromagnetics
#
run_on_cluster = True

if run_on_cluster:
	sys.path.append( '/central/home/gdrobert/Develompent/ceviche' )
import ceviche

eps_nought = 8.854 * 1e-12
mu_nought = 1.257 * 1e-6
c = 3.0 * 1e8
# z_free_space = 376.73
z_free_space = np.sqrt( mu_nought / eps_nought )

# def compute_binarization( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 2 / np.sqrt( total_shape ) ) * np.sqrt( np.sum( ( input_variable - 0.5 )**2 ) )
# def compute_binarization_gradient( input_variable ):
# 	total_shape = np.product( input_variable.shape )
# 	return ( 4 / np.sqrt( total_shape ) ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )

def compute_binarization( input_variable ):
	total_shape = np.product( input_variable.shape )
	return ( 1. / total_shape ) * np.sum( np.sqrt( ( input_variable - 0.5 )**2 ) )
def compute_binarization_gradient( input_variable ):
	total_shape = np.product( input_variable.shape )
	return ( 1. / total_shape ) * ( input_variable - 0.5 ) / np.sqrt( ( input_variable - 0.5 )**2 )	

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

class ColorReflectorOptimization2D():

	def __init__( self,
		device_size_voxels, coarsen_factor, mesh_size_nm,
		permittivity_bounds,
		wavelengths_um, wavelength_idx_to_is_reflective_by_state, random_seed,
		num_layers, designable_layer_indicators, non_designable_permittivity,
		save_folder ):
		
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

		self.focal_length_y_voxels = 0

		self.permittivity_bounds = permittivity_bounds
		self.min_relative_permittivity = permittivity_bounds[ 0 ]
		self.max_relative_permittivity = permittivity_bounds[ 1 ]

		self.wavelengths_um = wavelengths_um
		self.wavelength_intensity_scaling = self.wavelengths_um**2 / ( eps_nought * np.max( self.wavelengths_um )**2 )

		self.num_wavelengths = len( wavelengths_um )

		self.omega_values = 2 * np.pi * c / ( 1e-6 * wavelengths_um )

		self.wavelength_idx_to_is_reflective_by_state = wavelength_idx_to_is_reflective_by_state

		self.random_seed = random_seed
		np.random.seed( self.random_seed )

		assert( self.design_height_voxels % num_layers ) == 0, "Expected the number of layers to evenly divide the design region"

		self.num_layers = num_layers
		self.design_voxels_per_layer = int( self.design_height_voxels / num_layers )

		assert ( len( designable_layer_indicators ) == self.num_layers ), "The layer designability indicator length does not make sense!"
		assert ( len( non_designable_permittivity ) == len( designable_layer_indicators ) ), "Expected a different length for the non designable permittivity "

		self.designable_layer_indicators = np.array( designable_layer_indicators )
		self.non_designable_permittivity = np.array( non_designable_permittivity )
		self.non_designable_density = ( self.non_designable_permittivity - self.min_relative_permittivity ) / ( self.max_relative_permittivity - self.min_relative_permittivity )

		self.save_folder = save_folder

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

		sapphire_substrate_index = 1.55
		self.sapphire_substrate_permittivity = sapphire_substrate_index**2

		# self.gsst_thickness_um = 0.12
		self.gsst_thickness_um = 0.48
		self.gsst_thickness_voxels = int( self.gsst_thickness_um / self.mesh_size_um )

		gsst_index_values = [ 3.25, 4.5 ]
		self.gsst_permittivity_values = [ val**2 for val in gsst_index_values ]
		self.gsst_state = 0

		self.mgf2_capping_um = 0.12
		self.mgf2_capping_voxels = int( self.mgf2_capping_um / self.mesh_size_um )

		mgf2_capping_index = 1.35
		self.mgf2_capping_permittivity = mgf2_capping_index**2

		self.simulation_width_voxels = self.device_width_voxels + 2 * self.width_gap_voxels + 2 * self.pml_voxels
		self.simulation_height_voxels = (
			self.device_height_voxels + self.focal_length_y_voxels + self.height_gap_voxels_bottom + self.gsst_thickness_voxels + self.mgf2_capping_voxels +
			self.height_gap_voxels_top + 2 * self.pml_voxels )

		self.substrate_start = self.pml_voxels
		self.substrate_end = self.substrate_start + self.height_gap_voxels_bottom

		self.gsst_start = self.substrate_end
		self.gsst_end = self.gsst_start + self.gsst_thickness_voxels

		self.mgf2_start = self.gsst_end
		self.mgf2_end = self.mgf2_start + self.mgf2_capping_voxels

		self.device_width_start = int( 0.5 * ( self.simulation_width_voxels - self.device_width_voxels ) )
		self.device_width_end = self.device_width_start + self.device_width_voxels
		self.device_height_start = int( self.pml_voxels + self.height_gap_voxels_bottom + self.gsst_thickness_voxels + self.mgf2_capping_voxels + self.focal_length_y_voxels )
		self.device_height_end = self.device_height_start + self.device_height_voxels

		self.transmission_x_bounds = [ self.device_width_start, self.device_width_end ]
		self.transmission_x_bounds_normalization = [ int( self.device_width_start - 0.5 * self.width_gap_voxels ), int( self.device_width_end + 0.5 * self.width_gap_voxels ) ]

		self.fwd_src_y = int( self.device_height_end + 0.5 * self.height_gap_voxels_top )
		self.transmission_y = int( self.device_height_end + 0.75 * self.height_gap_voxels_top )
		self.normalize_transmission_y = int( self.device_height_end )

		self.rel_eps_simulation = np.ones( ( self.simulation_width_voxels, self.simulation_height_voxels ) )
		self.rel_eps_simulation[ :, self.substrate_start : self.substrate_end ] = self.sapphire_substrate_permittivity
		self.rel_eps_simulation[ :, self.gsst_start : self.gsst_end ] = self.gsst_permittivity_values[ self.gsst_state ]
		self.rel_eps_simulation[ :, self.mgf2_start : self.mgf2_end ] = self.mgf2_capping_permittivity

		fwd_src_x_range = np.arange( 0, self.simulation_width_voxels )
		fwd_src_y_range = self.fwd_src_y * np.ones( fwd_src_x_range.shape, dtype=int )

		self.fwd_source_ez = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		self.fwd_source_ez[ fwd_src_x_range, fwd_src_y_range ] = 1

		self.fwd_source_hz = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		self.fwd_source_hz[ fwd_src_x_range, fwd_src_y_range ] = z_free_space

		self.air_device = np.ones( ( self.device_width_voxels, self.device_height_voxels ) )

		self.normalize_transmission_ez()
		self.normalize_transmission_hz()

	def set_gsst_state( self, gsst_state ):
		self.gsst_state = gsst_state
		self.rel_eps_simulation[ :, self.gsst_start : self.gsst_end ] = self.gsst_permittivity_values[ self.gsst_state ]

	def compute_forward_fields_hz( self, omega, device_permittivity, normalization_permittivity=False ):
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		choose_permittivity = self.rel_eps_simulation

		if normalization_permittivity:
			choose_permittivity = np.ones( self.rel_eps_simulation.shape )

		simulation = ceviche.fdfd_hz( omega, self.mesh_size_m, choose_permittivity, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Ex, fwd_Ey, fwd_Hz = simulation.solve( self.fwd_source_hz )

		return fwd_Ex, fwd_Ey, fwd_Hz


	def compute_fom_from_fields_hz( self, fwd_Ex, fwd_Ey, fwd_Hz, src_scattered_Ex, src_scattered_Ey, src_scattered_Hz, normalization_mode=False ):
		reflection_hz_product = -np.conj( fwd_Ex ) * fwd_Hz
		fom = np.sum( np.real( reflection_hz_product[ self.transmission_x_bounds_normalization[ 0 ] : self.transmission_x_bounds_normalization[ 1 ], self.normalize_transmission_y ] ) )

		if not normalization_mode:
			scattered_Hz = (
				fwd_Hz[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
				src_scattered_Hz )

			scattered_Ex = (
				fwd_Ex[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
				src_scattered_Ex )

			scattered_Ey = (
				fwd_Ey[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
				src_scattered_Ey )

			reflection_hz_product = -np.conj( scattered_Ex ) * scattered_Hz
			fom = np.sum( np.real( reflection_hz_product ) )

		return fom


	def compute_fom_hz( self, omega_idx, device_permittivity ):
		fwd_Ex, fwd_Ey, fwd_Hz = self.compute_forward_fields_hz( self.omega_values[ omega_idx ], device_permittivity )

		return ( 1. / self.hz_transmission_normalization[ omega_idx ] ) * self.compute_fom_from_fields_hz(
			fwd_Ex, fwd_Ey, fwd_Hz,
			self.src_scattered_Ex_by_omega[ omega_idx ], self.src_scattered_Ey_by_omega[ omega_idx ], self.src_scattered_Hz_by_omega[ omega_idx ] )


	def compute_fom_and_gradient_hz( self, omega_idx, device_permittivity ):
		omega = self.omega_values[ omega_idx ]

		fwd_Ex, fwd_Ey, fwd_Hz = self.compute_forward_fields_hz( omega, device_permittivity )

		fom = ( 1. / self.hz_transmission_normalization[ omega_idx ] ) * self.compute_fom_from_fields_hz(
			fwd_Ex, fwd_Ey, fwd_Hz,
			self.src_scattered_Ex_by_omega[ omega_idx, : ],
			self.src_scattered_Ey_by_omega[ omega_idx, : ],
			self.src_scattered_Hz_by_omega[ omega_idx, : ]
		)

		scattered_Ex = (
			fwd_Ex[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
			self.src_scattered_Ex_by_omega[ omega_idx, : ] )

		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] = -np.conj( scattered_Ex )

		simulation = ceviche.fdfd_hz( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Ex, adj_Ey, adj_Hz = simulation.solve( adj_source )

		gradient = ( 1. / self.hz_transmission_normalization[ omega_idx ] ) * 2 * np.real( omega * eps_nought * ( fwd_Ex * adj_Ex + fwd_Ey * adj_Ey ) / ( 1j ) )

		return fom, gradient


	def compute_fom_and_gradient_ez( self, omega_idx, device_permittivity ):
		omega = self.omega_values[ omega_idx ]

		fwd_Hx, fwd_Hy, fwd_Ez = self.compute_forward_fields_ez( omega, device_permittivity )

		fom = ( 1. / self.ez_transmission_normalization[ omega_idx ] ) * self.compute_fom_from_fields_ez(
			fwd_Hx, fwd_Hy, fwd_Ez,
			self.src_scattered_Hx_by_omega[ omega_idx, : ],
			self.src_scattered_Hy_by_omega[ omega_idx, : ],
			self.src_scattered_Ez_by_omega[ omega_idx, : ]
		)

		scattered_Hx = (
			fwd_Hx[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
			self.src_scattered_Hx_by_omega[ omega_idx, : ] )

		adj_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		adj_source[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] = np.conj( scattered_Hx )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = ( 1. / self.ez_transmission_normalization[ omega_idx ] ) * 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / ( 1j ) )

		return fom, gradient

	def get_source_subtracted_fields_ez( self, omega_idx ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		Hx, Hy, Ez = self.compute_forward_fields_ez( self.omega_values[ omega_idx ], device_permittivity )

		return ( Hx - self.src_scattered_Hx_full_by_omega[ omega_idx ] ), ( Hy - self.src_scattered_Hy_full_by_omega[ omega_idx ] ), ( Ez - self.src_scattered_Ez_full_by_omega[ omega_idx ] )

	def get_current_fields_ez( self, omega_idx ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		return self.compute_forward_fields_ez( self.omega_values[ omega_idx ], device_permittivity )

	def get_current_fom_ez( self, omega_idx ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		return self.compute_fom_ez( omega_idx, device_permittivity )


	def compute_forward_fields_ez( self, omega, device_permittivity, normalization_permittivity=False ):
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		choose_permittivity = self.rel_eps_simulation

		if normalization_permittivity:
			choose_permittivity = np.ones( self.rel_eps_simulation.shape )

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, choose_permittivity, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( self.fwd_source_ez )

		return fwd_Hx, fwd_Hy, fwd_Ez

	def compute_fom_from_fields_ez( self, fwd_Hx, fwd_Hy, fwd_Ez, src_scattered_Hx, src_scattered_Hy, src_scattered_Ez, normalization_mode=False ):
		reflection_ez_product = fwd_Ez * np.conj( fwd_Hx )
		fom = np.sum( np.real( reflection_ez_product[ self.transmission_x_bounds_normalization[ 0 ] : self.transmission_x_bounds_normalization[ 1 ], self.normalize_transmission_y ] ) )

		if not normalization_mode:
			scattered_Ez = (
				fwd_Ez[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
				src_scattered_Ez )

			scattered_Hx = (
				fwd_Hx[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
				src_scattered_Hx )

			scattered_Hy = (
				fwd_Hy[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ] -
				src_scattered_Hy )

			reflection_ez_product = scattered_Ez * np.conj( scattered_Hx )

			fom = np.sum( np.real( reflection_ez_product ) )

		return fom

	def compute_fom_ez( self, omega_idx, device_permittivity ):
		fwd_Hx, fwd_Hy, fwd_Ez = self.compute_forward_fields_ez( self.omega_values[ omega_idx ], device_permittivity )

		return ( 1. / self.ez_transmission_normalization[ omega_idx ] ) * self.compute_fom_from_fields_ez(
			fwd_Hx, fwd_Hy, fwd_Ez,
			self.src_scattered_Hx_by_omega[ omega_idx ], self.src_scattered_Hy_by_omega[ omega_idx ], self.src_scattered_Ez_by_omega[ omega_idx ] )

	def normalize_transmission_ez( self ):
		set_normalization_mode = True
		self.ez_transmission_normalization = np.zeros( self.num_wavelengths )

		scattered_line_length = self.transmission_x_bounds[ 1 ] - self.transmission_x_bounds[ 0 ]

		self.src_scattered_Ez_by_omega = np.zeros( ( self.num_wavelengths, scattered_line_length ), dtype=np.complex )
		self.src_scattered_Hx_by_omega = np.zeros( ( self.num_wavelengths, scattered_line_length ), dtype=np.complex )
		self.src_scattered_Hy_by_omega = np.zeros( ( self.num_wavelengths, scattered_line_length ), dtype=np.complex )

		self.src_scattered_Ez_full_by_omega = np.zeros( ( [ self.num_wavelengths ] + list( self.rel_eps_simulation.shape ) ), dtype=np.complex )
		self.src_scattered_Hx_full_by_omega = np.zeros( ( [ self.num_wavelengths ] + list( self.rel_eps_simulation.shape ) ), dtype=np.complex )
		self.src_scattered_Hy_full_by_omega = np.zeros( ( [ self.num_wavelengths ] + list( self.rel_eps_simulation.shape ) ), dtype=np.complex )


		for wl_idx in range( 0, self.num_wavelengths ):
			fwd_Hx, fwd_Hy, fwd_Ez =  self.compute_forward_fields_ez(
				self.omega_values[ wl_idx ], self.air_device, set_normalization_mode )

			self.src_scattered_Ez_by_omega[ wl_idx, : ] = fwd_Ez[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ]
			self.src_scattered_Hx_by_omega[ wl_idx, : ] = fwd_Hx[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ]
			self.src_scattered_Hy_by_omega[ wl_idx, : ] = fwd_Hy[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ]

			self.src_scattered_Ez_full_by_omega[ wl_idx, : ] = fwd_Ez
			self.src_scattered_Hx_full_by_omega[ wl_idx, : ] = fwd_Hx
			self.src_scattered_Hy_full_by_omega[ wl_idx, : ] = fwd_Hy

			self.ez_transmission_normalization[ wl_idx ] = -self.compute_fom_from_fields_ez( fwd_Hx, fwd_Hy, fwd_Ez, None, None, None, set_normalization_mode )

	def normalize_transmission_hz( self ):
		set_normalization_mode = True
		self.hz_transmission_normalization = np.zeros( self.num_wavelengths )

		scattered_line_length = self.transmission_x_bounds[ 1 ] - self.transmission_x_bounds[ 0 ]

		self.src_scattered_Hz_by_omega = np.zeros( ( self.num_wavelengths, scattered_line_length ), dtype=np.complex )
		self.src_scattered_Ex_by_omega = np.zeros( ( self.num_wavelengths, scattered_line_length ), dtype=np.complex )
		self.src_scattered_Ey_by_omega = np.zeros( ( self.num_wavelengths, scattered_line_length ), dtype=np.complex )

		for wl_idx in range( 0, self.num_wavelengths ):
			fwd_Ex, fwd_Ey, fwd_Hz =  self.compute_forward_fields_hz(
				self.omega_values[ wl_idx ], self.air_device, set_normalization_mode )

			self.src_scattered_Hz_by_omega[ wl_idx, : ] = fwd_Hz[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ]
			self.src_scattered_Ex_by_omega[ wl_idx, : ] = fwd_Ex[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ]
			self.src_scattered_Ey_by_omega[ wl_idx, : ] = fwd_Ey[ self.transmission_x_bounds[ 0 ] : self.transmission_x_bounds[ 1 ], self.transmission_y ]

			self.hz_transmission_normalization[ wl_idx ] = -self.compute_fom_from_fields_hz( fwd_Ex, fwd_Ey, fwd_Hz, None, None, None, set_normalization_mode )
			print( self.hz_transmission_normalization[ wl_idx ] )

	def compute_net_fom( self, compute_ez ):
		fom_by_wl = []

		assert 1==0, "This is not ready!"

		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		for wl_idx in range( 0, self.num_wavelengths ):
			if compute_ez:
				get_fom = self.compute_fom_ez(
					self.omega_values[ wl_idx ], device_permittivity, self.wavelength_intensity_scaling[ wl_idx ] )
			else:
				get_fom = self.compute_fom_hz(
					self.omega_values[ wl_idx ], device_permittivity, self.wavelength_intensity_scaling[ wl_idx ] )

			fom_by_wl.append( get_fom )

		net_fom = np.product( fom_by_wl )

		return net_fom

	def verify_adjoint_against_finite_difference( self ):
		fd_x = int( 0.5 * self.device_width_voxels )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd = np.zeros( len( fd_y ) )
		omega_idx = 0#int( 0.5 * len( self.omega_values ) )
		fd_omega = self.omega_values[ omega_idx ]

		# fd_init_device = 1.5 * np.ones( ( self.device_width_voxels, self.device_height_voxels ) )
		fd_init_density = np.random.random( ( self.design_width_voxels, self.design_height_voxels ) )

		upsample_density = upsample( fd_init_density, self.coarsen_factor )
		fd_init_device = self.density_to_permittivity( upsample_density )


		get_fom, get_grad = self.compute_fom_and_gradient_hz(
			omega_idx, fd_init_device )


		fd_step_eps = 1e-6

		for fd_y_idx in range( 0, len( fd_y ) ):
			print( "Fd y idx = " + str( fd_y_idx ) + " out of " + str( len( fd_y ) - 1 ) )
			fd_device_permittivity = fd_init_device.copy()
			fd_device_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps

			# get_fom_step = self.compute_fom_hz( omega_idx, fd_device_permittivity )
			get_fom_step = self.compute_fom_hz( omega_idx, fd_device_permittivity )
			print( get_fom )
			print( get_fom_step )

			compute_fd[ fd_y_idx ] = ( get_fom_step - get_fom ) / fd_step_eps

			print( compute_fd[ fd_y_idx ] )

		extract_gradient = get_grad[ self.device_width_start : self.device_width_end, : ]

		line_gradient = extract_gradient[ fd_x, self.device_height_start : self.device_height_end ]

		self.average_adjoint_finite_difference_error = np.sqrt( np.mean( np.abs( line_gradient - compute_fd )**2 ) )
		self.average_adjoint_finite_difference_error_normalized = np.sqrt( np.mean( np.abs( ( line_gradient - compute_fd ) / compute_fd )**2 ) )

		print( "Average error = " + str( self.average_adjoint_finite_difference_error ) )
		print( "Average error normalized = " + str( self.average_adjoint_finite_difference_error_normalized ) )

		np.save( self.save_folder + "/fd_grad.npy", compute_fd )
		np.save( self.save_folder + "/adj_grad.npy", line_gradient )

		import matplotlib.pyplot as plt

		plt.plot( line_gradient, color='r', linewidth=2 )
		plt.plot( compute_fd, color='g', linewidth=2, linestyle='--' )
		plt.show()

		plt.plot( line_gradient / np.max( np.abs( line_gradient ) ), color='r', linewidth=2 )
		plt.plot( compute_fd / np.max( np.abs( compute_fd ) ), color='g', linewidth=2, linestyle='--' )
		plt.show()

		print( "The ratio is " + str( np.max( np.abs( line_gradient ) ) / np.max( np.abs( compute_fd ) ) ) )
		print( "The inverse ratio is " + str( np.max( np.abs( compute_fd ) ) / np.max( np.abs( line_gradient ) ) ) )

		plt.plot( line_gradient / compute_fd )
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

	def step_binarize( self, gradient, binarize_amount, binarize_max_movement ):

		density_for_binarizing = self.design_density.flatten()
		flatten_gradient = gradient.flatten()

		flatten_design_cuts = density_for_binarizing.copy()
		extract_binarization_gradient = compute_binarization_gradient( flatten_design_cuts )
		flatten_fom_gradients = flatten_gradient.copy()

		beta = binarize_max_movement
		projected_binarization_increase = 0

		c = flatten_fom_gradients

		initial_binarization = compute_binarization( flatten_design_cuts )

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

		print( initial_binarization )
		print( compute_binarization( proposed_design_variable.flatten() ) )

		return np.reshape( proposed_design_variable, self.design_density.shape )

	def get_device_permittivity( self ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )

		return device_permittivity

	def optimize( self, num_iterations, binarize=False, binarize_movement_per_step=0.01, binarize_max_movement_per_voxel=0.025 ):
		self.max_density_change_per_iteration_start = 0.03#0.05
		self.max_density_change_per_iteration_end = 0.01#0.005

		self.gradient_norm_evolution = np.zeros( num_iterations )
		self.fom_evolution = np.zeros( num_iterations )
		self.binarization_evolution = np.zeros( num_iterations )
		self.fom_by_wl_evolution = np.zeros( ( num_iterations, self.num_wavelengths ) )
		self.gradient_directions = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		for iter_idx in range( 0, num_iterations ):
			if ( iter_idx % 10 ) == 0:
				log_file = open( self.save_folder + "/log.txt", 'a' )
				log_file.write( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) + "\n" )
				log_file.close()

				print( "Iteration " + str( iter_idx ) + " out of " + str( num_iterations - 1 ) )

			import_density = upsample( self.design_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )

			gradient_by_state = []
			fom_by_state = []

			num_gsst_states = len( self.gsst_permittivity_values )
			for gsst_state in range( 0, num_gsst_states ):
				self.set_gsst_state( gsst_state )

				gradient_by_wl = []
				fom_by_wl = []

				for wl_idx in range( 0, self.num_wavelengths ):
					get_fom, get_grad = self.compute_fom_and_gradient_ez(
						wl_idx, device_permittivity )

					if not self.wavelength_idx_to_is_reflective_by_state[ gsst_state, wl_idx ]:
						get_fom = np.maximum( 1 - get_fom, 0 )
						get_grad *= -1

					upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

					scale_fom_for_wl = get_fom
					scale_gradient_for_wl = upsampled_device_grad

					print( get_fom )

					gradient_by_wl.append( scale_gradient_for_wl )
					fom_by_wl.append( scale_fom_for_wl )

				net_fom = np.product( fom_by_wl )
				net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

				# We are currently not doing a performance based weighting here, but we can add it in
				for wl_idx in range( 0, self.num_wavelengths ):
					wl_gradient = ( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
					weighting = net_fom / fom_by_wl[ wl_idx ]

					net_gradient += ( weighting * wl_gradient )

				gradient_by_state.append( net_gradient )
				fom_by_state.append( net_fom )

			fom_by_state = np.array( fom_by_state )
			state_performance_weights = ( 2. / num_gsst_states ) - fom_by_state**2 / np.sum( fom_by_state**2 )
			state_performance_weights = np.maximum( state_performance_weights, 0 )
			state_performance_weights /= np.sum( state_performance_weights )

			print( "fom by state = " + str( fom_by_state ) )

			state_weighted_gradient = np.zeros( net_gradient.shape )
			for gsst_state in range( 0, num_gsst_states ):
				state_weighted_gradient += state_performance_weights[ gsst_state ] * gradient_by_state[ gsst_state ]

			state_weighted_gradient = reinterpolate_average( state_weighted_gradient, self.coarsen_factor )

			#
			# Now, we should zero out non-designable regions and average over designable layers
			#
			state_weighted_gradient = self.layer_spacer_averaging( state_weighted_gradient )

			gradient_norm = vector_norm( state_weighted_gradient )

			self.fom_evolution[ iter_idx ] = net_fom
			self.fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
			self.gradient_norm_evolution[ iter_idx ] = gradient_norm

			norm_scaled_gradient = state_weighted_gradient / gradient_norm

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

	def save_optimization_data( self, file_base ):
		np.save( file_base + "_gradient_norm_evolution.npy", self.gradient_norm_evolution )
		np.save( file_base + "_fom_evolution.npy", self.fom_evolution )
		np.save( file_base + "_binarization_evolution.npy", self.binarization_evolution )
		np.save( file_base + "_fom_by_wl_evolution.npy", self.fom_by_wl_evolution )
		np.save( file_base + "_gradient_directions.npy", self.gradient_directions )
		np.save( file_base + "_optimized_density.npy", self.design_density )
		np.save( file_base + "_random_seed.npy", self.random_seed )

	def load_optimization_data( self, file_base ):
		self.gradient_norm_evolution = np.load( file_base + "_gradient_norm_evolution.npy" )
		self.fom_evolution = np.load( file_base + "_fom_evolution.npy" )
		self.binarization_evolution = np.load( file_base + "_binarization_evolution.npy" )
		self.fom_by_wl_evolution = np.load( file_base + "_fom_by_wl_evolution.npy" )
		self.gradient_directions = np.load( file_base + "_gradient_directions.npy" )
		self.design_density = np.load( file_base + "_optimized_density.npy" )
		self.random_seed = np.load( file_base + "_random_seed.npy" )



