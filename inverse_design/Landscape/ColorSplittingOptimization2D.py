#
# Math
#
import numpy as np

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
c = 3.0 * 1e8

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
		permittivity_bounds, focal_spots_x_voxels, focal_length_y_voxels,
		wavelengths_um, wavelength_idx_to_focal_idx, random_seed ):
		
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
		self.focal_spots_x_voxels = focal_spots_x_voxels
		self.focal_length_y_voxels = focal_length_y_voxels
		self.wavelengths_um = wavelengths_um
		self.wavelength_intensity_scaling = np.maximum( self.wavelengths_um )**2 / self.wavelengths_um**2

		self.num_wavelengths = len( wavelengths_um )

		self.omega_values = 2 * np.pi * c / ( 1e-6 * wavelengths_um )

		self.wavelength_idx_to_focal_idx = wavelength_idx_to_focal_idx

		self.random_seed = random_seed
		np.random.seed( self.random_seed )

		self.setup_simulation()

	def init_density_with_random( self, mean_density, sigma_density ):
		num_values = self.design_width_voxels * self.device_height_voxels

		random_array_normal_distribution = np.random.normal(
			loc=mean_density,
			scale=sigma_density, size=[ num_values ] )
		self.design_density = np.reshape( random_array_normal_distribution, [ self.design_width_voxels, self.design_height_voxels ] )

		self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

	def init_density_with_uniform( self, density_value ):
		assert ( ( density_value <= 1.0 ) and ( density_value >= 0.0 ) ), "Invalid density value specified!"

		self.design_density = density_value * np.ones( [ self.design_width_voxels, self.design_height_voxels ] )

	def init_density_directly( self, input_density ):
		assert ( ( input_density.shape[ 0 ] == self.design_width_voxels ) and ( input_density.shape[ 1 ] == self.design_height_voxels ) ), "Specified design has the wrong shape"

		self.design_density = input_density.copy()

	def init_density_with_this_class( self, this_class ):
		self.init_density_directly( this_class.design_density )

	def setup_simulation( self ):
		self.width_gap_voxels = int( 1.5 * np.maximum( wavelengths_um ) / self.mesh_size_um )
		self.height_gap_voxels_top = int( 2.0 * np.maximum( wavelengths_um ) / self.mesh_size_um )
		self.height_gap_voxels_bottom = self.width_gap_voxels
		self.pml_voxels = int( 1.5 * np.maximum( wavelengths_um ) / self.mesh_size_um )

		self.simluation_width_voxels = self.device_width_voxels + 2 * self.width_gap_voxels + 2 * self.pml_voxels
		self.simulation_height_voxels = self.device_height_voxels + self.focal_length_y_voxels + self.height_gap_voxels_bottom + self.height_gap_voxels_top + 2 * self.pml_voxels

		self.device_width_start = int( 0.5 * ( self.simluation_width_voxels - self.device_width_voxels ) )
		self.device_width_end = self.device_width_start + self.device_width_voxels
		self.device_height_start = int( self.pml_voxels + self.height_gap_voxels_bottom + self.focal_length_y_voxels )
		self.device_height_end = self.device_height_start + self.device_height_voxels

		self.fwd_src_y = int( self.pml_voxels + self.height_gap_voxels_bottom + self.focal_length_y_voxels + self.device_height_voxels + 0.75 * self.height_gap_voxels_top )
		self.focal_point_y = int( self.pml_voxels + self.height_gap_voxels_bottom )

		self.rel_eps_simulation = np.ones( ( self.simluation_width_voxels, self.simulation_height_voxels ) )

		self.fwd_src_x_range = np.arange( 0, self.simulation_width_cells )
		self.fwd_src_y_range = self.fwd_src_y * np.ones( self.fwd_src_x_range.shape, dtype=int )

		self.fwd_source = np.zeros( ( self.simulation_width_cells, self.simulation_height_cells ), dtype=np.complex )
		self.fwd_source[ fwd_src_x_range, fwd_src_y_range ] = 1

	def compute_forward_fields( self, omega, device_permittivity ):
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( self.fwd_source )

		return fwd_Ez

	def compute_fom( self, omega, device_permittivity, focal_point_x_loc ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		return fom

	def compute_fom_and_gradient( self, omega, device_permittivity, focal_point_x_loc ):
		fwd_Ez = self.compute_forward_fields( omega, device_permittivity )
		fom = np.abs( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )**2
		
		adj_source = np.zeros( ( self.simulation_width_cells, self.simulation_height_cells ), dtype=np.complex )
		adj_source[ focal_point_x_loc, self.focal_point_y ] = np.conj( fwd_Ez[ focal_point_x_loc, self.focal_point_y ] )

		adj_Hx, adj_Hy, adj_Ez = simulation.solve( adj_source )

		gradient = 2 * np.real( omega * eps_nought * fwd_Ez * adj_Ez / 1j )

		return fom, gradient

	# CODE DUPLICATION! FIX
	def compute_net_fom( self ):
		fom_by_wl = []

		for wl_idx in range( 0, self.num_wavelengths ):
			get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]
			get_fom, get_grad = self.compute_fom( self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ] )
			scale_fom_for_wl = get_fom * self.wavelength_intensity_scaling[ wl_idx ]
			fom_by_wl.append( scale_fom_for_wl )

		net_fom = np.product( fom_by_wl )

		return net_fom


	def verify_adjoint_against_finite_difference( self ):
		fd_x = int( 0.5 * self.simluation_width_voxels )
		fd_y = np.arange( 0, self.device_height_voxels )
		compute_fd = np.zeros( len( fd_y ) )
		fd_omega = self.omega_values[ int( 0.5 * len( self.omega_values ) ) ]

		fd_init_device = 1.5 * np.ones( ( self.device_width_voxels, self.device_height_voxels ) )

		focal_point_x = self.focal_spots_x_voxels[ 0 ]

		get_fom, get_grad = self.compute_fom_and_gradient(
			fd_omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ],
			self.fwd_src_y, focal_point_x, self.focal_point_y )

		fd_step_eps = 1e-4

		for fd_y_idx in range( 0, len( fd_y ) ):
			fd_device_permittivity = fd_init_device.copy()
			fd_device_permittivity[ fd_x, fd_y[ fd_y_idx ] ] += fd_step_eps

			get_fom_step = self.compute_fom( fd_omega, fd_device_permittivity, focal_point_x )

			compute_fd[ fd_y_idx ] = ( get_fom_step - get_fom ) / fd_step_eps

		self.average_adjoint_finite_difference_error = np.sqrt( np.mean( np.abs( get_grad[ fd_x, device_height_start : device_height_end ] - compute_fd )**2 ) )

	def density_to_permittivity( self, density ):
		return ( self.min_relative_permittivity + ( self.max_relative_permittivity - self.min_relative_permittivity ) * density )

	def optimize( self, num_iterations ):
		self.max_density_change_per_iteration_start = 0.05
		self.max_density_change_per_iteration_end = 0.005

		self.gradient_norm_evolution = np.zeros( num_iterations )
		self.fom_evolution = np.zeros( num_iterations )
		self.fom_by_wl_evolution = np.zeros( ( num_iterations, num_wavelengths ) )
		self.gradient_directions = np.zeros( ( num_iterations, self.design_width_voxels, self.design_height_voxels ) )

		for iter_idx in range( 0, num_iterations ):
		
			import_density = upsample( self.design_density, self.coarsen_factor )
			device_permittivity = self.density_to_permittivity( import_density )

			gradient_by_wl = []
			fom_by_wl = []

			for wl_idx in range( 0, self.num_wavelengths ):
				get_focal_point_idx = self.wavelength_idx_to_focal_idx[ wl_idx ]

				get_fom, get_grad = self.compute_fom_and_gradient( self.omega_values[ wl_idx ], device_permittivity, self.focal_spots_x_voxels[ get_focal_point_idx ] )

				upsampled_device_grad = get_grad[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

				scale_fom_for_wl = get_fom * self.wavelength_intensity_scaling[ wl_idx ]
				scale_gradient_for_wl = device_grad * self.wavelength_intensity_scaling[ wl_idx ]

				gradient_by_wl.append( scale_gradient_for_wl )
				fom_by_wl.append( scale_fom_for_wl )

			net_fom = np.product( fom_by_wl )
			net_gradient = np.zeros( gradient_by_wl[ 0 ].shape )

			for wl_idx in range( 0, self.num_wavelengths ):
				wl_gradient = ( self.max_relative_permittivity - self.min_relative_permittivity ) * gradient_by_wl[ wl_idx ]
				weighting = net_fom / fom_by_wl[ wl_idx ]

				net_gradient += ( weighting * wl_gradient )

			net_gradient = reinterpolate_average( net_gradient, self.coarsen_factor )

			gradient_norm = vector_norm( net_gradient )

			self.fom_evolution[ iter_idx ] = net_fom
			self.fom_by_wl_evolution[ iter_idx ] = np.array( fom_by_wl )
			self.gradient_norm_evolution[ iter_idx ] = gradient_norm

			norm_scaled_gradient = net_gradient / gradient_norm

			self.gradient_directions[ iter_idx ] = norm_scaled_gradient

			max_density_change = (
				self.max_density_change_per_iteration_start +
				( iter_idx / ( num_iterations - 1 ) ) * ( self.max_density_change_per_iteration_end - self.max_density_change_per_iteration_start )
			)

			self.design_density += max_density_change * norm_scaled_gradient / np.max( np.abs( norm_scaled_gradient ) )
			self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

	def save_optimization_data( self, file_base ):
		np.save( file_base + "_gradient_norm_evolution.npy", self.gradient_norm_evolution )
		np.save( file_base + "_fom_evolution.npy", self.fom_evolution )
		np.save( file_base + "_fom_by_wl_evolution.npy", self.fom_by_wl_evolution )
		np.save( file_base + "_gradient_directions.npy", self.gradient_directions )
		np.save( file_base + "_optimized_density.npy", self.design_density )



