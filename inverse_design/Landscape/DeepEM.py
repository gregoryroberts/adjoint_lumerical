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

iter_idx = 0
fom_evolution = []
binarization_evolution = []

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

eps_nought = 8.854 * 1e-12
mu_nought = 1.257 * 1e-6 
c = 3.0 * 1e8
small = 1e-10

def upsample( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k * factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		for y_idx in range( 0, output_block_size[ 1 ] ):
			output_block[ x_idx, y_idx ] = input_block[ int( x_idx / factor ), int( y_idx / factor ) ]

	return output_block

class DeepEM():

	def __init__( self,
		device_size_voxels, num_layers, coarsen_factor, mesh_size_nm,
		permittivity_bounds, focal_length_y_voxels,
		wavelength_um, random_seed, save_folder ):
		
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

		self.focal_length_y_voxels = focal_length_y_voxels
		self.wavelength_um = wavelength_um

		self.omega = 2 * np.pi * c / ( 1e-6 * wavelength_um )

		self.random_seed = random_seed
		np.random.seed( self.random_seed )

		assert( self.design_height_voxels % num_layers ) == 0, "Expected the number of layers to evenly divide the design region"

		self.num_layers = num_layers
		self.design_voxels_per_layer = int( self.design_height_voxels / num_layers )

		self.setup_simulation()

	def plot_geometry( self, opt_mask=None ):
		import matplotlib.pyplot as plt

		device_region = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ) )
		device_region[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = 100 * np.imag(
			upsample( self.density_to_permittivity( self.design_density ), self.coarsen_factor )
		)

		device_region[
			self.focal_x_start : self.focal_x_end,
			self.focal_point_y - 5 : self.focal_point_y + 5 ] = 2

		plt.subplot( 2, 2, 1 )
		plt.imshow( np.real( self.fwd_source ) )
		plt.title( 'Forward Source' )
		plt.subplot( 2, 2, 2 )
		plt.imshow( device_region )
		plt.title( 'Focal Y' )
		plt.subplot( 2, 2, 3 )
		plt.imshow( device_region )
		plt.title( 'Device Region' )
		plt.show()


	def gen_random_device( self ):
		num_random_values = self.design_width_voxels * self.num_layers

		random_design_values = np.random.random( num_random_values )

		def apply_sigmoid( variable_in, beta, eta ):
			numerator = np.add(np.tanh(beta * eta), np.tanh(np.multiply(beta, np.subtract(variable_in, eta))))

			numerator = np.tanh( beta * eta ) + np.tanh( beta * ( variable_in - eta ) )
			denominator = np.tanh( beta * eta ) + np.tanh( beta * ( 1 - eta ) )

			return ( numerator / denominator )

		beta = 0.0625 + 20.0 * np.random.random()
		eta = 0.1 + 0.8 * np.random.random()

		random_design_values = apply_sigmoid( random_design_values, beta, eta )

		self.design_density = np.ones( [ self.design_width_voxels, self.design_height_voxels ] )

		for layer_idx in range( 0, self.num_layers ):
			layer_start = layer_idx * self.design_voxels_per_layer
			layer_end = layer_start + self.design_voxels_per_layer

			random_values_start = layer_idx * self.design_width_voxels
			random_values_end = random_values_start + self.design_width_voxels

			fill_data = random_design_values[ random_values_start : random_values_end ]

			for internal_layer_idx in range( layer_start, layer_end ):
				self.design_density[ :, internal_layer_idx ] = fill_data

		self.design_density = np.maximum( 0, np.minimum( self.design_density, 1 ) )

	def setup_simulation( self ):
		self.width_gap_voxels = int( 1.0 * self.wavelength_um / self.mesh_size_um )
		self.height_gap_voxels_top = int( 1.5 * self.wavelength_um / self.mesh_size_um )
		self.height_gap_voxels_bottom = self.width_gap_voxels
		self.pml_voxels = int( 1.0 * self.wavelength_um / self.mesh_size_um )

		self.simulation_width_voxels = self.device_width_voxels + 2 * self.width_gap_voxels + 2 * self.pml_voxels
		self.simulation_height_voxels = self.device_height_voxels + np.maximum( self.focal_length_y_voxels, 0 ) + self.height_gap_voxels_bottom + self.height_gap_voxels_top + 2 * self.pml_voxels

		self.device_width_start = int( 0.5 * ( self.simulation_width_voxels - self.device_width_voxels ) )
		self.device_width_end = self.device_width_start + self.device_width_voxels
		self.device_height_start = int( self.pml_voxels + self.height_gap_voxels_bottom + np.maximum( self.focal_length_y_voxels, 0 ) )
		self.device_height_end = self.device_height_start + self.device_height_voxels

		self.fwd_src_y = int( self.pml_voxels + self.height_gap_voxels_bottom + np.maximum( self.focal_length_y_voxels, 0 ) + self.device_height_voxels + 0.75 * self.height_gap_voxels_top )
		self.focal_point_y = int( self.pml_voxels + self.height_gap_voxels_bottom - np.minimum( self.focal_length_y_voxels, 0 ) )

		self.focal_x_start = self.pml_voxels
		self.focal_x_end = self.focal_x_start + 2 * self.width_gap_voxels + self.device_width_voxels

		self.rel_eps_simulation = np.ones( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )

		fwd_src_x_range = np.arange( 0, self.simulation_width_voxels )
		fwd_src_y_range = self.fwd_src_y * np.ones( fwd_src_x_range.shape, dtype=int )

		self.fwd_source = np.zeros( ( self.simulation_width_voxels, self.simulation_height_voxels ), dtype=np.complex )
		self.fwd_source[ fwd_src_x_range, fwd_src_y_range ] = 1


	def get_device_efields( self ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		Ez = self.compute_forward_fields( self.omega, device_permittivity )

		return Ez[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ]

	def get_focal_efields( self ):
		import_density = upsample( self.design_density, self.coarsen_factor )
		device_permittivity = self.density_to_permittivity( import_density )
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		Ez = self.compute_forward_fields( self.omega, device_permittivity )

		return Ez[ self.focal_x_start : self.focal_x_end, self.focal_point_y ]

	def compute_forward_fields( self, omega, device_permittivity ):
		self.rel_eps_simulation[ self.device_width_start : self.device_width_end, self.device_height_start : self.device_height_end ] = device_permittivity

		simulation = ceviche.fdfd_ez( omega, self.mesh_size_m, self.rel_eps_simulation, [ self.pml_voxels, self.pml_voxels ] )
		fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( self.fwd_source )

		return fwd_Ez

	def density_to_permittivity( self, density ):
		return ( self.min_relative_permittivity + ( self.max_relative_permittivity - self.min_relative_permittivity ) * density )



