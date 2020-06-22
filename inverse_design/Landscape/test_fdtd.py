import ceviche
import matplotlib.pyplot as plt
import numpy as np

eps_nought = 8.854 * 1e-12
c = 3.0 * 1e8

mesh_size_nm = 15
density_coarsen_factor = 4
mesh_size_m = mesh_size_nm * 1e-9
lambda_min_nm = 500
lambda_max_nm = 600
num_lambda_values = 2

min_relative_permittivity = 1.5**2
max_relative_permittivity = 2.5**2

lambda_values_nm = np.linspace( lambda_min_nm, lambda_max_nm, num_lambda_values )
omega_values = 2 * np.pi * c / ( 1e-9 * lambda_values_nm )

pml_voxels = 40
device_width_voxels = 140
device_height_voxels = 40
device_voxels_total = device_width_voxels * device_height_voxels
mid_width_voxel = int( 0.5 * device_width_voxels )
mid_height_voxel = int( 0.5 * device_height_voxels )
width_gap_voxels = 50
height_gap_voxels_top = 75
height_gap_voxels_bottom = 50
focal_length_voxels = 100
simulation_width_voxels = device_width_voxels + 2 * width_gap_voxels + 2 * pml_voxels
simulation_height_voxels = device_height_voxels + focal_length_voxels + height_gap_voxels_bottom + height_gap_voxels_top + 2 * pml_voxels

device_width_start = int( 0.5 * ( simulation_width_voxels - device_width_voxels ) )
device_width_end = device_width_start + device_width_voxels
device_height_start = int( pml_voxels + height_gap_voxels_bottom + focal_length_voxels )
device_height_end = device_height_start + device_height_voxels

fwd_src_y = int( pml_voxels + height_gap_voxels_bottom + focal_length_voxels + device_height_voxels + 0.75 * height_gap_voxels_top )
focal_point_y = int( pml_voxels + height_gap_voxels_bottom )


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

def upsample( input_block, factor ):
	input_block_size = input_block.shape
	output_block_size = [ int( k * factor ) for k in input_block_size ]

	output_block = np.zeros( output_block_size, input_block.dtype )

	for x_idx in range( 0, output_block_size[ 0 ] ):
		for y_idx in range( 0, output_block_size[ 1 ] ):
			output_block[ x_idx, y_idx ] = input_block[ int( x_idx / factor ), int( y_idx / factor ) ]

	return output_block

def density_to_permittivity( density_in ):
	return ( min_relative_permittivity + ( max_relative_permittivity - min_relative_permittivity ) * density_in )

rel_eps_simulation = np.ones( ( simulation_width_voxels, simulation_height_voxels ) )

focal_points_x = [
	int( device_width_start + 0.25 * device_width_voxels ),
	int( device_width_start + 0.75 * device_width_voxels )
]

device_density = np.random.random( ( int( device_width_voxels / density_coarsen_factor ), int( device_height_voxels / density_coarsen_factor ) ) )
upsampled_density = upsample( device_density, density_coarsen_factor )
device_permittivity = density_to_permittivity( upsampled_density )
rel_eps_simulation[ device_width_start : device_width_end, device_height_start : device_height_end ] = device_permittivity

omega = omega_values[ int( 0.5 * num_lambda_values ) ]

simulation = ceviche.fdfd_ez( omega, mesh_size_m, rel_eps_simulation, [ pml_voxels, pml_voxels ] )

fwd_src_x = np.arange( 0, simulation_width_voxels )
fwd_src_y_line = fwd_src_y * np.ones( fwd_src_x.shape, dtype=int )

fwd_source = np.zeros( ( simulation_width_voxels, simulation_height_voxels ), dtype=np.complex )
fwd_source[ fwd_src_x, fwd_src_y_line ] = 1

fwd_Hx, fwd_Hy, fwd_Ez = simulation.solve( fwd_source )

# plt.imshow( np.abs( fwd_Ez )**2, cmap='Blues' )
# plt.show()



