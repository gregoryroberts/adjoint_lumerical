import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import kkr

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# from LayeredMWIRBridgesBayerFilterParameters import *
import LayeredMWIRBridgesBayerFilter

# import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a/api/python/lumapi.py" )
import lumapi

import functools
import h5py
# import matplotlib.pyplot as plt
import numpy as np
import time

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))
projects_directory_location += "/cluster/mwir_081320/"

fdtd_hook.load(projects_directory_location + "/dilate_base.fsp")

fdtd_hook.save(projects_directory_location + "/complex_index.fsp")

fdtd_hook.switchtolayout()
fdtd_hook.run()
index_data = np.squeeze( fdtd_hook.getresult( 'monitor', 'index_x' ) )
x_range = np.squeeze( fdtd_hook.getresult( 'monitor', 'x' ) )
y_range = np.squeeze( fdtd_hook.getresult( 'monitor', 'y' ) )
z_range = np.squeeze( fdtd_hook.getresult( 'monitor', 'z' ) )


A_lor = [ 0.066, 0.144, 0.16, 0.038, 0.15, 0.18, 0.46, 0.5, 0.63, 0.22, 0.11, 0.0586, 0.0087, 0.021, 0.025, 0.0046, 0.0198 ]
A_gauss = [ 0.242, 0.056, 0.028, 0.15, 0.5, 0.12, 0.08, 0.136, 0.876, 0.041 ]

omega_nought_lor = [ 607, 753.4, 809.4, 830, 981.9, 1018, 1063, 1158, 1255, 1408.8, 1507.7, 1659, 2420, 2884, 2930, 3062, 3503 ]
omega_nought_gauss = [ 264, 701, 732.7, 1113, 1188, 1291, 1365, 1459.2, 1731.7, 2963 ]

gamma_lor = [ 174, 28, 8, 52, 22, 141, 21, 38, 39, 12, 7.9, 169, 2495, 48, 51, 190, 201 ]
gamma_gauss = [ 345, 8.9, 9, 32, 59, 24, 99, 44, 35.2, 40 ]

def eps_lor_imag( omega_nought, A, gamma, omega ):
	numerator = A * gamma**2 * omega_nought * omega
	denominator = ( omega_nought**2 - omega**2 )**2 + gamma**2 * omega**2

	return ( numerator / denominator )

def eps_lor_real( omega_nought, A, gamma, omega ):
	numerator = A * gamma * omega_nought * ( omega_nought**2 - omega**2 )
	denominator = ( omega_nought**2 - omega**2 )**2 + gamma**2 * omega**2

	return ( numerator / denominator )

def eps_gauss( omega_nought, A, gamma, omega ):
	one_over_f = 2. * np.sqrt( np.log( 2. ) )
	term1 = A * np.exp( -np.abs( omega - omega_nought ) * one_over_f / gamma )
	term2 = A * np.exp( -( omega + omega_nought ) * one_over_f / gamma )

	return ( term1 + term2 )

def index_from_permittivity( epsilon ):
	eps_r = np.real( epsilon )
	eps_i = np.imag( epsilon )

	k = np.sqrt( 0.5 * eps_r * ( -1 + np.sqrt( 1 + ( eps_i / eps_r )**2 ) ) )
	n = eps_i / ( 2 * k )

	return ( n + 1j * k )

eps_infinity = 2.37
delta_eps_infinity = eps_infinity - 1.0

num_eval_wls = 100
# eval_wl_um = np.linspace( 2.5, 6.0, num_eval_wls )
eval_wl_um = np.linspace( 2.8, 5.5, num_eval_wls )

omega_values = 10000. / eval_wl_um
eps_values = eps_infinity * np.ones( len( omega_values ), dtype=np.complex )
eps_values_gauss = np.zeros( len( omega_values ), dtype=np.complex )

for lor_idx in range( 0, len( A_lor ) ):
	eps_values += 1j * eps_lor_imag( omega_nought_lor[ lor_idx ], A_lor[ lor_idx ], gamma_lor[ lor_idx ], omega_values )
	# eps_values += eps_lor_real( omega_nought_lor[ lor_idx ], A_lor[ lor_idx ], gamma_lor[ lor_idx ], omega_values )

for gauss_idx in range( 0, len( A_gauss ) ):
	eps_values += 1j * eps_gauss( omega_nought_gauss[ gauss_idx ], A_gauss[ gauss_idx ], gamma_gauss[ gauss_idx ], omega_values )
	# eps_values_gauss += 1j * eps_gauss( omega_nought_gauss[ gauss_idx ], A_gauss[ gauss_idx ], gamma_gauss[ gauss_idx ], omega_values )

pack_imag_eps = np.zeros( ( len( omega_values ), 3, 3 ) )
for idx in range( 0, len( omega_values ) ):
	pack_imag_eps[ idx, 0, 0 ] = np.imag( eps_values[ idx ] )
	pack_imag_eps[ idx, 1, 1 ] = np.imag( eps_values[ idx ] )
	pack_imag_eps[ idx, 2, 2 ] = np.imag( eps_values[ idx ] )

real_eps = delta_eps_infinity + kkr.kkr( ( omega_values[ 1 ] - omega_values[ 0 ] ), pack_imag_eps )
real_eps = np.real( np.squeeze( real_eps[ :, 0, 0 ] ) )

imag_eps = np.squeeze( pack_imag_eps[ :, 0, 0 ] )

full_perm = real_eps + 1j * imag_eps
full_index = index_from_permittivity( full_perm )

fdtd_hook.switchtolayout()

polymer_import = fdtd_hook.addimport()
polymer_import['name'] = 'polymer_import'
polymer_import['x span'] = 25 * 1e-6
polymer_import['y span'] = 25 * 1e-6
polymer_import['z min'] = 25 * 1e-6
polymer_import['z max'] = 50 * 1e-6

polymer_x = 1e-6 * np.linspace( -12.5, 12.5, 2 )
polymer_y = 1e-6 * np.linspace( -12.5, 12.5, 2 )
polymer_z = 1e-6 * np.linspace( 25, 50, 2 )

polymer_data = np.ones( ( 2, 2, 2 ), dtype=np.complex )

binarize_index = np.greater( index_data, 1.25 )

min_index = 1.0
# max_index = 1.5

max_index = np.mean( np.real( full_index ) ) + 1j * np.mean( np.imag( full_index ) )

T0 = np.zeros( ( 2, num_eval_wls ) )
T1 = np.zeros( ( 2, num_eval_wls ) )
T2 = np.zeros( ( 2, num_eval_wls ) )
T3 = np.zeros( ( 2, num_eval_wls ) )

pol_source_appendices = [ 'x', 'y' ]

for pol_idx in range( 0, 2 ):
	delta_n = max_index - min_index

	polymer_data[ : ] = max_index

	reassemble_index = min_index + delta_n * binarize_index

	fdtd_hook.switchtolayout()

	for disable_src_idx in range( 0, 2 ):
		fdtd_hook.select( 'forward_src_' + str( pol_source_appendices[ disable_src_idx ] ) )
		fdtd_hook.set( 'enabled', 0 )

	fdtd_hook.select( 'forward_src_' + str( pol_source_appendices[ pol_idx ] ) )
	fdtd_hook.set( 'enabled', 1 )

	fdtd_hook.select( 't0' )
	fdtd_hook.set( 'use source limits', 0 )
	fdtd_hook.set( 'frequency points', num_eval_wls )
	fdtd_hook.set( 'minimum wavelength', 2.5 * 1e-6 )
	fdtd_hook.set( 'maximum wavelength', 6.0 * 1e-6 )

	fdtd_hook.select( 't1' )
	fdtd_hook.set( 'use source limits', 0 )
	fdtd_hook.set( 'frequency points', num_eval_wls )
	fdtd_hook.set( 'minimum wavelength', 2.5 * 1e-6 )
	fdtd_hook.set( 'maximum wavelength', 6.0 * 1e-6 )

	fdtd_hook.select( 't2' )
	fdtd_hook.set( 'use source limits', 0 )
	fdtd_hook.set( 'frequency points', num_eval_wls )
	fdtd_hook.set( 'minimum wavelength', 2.5 * 1e-6 )
	fdtd_hook.set( 'maximum wavelength', 6.0 * 1e-6 )

	fdtd_hook.select( 't3' )
	fdtd_hook.set( 'use source limits', 0 )
	fdtd_hook.set( 'frequency points', num_eval_wls )
	fdtd_hook.set( 'minimum wavelength', 2.5 * 1e-6 )
	fdtd_hook.set( 'maximum wavelength', 6.0 * 1e-6 )


	fdtd_hook.select( 'polymer_import' )
	fdtd_hook.importnk2( polymer_data, polymer_x, polymer_y, polymer_z )

	fdtd_hook.select( 'design_import' )
	fdtd_hook.importnk2( reassemble_index, x_range, y_range, z_range )

	fdtd_hook.run()

	T0_data = fdtd_hook.getresult( 't0', 'T' )
	T1_data = fdtd_hook.getresult( 't1', 'T' )
	T2_data = fdtd_hook.getresult( 't2', 'T' )
	T3_data = fdtd_hook.getresult( 't3', 'T' )

	T0[ pol_idx, : ] = T0_data[ 'T' ]
	T1[ pol_idx, : ] = T1_data[ 'T' ]
	T2[ pol_idx, : ] = T2_data[ 'T' ]
	T3[ pol_idx, : ] = T3_data[ 'T' ]

np.save( projects_directory_location + "/t0_avg.npy", T0 )
np.save( projects_directory_location + "/t1_avg.npy", T1 )
np.save( projects_directory_location + "/t2_avg.npy", T2 )
np.save( projects_directory_location + "/t3_avg.npy", T3 )
