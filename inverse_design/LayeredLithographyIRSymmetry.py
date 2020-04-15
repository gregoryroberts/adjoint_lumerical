import matplotlib.pyplot as plt
import numpy as np
import sys

# data_dir = './symmetry_ir_data/'
data_dir = '../projects/layered_infrared_3layers_pol_insensitive_6x6x3p12um_f4_test_symmetry/'

forward_symmetry_focal = np.load( data_dir + 'forward_symmetry_focal_data_y.npy' )
forward_compare_focal = np.load( data_dir + 'forward_compare_focal_data_y.npy' )

choose_wl = 0

print( forward_symmetry_focal[ :, :, 0 ] )
print()
print( forward_compare_focal[ :, :, 0 ] )
print()
print( forward_symmetry_focal[ :, :, 0 ] - forward_compare_focal[ :, :, 0 ] )
print()

print( np.max( np.abs( forward_symmetry_focal - forward_compare_focal ) ) )

print()


forward_symmetry_fields = np.load( data_dir + 'forward_symmetry_fields_y.npy' )
forward_compare_fields = np.load( data_dir + 'forward_compare_fields_y.npy' )

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( forward_symmetry_fields[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( forward_symmetry_fields[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( forward_compare_fields[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( forward_compare_fields[ 1, choose_wl, 10, :, : ] ) )
plt.show()

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( forward_symmetry_fields[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( forward_symmetry_fields[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( forward_compare_fields[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( forward_compare_fields[ 0, choose_wl, 20, :, : ] ) )
plt.show()


plt.subplot( 2, 2, 1 )
plt.imshow( np.real( forward_symmetry_fields[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( forward_symmetry_fields[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( forward_compare_fields[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( forward_compare_fields[ 2, choose_wl, 15, :, : ] ) )
plt.show()

adjoint_symmetry_fields_0 = np.load( data_dir + 'adjoint_symmetry_fields_y_0.npy' )
adjoint_compare_fields_0 = np.load( data_dir + 'adjoint_compare_fields_y_0.npy' )

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_0[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_0[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_0[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_0[ 1, choose_wl, 10, :, : ] ) )
plt.show()

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_0[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_0[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_0[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_0[ 0, choose_wl, 20, :, : ] ) )
plt.show()


plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_0[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_0[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_0[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_0[ 2, choose_wl, 15, :, : ] ) )
plt.show()

# sys.exit( 0 )

adjoint_symmetry_fields_2 = np.load( data_dir + 'adjoint_symmetry_fields_y_2.npy' )
adjoint_compare_fields_2 = np.load( data_dir + 'adjoint_compare_fields_y_2.npy' )

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_2[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_2[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_2[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_2[ 1, choose_wl, 10, :, : ] ) )
plt.show()

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_2[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_2[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_2[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_2[ 0, choose_wl, 20, :, : ] ) )
plt.show()


plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_2[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_2[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_2[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_2[ 2, choose_wl, 15, :, : ] ) )
plt.show()


adjoint_symmetry_fields_x_3 = np.load( data_dir + 'adjoint_symmetry_fields_x_3.npy' )
adjoint_compare_fields_x_3 = np.load( data_dir + 'adjoint_compare_fields_x_3.npy' )

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_x_3[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_x_3[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_x_3[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_x_3[ 1, choose_wl, 10, :, : ] ) )
plt.show()

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_x_3[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_x_3[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_x_3[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_x_3[ 0, choose_wl, 20, :, : ] ) )
plt.show()


plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_x_3[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_x_3[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_x_3[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_x_3[ 2, choose_wl, 15, :, : ] ) )
plt.show()


adjoint_symmetry_fields_y_3 = np.load( data_dir + 'adjoint_symmetry_fields_y_3.npy' )
adjoint_compare_fields_y_3 = np.load( data_dir + 'adjoint_compare_fields_y_3.npy' )

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_y_3[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_y_3[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_y_3[ 1, choose_wl, 10, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_y_3[ 1, choose_wl, 10, :, : ] ) )
plt.show()

plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_y_3[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_y_3[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_y_3[ 0, choose_wl, 20, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_y_3[ 0, choose_wl, 20, :, : ] ) )
plt.show()


plt.subplot( 2, 2, 1 )
plt.imshow( np.real( adjoint_symmetry_fields_y_3[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 2 )
plt.imshow( np.imag( adjoint_symmetry_fields_y_3[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 3 )
plt.imshow( np.real( adjoint_compare_fields_y_3[ 2, choose_wl, 15, :, : ] ) )
plt.subplot( 2, 2, 4 )
plt.imshow( np.imag( adjoint_compare_fields_y_3[ 2, choose_wl, 15, :, : ] ) )
plt.show()

