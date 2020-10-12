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

fdtd_hook.load(projects_directory_location + "/bayer_filter_for_Ian.fsp")

from scipy import ndimage

# fdtd_hook.run()
index_data = fdtd_hook.get_result( 'monitor', 'index_x' )
x_range = fdtd_hook.get_result( 'monitor', 'x' )
y_range = fdtd_hook.get_result( 'monitor', 'y' )
z_range = fdtd_hook.get_result( 'monitor', 'z' )

print( index_data )
print( x_range )
print( y_range )
print( z_range )

# get_index = np.load( '/Users/gregory/Downloads/swarm_epoch_18_sio2_v8.npy' )
# threshold = np.greater_equal( get_index**2, 0.5 * ( min_real_permittivity + max_real_permittivity ) )

# for num_dilation in range( 0, 3 ):
# 	threshold = ndimage.binary_erosion( threshold ).astype( threshold.dtype )


