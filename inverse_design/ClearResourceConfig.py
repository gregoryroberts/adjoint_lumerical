import platform
import os
import re
import numpy as np

import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )

import lumapi
#
# Code from Conner // START
#

def configure_resources_for_cluster( fdtd_hook, node_hostnames, N_resources=2, N_threads_per_resource=8 ):
    '''
    Take in a list of hostnames (different nodes on the cluster), and configure
    them to have N_threads_per_resource.
    '''
    if len(node_hostnames) != N_resources:
        raise ValueError('Length of node_hostnames should be N_resources')

    # Use different MPIs depending on platform.
    if platform.system() == 'Windows':
        mpi_type = 'Remote: Intel MPI'
    else:
        mpi_type = 'Remote: MPICH2'
    # Delete all resources. Lumerical doesn't let us delete the last resource, so
    # we stop when it throws an Exception.
    while True:
        try:
            fdtd_hook.deleteresource("FDTD", 1)
        except lumapi.LumApiError:
            break
    # Change the one resource we have to have the proper number of threads.
    fdtd_hook.setresource("FDTD", 1, "processes", N_threads_per_resource)
    fdtd_hook.setresource('FDTD', 1, 'Job launching preset', mpi_type)
    fdtd_hook.setresource('FDTD', 1, 'hostname', node_hostnames[0])
    # Now add and configure the rest.
    for i in np.arange(1, N_resources):
        try:
            fdtd_hook.addresource("FDTD")
        except:
            pass
        finally:
            fdtd_hook.setresource("FDTD", i+1, "processes", N_threads_per_resource)
            fdtd_hook.setresource('FDTD', i+1, 'Job launching preset', mpi_type)
            fdtd_hook.setresource('FDTD', i+1, 'hostname', node_hostnames[i])


fdtd_hook = lumapi.FDTD( hide=True )
configure_resources_for_cluster( fdtd_hook, [ 'localhost' ], 1, 8 )



