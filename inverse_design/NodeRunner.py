import numpy as np
import os
import re
import sys
import time

# import imp
# imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/2020a_r6/api/python/lumapi.py" )

# import lumapi

# fdtd_hook = lumapi.FDTD( hide=True )

filebase = sys.argv[ 1 ]
id_number = sys.argv[ 2 ]

pattern = re.compile("ID" + id_number + "(.*?)\.READY")

while True:
    
    if os.path.isdir( filebase ):
        for filepath in os.listdir( filebase ):
            if pattern.match( filepath ):

                file_root = filebase + "/" + filepath[:-5]
                print(file_root)                
                os.remove( file_root + "READY" )
                
                lumerical_bin_mpiexec = "/central/home/gdrobert/Develompent/lumerical/2020a_r6/mpich2/nemesis/bin/"
                lumerical_bin_nemesis = "/central/home/gdrobert/Develompent/lumerical/2020a_r6/bin/"
                os.system(
                    lumerical_bin_mpiexec +  "mpiexec -n 8 " + lumerical_bin_nemesis + "fdtd-engine-mpich2nem -t 1 " + file_root + "fsp" )

                # fdtd_hook.load( file_root + "fsp" )
                # fdtd_hook.run()

                completed = open( file_root + "COMPLETED", "w" )
                completed.write( "COMPLETED\n" )
                completed.close()

    time.sleep( 1 )






