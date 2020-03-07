import numpy as np
import CMOSMetalBayerFilter2D

import scipy.signal

class OptimizationStateMultiDevice():

    def __init__( self, num_iterations, num_epochs, filename_prefix, num_devices ):
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.filename_prefix = filename_prefix

        self.num_devices = num_devices

        self.figure_of_merit = np.zeros( ( num_epochs, num_iterations, self.num_devices ) )

    #
    # In the multi-device case, we should have multiple pieces of data here to submit
    #
    def submit_figure_of_merit( self, data, iteration, epoch ):
        self.figure_of_merit[ epoch, iteration ] = data

    def load( self, filebase, epoch ):
        # load_figure_of_merit = np.load( filebase + "/" + self.filename_prefix + "figure_of_merit.npy" )
        # self.figure_of_merit[ 0 : ( epoch + 1 ), : ] = load_figure_of_merit[ 0 : ( epoch + 1 ), : ]

        self.load_design( filebase, epoch )

    def save( self, filebase, epoch ):
        np.save( filebase + "/" + self.filename_prefix + "figure_of_merit.npy", self.figure_of_merit )

        self.save_design( filebase, epoch )

    def reinterpolate( self, input_array, output_shape ):
        input_shape = input_array.shape

        assert len( input_shape ) == len( output_shape ), "Reinterpolate: expected the input and output to have same number of dimensions"

        output_array = input_array.copy()

        for axis_idx in range( 0, len( input_shape ) ):
            output_array = scipy.signal.resample( output_array, output_shape[ axis_idx ], axis=axis_idx )

        return output_array    
