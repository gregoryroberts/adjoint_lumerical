import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

import numpy as np

import scipy
from scipy import optimize

from Focusing2DSaveGradientDirectionsParameters import *

class Focusing2DFilter(device.Device):

    def __init__(self, size, permittivity_bounds, init_permittivity):
        super(Focusing2DFilter, self).__init__( size, permittivity_bounds, init_permittivity )

        self.x_dimension_idx = 0
        self.y_dimension_idx = 1
        self.z_dimension_idx = 2

        self.minimum_design_value = 0
        self.maximum_design_value = 1
        self.init_filters_and_variables()

        self.update_permittivity()


    def update_filters(self, epoch):
        self.sigmoid_beta = 0.0625 * (2**epoch)

        self.sigmoid_0 = sigmoid.Sigmoid( self.sigmoid_beta, self.sigmoid_eta )
        self.filters = [ self.sigmoid_0, self.scale_1 ]

    def init_filters_and_variables(self):
        self.num_filters = 2#5
        self.num_variables = 1 + self.num_filters

        # Start the sigmoids at weak strengths
        self.sigmoid_beta = 0.0625 * (2**epoch)
        self.sigmoid_eta = 0.5
        self.sigmoid_0 = sigmoid.Sigmoid( self.sigmoid_beta, self.sigmoid_eta )

        self.scale_1 = scale.Scale( self.permittivity_bounds )

        # Initialize the filter chain
        self.filters = [ self.sigmoid_0, self.scale_1 ]

        self.update_filters( 0 )

        self.init_variables()


    # In the step function, we should update the permittivity with update_permittivity
    def step( self, gradient, step_size ):
        self.w[0] = self.proposed_design_step( gradient, step_size )
        # Update the variable stack including getting the permittivity at the w[-1] position
        self.update_permittivity()
