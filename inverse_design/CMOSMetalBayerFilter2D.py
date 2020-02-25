import device as device
import layering as layering
import scale as scale
import sigmoid as sigmoid
import square_blur as square_blur

import numpy as np

import scipy
from scipy import optimize

from CMOSMetalBayerFilter2DParameters import *

class CMOSMetalBayerFilter2D(device.Device):

    def __init__(self, size, permittivity_bounds, init_permittivity, num_y_layers):
        super(CMOSMetalBayerFilter2D, self).__init__(size, permittivity_bounds, init_permittivity)

        self.x_dimension_idx = 0
        self.y_dimension_idx = 1
        self.z_dimension_idx = 2

        self.num_y_layers = num_y_layers
        self.flip_threshold = 0.5
        self.minimum_design_value = 0
        self.maximum_design_value = 1
        self.init_filters_and_variables()

        self.update_permittivity()


    #
    # Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
    #
    def update_permittivity(self):
        var0 = self.w[0]

        var1 = self.layering_y_0.forward( var0 )
        self.w[1] = var1

        var2 = self.sigmoid_1.forward( var1 )
        self.w[2] = var2

        scale_real_2 = self.scale_2[ 0 ]
        scale_imag_2 = self.scale_2[ 1 ]

        var3 = scale_real_2.forward( var2 ) + 1j * scale_imag_2.forward( var2 )
        self.w[3] = var3

        # var2 = self.layering_z_1.forward(var1)
        # self.w[2] = var2

        # var3 = self.max_blur_xy_2.forward(var2)
        # self.w[3] = var3

        # var4 = self.sigmoid_3.forward(var3)
        # self.w[4] = var4


    #
    # Need to also override the backpropagation function
    #
    def backpropagate(self, gradient_real, gradient_imag):
        scale_real_2 = self.scale_2[ 0 ]
        scale_imag_2 = self.scale_2[ 1 ]

        gradient = (
            scale_real_2.chain_rule( gradient_real, self.w[3], self.w[2] ) +
            scale_imag_2.chain_rule( gradient_imag, self.w[3], self.w[2] )
        )	

        gradient = self.sigmoid_1.chain_rule( gradient, self.w[2], self.w[1] )
        gradient = self.layering_y_0.chain_rule( gradient, self.w[1], self.w[0] )

        # gradient = self.sigmoid_3.chain_rule(gradient, self.w[4], self.w[3])
        # gradient = self.max_blur_xy_2.chain_rule(gradient, self.w[3], self.w[2])
        # gradient = self.layering_y_0.chain_rule(gradient, self.w[1], self.w[0])
        # gradient = self.sigmoid_0.chain_rule(gradient, self.w[1], self.w[0])

        return gradient

    def update_filters(self, epoch):
        self.sigmoid_beta = 0.0625 * (2**epoch) / ( 2**3 )
        # self.sigmoid_beta = 0.0625
        self.layering_y_0 = layering.Layering(self.y_dimension_idx, self.num_y_layers)

        self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
        # self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
        # self.filters = [self.layering_y_0, self.sigmoid_1, self.scale_2]#[self.layering_y_0, self.scale_1]# [self.sigmoid_0, self.layering_z_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4]
        self.filters = [self.layering_y_0, self.sigmoid_1, self.scale_2]
        # self.filters = [self.scale_1]

    def init_filters_and_variables(self):
        self.num_filters = 3#5
        self.num_variables = 1 + self.num_filters

        # z_voxel_layers = self.size[2]
        self.layering_y_0 = layering.Layering(self.y_dimension_idx, self.num_y_layers)

        # Start the sigmoids at weak strengths
        self.sigmoid_beta = 0.0625 / ( 2**3 )
        self.sigmoid_eta = 0.5
        self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
        # self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)


        # alpha = 8
        # self.blur_half_width = blur_half_width_voxels
        #
        # This notation is slightly confusing, but it is meant to be the
        # direction you blur when you are on the layer corresponding to x-
        # or y-layering.  So, if you are layering in x, then you blur in y
        # and vice versa.
        #
        # self.max_blur_xy_2 = square_blur.SquareBlur(
        # 	alpha,
        # 	[self.blur_half_width, self.blur_half_width, 0])

        scale_real_min = np.real( self.permittivity_bounds[0] )
        scale_real_max = np.real( self.permittivity_bounds[1] )
        scale_real_2 = scale.Scale([scale_real_min, scale_real_max])

        scale_imag_min = np.imag( self.permittivity_bounds[0] )
        scale_imag_max = np.imag( self.permittivity_bounds[1] )
        scale_imag_2 = scale.Scale([scale_imag_min, scale_imag_max])

        self.scale_2 = [ scale_real_2, scale_imag_2 ]

        # Initialize the filter chain
        # self.filters = [self.layering_y_0, self.sigmoid_1, self.scale_2]#[self.layering_y_0, self.scale_1]# [self.sigmoid_0, self.layering_z_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4]
        self.filters = [self.layering_y_0, self.sigmoid_1, self.scale_2]
        # self.filters = [self.scale_1]

        self.update_filters( 0 )

        self.init_variables()


    def proposed_design_step(self, gradient_real, gradient_imag, step_size, do_simulated_annealing, current_temperature, enforce_binarization_increase, binarization_increase, max_design_change_point):
        gradient = self.backpropagate(gradient_real, gradient_imag)
        gradient_norm = np.sqrt( np.sum( np.abs( gradient )**2 ) )
        normalized_direction = gradient / gradient_norm

        if do_simulated_annealing:

            normalized_direction = gradient / np.sqrt( np.sum( np.abs( gradient )**2 ) )

            perturbation_success = False

            random_direction = np.random.random( gradient.shape ) - 0.5
            random_direction /= np.sqrt( np.sum( np.abs( random_direction )**2 ) )

            flatten_perturbation = random_direction.flatten()
            flatten_direction = normalized_direction.flatten()

            difference_measure = np.sum( ( flatten_perturbation - flatten_direction ) * flatten_direction )

            annealing_probability = np.exp( 0.5 * difference_measure / current_temperature )
            flip_coin = np.random.random( 1 )

            print("Current middle annealing probability = " + str( np.exp( -0.5 * 1 / current_temperature ) ))

            perturbation_success = flip_coin[ 0 ] >= ( 1 - annealing_probability )

            if perturbation_success:
                print("Successful perturbation!")
                gradient = gradient_norm * random_direction


        def compute_binarization( input_variable ):
            return ( 2 / np.sqrt( len( input_variable ) ) ) * np.sum( ( input_variable - 0.5 )**2 )
        def compute_binarization_gradient( input_variable ):
            return ( 4 / len( input_variable ) ) * ( input_variable - 0.5 ) / compute_binarization( input_variable )
        depth = self.w[ 0 ].shape[ 1 ]
        half_depth = int( 0.5 * depth )
        design_variable_cut = self.w[ 0 ][ :, half_depth, 0 ]

        # if enforce_binarization_increase:

        #     design_variable_cut = np.real( design_variable_cut )
        #     gradient_cut = gradient[ :, half_depth, 0 ]
        #     gradient_cut = np.real( gradient_cut )            

        #     # Desired binarization increase
        #     alpha = 0.02
        #     # Maximum movement for each density variable
        #     beta = 0.01

        #     c = gradient_cut

        #     print( "Starting binarization = " + str( compute_binarization( design_variable_cut ) ) )

        #     b = -compute_binarization_gradient( design_variable_cut )

        #     lower_bounds = np.zeros( len( c ) )
        #     upper_bounds = np.zeros( len( c ) )

        #     for idx in range( 0, len( c ) ):
        #         upper_bounds[ idx ] = np.maximum( np.minimum( 1 - design_variable_cut[ idx ], beta ), 0 )
        #         lower_bounds[ idx ] = np.minimum( np.maximum( -beta, -design_variable_cut[ idx ] ), 0 )

        #     nu_low = 0
        #     num_high = 0

        #     b_min = b[ 0 ]
        #     b_max = b[ 0 ]

        #     b_min_idx = 0
        #     b_max_idx = 0

        #     dual_start = time.time()

        #     for idx in range( 1, dim ):
        #         if b[ idx ] < b_min:
        #             b_min_idx = idx
        #         if b[ idx ] > b_max:
        #             b_max_idx = idx

        #         b_min = np.minimum( b_min, b[ idx ] )
        #         b_max = np.maximum( b_max, b[ idx ] )


        #     lower_nu = -b_min / ( c[ b_min_idx ] + 1e-12 )
        #     upper_nu = b_max / ( c[ b_max_idx ] + 1e-12 )
        #     num_nu_steps = 1000

        #     nu_values = np.linspace( lower_nu, upper_nu, num_nu_steps )

        #     cur_value = -np.inf
        #     cur_x = np.zeros( dim )

        #     for nu_idx in range( 0, num_nu_steps ):
        #         get_nu = nu_values[ nu_idx ]

        #         constraint = c + get_nu * b
        #         get_x = np.zeros( dim )

        #         for idx in range( 0, dim ):
        #             if constraint[ idx ] < 0:
        #                 get_x[ idx ] = beta
        #             else:
        #                 get_x[ idx ] = -beta

        #         max_value = np.dot( constraint, get_x ) + alpha * get_nu

        #         if max_value > cur_value:
        #             cur_x = get_x.copy()

        #         cur_value = np.maximum( cur_value, max_value )

        #     projected_binarization_increase = np.dot( -b, cur_x )
        #     # If we scale everybody, it assumes everyone can actually move more than they thought..
        #     # For now, we will be ok with this!
        #     scale_x = alpha / projected_binarization_increase
        #     cur_x *= scale_x

        #     move_design_vector = np.zeros( self.w[ 0 ].shape )
        #     for y_idx in range( 0, depth ):
        #         move_design_vector[ :, y_idx, 0 ] = cur_x

        #     proposed_design_variable = self.w[ 0 ] - move_design_vector

        # proposed_design_variable = np.maximum(
        #                             np.minimum(
        #                                 proposed_design_variable,
        #                                 self.maximum_design_value),
        #                             self.minimum_design_value)

        # design_variable_cut = self.w[ 0 ][ :, half_depth, 0 ]
        # print( "Ending binarization = " + str( compute_binarization( design_variable_cut ) ) )


        proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)
        #
        # Note: you are implicitly doing each layer separately.. you can all of them together or you should problem change the
        # amount the binarization has to change based on which layer you are on
        #
        if enforce_binarization_increase:
                def compute_binarization( input_variable ):
                    return np.mean( np.abs( input_variable - 0.5 ) ) / 0.5

                depth = self.w[ 0 ].shape[ 1 ]
                design_variable_cut = self.w[ 0 ][ :, 5, 0 ]
                design_variable_cut = np.real( design_variable_cut )
                gradient_cut = gradient[ :, 0, 0 ]
                gradient_cut = np.real( gradient_cut )

                # cur_design_variable = np.real( self.get_design_variable().flatten() )
                # flatten_gradient = np.real( gradient.flatten() )
                cur_binarization = compute_binarization( design_variable_cut )

                def minimization_fun( variable ):
                    return -np.dot( gradient_cut, variable )

                def minimization_grad( variable ):
                    return -gradient_cut

                def constraint( variable ):
                    binarization = compute_binarization( design_variable_cut - variable )
                    delta_binarization = depth * ( binarization - cur_binarization )
                    return delta_binarization

                gradient_cut_scaled = -max_design_change_point * gradient_cut / np.max( np.abs( gradient_cut ) )

                lower_bound = design_variable_cut.copy() - 1
                upper_bound = design_variable_cut.copy()

                for idx in range( 0, len( lower_bound ) ):
                    gradient_direction = -gradient_cut_scaled[ idx ]
                    if gradient_direction < 0:
                        lower_bound[ idx ] = np.maximum( lower_bound[ idx ], gradient_direction )
                        upper_bound[ idx ] = np.minimum( upper_bound[ idx ], max_design_change_point )
                    else:
                        lower_bound[ idx ] = np.maximum( lower_bound[ idx ], -max_design_change_point )
                        upper_bound[ idx ] = np.minimum( upper_bound[ idx ], gradient_direction )

                # lower_bound = np.maximum( design_variable_cut - 1, np.maximum( -gradient_cut_scaled, -max_design_change_point ) )
                # upper_bound = np.minimum( design_variable_cut, np.minimum( -gradient_cut_scaled, max_design_change_point ) )

                # lower_bound =  np.maximum( design_variable_cut - 1, -max_design_change_point * np.ones( design_variable_cut.shape ) )
                # upper_bound = np.minimum( design_variable_cut, max_design_change_point * np.ones( design_variable_cut.shape ) )

                make_bounds = scipy.optimize.Bounds( lower_bound, upper_bound )
                make_constraint = scipy.optimize.NonlinearConstraint( constraint, binarization_increase, np.inf )


                optimization_solution = scipy.optimize.minimize( minimization_fun, np.zeros( len( design_variable_cut ) ), jac=minimization_grad, bounds=make_bounds, constraints=[ make_constraint] )
                # print( minimization_fun( optimization_solution.x ) )
                # print( -np.dot( flatten_gradient, optimization_solution.x ))
                # print( -np.dot( step_size * flatten_gradient, flatten_gradient ))
                # print()

                if np.any( np.isnan( optimization_solution.x ) ):
                    print("NaN")

                # import matplotlib.pyplot as plt
                # print(np.real(self.w[0][:, 0, 0]))
                # print(self.w[0].shape)
                # plt.plot( optimization_solution.x.flatten() / np.max( np.abs( optimization_solution.x.flatten() ) ), linewidth=2, color='b' )
                # plt.plot( design_variable_cut, linewidth=2, color='c' )
                # plt.plot( lower_bound.flatten() / np.max( np.abs( optimization_solution.x.flatten() ) ), linewidth=2, linestyle='--', color='m' )
                # plt.plot( upper_bound.flatten() / np.max( np.abs( optimization_solution.x.flatten() ) ), linewidth=2, linestyle='--', color='k' )
                # plt.plot( step_size * gradient_cut / np.max( np.abs( step_size * gradient_cut ) ), linewidth=2, linestyle=':', color='g' )
                # plt.show()

                move_design_vector = np.zeros( self.w[ 0 ].shape )
                for y_idx in range( 0, depth ):
                    move_design_vector[ :, y_idx, 0 ] = optimization_solution.x


                # print( move_design_vector )
                # print( step_size * gradient )
                # print()

                proposed_design_variable = self.w[ 0 ] - move_design_vector

        proposed_design_variable = np.maximum(
                                    np.minimum(
                                        proposed_design_variable,
                                        self.maximum_design_value),
                                    self.minimum_design_value)

        return proposed_design_variable


    # In the step function, we should update the permittivity with update_permittivity
    def step(self, gradient_real, gradient_imag, step_size, do_simulated_annealing, current_temperature, enforce_binarization_increase, binarization_increase, max_design_change_point):
        self.w[0] = self.proposed_design_step(gradient_real, gradient_imag, step_size, do_simulated_annealing, current_temperature, enforce_binarization_increase, binarization_increase, max_design_change_point)
        # Update the variable stack including getting the permittivity at the w[-1] position
        self.update_permittivity()


    def convert_to_binary_map(self, variable):
        return np.greater(variable, self.mid_permittivity)
