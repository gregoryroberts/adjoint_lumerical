import numpy as np
import FreeBayerFilter2D
import OptimizationStateMultiDevice

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )

class FreeOptimizationMultiDevice( OptimizationStateMultiDevice.OptimizationStateMultiDevice ):


    def __init__( self, num_iterations, num_epochs, step_size, device_size_um, permittivity_bounds, optimization_mesh_step_um, optimization_seed, bayer_filter_creator_fns, filename_prefix, num_devices ):
        super(FreeOptimizationMultiDevice, self).__init__( num_iterations, num_epochs, filename_prefix, num_devices )

        self.device_size_um = device_size_um
        self.optimization_mesh_step_um = optimization_mesh_step_um

        self.opt_device_size_voxels = [ 1 + int( self.device_size_um[ i ] / self.optimization_mesh_step_um ) for i in range( 0, len( self.device_size_um ) ) ]

        self.num_devices = num_devices

        self.opt_seed = self.reinterpolate( optimization_seed, self.opt_device_size_voxels )

        self.bayer_filter_creator_fns = bayer_filter_creator_fns

        self.bayer_filters = []
        for device_idx in range( 0, self.num_devices ):
            self.bayer_filters.append( self.bayer_filter_creator_fns[ device_idx ]( self.opt_device_size_voxels, self.opt_device_size_voxels[ 1 ] ) )
            self.bayer_filters[ device_idx ].set_design_variable( self.opt_seed )

        self.step_size = 0.025

    def load_design( self, filebase, epoch ):
        load_design_variable = np.load( filebase + "/" + self.filename_prefix + str( epoch ) + ".npy" )

        for device_idx in range( 0, self.num_devices ):
            self.bayer_filters[ device_idx ].set_design_variable( load_design_variable )

    def save_design( self, filebase, epoch ):
        np.save( filebase + "/" + self.filename_prefix + str( epoch ) + ".npy", self.bayer_filters[ 0 ].get_design_variable() )

    def update_epoch( self, epoch ):
        for device_idx in range( 0, self.num_devices ):
            self.bayer_filters[ device_idx ].update_filters( epoch )

    def update( self, gradients_real, gradients_imag, lsf_gradients_real, lsf_gradients_imag, epoch, iteration ):
        #
        # First, we need to know how to weight the contributions from each device.  We aren't doing this for the fabrication penalty because we need
        # to evaluate fabrication performance for each device.  But this code should replace the fabrication piece so we should probably take that
        # out of there.
        #
        performance_by_device = self.figure_of_merit[ epoch, iteration, : ]
        weighting_by_device = ( 2. / self.num_devices ) - performance_by_device**2 / np.sum( performance_by_device**2 )
        weighting_by_device = np.maximum( weighting_by_device, 0 )
        weighting_by_device /= np.sum( weighting_by_device )

        gradients_real_interpolate = []
        gradients_imag_interpolate = []

        lsf_gradients_real_interpolate = []
        lsf_gradients_imag_interpolate = []

        for device_idx in range( 0, self.num_devices ):
            gradients_real_interpolate.append(
                self.reinterpolate( np.squeeze( gradients_real[ device_idx ] ), self.opt_device_size_voxels )
            )
            gradients_imag_interpolate.append(
                self.reinterpolate( np.squeeze( gradients_imag[ device_idx ] ), self.opt_device_size_voxels )
            )

            lsf_gradients_real_interpolate.append(
                self.reinterpolate( np.squeeze( lsf_gradients_real[ device_idx ] ), self.opt_device_size_voxels )
            )
            lsf_gradients_imag_interpolate.append(
                self.reinterpolate( np.squeeze( lsf_gradients_imag[ device_idx ] ), self.opt_device_size_voxels )
            )


        combined_density_layer_gradient = np.zeros( self.bayer_filters[ 0 ].size )
        combined_lsf_layer_gradient = np.zeros( self.bayer_filters[ 0 ].size )

        for device_idx in range( 0, self.num_devices ):
            combined_density_layer_gradient += weighting_by_device[ device_idx ] * self.bayer_filters[ device_idx ].backpropagate(
                gradients_real_interpolate[ device_idx ],
                gradients_imag_interpolate[ device_idx ] )

            combined_lsf_layer_gradient += weighting_by_device[ device_idx ] * self.bayer_filters[ device_idx ].backpropagate(
                lsf_gradients_real_interpolate[ device_idx ],
                lsf_gradients_imag_interpolate[ device_idx ] )

        gradient_step = combined_density_layer_gradient
        lsf_gradient_step = combined_lsf_layer_gradient
        
        max_abs_gradient_step = np.max( np.abs( gradient_step ) )
        max_abs_lsf_gradient_step = np.max( np.abs( lsf_gradient_step ) )


        proposed_design_variable = self.bayer_filters[ 0 ].get_design_variable() - self.step_size * combined_density_layer_gradient / max_abs_gradient_step
        proposed_design_variable = np.minimum( np.maximum( proposed_design_variable, 0 ), 1 )

        for device_idx in range( 0, self.num_devices ):
            self.bayer_filters[ device_idx ].set_design_variable( proposed_design_variable )


    def assemble_index( self, device_idx ):
        return permittivity_to_index( self.bayer_filters[ device_idx ].get_permittivity() )
