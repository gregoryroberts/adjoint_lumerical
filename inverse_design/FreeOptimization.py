import numpy as np
import FreeBayerFilter2D
import OptimizationState

def permittivity_to_index( permittivity ):
	eps_real = np.real( permittivity )
	eps_imag = np.imag( permittivity )

	eps_mag = np.sqrt( eps_real**2 + eps_imag**2 )

	n = np.sqrt( ( eps_mag + eps_real ) / 2. )
	kappa = np.sqrt( ( eps_mag - eps_real ) / 2. )

	return ( n + 1j * kappa )

class FreeOptimization( OptimizationState.OptimizationState ):


    def __init__( self, num_iterations, num_epochs, step_size, device_size_um, permittivity_bounds, optimization_mesh_step_um, optimization_seed, filename_prefix ):
        super(FreeOptimization, self).__init__( num_iterations, num_epochs, filename_prefix )

        self.device_size_um = device_size_um
        self.optimization_mesh_step_um = optimization_mesh_step_um

        self.opt_device_size_voxels = [ 1 + int( self.device_size_um[ i ] / self.optimization_mesh_step_um ) for i in range( 0, len( self.device_size_um ) ) ]

        # self.opt_seed = np.zeros( self.opt_device_size_voxels )
        
        self.opt_seed = self.reinterpolate( optimization_seed, self.opt_device_size_voxels )

        self.bayer_filter = FreeBayerFilter2D.FreeBayerFilter2D( self.opt_device_size_voxels, permittivity_bounds, 0.0, self.opt_device_size_voxels[ 1 ] )
        self.bayer_filter.set_design_variable( self.opt_seed )

        self.step_size = 0.025

    def load_design( self, filebase, epoch ):
        load_design_variable = np.load( filebase + "/" + self.filename_prefix + str( epoch ) + ".npy" )
        self.bayer_filter.set_design_variable( load_design_variable )

    def save_design( self, filebase, epoch ):
        np.save( filebase + "/" + self.filename_prefix + str( epoch ) + ".npy", self.bayer_filter.get_design_variable() )

    def update_epoch( self, epoch ):
        self.bayer_filter.update_filters( epoch )

    def update( self, gradient_real, gradient_imag, lsf_gradient_real, lsf_gradient_imag, epoch, iteration ):
        gradient_real_interpolate = self.reinterpolate( np.squeeze( gradient_real ), self.opt_device_size_voxels )
        gradient_imag_interpolate = self.reinterpolate( np.squeeze( gradient_imag ), self.opt_device_size_voxels )

        backprop_grad = self.bayer_filter.backpropagate( gradient_real_interpolate, gradient_imag_interpolate )
        step_size = self.step_size / np.max( np.abs( backprop_grad ) )

        self.bayer_filter.step( gradient_real_interpolate, gradient_imag_interpolate, step_size )

    def assemble_index( self ):
        return permittivity_to_index( self.bayer_filter.get_permittivity() )
