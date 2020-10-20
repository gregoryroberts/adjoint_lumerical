import numpy as np
import kkr

def wavelength_um_to_wavenumber_cminv( wavelengths_um ):
	return ( 10000. / wavelengths_um )

def wavenumber_cminv_to_wavelength_um( wavenumbers_cminv ):
	return ( 10000. / wavenumbers_cminv )

def eps_lorentzian_imag( omega_nought, A, gamma, omega ):
	numerator = A * gamma**2 * omega_nought * omega
	denominator = ( omega_nought**2 - omega**2 )**2 + gamma**2 * omega**2

	return ( numerator / denominator )

def eps_gaussian_imag( omega_nought, A, gamma, omega ):
	one_over_f = 2. * np.sqrt( np.log( 2. ) )
	term1 = A * np.exp( -( ( omega - omega_nought ) * one_over_f / gamma )**2 )
	term2 = A * np.exp( -( ( omega + omega_nought ) * one_over_f / gamma )**2 )

	return ( term1 + term2 )

def index_from_permittivity( epsilon ):
	less_than_piece = 1.0 * np.less( np.imag( epsilon ), np.finfo(np.float64).eps )

	eps_r = np.real( epsilon )
	eps_i = np.imag( epsilon )

	k = np.sqrt( 0.5 * eps_r * ( -1 + np.sqrt( 1 + ( eps_i / eps_r )**2 ) ) )
	n = eps_i / ( 2 * k + np.finfo(np.float64).eps )

	return ( less_than_piece * np.sqrt( np.real( epsilon ) ) + ( 1.0 - less_than_piece ) * ( n + 1j * k ) )

#
# If you are doing this again, this can be a very easily generalized class of a dispersion model with these types of parameters.
#
class IPDipDispersion():

	def __init__( self ):
		self.A_lorentzian = [ 0.066, 0.144, 0.16, 0.038, 0.15, 0.18, 0.46, 0.5, 0.63, 0.22, 0.11, 0.0586, 0.0087, 0.021, 0.025, 0.0046, 0.0198 ]
		self.A_gaussian = [ 0.242, 0.056, 0.028, 0.15, 0.5, 0.12, 0.08, 0.136, 0.876, 0.041 ]

		self.omega_nought_lorentzian = [ 607, 753.4, 809.4, 830, 981.9, 1018, 1063, 1158, 1255, 1408.8, 1507.7, 1659, 2420, 2884, 2930, 3062, 3503 ]
		self.omega_nought_gaussian = [ 264, 701, 732.7, 1113, 1188, 1291, 1365, 1459.2, 1731.7, 2963 ]

		self.gamma_lorentzian = [ 174, 28, 8, 52, 22, 141, 21, 38, 39, 12, 7.9, 169, 2495, 48, 51, 190, 201 ]
		self.gamma_gaussian = [ 345, 8.9, 9, 32, 59, 24, 99, 44, 35.2, 40 ]

		self.eps_infinity = 2.37
		self.delta_eps_infinity = self.eps_infinity - 1.0

		self.compute_permittivity()

	def compute_permittivity( self ):

		min_omega = 0.1
		max_omega = np.maximum( np.max( self.omega_nought_gaussian + 10 * self.gamma_gaussian ), np.max( self.omega_nought_lorentzian + 10 * self.gamma_lorentzian ) )
		omega_resolution_cminv = 1

		self.num_omega = int( np.ceil( ( max_omega - min_omega ) / omega_resolution_cminv ) )

		self.omega_range_cminv = np.linspace( min_omega, max_omega, self.num_omega )
		self.lambda_range_um = wavenumber_cminv_to_wavelength_um( self.omega_range_cminv )

		self.eps_imag = np.zeros( self.num_omega, dtype=np.complex )

		for lorentzian_idx in range( 0, len( self.A_lorentzian ) ):
			self.eps_imag += eps_lorentzian_imag(
				self.omega_nought_lorentzian[ lorentzian_idx ], self.A_lorentzian[ lorentzian_idx ], self.gamma_lorentzian[ lorentzian_idx ], self.omega_range_cminv )

		for gaussian_idx in range( 0, len( self.A_gaussian ) ):
			self.eps_imag += eps_gaussian_imag(
				self.omega_nought_gaussian[ gaussian_idx ], self.A_gaussian[ gaussian_idx ], self.gamma_gaussian[ gaussian_idx ], self.omega_range_cminv )

		pack_imag_eps = np.zeros( ( self.num_omega, 3, 3 ), dtype=np.complex )
		for idx in range( 0, self.num_omega ):
			pack_imag_eps[ idx, 0, 0 ] = self.eps_imag[ idx ]
			pack_imag_eps[ idx, 1, 1 ] = self.eps_imag[ idx ]
			pack_imag_eps[ idx, 2, 2 ] = self.eps_imag[ idx ]

		self.eps_real = self.delta_eps_infinity + kkr.kkr( ( self.omega_range_cminv[ 1 ] - self.omega_range_cminv[ 0 ] ), pack_imag_eps )
		self.eps_real = np.real( np.squeeze( self.eps_real[ :, 0, 0 ] ) )

		self.full_eps = self.eps_real + 1j * self.eps_imag


	def average_permittivity( self, permittivity_range_um ):
		permittivity_range_cminv = wavelength_um_to_wavenumber_cminv( np.array( permittivity_range_um ) )

		lower_omega = np.min( permittivity_range_cminv )
		upper_omega = np.max( permittivity_range_cminv )

		find_arg_closest_lower = 0
		find_arg_closest_upper = self.num_omega - 1

		find_lower = np.greater_equal( self.omega_range_cminv - lower_omega, 0.0 )
		find_arg_closest_lower = 0
		while ( not find_lower[ find_arg_closest_lower ] ) and ( find_arg_closest_lower < self.num_omega ):
			find_arg_closest_lower += 1

		if not find_lower[ find_arg_closest_lower ]:
			raise ValueError( 'Invalid range specified to average permittivity' )
			return None

		find_upper = np.less_equal( upper_omega - self.omega_range_cminv, 0.0 )
		find_arg_closest_upper = self.num_omega - 1
		while ( find_upper[ find_arg_closest_upper] ) and ( find_arg_closest_upper >= 0 ):
			find_arg_closest_upper -= 1

		if find_upper[ find_arg_closest_upper ]:
			raise ValueError( 'Invalid range specified to average permittivity' )
			return None

		average_real_permittivity = np.mean( self.eps_real[ find_arg_closest_lower : find_arg_closest_upper ] )
		average_imag_permittivity = np.mean( self.eps_imag[ find_arg_closest_lower : find_arg_closest_upper ] )

		return ( average_real_permittivity + 1j * average_imag_permittivity )
