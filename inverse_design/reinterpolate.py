import scipy.signal

def reinterpolate( input_array, output_shape ):
    input_shape = input_array.shape

    assert len( input_shape ) == len( output_shape ), "Reinterpolate: expected the input and output to have same number of dimensions"

    output_array = input_array.copy()

    for axis_idx in range( 0, len( input_shape ) ):
        output_array = scipy.signal.resample( output_array, output_shape[ axis_idx ], axis=axis_idx )

    return output_array    
