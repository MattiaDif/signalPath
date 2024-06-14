import warnings
import numpy as np

class CustErr:
    """
    Class for custom errors/exceptions management
    """

    def __init__(self):
        pass



    def type_error(self, var, var_type):
        """
        Custom ype error check \n
        Input:
        - var: variable to check
        - var_type: object type \n
        Output:
        - error bool, True/False
        """
        if not isinstance(var, var_type):
            raise TypeError('Expected a ' + str(var_type))
        
        return isinstance(var, var_type)


    def numpy_dimension_error(self, ndarray, max, min):
        """
        Numpy array dimension error check \n
        Input:
        - ndarray: numpy.ndarray for dimension check
        - max: max allowed dimension
        - min: min allowed dimension
        """
        if max < min:
            raise ValueError('max value must be greater than min value')
        elif ndarray.ndim > max:
            raise AttributeError('Accepted max ' + str(max) + '-dimensional numpy.nddarray')
        elif ndarray.ndim < min:
            raise AttributeError('Accepted min ' + str(min) + '-dimensional numpy.nddarray')



    def range_error(self, data, bits, signed):
        """
        Check potential overflow/underflow during cast \n
        Input:
        - data: data to cast, numpy.ndarray of 1 or 2 dimensions
        - bits: number of bits on which the cast is performed. numeric
        - signed: if the cast is returning a signed or unsigned value, bool
        """
        if data.ndim == 1:
            max_data = max(data)
            min_data = min(data)
        elif data.ndim == 2:
            max_data = max(np.amax(data, axis=1))
            min_data = min(np.amin(data, axis=1))

        shift = 2**(bits-1)
        if signed == True:
            if max_data > (2**(bits-1))-1:
                warnings.warn("Probably overflow occurred")
            elif min_data < -(2**bits):
                warnings.warn("Probably underflow occurred")
        elif signed == False:
            if max_data+shift > (2**bits)-1:
                warnings.warn("Probably overflow occurred")
            elif min_data+shift < 0:
                warnings.warn("Probably underflow occurred")
