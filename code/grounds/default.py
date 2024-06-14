import numpy as np
from grounds.custerrors import CustErr
from scipy import interpolate



class BasicFunc(CustErr):
    """
    Class to provide basic methods inheritable by all classes
    """

    def __init__(self):
        super().__init__()


    def dict_to_ndarray(self, dictionary, key):
        """
        Convert dictionary key in numpy.ndarray \n
        Input:
        - f: dictionary
        - key: dictionary key \n
        Output:
        - data: converted dictionary key, numpy.ndarray
        """
        if dictionary[key].size > 1:
            var = np.array(list(dictionary[key]))
        else:
            var = np.array(dictionary[key])
        print('converted value to numpy.ndarray from dict')

        return var  #numpy.ndarray

    
    def concat_numpy(self, data, axis=-1):
        """
        Concatenate numpy ndarray \n
        Input:
        - data: data to be concatenated, list/tuple of numpy.ndarray
        - axis: axis along which to perforn concatenation \n
        Output:
        - data_conc: concatenated array, numpy.ndarray
        """
        if isinstance(data, list) or isinstance(data, tuple):
            if data[0].ndim == 1:
                data_conc = np.transpose(np.array(data[0], ndmin=2))
            else:
                data_conc = data[0]

            for i in range(1, len(data), 1):
                if data[i].ndim == 1:
                    data_temp = np.transpose(np.array(data[i], ndmin=2))
                else:
                    data_temp = data[i]
                data_conc = np.concatenate((data_conc, data_temp), axis=axis)
        else:
            raise TypeError('input data must be a numpy.ndarray list/tuple object')

        return data_conc
    

    def castUint16(self, data, step_size):
        """
        Cast input data to (data type object) np.uint16 \n
        Input:
        - data: data to cast, numpy.ndarray
        - step_size: value for data conversion based on the amplifier specifications, numeric
                     example (Intan RHS system): Voltage step size --> 0.195 µV = (2∗1.225)/((2^16-1)∗Gain) 
                     1.225 --> amplifier output limits, Gain --> amplifier gain, 2^16-1 --> ADC resolution \n
        Output:
        - cast_data: data coverted, numpy.ndarray
        """
        CustErr().type_error(data, np.ndarray)
        CustErr().range_error(data, 16, False)

        data_int16 = (data/step_size).astype(np.int16)

        if data.ndim == 1:
            data_cast = np.zeros((len(data)), dtype=np.uint16)
        else:
            rows, cols = data.shape
            data_cast = np.zeros((rows, cols), dtype=np.uint16)

        mask = int("0x8000", 16)
        for i, el in enumerate(data_int16):
            data_cast[i] = el^mask

        print(data)
        print(data_cast)
        print('uint16 cast done')
        
        return data_cast
    

    def reshape_numpy(self, data, n_cols):
        """
        Automatically reshape array. The function returns a matrix starting from an array \n
        Example: 
        array A of length 60 sample, n_rows = 3
        output --> matrix (numpy.ndarray) with dimensions (20, 3) in which matrix element
        (0,0) is the first element of A, (0,1) is the 2nd element of A, (0,2) is the 3rd 
        element of A, (1,0) is the 4th element of A, etc. \n
        Input:
        - data: numpy.ndarray to reshape
        - n_cols: number of rows of the reshaped matrix \n
        Output:
        - data_resh: reshaped data, numpy.ndarray
        """
        CustErr().type_error(data, np.ndarray)

        k = 0
        data_resh = np.zeros((int(len(data)/n_cols), n_cols))   #int truncates the value (move closer to 0)
        for i in range(0, int(len(data))-n_cols, n_cols):       #-n_cols is to ensure that every column has the same n° of elements    
            for j in range(0, n_cols, 1):
                data_resh[k, j] = data[i+j]
            k = k+1

        return data_resh


    def by_column(self, data):
        """
        Automatically organize array by columns by forcing 2 dimensions. \n
        Input:
        - data: numpy.ndarray to reshape \n
        Output:
        - data_dim: reorganized data, numpy.ndarray
        """
        CustErr().type_error(data, np.ndarray)
        CustErr().numpy_dimension_error(data, 2, 1)

        if data.ndim == 1:
            data_dim = np.transpose(np.array(data, ndmin=2))
            
            return data_dim
        else:
            shape     = data.shape
            if shape[0] < shape [1]:
                data_dim = np.transpose(data)
                
                return data_dim
                
            else:
                print('data already organized by columns')
                return data
        
    
    def downsampling(self, data, fs, fs_down):
        """
        Downsample data picking one sample each fs/fs_down samples \n
        Input:
        - data: array to downsample, numpy.ndarray
        - fs: original sampling frequency
        - fs_down: new sampling frequency, numeric \n
        Output:
        - data_down: data downsampled by a factor fs/fs_down
        - idx_down: indexes of element chosen during downsampling
        """
        CustErr().type_error(data, np.ndarray)
        down_coeff = fs//fs_down
        rows, cols = data.shape

        data_down = np.zeros((round(rows/down_coeff), cols))
        idx_down  = np.zeros((round(rows/down_coeff), cols))

        print(round(rows/down_coeff))

        for i in range(0, cols, 1):
            data_down[:,i] = np.array([el for idx, el in enumerate(data[:, i]) if idx%down_coeff == 0 and idx/down_coeff < round(rows/down_coeff)])
            idx_down[:,i]  = np.array([idx for idx, el in enumerate(data[:, i]) if idx%down_coeff == 0 and idx/down_coeff < round(rows/down_coeff)])
        #todo list comprehension to return two values

        return data_down, idx_down



    def interp1D(self, x, y, interp_vec):
        """
        1D linear interpolation \n
        Input:
        - x: x-axis vector (generally time) --> original x vector, numpy.ndarray
        - y: y-axis vector --> data to interpolate, numpy.ndarray
        - interp_vec: vector for the interpolation, numpy.ndarray \n
        Output:
        - data_inter: data downsampled by a factor fs/fs_down
        """
        CustErr().type_error(x, np.ndarray)
        CustErr().type_error(y, np.ndarray)
        CustErr().type_error(interp_vec, np.ndarray)
    
        _, cols    = y.shape
        data_inter = np.zeros((interp_vec.shape[0], cols))

        for i in range(0, cols, 1):
            f               = interpolate.interp1d(x[:,0], y[:,i], fill_value="extrapolate")
            data_inter[:,i] = f(interp_vec[:,0])
            
        return data_inter
        


