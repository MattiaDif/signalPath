from abc import ABC, abstractmethod
import h5py
from pathlib import Path
import numpy as np
import hdfdict as hdf
from grounds.custerrors import CustErr
from grounds.default import BasicFunc as bc
import warnings



#Parent class
class FileHandling(ABC):
    """
    Abstract class for files r/w
    """
    def __init__(self):
        pass


    @abstractmethod
    def _file_loading(self):
        """
        Load file 
        """
        pass


    @abstractmethod
    def file_write(self):
        """
        Write file
        """
        pass



#Child class h5 file
class h5(FileHandling):

    def __init__(self, filepath, mode):
        """
        - filepath: path of the folder that contains the .hdf5 file
        - mode: 'r', 'w', r+ (read/write), w- (write on created file), a (read/write and if file
          doesn't existe, create it) etc., string
        """
        super().__init__()
        self._filepath = filepath
        self._mode = mode
        self._file = self._file_loading()


    def __del__(self):
        print('h5py automatically closed the file')   


    # get template dictionary
    @property
    def get_dict(self):
        """
        get property to return extracted dictionary from h5/hdf5 file
        """
        print('h5/hdf5 dictionary gotten')

        return self._file


    def _file_loading(self):
        """
        Load .h5 ord hdf5 file
        Input:
        - filepath: path of the folder that contains the .hdf5 file
        - mode: 'r', 'w', r+ (read/write), w- (write on created file), a (read/write and if file
          doesn't existe, create it) etc., string
        Output:
        - f: the h5 file is converted into a dictionary-like container
        """
        path = Path(self._filepath)
        if (path.suffix in ['.h5', '.hdf5']): 
            f = h5py.File(self._filepath, self._mode)
            print(list(f.keys()))
        else:
            raise Exception('File must be an hdf5 file (.h5 or .hdf5) or file is empty')

        return f #dictionary-like container
    

    def file_write(self, my_dict):
        """
        Write a dictionary to .h5 or hdf5 file
        Input:
        - my_dict: dictionary to write
        """
        CustErr().type_error(my_dict, dict)
        hdf.dump(my_dict, self._file)



class txt(FileHandling):

    def __init__(self, filepath, mode):
        """
        - filepath: path of the folder that contains the .hdf5 file
        - mode: 'r', 'w', x (create file, returns error if already exists), a (read/write and if file
          doesn't existe, create it), + (read/write), t (text mode), b (binary mode), string
        """
        super().__init__()
        self._filepath = filepath
        self._mode = mode
        self._file = self._file_loading()

        
    def __del__(self):
        print('destructor txt class')
        self._file_close()


    def _file_loading(self):
        """
        Load .txt file
        Output:
        - f: txt file object
        """
        path = Path(self._filepath)
        if (path.suffix in ['.txt']): 
            f = open(self._filepath, self._mode)
        else:
            raise Exception('File must be a .txt file')
        
        return f    #txt file object


    def file_empty(self):
        """
        Empty .txt file
        Input: 
        - file: txt file object to truncate
        """
        self._file.truncate(0)

    
    def _file_close(self):
        """
        Close .txt file
        Input: 
        - file: file to close
        """
        self._file.close()
        print('TXT FILE CLOSED - B)')

    
    def file_write(self, w_mode, to_write):
        """
        Write string to .txt file, it empties the file before writing
        Input:
        - f: txt file object
        - mode: 's' (insert string in a single line), 'l' (for a list of string, each element is written)
        - to_write: string to write in txt
        """
        if w_mode == 's':
            self._file.write(to_write)
        elif w_mode == 'l':
            self._file.writelines(to_write)


    def write_numpy(self, data, fmt):
        """
        Write numpy array to .txt file, it empties the file before writing
        Input:
        - file: txt file or txt file object
        - data: numpy array to write element by element
        - fmt: format of element written, e.g. to write integer use '%d' 
          see docs about numpy.savetxt for more details
        """
        self.file_empty()
        print(data)
        np.savetxt(self._file, data, fmt=fmt, delimiter=',')
        print('numpy array written to txt')

    
    def read_numpy(self, dtype):
        """
        Read txt file line by line and store into numpy.ndarray
        Input:
        - file: txt file or txt file object
        - dtype: data-type of the resulting array
        Output:
        - data_read: numpy.ndarray 
        """
        data_read = np.loadtxt(self._file, dtype=dtype)
        print(data_read)
        print('numpy array read from txt')

        return data_read
    


class bin(FileHandling):
    
    def __init__(self, filepath, mode, n_byte=None):
        """
        - filepath: path of the folder that contains the .hdf5 file
        - mode: 'r', 'w', x (create file, returns error if already exists), a (read/write and if file
          doesn't existe, create it), + (read/write), t (text mode), b (binary mode), ab (append binary mode), string
        - n_byte: number of bytes to read at once, numeric
        """
        if n_byte == None:
            warnings.warn("bytes number not set")
        
        super().__init__()
        self._filepath = filepath
        self._mode     = mode
        self._n_byte   = n_byte
        self._file = self._file_loading()  



    def __del__(self):
        """
        Close .bin file
        """
        self._file_close()

    
    def _file_close(self):
        """
        Close .bin file
        Input: 
        - file: file to close
        """
        self._file.close()
        print('BIN FILE CLOSED - B)')
        

    def _file_loading(self):
        """
        Load .bin file and write into a list
        Output:
        - f: bin file object
        """
        path = Path(self._filepath)
        if (path.suffix in ['.bin']): #todo check .dat
            f = open(self._filepath, self._mode)                
        else:
            raise Exception('File must be a .bin file')
        return f
        

    def file_write(self, data): 
        """
        Write string to .bin file, it empties the file before writing
        Input:
        - data: data to write, byte
        """
        self._file.write(data)

    

    def read_numpy(self, MSB_pos, dtype=None):
        """
        Read bin file object and write into numpy.ndarray \n
        Input:
        - file: txt file or txt file object
        - dtype: data-type of the resulting array
        - MSB_pos: if MSB is at beginning (use "big") or at the end (use "little"), string \n
        Output:
        - data_read: numpy.ndarray 
        """
        basic = bc()

        data = self._file.read(self._n_byte)
        data_int = []
        
        while data:
            data_int.append(int.from_bytes(data, MSB_pos))  
            data = self._file.read(self._n_byte)

        data_read = np.array(data_int, dtype=dtype)

        return basic.by_column(data_read)     # numpy ndarray
        
