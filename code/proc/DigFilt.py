from abc import ABC, abstractmethod
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from postproc.DataAnalysis import Plots as pt
import warnings
import matplotlib.pyplot as plt




class DigFiltDesign(ABC):
    """
    Abstract class for digital filter design
    """
    def __init__(self, fs, data=None):
        """
        Inputs:
        - fs: sampling frequency, numeric
        Optional inputs:
        - data: data to filter, numpy.ndarray
        """
        if not isinstance(data, np.ndarray):     #todo improve this condition
            warnings.warn('data array is empty or not a numpy.ndarray', RuntimeWarning)
        else:
            self._rows, self._cols = data.shape
        
        self._data             = data
        self._fs               = fs
        self._results          = {}



    @abstractmethod
    def get_coeff(self):
        pass


    @abstractmethod
    def get_fresponse(self):
        pass

    
    @abstractmethod
    def filt(self):
        pass


    @property
    def filt_results(self):
        """
        get digital filtering results
        """
        print('digital filtering results gotten')
        return self._results


    def _plot_fresp(self, w, h, title_add='', plt_cutf = True):
        """
        Plot filter frequency response
        """
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response ' + title_add)
        ax1.plot(w, 20 * np.log10(abs(h)), 'orange')
        ax1.set_ylabel('Amplitude [dB]', color='orange')
        ax1.set_xlabel('Frequency [Hz]')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h, deg=False)) 
        ax2.plot(w, angles, 'turquoise')
        ax2.set_ylabel('Angle (rad)', color='turquoise')
        ax2.grid(True, color='grey', linewidth=0.5)
        ax2.axis('tight')
        if plt_cutf == True:
            if isinstance(self._cutf, list):
                plt.axvline(self._cutf[0], linestyle='--', color='lemonchiffon', linewidth=0.5)  # low cutoff frequency
                plt.axvline(self._cutf[1], linestyle='--', color='lemonchiffon', linewidth=0.5)  # high cutoff frequency
            else:
                plt.axvline(self._cutf, linestyle='--', color='lemonchiffon', linewidth=0.5)  # cutoff frequency
        


    def welch_psd(self, window='hamming', nperseg=128, noverlap=None, nfft=None, 
                return_onesided=True, axis=- 1, average='mean'):
        """
        Compute PSD with Welch method
        Optional input:
        - window: window type (e.g. 'hamming', 'boxcar', 'triangle', etc.), string or tuple or array-like
        - nperseg: segment length, numeric
        - noverlap: overlap between segments, if None --> nperseg // 2 numeric
        - nfft: number of FFT points, if None --> = nperseg, numeric
        - return_onesided: if True return one-sided spectrum, bool
        - axis: axis along which performing spectral density, numeric
        - average: method for averaging periodogram ('mean', 'median'), string
        Output:
        - Pxx: PSD, numpy ndarray
        - freq_axis: frequency axis, numpy ndarray

        ** see scipy.signal.welch for more details
        """             
        if nfft==None:
            Pxx                        = np.zeros((nperseg//2+1, self._cols))
            self._results['freq_axis'] = np.linspace(0, self._fs//2-1, nperseg//2+1)
        else:
            Pxx                        = np.zeros((nfft//2+1, self._cols))
            self._results['freq_axis'] = np.linspace(0, self._fs//2-1, nfft//2+1)
        #** if you do a=b=0 to initialize a variable --> they have the same mapping in memory, so they are the same

        for i in range(0, self._cols, 1):
            _, Pxx[:,i] = signal.welch(self._data[:,i], fs=self._fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                    return_onesided=return_onesided, axis=axis, average=average)
            
        self._results['Pxx'] = Pxx



class Butterworth(DigFiltDesign):
    """
    IIR Butterworth filter designer
    
    """
    def __init__(self, fs, N, cutf, data=None, ftype='highpass', fresp=True):
        """
        inputs:
        - fs: sampling frequency
        - N: filter order, numeric
        - cutf: cutoff frequency, numeric or list in case of bandpass-like filters, numeric or list depending on
          the filter type
        Optional inputs:
        - data: data to filter, numpy.ndarray
        - ftype: type of filter, string
        - fresp: if plot magnitude and phase, bool
        """
        super().__init__(fs, data)
        self._N          = N
        self._cutf       = cutf
        self._ftype      = ftype
        self._b, self._a = signal.butter(N=self._N, Wn=self._cutf, btype=self._ftype, fs=self._fs)
        self._w, self._h = signal.freqz(b=self._b, a=self._a, worN=self._fs*8, fs=self._fs)
        if fresp: 
            self._plot_fresp(self._w, self._h)


    # get property get coeff implementation
    def get_coeff(self):
        """
        get property to return filter coefficient
        """
        print('filter coeff gotten')
        print("b array: ")
        for i in range(0, len(self._b)):
            print(self._b[i])
        print("a array: ")
        for i in range(0, len(self._a)):
            print(self._a[i])

        return self._b, self._a
        

    # get property fresponse implementation
    def get_fresponse(self):
        """
        get property to return frequency response magnitude and frequency
        """
        print('freq response gotten')

        return self._w, self._h
    

    def filt(self, axis=-1):
        """
        Direct-form II transposed filter implementation
        Input:
        - axis: axis of the input data array along which to apply the linear filter, nmumeric

        Output: 
        - data_filt: filtered data, numpy ndarray
        """
        data_filt = np.zeros((self._rows, self._cols))

        for i in range(0, self._cols, 1):
            data_filt[:,i] = signal.lfilter(self._b, self._a, self._data[:,i], axis=axis)

        self._results['data_filt'] = data_filt


    def filtfilt(self, axis=-1, padtype='odd', padlen=None, padmet="pad"):
        """
        Zero-phase digital filtering
        Input:
        - axis: axis of the input data array along which to apply the linear filter, nmumeric
        - padtype: type of padding, string, allowed input:'odd', 'even', 'constant', numeric
        - padlen: padding length, if None padlen=3*max(len(a), len(b)), numeric
        - padmet: padding methodm string, allowed input: "pad", "gust" (Gustafsson’s method), numeric

        Output:
        - data_filt: filtered data, numpy ndarray
        """
        if padlen is None:
            padlen = 3*max(len(self._a), len(self._b))

        data_filt = np.zeros((self._rows, self._cols))

        for i in range(0, self._cols, 1):
            data_filt[:,i] = signal.filtfilt(self._b, self._a, x=self._data[:,i], axis=axis, padtype=padtype, 
                                                                                padlen=padlen, method=padmet)
            
        self._results['data_filt_filt'] = data_filt



class FIRRemez(DigFiltDesign):
    """
    FIR filter designer using the Remez exchange algorithm
    
    """
    def __init__(self, fs, numtaps, bands, gain, maxiter=25, data=None, ftype='bandpass', fresp=True):
        """
        inputs:
        - fs: sampling frequency in Hz, numeric 
        - numtaps: number of taps in the filter, numeric
        - bands: band edges, list
        - gain: desired gain for each band, list
        Optional inputs:
        - maxiter: maximum number of iteration
        - data: data to filter, numpy.ndarray
        - ftype: type of filter, string
        - fresp: if plot magnitude and phase, bool
        """
        super().__init__(fs, data)
        self._numtaps    = numtaps
        self._cutf       = bands
        self._gain       = gain
        self._maxiter    = maxiter
        self._ftype      = ftype

        self._b = signal.remez(numtaps=self._numtaps, bands=self._cutf, desired=self._gain,
                                            type=self._ftype, fs=self._fs, maxiter=self._maxiter)
        self._a = 1
        
        self._w, self._h = signal.freqz(b=self._b, a=self._a, worN=self._fs*8, fs=self._fs)
        if fresp: 
            self._plot_fresp(self._w, self._h)


    # get property get coeff implementation
    def get_coeff(self):
        """
        get property to return filter coefficient
        """
        print('filter coeff gotten')
        print("b array: ")
        for i in range(0, len(self._b)):
            print(self._b[i])
        print("a array: ")
        for i in range(0, 1):
            print(self._a)

        return self._b, self._a
        

    # get property fresponse implementation
    def get_fresponse(self):
        """
        get property to return frequency response magnitude and frequency
        """
        print('freq response gotten')

        return self._w, self._h
    

    def filt(self, axis=-1):
        """
        Direct-form II transposed filter implementation
        Input:
        - axis: axis of the input data array along which to apply the linear filter, nmumeric

        Output: 
        - data_filt: filtered data, numpy ndarray
        """
        data_filt = np.zeros((self._rows, self._cols))

        for i in range(0, self._cols, 1):
            data_filt[:,i] = signal.lfilter(self._b, self._a, self._data[:,i], axis=axis)

        self._results['data_filt'] = data_filt



class IIRHilbert(DigFiltDesign):
    """
    IIR HIlbert filter designer (from Simulink Hilber filter (r2021b))

    There is also an implementation taken from this paper (it is commented):
    ref --> "An Infinite Impulse Response (IIR) Hilbert Transformer Filter Design Technique for Audio"
            Harris et al., 2010
    """

    def __init__(self, fs, coeff_img, coeff_real, data=None):
        """
        Inputs:
        - fs: sampling frequency, numeric
        - coeff: filter coefficients, list or tuple
        Optional inputs:
        - data: data to filter, numpy.ndarray
        - fresp: if plot magnnitude and phase or not
        """
        super().__init__(fs, data)
        self._coeff_img  = coeff_img
        self._coeff_real = coeff_real
        self._nstage     = len(coeff_img)
        
        hi_tmp = 1
        hi     = 1
        for i in range(0, len(self._coeff_img), 1):
            w, hi_tmp  = hi_tmp*signal.freqz([0, self._coeff_img[i]],  1, worN=self._fs*8, fs=self._fs)
            hi         = hi_tmp*hi

        hr_tmp = 1
        hr     = 1
        for i in range(0, len(self._coeff_real), 1):
            _, hr_tmp  = hr_tmp*signal.freqz([0, self._coeff_real[i]],  1, worN=self._fs*8, fs=self._fs)
            hr         = hr_tmp*hr
        
        self._hr = hr
        self._hi = hi
        self._w  = w

        """
        #todo check fresp
        if fresp: 
            self._plot_fresp(self._w, self._hr, plt_cutf=False)
            self._plot_fresp(self._w, self._hi, plt_cutf=False)
            angles_r = np.unwrap(np.angle(-self._hr, deg=False))
            angles_i = np.unwrap(np.angle(self._hi, deg=False))

            fig, ax0 = plt.subplots()
            ax0.plot(w, 180*(angles_r-angles_i)/np.pi, 'g')
            ax0.set_ylabel('Angle (deg)', color='g')
            ax0.grid(True)
            ax0.axis('tight')
        """
            


    # get property get coeff implementation
    def get_coeff(self):
        """
        get property to return filter coefficient
        """
        print('filter coeff gotten')
        print("A0 array: ")
        print(self._coeff_real)
        print("A1 array: ")
        print(self._coeff_img)

        return self._coeff_real, self._coeff_img
        

    # get property fresponse implementation
    def get_fresponse(self):
        #todo
        pass
    

    def filt(self):
        """
        Minimum multiplier filter implementation (see Hilbert Filter block, Simulink r2021b)

        Output: 
        - data_filt: filtered data, numpy ndarray
        """
        data_filt = np.zeros((self._rows, self._cols))

        for k in range(0, self._cols, 1):
            if (len(self._coeff_img))%2 == 0:
                data_tmp = -self._data[:,k]
            else:
                data_tmp = self._data[:,k]
            
            data_filt[:,k] = self._sos_stage(data_tmp)

        self._results['data_filt'] = data_filt   



    def _sos_stage(self, input, sos=0):
        """
        Second order section of the minimum multiplier filter architecture (recursive method)
        
        Input:
        - input: data to filter, numpy ndarray
        Optional input:
        - sos: current stage (section) of the filtering, numeric

        Output:
        - out: filtered data, numpy ndarray
        """
        out    = np.zeros((self._rows))
        in_raw = np.insert(input, 0, [0, 0])
        print(in_raw.shape)

        for i in range(2, self._rows, 1):
            a        = in_raw[i] - out[i-2]
            if i < 10:
                print("a: " + str(a))
            out[i] = in_raw[i-2] + (a * self._coeff_img[sos])

        if sos == self._nstage - 1:
            print("tot stages: " + str(sos+1))
            return out
        else:
            return self._sos_stage(out, sos=sos+1)
        