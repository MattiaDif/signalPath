import numpy as np
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
import warnings
from grounds.custerrors import CustErr



class NeuroSigProc(CustErr):
    """
    Class for neural signal processing
    """
    def __init__(self, data, fs):
        """
        Inputs:
        - data: data to process, numpy ndarray
        - fs: sampling frequency, numeric
        """
        super().__init__()
        CustErr().type_error(data, np.ndarray)
        self._data             = data
        self._fs               = fs
        self._rows, self._cols = data.shape



class SpikeD(NeuroSigProc):
    """
    Spike detection class
    """
    def __init__(self, data, fs):
        """
        Inputs:
        - data: data to process, numpy ndarray
        - fs: sampling frequency, numeric
        """
        super().__init__(data, fs)
        self._results = {}


    # get spike detection results
    @property
    def spike_results(self):
        """
        get property to return SpikeD class results
        """
        print('Spike detection results gotten')
        return self._results


    def hard_th(self, trace_polarity, threshold, interspike):
        """
        hard threshold spike detection algorithm

        Input:
        - trace_polarity: to invert or not the data (-1 to invert)
        - threshold: single numeric value or list/tuple of values above which the timestamp is identified
        - interspike: minimum distance in samples between two consecutive spikes, list/tuple of values
                      or single numeric value

        Output: 
        - hard_th: time position in samples of the spike detected, numpy ndarray
        - hard_th_train: spiketrain, numpy ndarray

        !!!
        organize the input numpy.ndarray by columns: rows are the samples to process, columns are the
        number of channels
        !!!
        """
        #CustErr().type_error(data, np.ndarray)
        #CustErr().numpy_dimension_error(data, 2, 1)

        if self._data.ndim == 1:
            cols = 1
            spiketrains = np.zeros((self._data.size, cols))
        elif self._data.ndim > 1 and self._data.ndim < 3:
            _, cols = self._data.shape
            spiketrains = np.zeros((self._data[:,0].size, cols))

        locks = []
        
        if trace_polarity == -1:
            if cols > 1:
                for element in range(0, cols, 1):
                    temp_locks, _ = find_peaks(-self._data[:, element] , height=threshold[element], distance=interspike[element])
                    locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                    spiketrains[temp_locks, element] = 1
            else:
                element = 0
                temp_locks, _ = find_peaks(-self._data[:, element] , height=threshold, distance=interspike)
                locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                spiketrains[temp_locks, element] = 1

        elif trace_polarity == 1:
            if cols > 1:
                for element in range(0, cols, 1):
                    temp_locks, _ = find_peaks(self._data[:, element] , height=threshold[element], distance=interspike[element])
                    locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                    spiketrains[temp_locks, element] = 1
            else:
                element = 0
                temp_locks, _ = find_peaks(self._data[:, element], height=threshold, distance=interspike)
                locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                spiketrains[temp_locks, element] = 1
        else:
            raise ValueError("trace_polarity can only take value 1 or -1 ")

        self._results['hard_th']       = locks
        self._results['hard_th_train'] = spiketrains




class LFP(NeuroSigProc):
    """
    LFP analysis class
    """

    def __init__(self, data, fs, data_intra=None, fs_intra=None):
        """
        Inputs:
        - data: data to process, numpy ndarray
        - fs: sampling frequency, numeric
        - data_intra: intracellular data, numpy ndarray
        - fs_intra: intracellular sampling frequency, numeric
        """
        super().__init__(data, fs)

        if not isinstance(data_intra, np.ndarray):     #todo improve this condition
            warnings.warn('data array is empty', RuntimeWarning)
        
        self._data_intra = data_intra
        self._fs_intra   = fs_intra
        self._results    = {}



    def _plot_hilbert(self, analytic_sign, env_sign, inst_phase):
        """
        Plot Hilbert transform 

        Inputs:
        - analytic_sign: analytic signal, numpy ndarray
        - env_sign: signal envelope, numpy ndarray
        - inst_phase: instantaneous phase, numpy ndarray
        """
        time_down = np.linspace(0, len(self._data[:,0])-1, len(self._data[:,0]))/self._fs

        if isinstance(self._data_intra, np.ndarray):     #todo improve this condition
            time      = np.linspace(0, len(self._data_intra[:,0])-1, len(self._data_intra[:,0]))/self._fs_intra
        
        fig = plt.figure()
        if self._cols > 4:
            plot_rows = 4
            plot_cols = self._cols//4
        else:
            plot_rows = 1
            plot_cols = 1

        gs  = fig.add_gridspec(plot_rows, plot_cols, hspace=0.15, wspace=0)

        k   = 0
        if not isinstance(self._data_intra, np.ndarray):     #todo improve this condition
            ax1 = None
        else:
            ax2 = None
        for i in range(0, plot_cols, 1):
            for j in range(0, plot_rows, 1):
                fig.suptitle('Hilbert transform results')

                #todo handle number of rows and cols in grid, find a way to compute them
                if not isinstance(self._data_intra, np.ndarray):     #todo improve this condition
                    sgs = gs[j, i].subgridspec(2, 1, hspace=0)
                    ax0 = fig.add_subplot(sgs[0], sharex=ax1)      
                else:
                    sgs = gs[j, i].subgridspec(3, 1, hspace=0)
                    ax0 = fig.add_subplot(sgs[0], sharex=ax2)   

                ax1 = fig.add_subplot(sgs[1], sharex=ax0)
                if isinstance(self._data_intra, np.ndarray):
                    ax2 = fig.add_subplot(sgs[2], sharex=ax0)   
                ax0.plot(time_down, self._data[:,k], label='signal')
                ax0.plot(time_down, env_sign[:,k], label='envelope')
                ax0.plot(time_down, np.imag(analytic_sign[:,k]), label='imag')
                if i == 0:
                    ax0.set_ylabel("Volts")

                ax1.plot(time_down, inst_phase[:,k]+180)
                if i == 0:
                    ax1.set_ylabel("deg")

                if isinstance(self._data_intra, np.ndarray):     #todo improve this condition
                #todo hide x and y values axis if not necessary + legend
                    ax2.plot(time, self._data_intra[:,0])
                    if i == 0:
                        ax2.set_ylabel("Volts")
                if j == 4:                    
                    ax2.set_xlabel("Time (s)")  #todo does not work
                k = k + 1
                

    # get LFP analysis results
    @property
    def lfp_results(self):
        """
        get property to return LFP class results
        """
        print('LFP analysis results gotten')
        return self._results
    


    def lfp_hilbert(self, axis=-1, plot_hilbert=False):
        """
        Hilbert transform computation

        Inputs:
        - axis: axis along which doing the transformation, numeric
        - plot_hilbert: if plot the hilbert transform or not, bool

        Output:
        - hilbert: analytic signal, numpy ndarray
        - hilbert_env: envelope, numpy ndarray
        - hilbert_phase: instantaneous phase, numpy ndarray
        """
        
        analytic_signal     = np.zeros((self._rows, self._cols), dtype=np.complex64)
        envelope            = np.zeros((self._rows, self._cols))
        instantaneous_phase = np.zeros((self._rows, self._cols))

        for i in range(0, self._cols, 1):
            analytic_signal[:,i]     = hilbert(self._data[:,i], axis=axis)
            envelope[:,i]            = np.abs(analytic_signal[:,i])
            #instantaneous_phase[:,i] = np.angle(analytic_signal[:,i], deg=True)
            #! from theory, it is imag/real not real/imag
            #instantaneous_phase[:,i] = np.rad2deg((np.arctan2(np.real(analytic_signal[:,i]), np.imag(analytic_signal[:,i]))))
            instantaneous_phase[:,i] = np.rad2deg((np.arctan2(np.imag(analytic_signal[:,i]), self._data[:,i])))

        if plot_hilbert:
            self._plot_hilbert(analytic_signal, envelope, instantaneous_phase)

        self._results['hilbert']       = analytic_signal
        self._results['hilbert_env']   = envelope
        self._results['hilbert_phase'] = instantaneous_phase

