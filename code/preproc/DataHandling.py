from pprint import pprint
import webbrowser as wb
import matplotlib.pylab as plt
import numpy as np
from scipy.signal import find_peaks
from grounds.custerrors import CustErr


#data extraction class
class DataExtr(CustErr):
    """
    Class to extract features of neural data
    Currently:
    - spike timestamps extraction from spike traces (timestamps)
    - samples above threshold extraction (over_th)
    """

    # base constructor
    def __init__(self, data):
        """
        Inputs:
        - data: data to process, numpy ndarray
        """
        super().__init__()
        CustErr().type_error(data, np.ndarray)
        self._data    = data
        self._results = {}



    # get DataExtr results
    @property
    def dataextr_results(self):
        """
        get property to return DataExtr results
        """
        print('Spike detection results gotten')
        return self._results
    

    def timestamps(self, spike_traces, trace_polarity, threshold, interspike):
        """
        Timestamps extraction and spiketrains generation from the spike traces of the signal
        Input:
        - spike_traces: numpy.ndarray of traces of spikes 
        - trace_polarity: to invert or not the spike traces signal (-1 to invert)
        - threshold: single numeric value or list/tuple of values above which the timestamp is identified
        - interspike: minimum distance in samples between two consecutive spikes, list/tuple of values
                      or single numeric value
        !!!
        organize the input numpy.ndarray by columns: rows are the samples to process, columns are the
        number of channels
        !!!
        Output:
        - locks: nested list of numpy.ndarray about the timestamps of the spike
        - spiketrains: numpy.ndarray of the generated spiketrains 
        """

        if spike_traces.ndim == 1:
            cols = 1
            spiketrains = np.zeros((spike_traces.size, cols))
        elif spike_traces.ndim > 1 and spike_traces.ndim < 3:
            _, cols = spike_traces.shape
            spiketrains = np.zeros((spike_traces[:,0].size, cols))

        locks = []
        
        if trace_polarity == -1:
            if cols > 1:
                for element in range(0, cols, 1):
                    temp_locks, _ = find_peaks(-spike_traces[:, element] , height=threshold[element], distance=interspike[element])
                    locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                    spiketrains[temp_locks, element] = 1
            else:
                temp_locks, _ = find_peaks(-spike_traces , height=threshold, distance=interspike)
                locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                spiketrains[temp_locks] = 1
        elif trace_polarity == 1:
            if cols > 1:
                for element in range(0, cols, 1):
                    temp_locks, _ = find_peaks(spike_traces[:, element] , height=threshold[element], distance=interspike[element])
                    locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                    spiketrains[temp_locks, element] = 1
            else:
                temp_locks, _ = find_peaks(spike_traces, height=threshold, distance=interspike)
                locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
                spiketrains[temp_locks] = 1
        else:
            raise ValueError("trace_polarity can only take value 1 or -1 ")
        print('timestamps computed')

        self._results['timestamps']       = locks
        self._results['timestamps_train'] = spiketrains
    


    def over_th(self, trace_polarity, threshold):
        """
        detect samples above a threshold
        Input:
        - trace_polarity: to invert or not the data (-1 to invert)
        - threshold: single numeric value or list/tuple of values above which the timestamp is identified

        !!!
        organize the input numpy.ndarray by columns: rows are the samples to process, columns are the
        number of channels
        !!!
        Output:
        - locks: nested list of numpy.ndarray about the position of the samples above/below threshold
        """

        if self._data.ndim == 1:
            cols = 1
        elif self._data.ndim > 1 and self._data.ndim < 3:
            _, cols = self._data.shape

        locks = []
        
        if trace_polarity == -1:
            if cols > 1:
                for element in range(0, cols, 1):
                    temp_locks = [idx for idx, el in enumerate(-self._data[:, element]) if el >= threshold]
                    locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
            else:
                temp_locks = [idx for idx, el in enumerate(-self._data) if el >= threshold]
                locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
        elif trace_polarity == 1:
            if cols > 1:
                for element in range(0, cols, 1):
                    temp_locks = [idx for idx, el in enumerate(self._data[:, element]) if el >= threshold]
                    locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
            else:
                temp_locks = [idx for idx, el in enumerate(self._data) if el >= threshold]
                locks.append(np.transpose(np.array(temp_locks, ndmin=2)))
        else:
            raise ValueError("trace_polarity can only take value 1 or -1 ")
        print('thresholding computed')

        self._results['over_th'] = locks
    







