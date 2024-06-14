import matplotlib.pyplot as plt
import numpy as np
from grounds.custerrors import CustErr


class Plots(CustErr):
    """
    Class for custom plot based on matplotlib
    """

    def __init__(self):
        super().__init__()


    def _check_style(self, style, n_el):
        """
        Check if plot style is None
        Input:
        - style: style of the plot (e.g. '-', '*', etc.), string
        - n_el: number of element to plot, numeric
        Output:
        - style: style of the plot, string
        """
        if style is None:
            style = ['-']*n_el
        elif isinstance(style, list) and (len(style) < n_el or len(style) > n_el):
            raise AttributeError('dimension mismatch between style and data, check if the number of style parameters correspond to the number of data to plot')

        return style    #string/list of string


    def _check_color(self, color, n_el):
        """
        Check if plot color is None
        Input:
        - color: color of the plot in RGB format
        - n_el: number of element to plot, numeric
        Output:
        - col: color of the plot, RG
        """
        if color is None:
            col = []
            for c in range(0, n_el, 1):
                col.append((np.random.random(), np.random.random(), np.random.random()))
        elif isinstance(color, list) and (len(color) < n_el or len(color) > n_el):
            raise AttributeError('dimension mismatch between color and data, check if the number of color parameters correspond to the number of data to plot')
        else:
            col = color
        return col  #RGB tuple/list of RGB tuple element
    
    

    def plt_subplt(self, data, t_axis, cols, rows, t_locks=[], x_label=None, y_label=None, color=None,
                    style=None, x_ticks=True, y_ticks=True, link_x=False, fontsz=12, legend=()):
        """
        plot with subplots. The function iteratively outputs a plot with a specific number of subplots
        based on cols and rows parameters. To output a superimposed plot suite, the function needs to get
        a list/tuple of numpy.ndarray for data. 
        !!!
        organize the input numpy.ndarray by columns: rows are the points to plot, columns are the elements
        (i.e. rows are the sample of the data, columns are the number of channels) --> N.B. the first element of the
        list/tuple must be on the same kind (i.e, first element: all raw data, second element: all filtered data, etc.)
        !!!
        Required inputs:
        - data: data to plot (y-axis), all the data must have same number of samples, 
                numpy.ndarray or list/tuple of numpy.ndarray
        - t_axis: x-axis, numpy.ndarray 1-dim
        - t_locks: list of index of data (and t_axis) to plot, list of lists of numpy.ndarray
        - cols: number of columns for subplots
        - rows: number of rows for subplots
        Optional inputs:
        - x_label: label of x-axis, string
        - y_label: label of y-axis, string
        - color: pass a list/tuple of list/tuple of normalized rgb values according to the number of data to plot, 
                 if not passed, random colors are generated *
        - style: plot style, for numpy.ndarray data pass a single value (i.e. style = ['-']), for list/tuple data 
                 pass a list/tuple of values (i.e. color=['-', '*']) according to the number of data to plot
        - x_ticks: to show or not x ticks, bool
        - y_ticks: to show or not y ticks, bool
        - link_x: to link x-axis, bool
        - fontsz: font size of the labels 
        - legend: legend of the plot, tuple of string

        *                                 R                   G                   B
            i.e. color_list.append((np.random.random(), np.random.random(), np.random.random())) with all the value between 0 and 1
        """
        n_elements = rows*cols
        if n_elements <= 0:
            raise ValueError('cols and/or rows for the subplot must be at least equal to 1')

        if isinstance(data, np.ndarray):
            CustErr().numpy_dimension_error(data, 2, 1)

            plt.figure()

            style = self._check_style(style, 1)
            col   = self._check_color(color, 1)

            for i in range(0, n_elements, 1):
                if link_x:
                    if i > 0:
                        plt.subplot(rows, cols, i+1, sharex=ax0)
                    else:
                        ax0 = plt.subplot(rows, cols, 1)
                else:
                    plt.subplot(rows, cols, i+1)

                plt.ylabel(y_label, fontsize=fontsz)
                plt.xlabel(x_label, fontsize=fontsz)
                plt.tick_params('x', labelbottom=x_ticks)
                plt.tick_params('y', labelleft=y_ticks)
                if t_locks == [] and n_elements > 1:
                    plt.plot(t_axis, data[:, i], style[0], color=col[0])
                elif t_locks != [] and n_elements > 1:
                    plt.plot(t_axis[t_locks[i]], data[t_locks[i], i], style[0], color=col[0])
                elif t_locks == [] and n_elements == 1:
                    plt.plot(t_axis, data, style[0], color=col[0])
                else:
                    plt.plot(t_axis[t_locks], data[t_locks], style[0], color=col[0])
            
        elif isinstance(data, list) or isinstance(data, tuple):

            plt.figure()

            n_iter = len(data)  #elements of list/tuple
            style  = self._check_style(style, n_iter)
            col    = self._check_color(color, n_iter)

            for k in range(0, n_iter, 1):
                CustErr().type_error(data[k], np.ndarray)
                CustErr().numpy_dimension_error(data[k], 2, 1)

            for j in range(0, n_iter, 1):
                data_temp = data[j]
                if t_locks != []:
                    t_locks_temp = t_locks[j]
                else:
                    t_locks_temp = []
                for i in range(0, n_elements, 1):
                    if link_x:
                        if i > 0:
                            plt.subplot(rows, cols, i+1, sharex=ax0)
                        else:
                            ax0 = plt.subplot(rows, cols, 1)
                    else:
                        plt.subplot(rows, cols, i+1)

                    plt.ylabel(y_label, fontsize=fontsz)
                    plt.xlabel(x_label, fontsize=fontsz)
                    plt.tick_params('x', labelbottom=x_ticks)
                    plt.tick_params('y', labelleft=y_ticks)
                    if t_locks_temp == [] and n_elements > 1:
                        plt.plot(t_axis, data_temp[:, i], style[j], color=col[j])
                    elif t_locks_temp != [] and n_elements > 1:
                        plt.plot(t_axis[t_locks_temp[i]], data_temp[t_locks_temp[i], i], style[j], color=col[j])
                    elif t_locks_temp == [] and n_elements == 1:
                        plt.plot(t_axis, data_temp, style[j], color=col[j])
                    else:
                        plt.plot(t_axis[t_locks_temp], data_temp[t_locks_temp], style[j], color=col[j])
                
                if legend != ():
                    for i in range(0, n_elements, 1):
                        plt.subplot(rows, cols, i+1)
                        plt.gca().legend(legend)

        else:
            raise TypeError('input data must be a numpy.ndarray or a numpy.ndarray list/tuple object')
            

        
    def plt_single(self, data, t_axis, t_locks=[], style=None, color=None, x_label=None, y_label=None, title=None,
                        x_ticks=True, y_ticks=True, fontsz=12, legend=()):
        """
        simple plot without subplots. The function outputs a single plot superimposing the arrays inside data.
        The function needs a numpy.ndarray for data.
        !!!
        organize the input numpy.ndarray by columns: rows are the points to plot, columns are the elements
        (i.e. rows are the sample of the data, columns are the number of channels)
        !!!
        Required inputs:
        - data: data to plot (y-axis), numpy.ndarray organized by columns
        - t_axis: x-axis, numpy.ndarray 1-dim
        Optional inputs:
        - t_locks: list of index of data (and t_axis) to plot
        - x_label: label of x-axis, string
        - y_label: label of y-axis, string
        - title: plot title, string
        - color: pass a list/tuple of list/tuple of normalized rgb values according to the number of data to plot, if not passed, random colors are generated *
        - style: plot style, for numpy.ndarray data pass a single value (i.e. style = ['-']), for list/tuple data 
                 pass a list/tuple of values (i.e. color=['-', '*']) according to the number of data to plot
        - x_ticks: to show or not x ticks
        - y_ticks: to show or not y ticks
        - fontsz: font size of the labels and titles
        - legend: legend of the plot, tuple of string

        *                                 R                   G                   B
            i.e. color_list.append((np.random.random(), np.random.random(), np.random.random())) with all the value between 0 and 1
        """
        #todo handle legend + title in all functions
        CustErr().type_error(data, np.ndarray)
        CustErr().numpy_dimension_error(data, 2, 1)

        plt.figure()

        if data.ndim == 1:
            cols = 1
        else:
            _, cols = data.shape

        style = self._check_style(style, cols)
        col   = self._check_color(color, cols)

        plt.title(title,fontsize=fontsz)
        plt.ylabel(y_label,fontsize=fontsz)
        plt.xlabel(x_label,fontsize=fontsz)
        plt.tick_params('x', labelbottom=x_ticks)
        plt.tick_params('y', labelleft=y_ticks)

        for i in range(0, cols, 1):
            if t_locks == [] and cols > 1:
                plt.plot(t_axis, data[:, i], style[i], color=col[i])
            elif t_locks != [] and cols > 1:
                plt.plot(t_axis[t_locks[i]], data[t_locks[i], i], style[i], color=col[i])
            elif t_locks == [] and cols == 1:
                plt.plot(t_axis, data, style[i], color=col[i])
            else:
                plt.plot(t_axis[t_locks], data[t_locks], style[i], color=col[i])
            
        if legend != ():
            plt.gca().legend(legend)



    def plt_hist(self, data, cols, rows, nbins=None, x_label=None, y_label=None, fontsz=12, legend=()):
        """
        histogram with subplots. The function iteratively outputs a histogram with a specific number of subplots
        based on cols and rows parameters.
        !!!
        organize the input numpy.ndarray by columns: rows are the points to plot, columns are the elements
        (i.e. rows are the sample of the data, columns are the number of channels) --> N.B. the first element of the
        list/tuple must be on the same kind (i.e, first element: all raw data, second element: all filtered data, etc.)
        !!!
        Required inputs:
        - data: data to plot (y-axis), numpy.ndarray
        nbins
        - cols: number of columns for subplots
        - rows: number of rows for subplots
        Optional inputs:
        - nbins: number of bins of the histogram, if not set, the default value is used
        - x_label: label of x-axis, string
        - y_label: label of y-axis, string
        - color: color: pass a list/tuple of list/tuple of normalized rgb values according to the number of data to plot, 
                 if not passed, random colors are generated *
        - fontsz: font size of the labels 
        - legend: legend of the plot, tuple of string

        *                                 R                   G                   B
            i.e. color_list.append((np.random.random(), np.random.random(), np.random.random())) with all the value between 0 and 1
        """

        if isinstance(data, np.ndarray):
            CustErr().type_error(data, np.ndarray)
            CustErr().numpy_dimension_error(data, 2, 1)

            plt.figure()
            plt.ylabel(y_label, fontsize=fontsz)
            plt.xlabel(x_label, fontsize=fontsz)
            plt.hist(data, bins=nbins)


        elif isinstance(data, list) or isinstance(data, tuple):

            plt.figure()
            n_iter = len(data)  #elements of list/tuple

            for k in range(0, n_iter, 1):
                CustErr().type_error(data[k], np.ndarray)
                CustErr().numpy_dimension_error(data[k], 2, 1)

            for i in range(0, n_iter, 1):
                    data_temp = data[i]
                    plt.subplot(rows, cols, i+1)
                    plt.ylabel(y_label, fontsize=fontsz)
                    plt.xlabel(x_label, fontsize=fontsz)
                    plt.hist(data_temp, bins=nbins)

        else:
            raise TypeError('input data must be a numpy.ndarray or a numpy.ndarray list/tuple object')
        


    def plt_subplt_nolocks(self, data, t_axis, cols, rows, x_label=None, y_label=None, color=None,
                    style=None, x_ticks=True, y_ticks=True, link_x=False, fontsz=12, legend=()):
        """
        plot with subplots without locks. The function iteratively outputs a plot with a specific number of subplots
        based on cols and rows parameters. To output a superimposed plot suite, the function needs to get
        a list/tuple of numpy.ndarray for data. 
        !!!
        organize the input numpy.ndarray by columns: rows are the points to plot, columns are the elements
        (i.e. rows are the sample of the data, columns are the number of channels) --> N.B. the first element of the
        list/tuple must be on the same kind (i.e, first element: all raw data, second element: all filtered data, etc.)
        !!!
        Required inputs:
        - data: data to plot (y-axis), list/tuple of numpy.ndarray
        - t_axis: x-axis support for each y-axis, list/tuple of numpy.ndarray
        - cols: number of columns for subplots
        - rows: number of rows for subplots
        Optional inputs:
        - x_label: label of x-axis, string
        - y_label: label of y-axis, string
        - color: pass a list/tuple of list/tuple of normalized rgb values according to the number of data to plot, 
                 if not passed, random colors are generated *
        - style: plot style, for numpy.ndarray data pass a single value (i.e. style = ['-']), for list/tuple data 
                 pass a list/tuple of values (i.e. color=['-', '*']) according to the number of data to plot
        - x_ticks: to show or not x ticks, bool
        - y_ticks: to show or not y ticks, bool
        - link_x: to link x-axis, bool
        - fontsz: font size of the labels 
        - legend: legend of the plot, tuple of string

        *                                 R                   G                   B
            i.e. color_list.append((np.random.random(), np.random.random(), np.random.random())) with all the value between 0 and 1
        """
        n_elements = rows*cols
          
        if isinstance(data, list) or isinstance(data, tuple):

            plt.figure()

            n_iter = len(data)  #elements of list/tuple
            style  = self._check_style(style, n_iter)
            col    = self._check_color(color, n_iter)

            for k in range(0, n_iter, 1):
                CustErr().type_error(data[k], np.ndarray)
                CustErr().numpy_dimension_error(data[k], 2, 1)

            for j in range(0, n_iter, 1):
                data_temp   = data[j]
                t_axis_temp = t_axis[j]
                for i in range(0, n_elements, 1):
                    if link_x:
                        if i > 0:
                            plt.subplot(rows, cols, i+1, sharex=ax0)
                        else:
                            ax0 = plt.subplot(rows, cols, 1)
                    else:
                        plt.subplot(rows, cols, i+1)
                        
                    plt.ylabel(y_label, fontsize=fontsz)
                    plt.xlabel(x_label, fontsize=fontsz)
                    plt.tick_params('x', labelbottom=x_ticks)
                    plt.tick_params('y', labelleft=y_ticks)
                    if n_elements > 1:
                        plt.plot(t_axis_temp[:,i], data_temp[:,i], style[j], color=col[j])
                    elif n_elements == 1:
                        plt.plot(t_axis_temp, data_temp, style[j], color=col[j])
                
                if legend != ():
                    for i in range(0, n_elements, 1):
                        plt.subplot(rows, cols, i+1)
                        plt.gca().legend(legend)

        else:
            raise TypeError('input data must be a numpy.ndarray or a numpy.ndarray list/tuple object')
            