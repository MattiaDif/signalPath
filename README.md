```diff
!WARNING: THIS SOFTWARE IS UNDER DEVELOPMENT
!ALWAYS REFER TO THE *DEV* BRANCH
```
If you need any help or get a strange error while using this software, feel free to ask. It's still under development, I'm plannig to get a first release within the next couple of months (by end of August), documentation included. Every contribute is welcomed!

# signalPath
*signalPath* is a custom pipeline for designing neuroengineering algorithms. It collects various filters, spike detection algorithms, Hilbert transform, etc. to create a unified framework for neural signal processing.

The software is in its early stages of the development. Over time additional features will be included, such as tools for statistical analysis.

The goal is to create a simplified environment consisting of a series of ready-to-use functions collected in a file named `tool.py`.

## What's included
- txt, h5/hdf5, bin file handling
- Butterworth, Remez, IIR Hilbert filter
- Spike detection algorithms
- Hilbert transform
- Custom plot suite

## What's in progress
- Extensive unit testing
- Plot suite graphic improvements
- tool.py development (set of ready to use functions which includes all software funtionalities)
- Descritpive statistics class
- Software validation for Python > 3.8

## Installation
To clone this repo open your terminal and run:

`git clone https://github.com/MattiaDif/signalPath.git`

Then go into code/ folder and run:

`./prj_init.py`

to install required packages. It works for both Windows and Linux systems.


