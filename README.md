```diff
! WARNING! THE SOFTWARE IS UNDER DEVELOPMENT
```
If you need any help during or get a strange error during or while using this software, feel free to ask for help. It's under development so it'll break! Every contribute is welcomed! :)

**ALWAYS REFER TO THE *DEV* BRANCH TO FORK OR WHATEVER!**

After downloading the software, be sure to have the following Python packages installed (preferably in a virtual environment with Python 3.8):
- scipy
- numpy
- matplotlib
- h5py
- warnings
- MEArec
- webbrowser

<br />


# neuroWare
A project which aspires to be a tool to simplify the development process of digital systems. The goal is to develop specific functionalities in order to get a clear comparison between the outputs of the digital implementation (for microcontroller, FPGA, etc.) and of the exact replica designed in Python. Doing this, a consistent validation of the architecture implementation is gotten.

Initially, this software aims to be very specific for the neuroengineering field. Indeed, spike detection algorithms, LFP analysis algorithms, IIR filters, etc. will be implemented. However, nothing prevents extending the capabilities making the tool exploitable for different applications. 

The software is fully integrated with [MEArec](https://github.com/alejoe91/MEArec.git), to generate synthetic biophysical extracellular neural recording on Multi-Electrode Arrays for testing.

To give you an example to clarify what I mean, I show an image below:
<br />
<br />

<img src="https://github.com/MattiaDif/neuroWare/blob/main/img/filtering.png" width="800">

<p>
    <b>Fig.1 - Filtering comparison</b></figcaption>
</p>

<br />


The picture shows a comparison between a filtering output implemented in Python (available in the neuroWare software) and the output of the exact replica implemented in VHDL.

So, what's the point? I make you an example: do you need a IIR filter to be implemented on FPGA? Cool, desing it in HDL, implement the same filter in Python, give to both implementations the same input testbench (in Fig. 1, MEArec data have been used) and compare the two outputs, that's all. I'm doing this to validate my VHDL architecture. The point is to have software support.

<br />
