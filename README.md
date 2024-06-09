# Tcl-Vivado-init

This repo offers support to anyone who wants to start handling Xiinx Vivado project via Tcl scripts. This is very convenient since it allows to automatize the processes of, but not limited to, project initialization, syhnthesis, implementation, and bistream generation. 

Everything has been tested in Xilinx Vivado 2021.2.

## Installation

To clone this repo open your terminal and run:

`git clone https://github.com/MattiaDif/Tcl-Vivado.git`

## Description

All the .vhd code are just example files, they have not been tested and/or validated.

The /code folder includes:
- /src folder: example adder.vhd and top_level.vhd file
- /sim folder: example adder_tb.vhd file
- /xdc folder: constraint file of Digilent Arty A7
- init_prj.tcl --> Tcl scipt for Vivado project initialization
- bistream_prj.tcl --> Tcl script for bistream generation

# How to use

1) Open Xilinx Vivado
2) Open the Tcl console (usually found at the bottom of the Vivado window)
3) Locate the script using the command: `cd "path_to_your_script"`
4) Source the script using the command to initialize the project: `source "path_to_your_script/init_prj.tcl"`
5) Source the script using the command to generate bitstream: `source "path_to_your_script/bitstream_prj.tcl"`


## NOTES
To contribute refers to the dev branch! Thanks and see you around! :)

