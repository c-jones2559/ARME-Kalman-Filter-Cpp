# ARME Kalman Filter

## Overview
This repository consists of files from [this project](https://github.com/arme-project/ARME-Kalman-Filter-Model-Parameter-Tuning) which have been converted from Python to C++. Please see the original project for more details about the functionality of the files.

### Deviations from original project
Note that the files you execute are still in Python, but they will call back end functions in C++. This is so that all previous functionality with MatPlotLib graphs will be retained.
The original files would execute in Google Colab after you uploaded the `virtuoso.csv` file. This version is hard-coded to use the local `virtuoso.csv` file. This can be changed in the future.
If you do run the test script, it may appear that it has frozen but rest assured it is just taking a while. The pure Python scripts usually take over 20 minutes to execute. The C++ ones usually take less then one minute.
The original scripts were `.ipynb` files. My current implementation uses `.py` files instead. This makes them easier to execute, but less pretty. Please see the instructions for converting these files below if you wish to change format.


### Dependencies
- C++20 compatible compiler. (gcc, clang, etc.)
- CMake v3.14+
- Python v3.14+
- OpenBLAS
- NumCpp (Automatically installed.)
- PyBind11 (Automatically installed.)
- Jupytext (Optional)

### Instructions:
1. Ensure dependencies are installed.

2. Clone this git repo.
`git clone https://github.com/c-jones2559/arme-kalman-filter`

3. Create and navigate into the build folder:
`mkdir build && cd build`

4. Compile the project. This will automatically fetch NumCpp and PyBind11.
`cmake .. && make`

5. You can now call the C++ functions in a Python script by importing them. See the Python scripts in `src/` for info about how to do this.

If you want to run the test script: (Note the test script has the graphs disabled, but may still save files of the graphs.)
`cd ../tests`
`python test.py -t [optart, optimise, optreal, heatmap, all (default)] -m [py, cpp, both (default)]`

If you want to run an individual script: (This will open the graphs in a new window as they are created.)
`cd ../src`
`python [RD_optart_results_cpp.py, RD_optimise_cpp.py, RD_optreal_results_cpp.py, RD_optreal_heatmap_plots_cpp.py]`

If you want to convert a `.py` file to `.ipynb`:
`jupytext --to ipynb your_script.py`
If you want to convert an `.ipynb` file to `.py`:
`jupytext --to py your_notebook.ipynb

