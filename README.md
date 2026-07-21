# ARME Kalman Filter C++

## Overview
This repository consists of files from [this project](https://github.com/arme-project/ARME-Kalman-Filter-Model-Parameter-Tuning) which have been converted from Python to C++. Please see the original project for more details about the functionality of the files.

Note that the files you execute are still in Python, but they will call back end functions in C++. This is so that all previous functionality with MatPlotLib graphs will be retained.

### Deviations from original project
- The original files would execute in Google Colab after you uploaded the `virtuoso.csv` file, but this meant you couldn't run the files locally. This version is hard-coded to use the local `virtuoso.csv` file. This can be changed in the future.
- When you run the test script, it may appear frozen but it is just taking a while. The pure Python scripts usually take over 20 minutes to execute. The C++ ones usually take less than one minute.
- The original scripts were `.ipynb` files. My current implementation uses `.py` files instead. This makes them easier to execute, but less pretty. Please see the instructions for converting these files below if you wish to change format.

### Project structure
```text
├──  oldPythonFiles                       # Files from the old repo. Used in the test script for comparison.
│   ├──  'Ensemble Functions.py'
│   ├──  RD_optart_results.py
│   ├──  RD_optimise.py
│   ├──  RD_optreal_heatmap_plots.py
│   └──  RD_optreal_results.py
├── 󰣞 src
│   ├──  bindings.cpp                     # The bridge which handles format conversions.
│   ├──  EnsembleFunctions.cpp            # The core logic functions.
│   ├──  EnsembleFunctions.hpp            # Header for the core functions.
│   ├──  rapidcsv.h                       # Lightweight csv parser.
│   ├──  RD_optart_results_cpp.py         # One of the original files, but now calls C++.
│   ├──  RD_optimise_cpp.py               # One of the original files, but now calls C++.
│   ├──  RD_optreal_heatmap_plots_cpp.py  # One of the original files, but now calls C++.
│   └──  RD_optreal_results_cpp.py        # One of the original files, but now calls C++.
├──  tests
│   ├──  test.py                          # Testing suite to compare Python vs C++.
├──  CMakeLists.txt                       # Compilation instructions.
├── 󰂺 README.md                            # The file you're reading now!
└──  virtuoso.csv                         # Sample dataset.
```


## Instructions

### Dependencies
- C++20 compatible compiler. (gcc, clang, etc.)
- CMake v3.14+
- Python v3.10+
- OpenBLAS
- NumCpp (Automatically installed.)
- PyBind11 (Automatically installed.)
- Jupytext (Optional.)

### Building
1. Ensure dependencies are installed.

2. Clone this git repo.
```bash
git clone https://github.com/c-jones2559/arme-kalman-filter-cpp
```

3. Create and navigate into the build folder:
```bash
mkdir build && cd build
```

4. Compile the project. This will automatically fetch NumCpp and PyBind11.
```bash
cmake .. && make
```

5. You can now call the C++ functions in a Python script by importing them. See the Python scripts in `src/` for examples on how I did this.

### Test script
If you want to run the test script:
*(Note: The test script has graphs disabled, but may still save graph files to disk.)*

The script accepts two optional parameters:
*   `--test` (`-t`): Specifies which file to test.
    *   Options: `optart`, `optimise`, `optreal`, `heatmap`, `all` (default)
*   `--mode` (`-m`): Specifies which language implementation to test.
    *   Options: `py`, `cpp`, `both` (default)

```bash
cd ../tests
python test.py -t [test] -m [mode]
```

Example output:
```text
Starting benchmark...
----------------------------------------

>>> Running: optreal
C++ took 10.84 seconds.
Python took 589.93 seconds.
C++ is faster by 579.09 seconds. (took 98.2% less time)

>>> Running: heatmap
C++ took 34.43 seconds.
Python took 588.23 seconds.
C++ is faster by 553.80 seconds. (took 94.1% less time)

>>> Running: optimise
C++ took 3.02 seconds.
Python took 23.71 seconds.
C++ is faster by 20.69 seconds. (took 87.3% less time)

>>> Running: optart
C++ took 3.86 seconds.
Python took 7.91 seconds.
C++ is faster by 4.05 seconds. (took 51.3% less time)

----------------------------------------
All done in 1261.92 seconds.
```

### Individual scripts
If you want to run an individual script: (This will open the graphs in new windows as they are created.)
```bash
cd ../src
python [RD_optart_results_cpp.py, RD_optimise_cpp.py, RD_optreal_results_cpp.py, RD_optreal_heatmap_plots_cpp.py]
```

### Conversions
If you want to convert a `.py` file to `.ipynb`:
```bash
jupytext --to ipynb your_script.py
```
If you want to convert an `.ipynb` file to `.py`:
```bash
jupytext --to py your_notebook.ipynb
```

