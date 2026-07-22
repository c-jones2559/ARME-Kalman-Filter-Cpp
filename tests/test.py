import argparse
import subprocess
import time
import sys
import os

TESTS = {
    'optreal': {'cpp': '../src/RD_optreal_results_cpp.py', 'py': '../oldPythonFiles/RD_optreal_results.py'},
    'heatmap': {'cpp': '../src/RD_optreal_heatmap_plots_cpp.py', 'py': '../oldPythonFiles/RD_optreal_heatmap_plots.py'},
    'optimise': {'cpp': '../src/RD_optimise_cpp.py', 'py': '../oldPythonFiles/RD_optimise.py'},
    'optart': {'cpp': '../src/RD_optart_results_cpp.py', 'py': '../oldPythonFiles/RD_optart_results.py'}
}

def time_script(script_name, data_file=None):
    """Runs a script headlessly and returns the execution time."""
    custom_env = os.environ.copy()
    custom_env['MPLBACKEND'] = 'Agg'

    cmd = [sys.executable, script_name]

    if data_file and '_cpp.py' in script_name and ('optreal' in script_name or 'heatmap' in script_name):
        cmd.extend(['-f', data_file])

    start = time.perf_counter()
    subprocess.run(cmd, capture_output=True, env=custom_env)
    return time.perf_counter() - start

def run_benchmark(name, mode, data_file):
    print(f"\n>>> Running: {name}")
    times = {}

    if mode in ['cpp', 'both']:
        times['cpp'] = time_script(TESTS[name]['cpp'], data_file)
        print(f"C++ took {times['cpp']:.2f} seconds.")

    if mode in ['py', 'both']:
        times['py'] = time_script(TESTS[name]['py'], data_file)
        print(f"Python took {times['py']:.2f} seconds.")

    if mode == 'both':
        t1, t2 = times['cpp'], times['py']
        if t1 >= t2:
            print(f"Python is faster by {t1-t2:.2f} seconds. (took {((t1-t2)/t1)*100:.1f}% less time)")
        else:
            print(f"C++ is faster by {t2-t1:.2f} seconds. (took {((t2-t1)/t2)*100:.1f}% less time)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run benchmarking scripts.")
    parser.add_argument('-t', '--test', choices=list(TESTS.keys()) + ['all'], default='all', 
                        help="Specific test to run, or 'all'.")
    parser.add_argument('-m', '--mode', choices=['py', 'cpp', 'both'], default='both', 
                        help="Run Python only, C++ only, or compare both.")
    parser.add_argument('-f', '--file', type=str, default=None,
                        help="Specific dataset file to pass to the C++ real-data scripts.")
    
    args = parser.parse_args()

    print("Starting benchmark...\n" + "-"*40)
    start_total = time.perf_counter()
    
    if args.test == 'all':
        for test in TESTS:
            run_benchmark(test, args.mode, args.file)
    else:
        run_benchmark(args.test, args.mode, args.file)

    print("\n" + "-"*40 + f"\nAll done in {time.perf_counter() - start_total:.2f} seconds.")
