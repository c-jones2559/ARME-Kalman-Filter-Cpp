import subprocess
import time
import sys
import os

def time_script(script_name):
    # Copy your current environment and slap the headless Matplotlib rule in
    custom_env = os.environ.copy()
    custom_env['MPLBACKEND'] = 'Agg'
    
    start = time.perf_counter()
    # Pass the custom environment into the subprocess
    subprocess.run([sys.executable, script_name], capture_output=True, env=custom_env)
    end = time.perf_counter()
    
    return end - start

time1 = time_script('../src/RD_optreal_heatmap_plots_cpp.py')
print(f"C++ took {time1:.2f} seconds.")

time2 = time_script('../oldPythonFiles/RD_optreal_heatmap_plots.py')
print(f"Python took {time2:.2f} seconds.")

if time1 >= time2:
    print(f"Python is faster by {time1-time2:.2f} seconds. (took {((time1-time2)/time1)*100:.1f}% less time)")
else:
    print(f"C++ is faster by {time2-time1:.2f} seconds. (took {((time2-time1)/time2)*100:.1f}% less time)")
