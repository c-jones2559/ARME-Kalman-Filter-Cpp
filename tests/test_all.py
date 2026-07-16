import subprocess
import sys
import time

test_scripts = [
    'RD_optreal_results_compare.py',
    'RD_optreal_heatmap_plots_compare.py',
    'RD_optimise_compare.py',
    'RD_optart_results_compare.py'
]

print("Right, kicking off the benchmark suite...\n" + "-"*40)
start_total = time.perf_counter()

for script in test_scripts:
    print(f"\n>>> Running: {script}")
    subprocess.run([sys.executable, script])

total_time = time.perf_counter() - start_total
print("\n" + "-"*40)
print(f"All done in {total_time:.2f} seconds.")
