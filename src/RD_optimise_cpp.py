# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3.13 (XPython)
#     language: python
#     name: xpython
# ---

# %%
# OPTIMISE ARTIFICIAL DATA
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# import c++ functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ensemble_backend

# choose opt method
opt_method = 'L-BFGS-B'  # options: 'L-BFGS-B', 'Nelder-Mead', etc... maybe try others?

# choose what to optimise
optimize_flags = {
    "sigma_w": True,
    "sigma_v": True,
    "sigma_alpha": True,
    "alpha_KF_init": True  
}

########## TEST CASES INTERPRETATION ##########
# perfect input: input ideal=True, static=True 
# dynamic alpha: ideal=True, dynamic=True
# realistic input: ideal=False, static=True
# hard case: ideal=False, dynamic=True
###############################################

# choose performance
ideal = False # (use s_true in run KF)
realistic = not(ideal) # (use s_win in run KF)

# choose static or dynamic alpha
static = False
dynamic = not(static)

# artificial data
N = 100
K = 4
w = 5
Tmkp = {str(k): [np.nan] + N * [500] for k in range(1, K + 1)} # set T_i,n to 500ms for all players 
# static alpha
if static:
    alpha_true = 0.25
    alpha = {f"{i}{j}": [np.nan] + N * [alpha_true] for i in range(1, K+1) for j in range(1, K+1) if i != j}
# dynamic alpha; sinusoidal time-varying alpha_true 
if dynamic:
    alpha = {}
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if i != j:
                # variation between 0.1 and 0.5 over time
                alpha[f"{i}{j}"] = [np.nan] + [
                    0.3 + 0.2 * np.sin(2 * np.pi * n / N) for n in range(N)
                ]
sigma_v_true = 40
sigma_v_dict = {str(k): sigma_v_true for k in range(1, K + 1)} 


s_true, s_win, r, A, t, T = ensemble_backend.generate_ensemble_data(N, Tmkp, alpha, sigma_v_dict, w)
players = [str(k) for k in range(1, K + 1)]
pairs = [str(i) + str(j) for i in players for j in players if i != j]
n_alpha = len(pairs)

# initial values
s_w_init = 1e-1
s_win_values = np.concatenate([s_win[p][~np.isnan(s_win[p])] for p in players])
# s_v_init = (np.nanstd(s_win_values))**2 # estimate sigma_v_init from s_win (std of observed s_win across all players)
s_v_init = 40
s_alpha_init = 0.3
alpha_init = 1 / K

# else values (if not being optimised); default in og code 
s_w_else = 1e-1
s_v_else = 40
s_alpha_else = 0.3
alpha_else = 1 / K

# initialise params and bounds
def get_initial_params(opt_flags, K, n_alpha):
    params = []
    bounds = []

    if opt_flags["sigma_w"]:
        params.append(s_w_init) 
        bounds.append((1e-7, 1e2))

    if opt_flags["sigma_v"]:
        params.append(s_v_init) 
        bounds.append((1, 1e6))

    if opt_flags["sigma_alpha"]:
        params.append(s_alpha_init) 
        bounds.append((1e-4, 1.0))

    if opt_flags["alpha_KF_init"]:
        params.append(alpha_init) 
        bounds.append((0.0, 1.0))

    return np.array(params), bounds # sigma_w, sigma_v, sigma_alpha_init, alpha_KF_init

# Initialise the C++ optimizer class ONCE before the optimize loop.
# It pre-loads s_win, A, and s_true so they never cross the bridge again.
s_input = s_win if realistic else s_true
kf_optimizer = ensemble_backend.KFOptimizer(s_input, A, s_true, K, w)

# Updated loss fct
def loss_function(params, opt_flags):
    idx = 0

    sigma_w = params[idx] if opt_flags["sigma_w"] else s_w_else
    idx += int(opt_flags["sigma_w"])

    sigma_v = params[idx] if opt_flags["sigma_v"] else s_v_else
    idx += int(opt_flags["sigma_v"])

    sigma_alpha = params[idx] if opt_flags["sigma_alpha"] else s_alpha_else
    idx += int(opt_flags["sigma_alpha"])

    alpha_KF_val = params[idx] if opt_flags["alpha_KF_init"] else alpha_else

    try:
        # Call the blazing fast C++ method directly. 
        # We are only passing 5 standard data types (floats/bools) across the bridge now!
        mse = kf_optimizer.loss(sigma_w, sigma_v, sigma_alpha, alpha_KF_val, ideal)
        return mse
        
    except Exception as e:
        print(f"Error during model eval: {e}")
        return 1e6

# optimisation step
initial_params, bounds = get_initial_params(optimize_flags, K, n_alpha)

# Only need the flags now!
args = (optimize_flags,)

result = minimize(
    fun=loss_function,
    x0=initial_params,
    args=args,
    bounds=bounds if opt_method != 'Nelder-Mead' else None,
    method=opt_method,
    options={"disp": True, "maxiter": 500}
)


# %%
# RESULTS
print("\n Success")
print("Final MSE:", result.fun)
idx = 0
if optimize_flags["sigma_w"]:
    print("sigma_w:", result.x[idx]); idx += 1
if optimize_flags["sigma_v"]:
    print("sigma_v:", result.x[idx]); idx += 1
    print("sigma_v_true:", sigma_v_true)
if optimize_flags["sigma_alpha"]:
    print("sigma_alpha:", result.x[idx]); idx += 1
if optimize_flags["alpha_KF_init"]:
    print("alpha_KF_init:", result.x[idx])
    if static:
        print("alpha_true:", alpha_true)
    else:
        pass

# %%
# S OPT VS S TRUE
# reconstruct optimal params
idx = 0
sigma_w = result.x[idx] if optimize_flags["sigma_w"] else 1e-2
idx += int(optimize_flags["sigma_w"])

sigma_v = result.x[idx] if optimize_flags["sigma_v"] else 50.0
Sigma_v = np.diag([sigma_v] * K)
idx += int(optimize_flags["sigma_v"])

sigma_alpha = result.x[idx] if optimize_flags["sigma_alpha"] else 0.3
Sigma_alpha = np.diag([sigma_alpha] * n_alpha)
idx += int(optimize_flags["sigma_alpha"])

alpha_KF_val = result.x[idx] if optimize_flags["alpha_KF_init"] else 1 / K
alpha_KF_init = np.array([alpha_KF_val] * n_alpha)

# run final KF; alpha_KF_predict & s_KF_predict
_, _, s_hat, _, alpha_hat, *_ = ensemble_backend.KF_ensemble(
    s=s_win,
    A=A,
    Sigma_v_init=Sigma_v,
    Sigma_w=sigma_w,
    alpha_KF_init=alpha_KF_init,
    Sigma_alpha_init=Sigma_alpha,
    w=w
)

# plot
fig, axs = plt.subplots(K, 1, figsize=(10, 2.5 * K), sharex=True)
fig.suptitle("Estimated s vs. True s", fontsize=16)

for i, player in enumerate(players):
    axs[i].plot(s_true[player], label=f"True $s_{{{player}}}$", linestyle='--', linewidth=2)
    axs[i].plot(s_hat[player], label=f"KF estimated $s_{{{player}}}$", linewidth=2)
    axs[i].legend()
    axs[i].set_ylabel("s")

axs[-1].set_xlabel("Time Step")
plt.tight_layout()
plt.show()

# METRICS
mets = ensemble_backend.metrics_ensemble(s_hat, s_true)
def print_mets(mets):
    print("Metrics:\n")
    for player, vals in mets.items():
        corr = vals['corr']
        std = np.sqrt(vals['std'])
        print(f"Player {player}:")
        print(f"  Correlation = {corr}")
        print(f"  Std. Dev. of Error = {std}")
        print()
print_mets(mets)

# %%
# HEATMAP
# grid for sigma_w and sigma_v
sigma_w_vals = np.logspace(-7, 4, 60)  # Log scale: 1e-7 to 1e4
sigma_v_vals = np.logspace(0, 8, 60)  # Log scale: 1 to 1e8
loss_surface = np.zeros((len(sigma_w_vals), len(sigma_v_vals)))

# fix other params
alpha_KF_val = 1 / K
alpha_KF_init = np.array([alpha_KF_val] * n_alpha)
Sigma_alpha = np.diag([0.3] * n_alpha)

# loss across grid
for i, sigma_w in enumerate(sigma_w_vals):
    for j, sigma_v in enumerate(sigma_v_vals):
        Sigma_v = np.diag([sigma_v] * K)
        try:
            _, _, s_hat, *_ = ensemble_backend.KF_ensemble(
                s=s_win,
                A=A,
                Sigma_v_init=Sigma_v,
                Sigma_w=sigma_w,
                alpha_KF_init=alpha_KF_init,
                Sigma_alpha_init=Sigma_alpha,
                w=w
            )

            s_hat_matrix = np.array([s_hat[p] for p in players])
            s_true_matrix = np.array([s_true[p] for p in players])
            valid_mask = ~np.isnan(s_hat_matrix) & ~np.isnan(s_true_matrix)

            if np.any(valid_mask):
                mse = np.mean((s_hat_matrix[valid_mask] - s_true_matrix[valid_mask]) ** 2)
                loss_surface[i, j] = mse
            else:
                loss_surface[i, j] = np.nan

        except Exception as e:
            loss_surface[i, j] = np.nan  # ignore failed runs

# plot heatmap
plt.figure(figsize=(10, 6))
contour = plt.contourf(sigma_w_vals, sigma_v_vals, loss_surface, levels=50, cmap="viridis")

plt.xscale('log')
plt.yscale('log')
plt.colorbar(contour, label="MSE")
plt.xlabel(r"$\sigma_w$ (process noise)")
plt.ylabel(r"$\sigma_v$ (observation noise)")
plt.title("Loss (MSE) ")
plt.tight_layout()
plt.show()

# %%
# ALPHA EST VS ALPHA TRUE
# plot alpha_true vs alpha_KF for specific player pairs
i = 1
j = 2

pair = f"{i}{j}"
alpha_true_vals = alpha[pair]
alpha_hat_vals = alpha_hat.get(pair, [np.nan] * (N + 1))


plt.plot(alpha_true_vals, label="True α", linestyle="--", linewidth=2)
plt.plot(alpha_hat_vals, label="Estimated α", linewidth=2)
plt.title(f"α_{i}→{j}")
plt.legend()
plt.ylim(0, 1)


plt.show()

# %%
