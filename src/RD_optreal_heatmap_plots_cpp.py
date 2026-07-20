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
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="DwbA4GMd24Ly" outputId="3d67e254-761d-473c-e04f-143d3e0bb93d"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
from scipy.linalg import block_diag
from scipy.optimize import minimize

# import c++ functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ensemble_backend

# PLOS guideline safe
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# upload file
filename = '../virtuoso.csv'

def lag1_ac(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan

def compute_std_and_lag1(A_true, r_est):
    # compute mean std and lag-1 autocorr of async errors
    stds, lags = [], []
    for pair in A_true:
        pred = r_est[pair[0]] - r_est[pair[1]]
        true = A_true[pair]
        mask = ~np.isnan(pred) & ~np.isnan(true)
        if mask.sum() > 1:
            # resid = true[mask] - pred[mask]
            resid = pred[mask]
            stds.append(np.std(resid))
            lags.append(lag1_ac(resid))
    return np.mean(stds), np.mean(lags)

def run_model(params, r_data, s_data, A, w):
    # run KF with params dict, return r_est only
    s_hat, _, _ = ensemble_backend.KF_ensemble_2(
        s=s_data,
        A=A,
        Sigma_v_init=np.diag([params['sigma_v']] * len(s_data)),
        Sigma_w=params['sigma_w'],
        alpha_KF_init=[params['alpha_init']] * (len(s_data)*(len(s_data)-1)),
        Sigma_alpha_init=np.diag([params['sigma_alpha']] * (len(s_data)*(len(s_data)-1))),
        w=w
    )
    r_est = ensemble_backend.r_from_s_ensemble(s_hat, r_data, w)
    return r_est

# heatmap plots only
leaders = ['VN1', 'VN2']
reps = [1, 2, 5, 8, 11, 12]
conds = ['DP', 'NR', 'SP']
rep = 1
w = 5
weight = 1.0

# param ranges
sigma_v_values = np.logspace(1, 11, 60)
sigma_w_values = np.logspace(-6, 10, 60)

for leader in leaders:
    for rep in reps:

        r_dp, r_nr, r_sp, s_dp, s_nr, s_sp, A_dp, A_nr, A_sp, _, _, _ = ensemble_backend.process_ensemble_data(
            leader=leader,
            rep=rep,
            w=w
        )

        conditions_data = {
            'DP': {'r_data': r_dp, 's_data': s_dp, 'A': A_dp},
            'NR': {'r_data': r_nr, 's_data': s_nr, 'A': A_nr},
            'SP': {'r_data': r_sp, 's_data': s_sp, 'A': A_sp}
        }

        # fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=600)
        fig = plt.figure(figsize=(13.5, 4), dpi=600)

        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.45)

        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])

        for cond in conds:
            r_data = conditions_data[cond]['r_data']
            s_data = conditions_data[cond]['s_data']
            A = conditions_data[cond]['A']

        for ax, cond in zip(axes, conds):
            r_data = conditions_data[cond]['r_data']
            s_data = conditions_data[cond]['s_data']
            A = conditions_data[cond]['A']

            # Initialize the optimizer
            K = 4 
            kf_optimizer = ensemble_backend.KFOptimizer(s_data, A, r_data, s_data, K, w)

            # Calculate the entire grid in C++ with a single function call
            loss_grid = kf_optimizer.combined_loss_grid(sigma_w_values, sigma_v_values, 0.3, 0.25, weight)

            # plot combined loss heatmap
            S_W, S_V = np.meshgrid(sigma_w_values, sigma_v_values)
            h = ax.pcolormesh(S_W, S_V, loss_grid, shading='auto', cmap='viridis', rasterized=True)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\sigma^{(w)}$")
            ax.set_ylabel(r"$\sigma^{(v)}$")
            ax.set_title(cond)

        fig.suptitle(f'Leader {leader}, Repetition {rep}')
        cbar = fig.colorbar(h, cax=cax)
        cbar.set_label('Combined Loss')

        # show all heatmaps for VN1 rep 1
        if leader == 'VN1' and rep == 1:
            plt.show()

            # std & lag-1 autocorr. heatmaps
            std_map = np.zeros_like(loss_grid)
            lag_map = np.zeros_like(loss_grid)

            for i, sigma_v in enumerate(sigma_v_values):
                for j, sigma_w in enumerate(sigma_w_values):
                    params_test = {
                        'sigma_v': sigma_v,
                        'sigma_w': sigma_w,
                        'sigma_alpha': 0.3,
                        'alpha_init': 0.25
                    }
                    try:
                        r_est = run_model(params_test, r_data, s_data, A, w)
                        std_val, lag_val = compute_std_and_lag1(A, r_est)
                        std_map[i,j] = std_val
                        lag_map[i,j] = lag_val
                    except np.linalg.LinAlgError:
                        std_map[i,j] = np.nan
                        lag_map[i,j] = np.nan

            # plot std of asynchronies
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im1 = ax.pcolormesh(S_W, S_V, std_map, shading='auto', cmap='plasma')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\sigma^{(w)}$")
            ax.set_ylabel(r"$\sigma^{(v)}$")
            ax.set_title(f'Std of Asynchronies ({cond})')
            plt.colorbar(im1, ax=ax, label='Std')
            plt.show()

            # plot lag-1 autocorrelation of asynchronies
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im2 = ax.pcolormesh(S_W, S_V, lag_map, shading='auto', cmap='magma')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\sigma^{(w)}$")
            ax.set_ylabel(r"$\sigma^{(v)}$")
            ax.set_title(f'Lag-1 Autocorr. of Asynchronies ({cond})')
            plt.colorbar(im2, ax=ax, label='Lag-1 Corr')
            plt.show()
        else:
            pass

        fname = f'{leader}_Rep{rep}'
        plt.tight_layout()
        plt.savefig(fname + '.pdf')
        plt.close()





