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

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="CD-zIOndpaiq" outputId="48e4e019-24b9-481f-816e-60bbd31819ec"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# import c++ functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ensemble_backend

# run KF with init params (not optimised)
def run_kf_with_initial_params(s_win, s_true, A, K, n_alpha, w, players):
    Sigma_v = np.diag([40.0] * K)
    Sigma_alpha = np.diag([0.3] * n_alpha)
    alpha_KF_init = np.array([1 / K] * n_alpha)
    sigma_w = 1e-1

    _, _, s_hat, *_ = ensemble_backend.KF_ensemble(
        s=s_win,
        A=A,
        Sigma_v_init=Sigma_v,
        Sigma_w=sigma_w,
        alpha_KF_init=alpha_KF_init,
        Sigma_alpha_init=Sigma_alpha,
        w=w
    )
    return ensemble_backend.metrics_ensemble(s_hat, s_true)


# single expt
def run_single_experiment(K, N, sigma_v_true, alpha_type, ideal, optimize_flags, w, plot=True, optimize=True):
    players = [str(k) for k in range(1, K + 1)]
    pairs = [str(i) + str(j) for i in players for j in players if i != j]
    n_alpha = len(pairs)

    if alpha_type == "static":
        alpha_val = 0.25
        alpha = {f"{i}{j}": [np.nan] + N * [alpha_val] for i in players for j in players if i != j}
    elif alpha_type == "dynamic":
        alpha = {f"{i}{j}": [np.nan] + [0.3 + 0.2 * np.sin(2 * np.pi * n / N) for n in range(N)]
                 for i in players for j in players if i != j}

    # generate data
    Tmkp = {k: [np.nan] + N * [500] for k in players}
    sigma_v_dict = {k: sigma_v_true for k in players}
    s_true, s_win, r, A, t, T = ensemble_backend.generate_ensemble_data(N, Tmkp, alpha, sigma_v_dict, w)
    s_input = s_true if ideal else s_win

    s_win_values = np.concatenate([s_win[p][~np.isnan(s_win[p])] for p in players])
    s_v_init = np.nanstd(s_win_values)
    s_alpha_init = 0.3
    alpha_init = 1 / K
    s_w_init = 1e-1

    def get_initial_params():
        params = []
        bounds = []
        if optimize_flags["sigma_w"]:
            params.append(s_w_init)
            bounds.append((1e-4, 1e2))
        if optimize_flags["sigma_v"]:
            params.append(s_v_init)
            bounds.append((1.0, 1e3))
        if optimize_flags["sigma_alpha"]:
            params.append(s_alpha_init)
            bounds.append((1e-3, 0.5))
        if optimize_flags["alpha_KF_init"]:
            params.append(alpha_init)
            bounds.append((0.0, 1.0))
        return np.array(params), bounds

    def loss_function(params):
        idx = 0
        try:
            sigma_w = params[idx] if optimize_flags["sigma_w"] else 1e-1
            idx += int(optimize_flags["sigma_w"])
            sigma_v = params[idx] if optimize_flags["sigma_v"] else 40.0
            Sigma_v = np.diag([sigma_v] * K)
            idx += int(optimize_flags["sigma_v"])
            sigma_alpha = params[idx] if optimize_flags["sigma_alpha"] else 0.3
            Sigma_alpha = np.diag([sigma_alpha] * n_alpha)
            idx += int(optimize_flags["sigma_alpha"])
            alpha_KF_val = params[idx] if optimize_flags["alpha_KF_init"] else 1 / K
            alpha_KF_init = np.array([alpha_KF_val] * n_alpha)

            _, _, s_hat, *_ = ensemble_backend.KF_ensemble(
                s=s_input,
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
            if not np.any(valid_mask):
                return 1e6
            mse = np.mean((s_hat_matrix[valid_mask] - s_true_matrix[valid_mask]) ** 2)
            return mse
        except Exception as e:
            print(f"Loss function error: {e}")
            return 1e6

    if optimize:
        initial_params, bounds = get_initial_params()
        result = minimize(
            fun=loss_function,
            x0=initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={"disp": False, "maxiter": 300}
        )

        # re-run KF with opt
        idx = 0
        sigma_w = result.x[idx] if optimize_flags["sigma_w"] else 1e-1
        idx += int(optimize_flags["sigma_w"])
        sigma_v = result.x[idx] if optimize_flags["sigma_v"] else 40.0
        Sigma_v = np.diag([sigma_v] * K)
        idx += int(optimize_flags["sigma_v"])
        sigma_alpha = result.x[idx] if optimize_flags["sigma_alpha"] else 0.3
        Sigma_alpha = np.diag([sigma_alpha] * n_alpha)
        idx += int(optimize_flags["sigma_alpha"])
        alpha_KF_val = result.x[idx] if optimize_flags["alpha_KF_init"] else 1 / K
        alpha_KF_init = np.array([alpha_KF_val] * n_alpha)
    else:
        # use og
        sigma_w = 1e-1
        Sigma_v = np.diag([40.0] * K)
        Sigma_alpha = np.diag([0.3] * n_alpha)
        alpha_KF_init = np.array([1 / K] * n_alpha)

    # run KF
    _, _, s_hat, _, alpha_hat, *_ = ensemble_backend.KF_ensemble(
        s=s_win,
        A=A,
        Sigma_v_init=Sigma_v,
        Sigma_w=sigma_w,
        alpha_KF_init=alpha_KF_init,
        Sigma_alpha_init=Sigma_alpha,
        w=w
    )


    # run bGLS for alpha and s timeseries
    # o_rm = np.array([s_true[p] for p in players]).T   # onset matrix
    alpha_bgls, _, _ = ensemble_backend.bGLS_ensemble(np.array(list(t.values())).T[1:, ])
    s_bgls = ensemble_backend.s_from_bGLS_ensemble(alpha_bgls, A)

    # plot s true vs s KF vs s bGLS
    if plot:
        met = pd.concat([pd.DataFrame(ensemble_backend.metrics_ensemble(s_hat, s_true)), pd.DataFrame(ensemble_backend.metrics_ensemble(s_bgls, s_true))], axis = 0)
        met.index = ['KF corr', 'KF var', 'bGLS corr', 'bGLS var']
        fig, axs = plt.subplots(K, 1, figsize=(10, 2.5 * K), sharex=True)
        fig.suptitle("Estimated s vs. True s", fontsize=16)
        for i, player in enumerate(players):
            axs[i].plot(s_true[player], color = 'blueviolet', label=f"True $s_{{{player}}}$", linestyle='--', linewidth=2)
            axs[i].plot(s_hat[player], color = 'deepskyblue', label=f"KF estimated $s_{{{player}}}$", linewidth=2)
            axs[i].plot(s_bgls[player], color = 'crimson', label=f"bGLS estimated $s_{{{player}}}$", linewidth=2)
            axs[i].legend()
            axs[i].set_ylabel("s")
        axs[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.show()

        # plot trajectories
        fig, axs = plt.subplots(len(pairs), 1, figsize=(10, 2.5 * len(pairs)), sharex=True)
        fig.suptitle("Alpha Timeseries", fontsize=16)
        for idx, pair in enumerate(pairs):
            axs[idx].plot(alpha[pair], label=f"True α{pair}", color="blueviolet", linestyle="--")
            axs[idx].plot(alpha_hat[pair], label=f"KF α{pair}", color="green")
            axs[idx].hlines(alpha_bgls[pair], xmin=0, xmax=N, color="crimson", linestyle=":", label="bGLS α")
            axs[idx].legend(loc="upper right")
            axs[idx].set_ylabel(f"α{pair}")
        axs[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.show()

        print(met)

    return ensemble_backend.metrics_ensemble(s_hat, s_true)

# run multiple expts & average metrics
def run_multiple_experiments(n_runs, config, label="", plot_first=True):
    players = [str(k) for k in range(1, config['K'] + 1)]
    all_metrics_opt = []
    all_metrics_init = []

    for i in range(n_runs):
        print(f"Running {label} | Iter {i+1}/{n_runs}")
        plot = (i == 0 and plot_first)

        m_opt = run_single_experiment(**config, plot=plot, optimize=True)
        all_metrics_opt.append(m_opt)

        m_init = run_single_experiment(**config, plot=False, optimize=False)
        all_metrics_init.append(m_init)

    def avg_metrics(metric_list):
        result = {}
        for p in players:
            corr_vals = [m[p]['corr'] for m in metric_list]
            std_vals = [m[p]['std'] for m in metric_list]
            result[p] = {
                'corr': np.mean(corr_vals),
                'std': np.mean(std_vals)
            }
        return result

    avg_opt = avg_metrics(all_metrics_opt)
    avg_init = avg_metrics(all_metrics_init)

    data = []
    for p in players:
        corr_opt = avg_opt[p]['corr']
        std_opt = avg_opt[p]['std']
        corr_init = avg_init[p]['corr']
        std_init = avg_init[p]['std']

        corr_improvement = 100 * (corr_opt - corr_init) / abs(corr_init)
        std_improvement = 100 * (std_init - std_opt) / abs(std_init)

        data.append([
            p,
            round(corr_opt, 3),
            round(std_opt, 3),
            round(corr_init, 3),
            round(std_init, 3),
            round(corr_improvement, 1),
            round(std_improvement, 1)
        ])

    df = pd.DataFrame(data, columns=[
        "Player",
        "Corr (Optimised)", "Std (Optimised)",
        "Corr (Original)", "Std (Original)",
        "% improv. Corr", "% improv. Std"
    ])

    print(f"\n Average Metrics: {label}\n")
    print(df.to_string(index=False))
    return df


# run examples
optimize_flags = {
    "sigma_w": True,
    "sigma_v": True,
    "sigma_alpha": True,
    "alpha_KF_init": True
}

sigma_v_true = 100
configs = [
    {"K": 4, "N": 100, "sigma_v_true": sigma_v_true, "alpha_type": "static", "ideal": True, "optimize_flags": optimize_flags, "w": 5},
    {"K": 4, "N": 100, "sigma_v_true": sigma_v_true, "alpha_type": "static", "ideal": False, "optimize_flags": optimize_flags, "w": 5},
    {"K": 4, "N": 100, "sigma_v_true": sigma_v_true, "alpha_type": "dynamic", "ideal": True, "optimize_flags": optimize_flags, "w": 5},
    {"K": 4, "N": 100, "sigma_v_true": sigma_v_true, "alpha_type": "dynamic", "ideal": False, "optimize_flags": optimize_flags, "w": 5},
]

titles = [
    "Static Alpha; Ideal Input",
    "Static Alpha; Realistic Input",
    "Dynamic Alpha; Ideal Input",
    "Dynamic Alpha; Realistic Input"
]

all_results = []

# change no. of iterations - KEPT AT 2 FOR TESTING
for cfg, title in zip(configs, titles):
    df = run_multiple_experiments(n_runs=2, config=cfg, label=title)  # ← Change n_runs here
    all_results.append((title, df))

