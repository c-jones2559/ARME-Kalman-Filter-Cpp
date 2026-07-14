import ensemble_backend
import numpy as np

print("--- Testing ensemble_backend ---")

# ==========================================
# 1. Test Utility Functions (cov2cor / cor2cov)
# ==========================================
V = np.array([[1.0, 0.5], [0.5, 1.0]])
stdevs, V_cor = ensemble_backend.cov2cor(V)
V_back = ensemble_backend.cor2cov(stdevs, V_cor)

assert np.isclose(V[0, 1], V_back[0, 1])
print("✅ [OK] cov2cor & cor2cov conversions")

# ==========================================
# 2. Test Main Pipeline (estimate_ensemble)
# ==========================================
# Mocking up dummy data that matches the C++ expected shapes
N_steps = 20
players = ["1", "2"]
pairs = ["12", "21"]

# s and r are expected to be dicts of Nx1 NumPy arrays (column vectors)
s_dict = {p: np.random.rand(N_steps, 1) for p in players}
r_dict = {p: np.random.rand(N_steps, 1) for p in players}

# t and A are expected to be dicts of standard 1D lists/arrays
t_dict = {p: np.linspace(0, 10, N_steps).tolist() for p in players}
A_dict = {pair: np.random.rand(N_steps).tolist() for pair in pairs}

# Parameters
w = 5.0
Sigma_v_init = np.array([[0.1, 0.0], [0.0, 0.1]]) # 2x2 covariance matrix for 2 players
Sigma_w = 0.1
alpha_KF_init = 0.25
Sigma_alpha_init = 0.3
est_Sigma_v = False
w_KF = 5.0

print("⏳ Running estimate_ensemble (this executes in C++)...")
try:
    estimates = ensemble_backend.estimate_ensemble(
        s_dict,
        r_dict,
        A_dict,
        t_dict,
        w,
        Sigma_v_init,
        Sigma_w,
        alpha_KF_init,
        Sigma_alpha_init,
        est_Sigma_v,
        w_KF
    )
    
    # Verify bGLS outputs
    bGLS_alphas = estimates.bGLS.alpha
    assert isinstance(bGLS_alphas, dict), "bGLS alpha should be a dictionary"
    
    # Verify KF outputs
    kf_s_pred = estimates.KF.s_pred
    assert isinstance(kf_s_pred, dict), "KF s_pred should be a dictionary"
    
    print("✅ [OK] estimate_ensemble pipeline executed successfully")
    print(f"   -> bGLS Motor Noise (sigma_m): {estimates.bGLS.sigma_m:.4f}")
    print(f"   -> KF Predicted 's' shape for player 1: {kf_s_pred['1'].shape}")

except Exception as e:
    print(f"❌ [FAIL] estimate_ensemble threw a wobbly: {e}")

print("--- All tests finished! ---")
