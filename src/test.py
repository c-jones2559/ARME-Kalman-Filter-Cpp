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
# 2. Test bGLS_ensemble
# ==========================================
# bGLS expects an onset matrix (Time x Players)
# Let's create a mock dataset: 50 time steps, 4 players
N_steps = 50
K = 4
# Create a dummy onset matrix (Time x Players)
# We add a bit of structure so the GLS doesn't just return zeros
o_rm = np.linspace(0, 100, N_steps).reshape(-1, 1) + np.random.normal(0, 0.1, (N_steps, K))

print("⏳ Running bGLS_ensemble...")
try:
    # bGLS returns: (alphas_dict, sigma_m, sigma_t)
    alphas, sm, st = ensemble_backend.bGLS_ensemble(o_rm)
    
    assert isinstance(alphas, dict), "bGLS should return a dictionary of alphas"
    assert len(alphas) == K * (K - 1), "Incorrect number of alpha pairs"
    print("✅ [OK] bGLS_ensemble executed successfully")
    print(f"   -> Calculated Motor Noise (sm): {sm:.4f}")
    print(f"   -> Calculated Timekeeper Noise (st): {st:.4f}")
    
except Exception as e:
    print(f"❌ [FAIL] bGLS_ensemble failed: {e}")

# ==========================================
# 3. Test Main Pipeline (estimate_ensemble)
# ==========================================
# ... [Keep your existing estimate_ensemble test code here] ...
N_steps = 20
players = ["1", "2"]
pairs = ["12", "21"]
s_dict = {p: np.random.rand(N_steps, 1) for p in players}
r_dict = {p: np.random.rand(N_steps, 1) for p in players}
t_dict = {p: np.linspace(0, 10, N_steps).tolist() for p in players}
A_dict = {pair: np.random.rand(N_steps).tolist() for pair in pairs}
Sigma_v_init = np.eye(2) * 0.1

print("⏳ Running estimate_ensemble...")
estimates = ensemble_backend.estimate_ensemble(
    s_dict, r_dict, A_dict, t_dict, 5.0, Sigma_v_init, 0.1, 0.25, 0.3, False, 5.0
)
print("✅ [OK] estimate_ensemble pipeline executed successfully")

print("--- All tests finished! ---")
