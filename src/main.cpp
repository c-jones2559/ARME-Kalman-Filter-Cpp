#include <iostream>
#include <cassert>
#include <cmath>
#include "EnsembleFunctions.hpp"

// Helper to check if doubles are close enough
bool is_close(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

void test_estimate_ensemble_pipeline() {
    int N = 10; // Number of onsets
    int K = 4;  // Number of players
    
    // 1. Setup dummy s and r maps (4 players)
    std::unordered_map<std::string, nc::NdArray<double>> s;
    std::unordered_map<std::string, nc::NdArray<double>> r;
    for (int i = 1; i <= K; ++i) {
        s[std::to_string(i)] = nc::zeros<double>(N + 1, 1);
        r[std::to_string(i)] = nc::ones<double>(N + 1, 1);
    }

    // 2. Setup A and t maps
    std::unordered_map<std::string, std::vector<double>> A;
    std::unordered_map<std::string, std::vector<double>> t;
    for (int i = 1; i <= K; ++i) {
        t[std::to_string(i)] = std::vector<double>(N + 1, 0.0);
        for (int j = 1; j <= K; ++j) {
            if (i != j) {
                A[std::to_string(i) + std::to_string(j)] = std::vector<double>(N + 1, 0.0);
            }
        }
    }

    // 3. Define initial params
    nc::NdArray<double> Sigma_v_init = nc::eye<double>(K);
    double w = 5.0;
    
    // 4. Run the estimate
    std::cout << "Attempting to run estimate_ensemble..." << std::endl;
    Estimates results = estimate_ensemble(s, r, A, t, w, Sigma_v_init, 0.1, 0.25, 0.3, false, 5);
    
    std::cout << "Success! Pipeline finished." << std::endl;
}

int main() {
    std::cout << "--- Starting Ensemble C++ Verification Suite ---" << std::endl;

    // 1. Test Matrix Math (NumCpp Check)
    {
        nc::NdArray<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
        double sum = nc::sum(mat).item();
        assert(std::abs(sum - 10.0) < 0.001);
        std::cout << "  [OK] Matrix math" << std::endl;
    }

    // 2. Test Covariance/Correlation
    {
        nc::NdArray<double> V = {{1.0, 0.5}, {0.5, 1.0}};
        auto [stdevs, V_cor] = cov2cor(V);
        auto V_back = cor2cov(stdevs, V_cor);
        assert(is_close(V(0, 1), V_back(0, 1)));
        std::cout << "  [OK] Cov2Cor conversions" << std::endl;
    }

    // 3. Test Generate Ensemble Data (Minimal)
    {
        int N = 5;
        std::unordered_map<std::string, std::vector<double>> Tmkp = {{"1", {0, 0, 1, 1, 1, 1}}};
        std::unordered_map<std::string, std::vector<double>> alpha = {{"12", {0, 0, 0, 0, 0, 0}}};
        std::unordered_map<std::string, double> sigma_v = {{"1", 0.01}};
        auto result = generate_ensemble_data(N, Tmkp, alpha, sigma_v, 2);
        assert(result[0].at("1").size() == 6); 
        std::cout << "  [OK] Generate Ensemble Data" << std::endl;
    }
    test_estimate_ensemble_pipeline();

    // 4. Test bGLS_ensemble (Minimal)
    {
        nc::NdArray<double> o_rm = {{1,2, 3, 4}, {11,12, 13, 14}, {21,22, 23, 24}, {31,32, 33, 34}};
        auto [alphas, sm, st] = bGLS_ensemble(o_rm);
        assert(alphas.size() > 0);
        std::cout << "  [OK] bGLS Ensemble" << std::endl;
    }

    std::cout << "\n--- All internal tests passed! You're a legend. ---" << std::endl;
    return 0;
}
