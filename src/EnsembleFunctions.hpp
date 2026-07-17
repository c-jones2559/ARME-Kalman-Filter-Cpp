#pragma once 

#include "NumCpp.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>

#include <array>

// Structs
struct VirtuosoRow {
    std::string condition;
    std::string leader;
    int repetition;
    std::vector<double> ioi_values;
};

struct EstBGLS {
    std::unordered_map<std::string, double> alpha;
    double sigma_v;
    double sigma_m;
    std::unordered_map<std::string, nc::NdArray<double>> s;
    std::unordered_map<std::string, nc::NdArray<double>> r;
};

struct EstKF {
    std::unordered_map<std::string, nc::NdArray<double>> alpha_pred;
    std::vector<nc::NdArray<double>> sigma2_alpha_pred;
    std::unordered_map<std::string, nc::NdArray<double>> s_pred;
    std::vector<nc::NdArray<double>> sigma2_s_pred;
    std::vector<nc::NdArray<double>> alpha_update;
    std::vector<nc::NdArray<double>> sigma2_alpha_update;
    std::vector<nc::NdArray<double>> gain;
    std::vector<nc::NdArray<double>> sigma_v;
    std::unordered_map<std::string, nc::NdArray<double>> r;
};

struct Estimates {
    EstBGLS bGLS;
    EstKF KF;
};

// Function Signatures

std::array<std::unordered_map<std::string, nc::NdArray<double>>, 6> generate_ensemble_data(
    int N,
    const std::unordered_map<std::string, std::vector<double>>& Tmkp,
    const std::unordered_map<std::string, std::vector<double>>& alpha,
    const std::unordered_map<std::string, double>& sigma_v,
    double w);

std::tuple<std::unordered_map<std::string, double>, double, double> bGLS_ensemble(nc::NdArray<double> o_rm);

std::tuple<std::vector<nc::NdArray<double>>,
           std::vector<nc::NdArray<double>>,
           std::unordered_map<std::string, nc::NdArray<double>>,
           std::vector<nc::NdArray<double>>,
           std::unordered_map<std::string, nc::NdArray<double>>,
           std::vector<nc::NdArray<double>>,
           std::vector<nc::NdArray<double>>,
           std::vector<nc::NdArray<double>>>
KF_ensemble(const std::unordered_map<std::string, nc::NdArray<double>>& s,
            const std::unordered_map<std::string, nc::NdArray<double>>& A,
            const nc::NdArray<double>& Sigma_v_init,
            double Sigma_w,
            nc::NdArray<double> alpha_KF_init,
            nc::NdArray<double> Sigma_alpha_init,
            bool est_Sigma_v = false,
            double w = 5);

std::tuple<std::unordered_map<std::string, nc::NdArray<double>>, 
           std::vector<nc::NdArray<double>>, 
           double> 
KF_ensemble_2(const std::unordered_map<std::string, nc::NdArray<double>>& s,
            const std::unordered_map<std::string, nc::NdArray<double>>& A,
            const nc::NdArray<double>& Sigma_v_init,
            double Sigma_w,
            nc::NdArray<double> alpha_KF_init,
            nc::NdArray<double> Sigma_alpha_init,
            bool est_Sigma_v = false,
            double w = 5);

nc::NdArray<double> cor2cov(const nc::NdArray<double>& stdevs, const nc::NdArray<double>& V_cor);

std::tuple<nc::NdArray<double>, nc::NdArray<double>> cov2cor(const nc::NdArray<double>& V);

std::tuple<std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>, 
           std::unordered_map<std::string, nc::NdArray<double>>> 
process_ensemble_data(std::string leader, int rep, int w = 5);

std::unordered_map<std::string, std::unordered_map<std::string, nc::NdArray<double>>> metrics_ensemble(
    const std::unordered_map<std::string, nc::NdArray<double>>& s_pred,
    const std::unordered_map<std::string, nc::NdArray<double>>& s_ref);

std::unordered_map<std::string, nc::NdArray<double>> s_from_bGLS_ensemble(
    const std::unordered_map<std::string, double>& alpha_est,
    const std::unordered_map<std::string, nc::NdArray<double>>& A);

std::unordered_map<std::string, nc::NdArray<double>> r_from_s_ensemble(
    const std::unordered_map<std::string, nc::NdArray<double>>& s_est,
    const std::unordered_map<std::string, nc::NdArray<double>>& r,
    int w = 5);

Estimates estimate_ensemble(
    const std::unordered_map<std::string, nc::NdArray<double>>& s,
    const std::unordered_map<std::string, nc::NdArray<double>>& r,
    std::unordered_map<std::string, std::vector<double>> A,
    const std::unordered_map<std::string, std::vector<double>>& t,
    double w,
    const nc::NdArray<double>& Sigma_v_init,
    double Sigma_w,
    double alpha_KF_init,
    double Sigma_alpha_init,
    bool est_Sigma_v,
    double w_KF);

std::tuple<std::unordered_map<std::string, std::vector<nc::NdArray<double>>>,
           std::unordered_map<std::string, nc::NdArray<double>>> 
make_avg_alpha_ensemble(std::vector<std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>>> estimates);
