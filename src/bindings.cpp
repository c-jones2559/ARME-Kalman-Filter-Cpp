#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> 
#include "EnsembleFunctions.hpp"
#include <cmath>

namespace py = pybind11;

// --- NumCpp to NumPy Bridge ---

// --- Add this to force contiguous memory and type casting ---
using PyInArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// 1. Convert NumCpp array to Python NumPy array (Hardened deep copy)
py::array_t<double> nc_to_py(const nc::NdArray<double>& arr) {
    // Safely bail out if the C++ array is completely empty
    if (arr.size() == 0 || arr.isempty()) {
        return py::array_t<double>();
    }
    
    // Squeeze to 1D if it's a row vector to match NumPy conventions
    if (arr.numRows() == 1) {
        py::array_t<double> py_arr({arr.numCols()});
        py::buffer_info buf = py_arr.request();
        if (buf.ptr != nullptr && arr.data() != nullptr) {
            std::memcpy(buf.ptr, arr.data(), arr.size() * sizeof(double));
        }
        return py_arr;
    } 
    // Squeeze to 1D if it's a column vector
    else if (arr.numCols() == 1) {
        py::array_t<double> py_arr({arr.numRows()});
        py::buffer_info buf = py_arr.request();
        if (buf.ptr != nullptr && arr.data() != nullptr) {
            std::memcpy(buf.ptr, arr.data(), arr.size() * sizeof(double));
        }
        return py_arr;
    }
    
    // Default 2D matrix case
    py::array_t<double> py_arr({arr.numRows(), arr.numCols()});
    py::buffer_info buf = py_arr.request();
    
    if (buf.ptr != nullptr && arr.data() != nullptr) {
        std::memcpy(buf.ptr, arr.data(), arr.size() * sizeof(double));
    }
    return py_arr;
}

// 2. Convert Python NumPy array to NumCpp array (Hardened + Auto-Squeeze)
nc::NdArray<double> py_to_nc(PyInArr arr) {
    py::buffer_info buf = arr.request();
    
    // Handle completely empty arrays or scalars gracefully
    if (buf.size == 0 || buf.ndim == 0) {
        return nc::NdArray<double>();
    }

    nc::uint32 rows = 1;
    nc::uint32 cols = 1;
    
    if (buf.ndim == 1) {
        cols = buf.shape[0];
    } else if (buf.ndim == 2) {
        rows = buf.shape[0];
        cols = buf.shape[1];
    } else if (buf.ndim == 3) {
        // Python sent a 3D tensor.
        if (buf.shape[0] == 1) {
            rows = buf.shape[1];
            cols = buf.shape[2];
        } else if (buf.shape[1] == 1) {
            rows = buf.shape[0];
            cols = buf.shape[2];
        } else if (buf.shape[2] == 1) {
            rows = buf.shape[0];
            cols = buf.shape[1];
        } else {
            throw std::runtime_error("3D array provided cannot be squeezed into a 2D matrix (no dimension is 1).");
        }
    } else {
        throw std::runtime_error("Only up to 3D arrays (with a squeezable dimension) are supported.");
    }

    nc::NdArray<double> res(rows, cols);
    if (buf.ptr != nullptr) {
        std::memcpy(res.data(), buf.ptr, buf.size * sizeof(double));
    }
    return res;
}

class KFOptimizer {
private:
    std::unordered_map<std::string, nc::NdArray<double>> s_data;
    std::unordered_map<std::string, nc::NdArray<double>> A_data;
    std::unordered_map<std::string, nc::NdArray<double>> r_data; 
    std::unordered_map<std::string, nc::NdArray<double>> s_true_data;
    int K;
    double w;
    std::vector<std::string> players;
    std::vector<std::string> pairs;

public:
    // Constructor: Takes the heavy data ONCE and translates it to NumCpp
    KFOptimizer(std::unordered_map<std::string, PyInArr> py_s,
                std::unordered_map<std::string, PyInArr> py_A,
                std::unordered_map<std::string, PyInArr> py_r,
                std::unordered_map<std::string, PyInArr> py_s_true,
                int K_in, double w_in) {
        for (auto& [k, v] : py_s) s_data[k] = py_to_nc(v);
        for (auto& [k, v] : py_A) A_data[k] = py_to_nc(v);
        for (auto& [k, v] : py_r) r_data[k] = py_to_nc(v);
        for (auto& [k, v] : py_s_true) s_true_data[k] = py_to_nc(v);
        K = K_in;
        w = w_in;

        // Pre-compute string keys to avoid expensive concat in loops
        for (int i = 1; i <= K; i++) {
            players.push_back(std::to_string(i));
            for (int j = 1; j <= K; j++) {
                if (i != j) {
                    pairs.push_back(std::to_string(i) + std::to_string(j));
                }
            }
        }
    }

    // Original artificial loss
    double loss(double sigma_w, double sigma_v, double sigma_alpha, double alpha_KF_val, bool ideal) {
        int n_alpha = K * (K - 1);
        
        nc::NdArray<double> Sigma_v = nc::eye<double>(K) * sigma_v;
        nc::NdArray<double> Sigma_alpha = nc::eye<double>(n_alpha) * sigma_alpha;
        nc::NdArray<double> alpha_KF_init = nc::full<double>(nc::Shape(n_alpha, 1), alpha_KF_val);

        // Run KF locally in C++ without crossing the Python bridge!
        auto s_input = ideal ? s_true_data : s_data;
        auto result = KF_ensemble(s_input, A_data, Sigma_v, sigma_w, alpha_KF_init, Sigma_alpha, false, w);

        auto s_hat = std::get<2>(result); // Extract s_KF_predict

        // Calculate MSE purely in C++
        double mse = 0.0;
        int valid_count = 0;
        
        for (const auto& player : players) {
            auto hat_arr = s_hat[player];
            auto true_arr = s_true_data[player];
            for(nc::uint32 i = 0; i < hat_arr.size(); ++i) {
                if(!std::isnan(hat_arr[i]) && !std::isnan(true_arr[i])) {
                    mse += std::pow(hat_arr[i] - true_arr[i], 2);
                    valid_count++;
                }
            }
        }
        
        return valid_count > 0 ? mse / valid_count : 1e6;
    }

    // New real data combined loss
    double combined_loss_eval(double sigma_w, double sigma_v, double sigma_alpha, double alpha_KF_val, double weight) {
        int n_alpha = K * (K - 1);
        nc::NdArray<double> Sigma_v = nc::eye<double>(K) * sigma_v;
        nc::NdArray<double> Sigma_alpha = nc::eye<double>(n_alpha) * sigma_alpha;
        nc::NdArray<double> alpha_KF_init = nc::full<double>(nc::Shape(n_alpha, 1), alpha_KF_val);

        auto result = KF_ensemble_2(s_data, A_data, Sigma_v, sigma_w, alpha_KF_init, Sigma_alpha, false, w);
        auto s_hat_map = std::get<0>(result);

        // mse for s timeseries
        std::vector<double> mse_s;
        for (const auto& player : players) {
            nc::NdArray<double> pred = nc::roll(s_hat_map.at(player), 1);
            const auto& r_vec = r_data.at(player);
            double sum_sq_err = 0.0;
            int valid_count = 0;

            for (nc::uint32 i = 0; i < pred.size(); ++i) {
                if (!std::isnan(pred[i]) && !std::isnan(r_vec[i])) {
                    double diff = pred[i] - r_vec[i];
                    sum_sq_err += diff * diff;
                    valid_count++;
                }
            }
            mse_s.push_back(valid_count > 0 ? (sum_sq_err / valid_count) : nc::constants::nan);
        }

        // mse for asynchronies
        std::unordered_map<std::string, nc::NdArray<double>> r_est = r_from_s_ensemble(s_hat_map, r_data, static_cast<int>(w));
        
        std::vector<double> mse_async;
        for (const auto& pair : pairs) {
            nc::NdArray<double> true_A = A_data.at(pair);
            nc::NdArray<double> pred_A = r_est.at(pair.substr(0, 1)) - r_est.at(pair.substr(1, 1));

            double sum_sq_err = 0.0;
            int valid_count = 0;

            for (nc::uint32 i = 0; i < true_A.size(); ++i) {
                if (!std::isnan(true_A[i]) && !std::isnan(pred_A[i])) {
                    double diff = pred_A[i] - true_A[i];
                    sum_sq_err += diff * diff;
                    valid_count++;
                }
            }
            mse_async.push_back(valid_count > 0 ? (sum_sq_err / valid_count) : nc::constants::nan);
        }

        double sum_combined = 0.0;
        int valid_combined = 0;
        
        // Emulating Python's zip(mse_s, mse_async[:len(players)])
        for (size_t i = 0; i < players.size(); ++i) {
            if (!std::isnan(mse_s[i]) && !std::isnan(mse_async[i])) { 
                double val = mse_s[i] + weight * mse_async[i];
                sum_combined += val;
                valid_combined++;
            }
        }

        return valid_combined > 0 ? sum_combined / valid_combined : 1e6;
    }

    // New real data likelihood loss
    double likelihood_loss_eval(double sigma_w, double sigma_v, double sigma_alpha, double alpha_KF_val) {
        int n_alpha = K * (K - 1);
        nc::NdArray<double> Sigma_v = nc::eye<double>(K) * sigma_v;
        nc::NdArray<double> Sigma_alpha = nc::eye<double>(n_alpha) * sigma_alpha;
        nc::NdArray<double> alpha_KF_init = nc::full<double>(nc::Shape(n_alpha, 1), alpha_KF_val);

        auto result = KF_ensemble_2(s_data, A_data, Sigma_v, sigma_w, alpha_KF_init, Sigma_alpha, false, w);
        double log_l = std::get<2>(result);
        
        return -log_l;
    }

    // New function to evaluate the entire grid in C++
    py::array_t<double> combined_loss_grid(PyInArr py_sigma_w, PyInArr py_sigma_v, double sigma_alpha, double alpha_KF_val, double weight) {
        auto buf_w = py_sigma_w.request();
        auto buf_v = py_sigma_v.request();
        double* ptr_w = static_cast<double*>(buf_w.ptr);
        double* ptr_v = static_cast<double*>(buf_v.ptr);
        int len_w = buf_w.size;
        int len_v = buf_v.size;

        // Allocate a 2D NumPy array to return to Python
        py::array_t<double> loss_grid({len_v, len_w});
        auto buf_out = loss_grid.request();
        double* ptr_out = static_cast<double*>(buf_out.ptr);

        for (int i = 0; i < len_v; ++i) {
            for (int j = 0; j < len_w; ++j) {
                try {
                    // Call the C++ eval function directly
                    double loss = combined_loss_eval(ptr_w[j], ptr_v[i], sigma_alpha, alpha_KF_val, weight);
                    ptr_out[i * len_w + j] = loss;
                } catch (...) {
                    // Catch any linear algebra crashes and insert NaN
                    ptr_out[i * len_w + j] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
        return loss_grid;
    }
};

// --- Module Definition ---

PYBIND11_MODULE(ensemble_backend, m) {
    m.doc() = "Lightning fast C++ backend for ensemble tracking algorithms";
    py::class_<KFOptimizer>(m, "KFOptimizer")
        .def(py::init<std::unordered_map<std::string, PyInArr>, 
                      std::unordered_map<std::string, PyInArr>, 
                      std::unordered_map<std::string, PyInArr>, 
                      std::unordered_map<std::string, PyInArr>, 
                      int, double>())
        .def("loss", &KFOptimizer::loss)
        .def("combined_loss_eval", &KFOptimizer::combined_loss_eval)
        .def("likelihood_loss_eval", &KFOptimizer::likelihood_loss_eval)
        .def("combined_loss_grid", &KFOptimizer::combined_loss_grid);

    // Bind EstBGLS
    py::class_<EstBGLS>(m, "EstBGLS")
        .def_readonly("alpha", &EstBGLS::alpha)
        .def_readonly("sigma_v", &EstBGLS::sigma_v)
        .def_readonly("sigma_m", &EstBGLS::sigma_m)
        .def_property_readonly("s", [](const EstBGLS& e) {
            std::unordered_map<std::string, PyInArr> py_s;
            for (const auto& [key, val] : e.s) py_s[key] = nc_to_py(val);
            return py_s;
        })
        .def_property_readonly("r", [](const EstBGLS& e) {
            std::unordered_map<std::string, PyInArr> py_r;
            for (const auto& [key, val] : e.r) py_r[key] = nc_to_py(val);
            return py_r;
        });

    // Bind EstKF
    py::class_<EstKF>(m, "EstKF")
        .def_property_readonly("alpha_pred", [](const EstKF& e) {
            std::unordered_map<std::string, PyInArr> py_map;
            for (const auto& [key, val] : e.alpha_pred) py_map[key] = nc_to_py(val);
            return py_map;
        })
        .def_property_readonly("sigma2_alpha_pred", [](const EstKF& e) {
            std::vector<PyInArr> py_vec;
            for (const auto& val : e.sigma2_alpha_pred) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("s_pred", [](const EstKF& e) {
            std::unordered_map<std::string, PyInArr> py_map;
            for (const auto& [key, val] : e.s_pred) py_map[key] = nc_to_py(val);
            return py_map;
        })
        .def_property_readonly("sigma2_s_pred", [](const EstKF& e) {
            std::vector<PyInArr> py_vec;
            for (const auto& val : e.sigma2_s_pred) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("alpha_update", [](const EstKF& e) {
            std::vector<PyInArr> py_vec;
            for (const auto& val : e.alpha_update) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("sigma2_alpha_update", [](const EstKF& e) {
            std::vector<PyInArr> py_vec;
            for (const auto& val : e.sigma2_alpha_update) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("gain", [](const EstKF& e) {
            std::vector<PyInArr> py_vec;
            for (const auto& val : e.gain) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("sigma_v", [](const EstKF& e) {
            std::vector<PyInArr> py_vec;
            for (const auto& val : e.sigma_v) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("r", [](const EstKF& e) {
            std::unordered_map<std::string, PyInArr> py_map;
            for (const auto& [key, val] : e.r) py_map[key] = nc_to_py(val);
            return py_map;
        });

    // Bind Estimates container
    py::class_<Estimates>(m, "Estimates")
        .def_readonly("bGLS", &Estimates::bGLS)
        .def_readonly("KF", &Estimates::KF);

    // Bind Core Pipeline Execution
    m.def("estimate_ensemble", [](
          std::unordered_map<std::string, PyInArr> py_s,
          std::unordered_map<std::string, PyInArr> py_r,
          std::unordered_map<std::string, std::vector<double>> A,
          std::unordered_map<std::string, std::vector<double>> t,
          double w,
          PyInArr py_Sigma_v_init,
          double Sigma_w,
          double alpha_KF_init,
          double Sigma_alpha_init,
          bool est_Sigma_v,
          double w_KF) {
          
          std::unordered_map<std::string, nc::NdArray<double>> cpp_s;
          for (auto& [k, v] : py_s) cpp_s[k] = py_to_nc(v);
          
          std::unordered_map<std::string, nc::NdArray<double>> cpp_r;
          for (auto& [k, v] : py_r) cpp_r[k] = py_to_nc(v);

          nc::NdArray<double> cpp_Sigma_v_init = py_to_nc(py_Sigma_v_init);

          return estimate_ensemble(cpp_s, cpp_r, A, t, w, cpp_Sigma_v_init, Sigma_w, alpha_KF_init, Sigma_alpha_init, est_Sigma_v, w_KF);
          
    }, "Runs the full bGLS and Kalman Filter ensemble estimation pipeline.");

// --- Utility Functions ---
    
    m.def("cov2cor", [](PyInArr py_V) {
        nc::NdArray<double> cpp_V = py_to_nc(py_V);
        auto [stdevs, V_cor] = cov2cor(cpp_V);
        return py::make_tuple(nc_to_py(stdevs), nc_to_py(V_cor));
        
    }, "Convert covariance matrix to standard deviations and correlation matrix.");

    m.def("cor2cov", [](PyInArr py_stdevs, PyInArr py_V_cor) {
        nc::NdArray<double> cpp_stdevs = py_to_nc(py_stdevs);
        nc::NdArray<double> cpp_V_cor = py_to_nc(py_V_cor);
        nc::NdArray<double> result = cor2cov(cpp_stdevs, cpp_V_cor);
        return nc_to_py(result);
    }, "Convert standard deviations and correlation matrix to covariance matrix.");

    m.def("generate_ensemble_data", [](int N,
                                       const std::unordered_map<std::string, std::vector<double>>& Tmkp,
                                       const std::unordered_map<std::string, std::vector<double>>& alpha,
                                       const std::unordered_map<std::string, double>& sigma_v,
                                       double w) {
        
        auto result = generate_ensemble_data(N, Tmkp, alpha, sigma_v, w);
        auto map_nc_to_py = [](const std::unordered_map<std::string, nc::NdArray<double>>& in_map) {
            py::dict out;
            for (const auto& kv : in_map) {
                out[py::str(kv.first)] = nc_to_py(kv.second);
            }
            return out;
        };
        
        return py::make_tuple(
            map_nc_to_py(result[0]), // s_true
            map_nc_to_py(result[1]), // s_win
            map_nc_to_py(result[2]), // r
            map_nc_to_py(result[3]), // A
            map_nc_to_py(result[4]), // t
            map_nc_to_py(result[5])  // T
        );
    }, "Generates synthetic ensemble data");

    m.def("KF_ensemble", [](std::unordered_map<std::string, PyInArr> py_s,
                            std::unordered_map<std::string, PyInArr> py_A,
                            PyInArr py_Sigma_v_init,
                            double Sigma_w,
                            PyInArr py_alpha_KF_init,
                            PyInArr py_Sigma_alpha_init,
                            bool est_Sigma_v,
                            double w) {
        
        std::unordered_map<std::string, nc::NdArray<double>> cpp_s;
        for (auto& [k, v] : py_s) cpp_s[k] = py_to_nc(v);

        std::unordered_map<std::string, nc::NdArray<double>> cpp_A;
        for (auto& [k, v] : py_A) cpp_A[k] = py_to_nc(v);
        
        nc::NdArray<double> cpp_Sigma_v_init = py_to_nc(py_Sigma_v_init);
        nc::NdArray<double> cpp_alpha_KF_init = py_to_nc(py_alpha_KF_init);
        nc::NdArray<double> cpp_Sigma_alpha_init = py_to_nc(py_Sigma_alpha_init);

        auto result = KF_ensemble(cpp_s, cpp_A, cpp_Sigma_v_init, Sigma_w, cpp_alpha_KF_init, cpp_Sigma_alpha_init, est_Sigma_v, w);

        auto map_nc_to_py = [](const std::unordered_map<std::string, nc::NdArray<double>>& in_map) {
            py::dict out;
            for (const auto& kv : in_map) out[py::str(kv.first)] = nc_to_py(kv.second);
            return out;
        };
        auto vec_nc_to_py = [](const std::vector<nc::NdArray<double>>& in_vec) {
            py::list out;
            for (const auto& v : in_vec) out.append(nc_to_py(v));
            return out;
        };

        return py::make_tuple(
            vec_nc_to_py(std::get<0>(result)),
            vec_nc_to_py(std::get<1>(result)),
            map_nc_to_py(std::get<2>(result)),
            vec_nc_to_py(std::get<3>(result)),
            map_nc_to_py(std::get<4>(result)),
            vec_nc_to_py(std::get<5>(result)),
            vec_nc_to_py(std::get<6>(result)),
            vec_nc_to_py(std::get<7>(result))
        );
    }, "Standalone KF_ensemble wrapper",
       py::arg("s"), py::arg("A"), py::arg("Sigma_v_init"), py::arg("Sigma_w") = 0.1,
       py::arg("alpha_KF_init"), py::arg("Sigma_alpha_init"), py::arg("est_Sigma_v") = false, py::arg("w") = 5.0);

    m.def("KF_ensemble_2", [](std::unordered_map<std::string, PyInArr> py_s,
                              std::unordered_map<std::string, PyInArr> py_A,
                              PyInArr py_Sigma_v_init,
                              double Sigma_w,
                              PyInArr py_alpha_KF_init,
                              PyInArr py_Sigma_alpha_init,
                              bool est_Sigma_v,
                              double w) {
        
        std::unordered_map<std::string, nc::NdArray<double>> cpp_s;
        for (auto& [k, v] : py_s) cpp_s[k] = py_to_nc(v);

        std::unordered_map<std::string, nc::NdArray<double>> cpp_A;
        for (auto& [k, v] : py_A) cpp_A[k] = py_to_nc(v);
        
        nc::NdArray<double> cpp_Sigma_v_init = py_to_nc(py_Sigma_v_init);
        nc::NdArray<double> cpp_alpha_KF_init = py_to_nc(py_alpha_KF_init);
        nc::NdArray<double> cpp_Sigma_alpha_init = py_to_nc(py_Sigma_alpha_init);

        auto result = KF_ensemble_2(cpp_s, cpp_A, cpp_Sigma_v_init, Sigma_w, cpp_alpha_KF_init, cpp_Sigma_alpha_init, est_Sigma_v, w);

        auto map_nc_to_py = [](const std::unordered_map<std::string, nc::NdArray<double>>& in_map) {
            py::dict out;
            for (const auto& kv : in_map) out[py::str(kv.first)] = nc_to_py(kv.second);
            return out;
        };
        auto vec_nc_to_py = [](const std::vector<nc::NdArray<double>>& in_vec) {
            py::list out;
            for (const auto& v : in_vec) out.append(nc_to_py(v));
            return out;
        };

        return py::make_tuple(
            map_nc_to_py(std::get<0>(result)),
            vec_nc_to_py(std::get<1>(result)),
            std::get<2>(result)
        );
    }, "Standalone KF_ensemble_2 wrapper",
       py::arg("s"), py::arg("A"), py::arg("Sigma_v_init"), py::arg("Sigma_w") = 0.1,
       py::arg("alpha_KF_init"), py::arg("Sigma_alpha_init"), py::arg("est_Sigma_v") = false, py::arg("w") = 5.0);

    m.def("metrics_ensemble", [](std::unordered_map<std::string, PyInArr> py_s_pred,
                                 std::unordered_map<std::string, PyInArr> py_s_ref) {
        
        std::unordered_map<std::string, nc::NdArray<double>> cpp_s_pred;
        std::unordered_map<std::string, nc::NdArray<double>> cpp_s_ref;
        
        for (auto& [k, v] : py_s_pred) cpp_s_pred[k] = py_to_nc(v);
        for (auto& [k, v] : py_s_ref) cpp_s_ref[k] = py_to_nc(v);

        auto cpp_metrics = metrics_ensemble(cpp_s_pred, cpp_s_ref);
        
        py::dict py_metrics;
        for (const auto& [player, vals] : cpp_metrics) {
            py::dict py_vals;
            for (const auto& [metric_name, val] : vals) {
                py_vals[py::str(metric_name)] = nc_to_py(val);
            }
            py_metrics[py::str(player)] = py_vals;
        }
        return py_metrics;
        
    }, "Calculates correlation and standard deviation metrics");

    m.def("bGLS_ensemble", [](PyInArr py_o_rm) {
        nc::NdArray<double> cpp_o_rm = py_to_nc(py_o_rm);
        auto result = bGLS_ensemble(cpp_o_rm);
        return py::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
        
    }, "Runs bGLS on the given onset matrix", py::arg("o_rm"));

    m.def("s_from_bGLS_ensemble", [](std::unordered_map<std::string, double> alpha_est,
                                     std::unordered_map<std::string, PyInArr> py_A) {
        std::unordered_map<std::string, nc::NdArray<double>> cpp_A;
        for (auto& [k, v] : py_A) {
            cpp_A[k] = py_to_nc(v);
        }

        auto cpp_s = s_from_bGLS_ensemble(alpha_est, cpp_A);

        py::dict py_s;
        for (const auto& [k, v] : cpp_s) {
            py_s[py::str(k)] = nc_to_py(v);
        }
        return py_s;
        
    }, "Obtain tracking of s from alpha estimate from bGLS", py::arg("alpha_est"), py::arg("A"));

    m.def("r_from_s_ensemble", [](std::unordered_map<std::string, PyInArr> py_s_est,
                                  std::unordered_map<std::string, PyInArr> py_r,
                                  int w) {
        
        std::unordered_map<std::string, nc::NdArray<double>> cpp_s_est;
        for (auto& [k, v] : py_s_est) cpp_s_est[k] = py_to_nc(v);

        std::unordered_map<std::string, nc::NdArray<double>> cpp_r;
        for (auto& [k, v] : py_r) cpp_r[k] = py_to_nc(v);

        auto cpp_r_est = r_from_s_ensemble(cpp_s_est, cpp_r, w);

        py::dict py_r_est;
        for (const auto& [k, v] : cpp_r_est) {
            py_r_est[py::str(k)] = nc_to_py(v);
        }
        return py_r_est;
        
    }, "Reconstruct r from estimated s", py::arg("s_est"), py::arg("r"), py::arg("w") = 5);

    m.def("likelihood_loss", [](std::unordered_map<std::string, std::vector<double>> r_dict,
                                std::unordered_map<std::string, std::vector<double>> s_win,
                                std::unordered_map<std::string, std::vector<double>> A,
                                double w,
                                std::unordered_map<std::string, double> params) {
        
        return likelihood_loss(r_dict, s_win, A, w, KF_ensemble_2, params);
        
    }, "Calculates likelihood loss utilizing the KF_ensemble_2 routine",
       py::arg("r_dict"), py::arg("s_win"), py::arg("A"), py::arg("w"), py::arg("params"));

    m.def("combined_loss", [](std::unordered_map<std::string, PyInArr> py_r_dict,
                            std::unordered_map<std::string, PyInArr> py_s_win,
                            std::unordered_map<std::string, PyInArr> py_A,
                            double w,
                            std::unordered_map<std::string, double> params,
                            double weight) {
        
        std::unordered_map<std::string, std::vector<double>> r_dict, s_win, A;
        auto to_vec = [](PyInArr arr) {
            auto buf = arr.request();
            double* ptr = static_cast<double*>(buf.ptr);
            return std::vector<double>(ptr, ptr + buf.size);
        };

        for (auto& [k, v] : py_r_dict) r_dict[k] = to_vec(v);
        for (auto& [k, v] : py_s_win)  s_win[k]  = to_vec(v);
        for (auto& [k, v] : py_A)       A[k]      = to_vec(v);

        return combined_loss(r_dict, s_win, A, w, KF_ensemble_2, params, weight);
        
    }, "Calculates combined loss", 
        py::arg("r_dict"), py::arg("s_win"), py::arg("A"), py::arg("w"), py::arg("params"), py::arg("weight") = 1.0);

    m.def("process_ensemble_data", [](std::string leader, int rep, int w) {
        auto result = process_ensemble_data(leader, rep, w);
        auto map_nc_to_py = [](const std::unordered_map<std::string, nc::NdArray<double>>& in_map) {
            py::dict out;
            for (const auto& kv : in_map) {
                out[py::str(kv.first)] = nc_to_py(kv.second);
            }
            return out;
        };
        
        return py::make_tuple(
            map_nc_to_py(std::get<0>(result)), // r_dp
            map_nc_to_py(std::get<1>(result)), // r_nr
            map_nc_to_py(std::get<2>(result)), // r_sp
            map_nc_to_py(std::get<3>(result)), // s_dp_win_arr
            map_nc_to_py(std::get<4>(result)), // s_nr_win_arr
            map_nc_to_py(std::get<5>(result)), // s_sp_win_arr
            map_nc_to_py(std::get<6>(result)), // A_dp
            map_nc_to_py(std::get<7>(result)), // A_nr
            map_nc_to_py(std::get<8>(result)), // A_sp
            map_nc_to_py(std::get<9>(result)), // t_dp
            map_nc_to_py(std::get<10>(result)), // t_nr
            map_nc_to_py(std::get<11>(result))  // t_sp
        );
    }, "Processes virtuoso ensemble data", 
       py::arg("leader"), py::arg("rep"), py::arg("w") = 5);

}
