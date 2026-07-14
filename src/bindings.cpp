#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> 
#include "EnsembleFunctions.hpp"

namespace py = pybind11;

// --- NumCpp to NumPy Bridge ---

// 1. Convert NumCpp array to Python NumPy array
py::array_t<double> nc_to_py(const nc::NdArray<double>& arr) {
    return py::array_t<double>(
        {arr.numRows(), arr.numCols()}, // shape
        {arr.numCols() * sizeof(double), sizeof(double)}, // strides (C-style contiguous)
        arr.data() // raw pointer to the data
    );
}

// 2. Convert Python NumPy array to NumCpp array
nc::NdArray<double> py_to_nc(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    
    // Check dimensions to handle 1D and 2D arrays gracefully
    nc::uint32 rows = buf.ndim == 2 ? buf.shape[0] : 1;
    nc::uint32 cols = buf.ndim == 2 ? buf.shape[1] : buf.shape[0];
    
    nc::NdArray<double> res(rows, cols);
    std::memcpy(res.data(), buf.ptr, buf.size * sizeof(double));
    return res;
}


// --- Module Definition ---

PYBIND11_MODULE(ensemble_backend, m) {
    m.doc() = "Lightning fast C++ backend for ensemble tracking algorithms";

    // Bind EstBGLS
    py::class_<EstBGLS>(m, "EstBGLS")
        .def_readonly("alpha", &EstBGLS::alpha)
        .def_readonly("sigma_v", &EstBGLS::sigma_v)
        .def_readonly("sigma_m", &EstBGLS::sigma_m)
        .def_property_readonly("s", [](const EstBGLS& e) {
            std::unordered_map<std::string, py::array_t<double>> py_s;
            for (const auto& [key, val] : e.s) py_s[key] = nc_to_py(val);
            return py_s;
        })
        .def_property_readonly("r", [](const EstBGLS& e) {
            std::unordered_map<std::string, py::array_t<double>> py_r;
            for (const auto& [key, val] : e.r) py_r[key] = nc_to_py(val);
            return py_r;
        });

    // Bind EstKF
    py::class_<EstKF>(m, "EstKF")
        .def_property_readonly("alpha_pred", [](const EstKF& e) {
            std::unordered_map<std::string, py::array_t<double>> py_map;
            for (const auto& [key, val] : e.alpha_pred) py_map[key] = nc_to_py(val);
            return py_map;
        })
        .def_property_readonly("sigma2_alpha_pred", [](const EstKF& e) {
            std::vector<py::array_t<double>> py_vec;
            for (const auto& val : e.sigma2_alpha_pred) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("s_pred", [](const EstKF& e) {
            std::unordered_map<std::string, py::array_t<double>> py_map;
            for (const auto& [key, val] : e.s_pred) py_map[key] = nc_to_py(val);
            return py_map;
        })
        .def_property_readonly("sigma2_s_pred", [](const EstKF& e) {
            std::vector<py::array_t<double>> py_vec;
            for (const auto& val : e.sigma2_s_pred) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("alpha_update", [](const EstKF& e) {
            std::vector<py::array_t<double>> py_vec;
            for (const auto& val : e.alpha_update) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("sigma2_alpha_update", [](const EstKF& e) {
            std::vector<py::array_t<double>> py_vec;
            for (const auto& val : e.sigma2_alpha_update) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("gain", [](const EstKF& e) {
            std::vector<py::array_t<double>> py_vec;
            for (const auto& val : e.gain) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("sigma_v", [](const EstKF& e) {
            std::vector<py::array_t<double>> py_vec;
            for (const auto& val : e.sigma_v) py_vec.push_back(nc_to_py(val));
            return py_vec;
        })
        .def_property_readonly("r", [](const EstKF& e) {
            std::unordered_map<std::string, py::array_t<double>> py_map;
            for (const auto& [key, val] : e.r) py_map[key] = nc_to_py(val);
            return py_map;
        });

    // Bind Estimates container
    py::class_<Estimates>(m, "Estimates")
        .def_readonly("bGLS", &Estimates::bGLS)
        .def_readonly("KF", &Estimates::KF);

    // Bind Core Pipeline Execution
    m.def("estimate_ensemble", [](
          std::unordered_map<std::string, py::array_t<double>> py_s,
          std::unordered_map<std::string, py::array_t<double>> py_r,
          std::unordered_map<std::string, std::vector<double>> A,
          std::unordered_map<std::string, std::vector<double>> t,
          double w,
          py::array_t<double> py_Sigma_v_init,
          double Sigma_w,
          double alpha_KF_init,
          double Sigma_alpha_init,
          bool est_Sigma_v,
          double w_KF) {
          
          // Convert incoming Python dicts of NumPy arrays to C++ dicts of NumCpp arrays
          std::unordered_map<std::string, nc::NdArray<double>> cpp_s;
          for (auto& [k, v] : py_s) cpp_s[k] = py_to_nc(v);
          
          std::unordered_map<std::string, nc::NdArray<double>> cpp_r;
          for (auto& [k, v] : py_r) cpp_r[k] = py_to_nc(v);

          nc::NdArray<double> cpp_Sigma_v_init = py_to_nc(py_Sigma_v_init);

          // Execute backend C++ logic
          return estimate_ensemble(cpp_s, cpp_r, A, t, w, cpp_Sigma_v_init, Sigma_w, alpha_KF_init, Sigma_alpha_init, est_Sigma_v, w_KF);
          
    }, "Runs the full bGLS and Kalman Filter ensemble estimation pipeline.");
// --- Utility Functions ---
    
    m.def("cov2cor", [](py::array_t<double> py_V) {
        // Convert incoming NumPy array to NumCpp
        nc::NdArray<double> cpp_V = py_to_nc(py_V);
        
        // Call the C++ function (returns a std::tuple)
        auto [stdevs, V_cor] = cov2cor(cpp_V);
        
        // Convert both NumCpp arrays back to NumPy and return as a Python tuple
        return py::make_tuple(nc_to_py(stdevs), nc_to_py(V_cor));
        
    }, "Convert covariance matrix to standard deviations and correlation matrix.");


    // Might as well chuck cor2cov in there too so you don't hit the exact same wall in two minutes
    m.def("cor2cov", [](py::array_t<double> py_stdevs, py::array_t<double> py_V_cor) {
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
        
        // Quick local lambda to translate the maps
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


m.def("KF_ensemble", [](std::unordered_map<std::string, py::array_t<double>> py_s,
                            std::unordered_map<std::string, py::array_t<double>> py_A,
                            py::array_t<double> py_Sigma_v_init,
                            double Sigma_w,
                            py::array_t<double> py_alpha_KF_init,
                            py::array_t<double> py_Sigma_alpha_init,
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

        // Local lambdas for translating the outputs
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

        // Note the swapped tuple indices to match the new C++ return order (0 and 4)
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

m.def("metrics_ensemble", [](std::unordered_map<std::string, py::array_t<double>> py_s_pred,
                                 std::unordered_map<std::string, py::array_t<double>> py_s_ref) {
        
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
}
