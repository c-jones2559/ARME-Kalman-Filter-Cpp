#include "EnsembleFunctions.hpp"
#include "NumCpp.hpp"
#include <NumCpp/Functions/corrcoef.hpp>
#include <NumCpp/Functions/full.hpp>
#include <NumCpp/NdArray/NdArrayCore.hpp>
#include "rapidcsv.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <iostream>

using std::unordered_map;

/*
 * GENERATE ENSEMBLE DATA
 */

std::array<std::unordered_map<std::string, nc::NdArray<double>>, 6> generate_ensemble_data(int N,
                                                                                 const std::unordered_map<std::string, std::vector<double>>& Tmkp,
                                                                                 const std::unordered_map<std::string, std::vector<double>>& alpha,
                                                                                 const std::unordered_map<std::string, double>& sigma_v,
                                                                                 double w) {
    int K = Tmkp.size(); // Number of players

    std::vector<std::string> players;
    for (int i = 1; i <= K; i++) {
        players.push_back(std::to_string(i));
    }
    std::vector<std::string> pairs;
    for (int i = 1; i <= K; i++) {
        for (int j = 1; j <= K; j++) {
            if (i == j) continue;
            pairs.push_back({std::to_string(i) + std::to_string(j)});
        }
    }


    // Define variables -- with NaN value at n = 0
    std::unordered_map<std::string, std::vector<double>> T;
    for (const auto& player : players) {
        T.insert({player, {nc::constants::nan}});
    }
    std::unordered_map<std::string, std::vector<double>> t;
    for (const auto& player : players) {
        t.insert({player, {nc::constants::nan}});
    }
    std::unordered_map<std::string, std::vector<double>> A;
    for (const auto& pair : pairs) {
        A.insert({pair, {nc::constants::nan}});
    }

    // Initialise values at n = 1 -- but it initialises at 0??
    for (const auto& player : players) {
        T[player].push_back(0);
        t[player].push_back(nc::random::normal<double>(0.0, sigma_v.at(player)));
    }

    for (const auto& pair: pairs) {
        A[pair].push_back(t[pair.substr(0,1)][1] - t[pair.substr(1,1)][1]);
    }

    // Simulate onsets
    for (int n = 2; n <= N; n++) {
        for (const auto& player : players) {
            T[player].push_back(T[player][n - 1] + Tmkp.at(player)[n]);

            double aux = 0;
            for (const auto& player2 : players) {
                if (player2 != player) {
                    aux += alpha.at(player + player2)[n] * A[player + player2][n - 1];
                }
            }

            t[player].push_back(t[player][n - 1] + Tmkp.at(player)[n] - aux + nc::random::normal(0.0, sigma_v.at(player)));
        }

        for (const auto& pair : pairs) {
            A[pair].push_back(t[std::string(1, pair[0])][n] - t[std::string(1, pair[1])][n]);
        }
    }

    // Convert to numpy array
    std::unordered_map<std::string, nc::NdArray<double>> T_num;
    for (const auto& player : players) {
        nc::NdArray<double> numArr(T[player]);
        T_num[player] = numArr;
    }
    std::unordered_map<std::string, nc::NdArray<double>> t_num;
    for (const auto& player : players) {
        nc::NdArray<double> numArr(t[player]);
        t_num[player] = numArr;
    }
    std::unordered_map<std::string, nc::NdArray<double>> A_num;
    for (const auto& pair : pairs) {
        nc::NdArray<double> numArr(A[pair]);
        A_num[pair] = numArr;
    }


    // Compute r and s - NaN for n = 0 and n = 1
    std::unordered_map<std::string, std::vector<double>> r;
    for (const auto& player : players) {
        r[player] = {nc::constants::nan, nc::constants::nan};
    }
    std::unordered_map<std::string, std::vector<double>> s_true;
    for (const auto& player : players) {
        s_true[player] = {nc::constants::nan, nc::constants::nan};
    }
    std::unordered_map<std::string, std::vector<double>> s_win;
    for (const auto& player : players) {
        s_win[player] = {nc::constants::nan, nc::constants::nan};
    }

    for (int n = 2; n <= N; n++) {
        for (const auto& player : players) {
            r[player].push_back(t_num[player][n] - t_num[player][n - 1]);
            s_true[player].push_back(r[player][n] - Tmkp.at(player)[n]);
            if (n <= w) {
                std::vector<double> slice(r[player].begin() + 2, r[player].end());
                double sum = std::accumulate(slice.begin(), slice.end(), 0.0); // 0.0 ensures double precision
                double mean = sum / slice.size();
                s_win[player].push_back(r[player][n] - mean);
            } else {
                std::vector<double> slice(r[player].begin() + n - w + 1, r[player].end());
                double sum = std::accumulate(slice.begin(), slice.end(), 0.0);
                double mean = sum / slice.size();
                s_win[player].push_back(r[player][n] - mean);
            }
        }
    }
    
    // Convert to numpy array
    std::unordered_map<std::string, nc::NdArray<double>> r_num;
    for (const auto& player : players) {
        nc::NdArray<double> numArr(r[player]);
        r_num[player] = numArr;
    }
    std::unordered_map<std::string, nc::NdArray<double>> s_true_num;
    for (const auto& player : players) {
        nc::NdArray<double> numArr(s_true[player]);
        s_true_num[player] = numArr;
    }
    std::unordered_map<std::string, nc::NdArray<double>> s_win_num;
    for (const auto& player : players) {
        nc::NdArray<double> numArr(s_win[player]);
        s_win_num[player] = numArr;
    }

    return {s_true_num, s_win_num, r_num, A_num, t_num, T_num};
}


/*
 * ENSEMBLE bGLS (Peter's code)
 */

std::tuple<std::unordered_map<std::string, double>, double, double> bGLS_ensemble(nc::NdArray<double> o_rm) {

    // Initialise matrix of Asynchronies
    int async_x = o_rm.numRows() - 1; // height of asynchrony array
    int async_y = o_rm.numCols(); // width of asynchrony array
    // nc::NdArray<double> am = nc::zeros<double>((async_x,async_y,async_y));
    std::vector<nc::NdArray<double>> am(async_x, nc::zeros<double>(async_y, async_y));

    for (int i = 0; i < async_x; i++) {
        for (int j = 0; j < async_y; j++) {
            for (int k = 0; k < async_y; k++) {
                am[i](k,j) = o_rm(i + 1, k) - o_rm(i + 1, j); // filling array
            }
        }
    }

    // Calculate Inter-Onset Intervals
    int ioi_x = o_rm.numRows() - 1; // IOI array height (also number of onsets)
    int ioi_y = o_rm.numCols(); // IOI array width

    nc::NdArray<double> Rm = nc::zeros<double>(ioi_x, ioi_y);
    for (int i = 0; i < ioi_y; i++) {
        for (int j = 0; j < ioi_x; j++) {
            Rm(j, i) = o_rm(j+1, i) - o_rm(j, i);
        }
    }


    int number_of_players = ioi_y;
    nc::NdArray<double> alphas = nc::zeros<double>(number_of_players, number_of_players); // Initialise alpha matrix

    o_rm = o_rm(nc::Slice(1, o_rm.numRows()), o_rm.cSlice());

    double k11;
    double k12;

    for (int sub = 0; sub < number_of_players; sub++) { // iterates through subjects
        nc::NdArray<double> R = Rm(nc::Slice(0, Rm.numRows()), sub);
        nc::NdArray<double> As = nc::zeros<double>(R.numRows(), (number_of_players-1));

        std::vector<int> others;
        for (int i = 0; i < number_of_players; i++) {
            if (i != sub) {
                others.push_back(i);
            }
        }

        for (int k = 0; k < async_x; k++) {
            for (int l = 0; l < number_of_players-1; l++) {
                As(k,l) = (am[k](sub, others[l]));
            }
        }
        nc::NdArray<double> meanA = nc::zeros<double>(1, others.size());
        for (int i = 0; i < others.size(); i++) {
            nc::NdArray<double> A_current = As(nc::Slice(0, As.numRows()), i); // average asynchrony for subject
            double sum = 0;
            for (int j = 0; j < A_current.size(); j++) {
                sum += A_current(0, j);
            }
            meanA(0, i) = sum;
        }

        double sum = std::accumulate(R.begin(), R.end(), 0.0); // 0.0 ensures double precision
        double meanR = sum / R.size();

        // bGLS!
        double iterations = 20;
        double thresh = 0.001;
        int N = R.numRows()-1;
        if (N < 1) {
            std::cout << "Warning: N is too small (" << N << ") to perform bGLS iteration." << std::endl;
            continue; 
        }
        int P = number_of_players-1;
        for (int p = 0; p < P; p++) {
            double current_mean = meanA(0, p);

            for (nc::uint32 i = 0; i < As.numRows(); i++) {
                As(i, p) = As(i, p) - current_mean;
            }
        }

        nc::NdArray<double> b3 = R(nc::Slice(1, R.numRows()), R.cSlice()) - meanR;
        nc::NdArray<double> a3 = As(nc::Slice(0, As.numRows() - 1), As.cSlice());

        k11 = 1;
        k12 = 0;

        nc::NdArray<double> zold = nc::full<double>(nc::Shape(P, 1), -9999.0);
        for (int iteration = 0; iteration  < iterations; iteration ++) {
            nc::NdArray<double> cc = nc::zeros<double>(N, N);
            for (int i = 0; i < N; ++i) {
                cc(i, i) = k11;
                if (i > 0) {
                    cc(i, i - 1) = k12;
                }
                if (i < N - 1) {
                    cc(i, i + 1) = k12;
                }
            }
            nc::NdArray<double> ic = nc::linalg::inv(cc);
            nc::NdArray<double> a3T = a3.transpose();
            nc::NdArray<double> step1 = nc::dot(a3T, ic);
            nc::NdArray<double> step2 = nc::dot(step1, a3);
            nc::NdArray<double> step3 = nc::dot(step1, b3);
            // Correct GLS math: z = (A^T C^{-1} A)^{-1} (A^T C^{-1} b)
            nc::NdArray<double> z = nc::dot(nc::linalg::pinv(step2), step3);
            // Correct residual calculation using matrix multiplication: d = (A z - b)^T
            nc::NdArray<double> d = (nc::dot(a3, z) - b3).transpose();

            nc::NdArray<double> slice1 = d(0, nc::Slice(0, d.numCols()-1));
            nc::NdArray<double> slice2 = d(0, nc::Slice(1, d.numCols()));
            nc::NdArray<double> k = nc::cov(nc::vstack({slice1, slice2})); // estimate residual acvf
            k11 = (k(0, 0)+k(1, 1))/2;
            k12 = k(0, 1);
            // apply bounds
            if (k12 > 0) {
                k12 = 0;
            }
            if (k11 < (-3)*k12) {
                k11 = (-3)*k12;
            }
            if (nc::sum(abs(z-zold)) < thresh) {
                break;
            }
            zold = z;
        }
        std::vector<double> temp;
        int count = 0;
        for (int i = 0; i < number_of_players; i++) {
            if (i == sub) {
                temp.push_back(0.0);
            } else {
                temp.push_back(zold(count, 0));
                count++;
            }
        }
        nc::NdArray<double> finalz(temp);
        finalz.reshape(1, number_of_players);

        // insert alpha row into alpha matrix
        for (int i = 0; i < number_of_players; i++) {
            alphas(sub, i) = finalz(0, i);
        }
    }
    double sm = nc::sqrt(-k12); // Motor noise calculation
    double st = nc::sqrt(k11-2*(pow(sm, 2))); // Timekeeper noise calculation

    std::unordered_map<std::string, double> alphas_dict;

    for (int player1 = 0; player1 < number_of_players; player1++) {
        for (int player2 = 0; player2 < number_of_players; player2++) {
            if (player1 != player2) {
                alphas_dict[std::to_string(player1 + 1) + std::to_string(player2 + 1)] = alphas(player1, player2);
                // alphas_dict[std::to_string(player1 + 1) + std::to_string(player2 + 1)];
                // alphas_dict.insert({
                //     std::to_string(player1 + 1) + std::to_string(player2 + 1),
                //     alphas(player1, player2)});
            }
        }
    }

    return {alphas_dict, sm, st};
}

/*
 * ENSEMBLE KALMAN FILTER
 */

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
                       bool est_Sigma_v, 
                       double w) {
    int K = s.size(); // Number of players

    std::vector<std::string> players;
    for (int i = 1; i <= K; i++) {
        players.push_back(std::to_string(i));
    }
    std::vector<std::string> pairs;
    for (int i = 1; i <= K; i++) {
        for (int j = 1; j <= K; j++) {
            if (i == j) continue;
            pairs.push_back({std::to_string(i) + std::to_string(j)});
        }
    }

    double N = s.at("1").size() - 1; 

    int P = K * (K - 1); // Dimension of your state vector (12)

    // Initialise predictions for alpha (with NaN for n = 0 and n = 1)
    std::vector<nc::NdArray<double>> alpha_KF_predict;
    alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, 1), nc::constants::nan));
    alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, 1), nc::constants::nan));

    std::vector<nc::NdArray<double>> Sigma_alpha_KF_predict;
    Sigma_alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, P), nc::constants::nan));
    Sigma_alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, P), nc::constants::nan));

    // Initialise predictions for s (with NaN for n = 0 and n = 1)
    std::vector<nc::NdArray<double>> s_KF_predict;
    s_KF_predict.push_back(nc::full<double>(nc::Shape(K, 1), nc::constants::nan));
    s_KF_predict.push_back(nc::full<double>(nc::Shape(K, 1), nc::constants::nan));

    std::vector<nc::NdArray<double>> Sigma_s_KF_predict;
    Sigma_s_KF_predict.push_back(nc::full<double>(nc::Shape(K, K), nc::constants::nan));
    Sigma_s_KF_predict.push_back(nc::full<double>(nc::Shape(K, K), nc::constants::nan));

    // Initialise updates for alpha (with n = 0 and n = 1)
    std::vector<nc::NdArray<double>> gain_KF;
    gain_KF.push_back(nc::full<double>(nc::Shape(P, K), nc::constants::nan));
    gain_KF.push_back(nc::full<double>(nc::Shape(P, K), nc::constants::nan));

    std::vector<nc::NdArray<double>> alpha_KF_update;
    alpha_KF_update.push_back(nc::full<double>(nc::Shape(P, 1), nc::constants::nan));
    
    // Reshape the 1D python array into a column vector
    nc::NdArray<double> alpha_init_col = alpha_KF_init;
    alpha_init_col.reshape(P, 1);
    alpha_KF_update.push_back(alpha_init_col);

    std::vector<nc::NdArray<double>> Sigma_alpha_KF_update;
    Sigma_alpha_KF_update.push_back(nc::full<double>(nc::Shape(P, P), nc::constants::nan));
    // Use the 2D diagonal matrix Python sent us directly
    Sigma_alpha_KF_update.push_back(Sigma_alpha_init);

    // Initialise dynamic estimation of sigma_v [TESTING] (with n = 0 and n = 1)
    std::vector<nc::NdArray<double>> Sigma_v;
    Sigma_v.push_back(nc::full<double>(nc::Shape(K, K), nc::constants::nan));
    Sigma_v.push_back(Sigma_v_init);

    // Initialise matrix F
    std::vector<nc::NdArray<double>> F;
    F.push_back(nc::full<double>(nc::Shape(K, P), nc::constants::nan));
    F.push_back(nc::full<double>(nc::Shape(K, P), nc::constants::nan));

    for (int n = 2; n <= N; n++) {
        // Build matrix F_n
        std::vector<nc::NdArray<double>> F_list;
        for (const auto& player1 : players) {
            std::vector<double> A_i;
            for (const auto& player2 : players) {
                if (player2 != player1) {
                    A_i.push_back(-A.at(player1 + player2)[n - 1]);
                }
            }
            F_list.push_back(nc::NdArray<double>(A_i));
        }

        // block diag
        int rows = 0, cols = 0;
        for (const auto& list : F_list) {
            rows += list.numRows();
            cols += list.numCols();
        }
        nc::NdArray<double> bigF = nc::zeros<double>(rows, cols);

        int rows_count = 0, cols_count = 0;
        for (const auto& list : F_list) {
            for (int i = 0; i < list.numRows(); i++) {
                for (int j = 0; j < list.numCols(); j++) {
                    bigF.at(i + rows_count, j + cols_count) =
                        list.at(i, j);
                }
            }
            rows_count += list.numRows();
            cols_count += list.numCols();
        }

        F.push_back(bigF);


        // Make vector with s at time n
        nc::NdArray<double> s_n_vec(K, 1);
        for (int i = 0; i < K; i++) {
            s_n_vec(i, 0) = s.at(players[i])[n]; 
        }

        // Predict alpha
        alpha_KF_predict.push_back({alpha_KF_update[n - 1]});
        Sigma_alpha_KF_predict.push_back(Sigma_alpha_KF_update[n - 1] + Sigma_w);

        // Predict s
        // if (n >= F.size() || n >= alpha_KF_predict.size()) {
        //     std::cout << "n: " << n << std::endl;
        //     std::cout << "F.size(): " << F.size() << std::endl;
        //     std::cout << "alpha_KF_predict.size(): " << alpha_KF_predict.size() << std::endl;
        // }
        s_KF_predict.push_back(nc::dot(F[n], alpha_KF_predict[n]));
        Sigma_s_KF_predict.push_back(nc::dot(F[n], nc::dot(Sigma_alpha_KF_predict[n], F[n].transpose())) + Sigma_v[n - 1]);

        // Update alpha
        gain_KF.push_back(nc::dot(Sigma_alpha_KF_predict[n], nc::dot(F[n].transpose(), nc::linalg::inv(Sigma_s_KF_predict[n]))));
        alpha_KF_update.push_back(alpha_KF_predict[n] + nc::dot(gain_KF[n], (s_n_vec - s_KF_predict[n])));
        Sigma_alpha_KF_update.push_back(Sigma_alpha_KF_predict[n] - nc::dot(gain_KF[n], nc::dot(F[n], Sigma_alpha_KF_predict[n])));

        if (est_Sigma_v) {
            if (n <= w) {
                Sigma_v.push_back(Sigma_v_init);
            } else {
                // s_KF_predict[n - w - 1 + 2:n + 1]
                std::vector<nc::NdArray<double>> part1 (s_KF_predict.begin() + n - w - 1 + 2, s_KF_predict.begin() + n + 1);

                // [s[player][n - w - 1 + 2:n + 1] for player in players]
                std::vector<nc::NdArray<double>> part2;
                for (const auto& player : players) {
                    nc::NdArray<double> temp (s.at(player).begin() + n - w - 1 + 2, s.at(player).begin() + n + 1);
                    part2.push_back(temp);
                }

                std::vector<nc::NdArray<double>> new_part;
                for (int i = 0; i < part1.size(); i++) {
                    new_part.push_back(part1[i] - part2[i]);
                }

                int W = new_part.size();
                nc::NdArray<double> new_matrix = nc::zeros<double>(K, W);
                for (int i = 0; i < W; i++) {
                    for (int j = 0; j < K; j++) {
                        new_matrix(i, j) = new_part[j](i, 0);
                    }
                }

                Sigma_v.push_back(nc::diag(nc::var(new_matrix)));
            }
        } else {
            Sigma_v.push_back(Sigma_v_init);
        }
    }

    // Prepare outputs (the most important ones)
    std::unordered_map<std::string, nc::NdArray<double>> s_KF_predict_unordered_map;
    for (int i = 0; i < K; i++) {
        nc::NdArray<double> row = nc::zeros<double>(1, s_KF_predict.size());
        for (size_t j = 0; j < s_KF_predict.size(); j++) {
            row(0, j) = s_KF_predict[j](i, 0);
        }
        s_KF_predict_unordered_map[players[i]] = row;
    }

    std::unordered_map<std::string, nc::NdArray<double>> alpha_KF_predict_unordered_map;
    for (int i = 0; i < P; i++) {
        nc::NdArray<double> row = nc::zeros<double>(1, alpha_KF_predict.size());
        for (size_t j = 0; j < alpha_KF_predict.size(); j++) {
            row(0, j) = alpha_KF_predict[j](i, 0);
        }
        alpha_KF_predict_unordered_map[pairs[i]] = row;
    }

    return {alpha_KF_update, Sigma_alpha_KF_predict, s_KF_predict_unordered_map, Sigma_s_KF_predict, alpha_KF_predict_unordered_map, Sigma_alpha_KF_update, gain_KF, Sigma_v};
}

/*
 * ENSEMBLE KALMAN FILTER 2: RD_optreal_results.py uses different format so this is for that
 */

std::tuple<std::unordered_map<std::string, nc::NdArray<double>>, 
           std::vector<nc::NdArray<double>>, 
           double> 
           KF_ensemble_2(const std::unordered_map<std::string, nc::NdArray<double>>& s,
                       const std::unordered_map<std::string, nc::NdArray<double>>& A,
                       const nc::NdArray<double>& Sigma_v_init,
                       double Sigma_w, 
                       nc::NdArray<double> alpha_KF_init, 
                       nc::NdArray<double> Sigma_alpha_init,
                       bool est_Sigma_v, 
                       double w) {
    int K = s.size(); // Number of players

    std::vector<std::string> players;
    for (int i = 1; i <= K; i++) {
        players.push_back(std::to_string(i));
    }
    std::vector<std::string> pairs;
    for (int i = 1; i <= K; i++) {
        for (int j = 1; j <= K; j++) {
            if (i == j) continue;
            pairs.push_back({std::to_string(i) + std::to_string(j)});
        }
    }

    double N = s.at("1").size() - 1; 

    int P = K * (K - 1); // Dimension of your state vector (12)

    // Initialise predictions for alpha (with NaN for n = 0 and n = 1)
    std::vector<nc::NdArray<double>> alpha_KF_predict;
    alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, 1), nc::constants::nan));
    alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, 1), nc::constants::nan));

    std::vector<nc::NdArray<double>> Sigma_alpha_KF_predict;
    Sigma_alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, P), nc::constants::nan));
    Sigma_alpha_KF_predict.push_back(nc::full<double>(nc::Shape(P, P), nc::constants::nan));

    // Initialise predictions for s (with NaN for n = 0 and n = 1)
    std::vector<nc::NdArray<double>> s_KF_predict;
    s_KF_predict.push_back(nc::full<double>(nc::Shape(K, 1), nc::constants::nan));
    s_KF_predict.push_back(nc::full<double>(nc::Shape(K, 1), nc::constants::nan));

    std::vector<nc::NdArray<double>> Sigma_s_KF_predict;
    Sigma_s_KF_predict.push_back(nc::full<double>(nc::Shape(K, K), nc::constants::nan));
    Sigma_s_KF_predict.push_back(nc::full<double>(nc::Shape(K, K), nc::constants::nan));

    // Initialise updates for alpha (with n = 0 and n = 1)
    std::vector<nc::NdArray<double>> gain_KF;
    gain_KF.push_back(nc::full<double>(nc::Shape(P, K), nc::constants::nan));
    gain_KF.push_back(nc::full<double>(nc::Shape(P, K), nc::constants::nan));

    std::vector<nc::NdArray<double>> alpha_KF_update;
    alpha_KF_update.push_back(nc::full<double>(nc::Shape(P, 1), nc::constants::nan));
    
    // Reshape the 1D python array into a column vector
    nc::NdArray<double> alpha_init_col = alpha_KF_init;
    alpha_init_col.reshape(P, 1);
    alpha_KF_update.push_back(alpha_init_col);

    std::vector<nc::NdArray<double>> Sigma_alpha_KF_update;
    Sigma_alpha_KF_update.push_back(nc::full<double>(nc::Shape(P, P), nc::constants::nan));
    // Use the 2D diagonal matrix Python sent us directly
    Sigma_alpha_KF_update.push_back(Sigma_alpha_init);

    // Initialise dynamic estimation of sigma_v [TESTING] (with n = 0 and n = 1)
    std::vector<nc::NdArray<double>> Sigma_v;
    Sigma_v.push_back(nc::full<double>(nc::Shape(K, K), nc::constants::nan));
    Sigma_v.push_back(Sigma_v_init);

    // Initialise matrix F
    std::vector<nc::NdArray<double>> F;
    F.push_back(nc::full<double>(nc::Shape(K, P), nc::constants::nan));
    F.push_back(nc::full<double>(nc::Shape(K, P), nc::constants::nan));

    double log_l = 0.0; // RD

    for (int n = 2; n <= N; n++) {
        // Build matrix F_n
        std::vector<nc::NdArray<double>> F_list;
        for (const auto& player1 : players) {
            std::vector<double> A_i;
            for (const auto& player2 : players) {
                if (player2 != player1) {
                    A_i.push_back(-A.at(player1 + player2)[n - 1]);
                }
            }
            F_list.push_back(nc::NdArray<double>(A_i));
        }

        // block diag
        int rows = 0, cols = 0;
        for (const auto& list : F_list) {
            rows += list.numRows();
            cols += list.numCols();
        }
        nc::NdArray<double> bigF = nc::zeros<double>(rows, cols);

        int rows_count = 0, cols_count = 0;
        for (const auto& list : F_list) {
            for (int i = 0; i < list.numRows(); i++) {
                for (int j = 0; j < list.numCols(); j++) {
                    bigF.at(i + rows_count, j + cols_count) =
                        list.at(i, j);
                }
            }
            rows_count += list.numRows();
            cols_count += list.numCols();
        }

        F.push_back(bigF);


        // Make vector with s at time n
        nc::NdArray<double> s_n_vec(K, 1);
        for (int i = 0; i < K; i++) {
            s_n_vec(i, 0) = s.at(players[i])[n]; 
        }

        // Predict alpha
        alpha_KF_predict.push_back({alpha_KF_update[n - 1]});
        Sigma_alpha_KF_predict.push_back(Sigma_alpha_KF_update[n - 1] + Sigma_w);

        // Predict s
        s_KF_predict.push_back(nc::dot(F[n], alpha_KF_predict[n]));
        Sigma_s_KF_predict.push_back(nc::dot(F[n], nc::dot(Sigma_alpha_KF_predict[n], F[n].transpose())) + Sigma_v[n - 1]);

        // RD: Innovation
        nc::NdArray<double> innov = s_n_vec - s_KF_predict[n];
        auto S = Sigma_s_KF_predict[n]; // Reuse the exact matrix you just computed
        auto S_inv = nc::linalg::inv(S);
        
        // RD: Log-likelihood contribution
        double det_part = nc::log(nc::linalg::det(2 * nc::constants::pi * S));
        double innov_part = nc::dot(innov.transpose(), nc::dot(S_inv, innov)).item();
        log_l += -0.5 * (det_part + innov_part);

        // Update alpha
        gain_KF.push_back(nc::dot(Sigma_alpha_KF_predict[n], nc::dot(F[n].transpose(), nc::linalg::inv(Sigma_s_KF_predict[n]))));
        alpha_KF_update.push_back(alpha_KF_predict[n] + nc::dot(gain_KF[n], (s_n_vec - s_KF_predict[n])));
        Sigma_alpha_KF_update.push_back(Sigma_alpha_KF_predict[n] - nc::dot(gain_KF[n], nc::dot(F[n], Sigma_alpha_KF_predict[n])));

        if (est_Sigma_v) {
            if (n <= w) {
                Sigma_v.push_back(Sigma_v_init);
            } else {
                // s_KF_predict[n - w - 1 + 2:n + 1]
                std::vector<nc::NdArray<double>> part1 (s_KF_predict.begin() + n - w - 1 + 2, s_KF_predict.begin() + n + 1);

                // [s[player][n - w - 1 + 2:n + 1] for player in players]
                std::vector<nc::NdArray<double>> part2;
                for (const auto& player : players) {
                    nc::NdArray<double> temp (s.at(player).begin() + n - w - 1 + 2, s.at(player).begin() + n + 1);
                    part2.push_back(temp);
                }

                std::vector<nc::NdArray<double>> new_part;
                for (int i = 0; i < part1.size(); i++) {
                    new_part.push_back(part1[i] - part2[i]);
                }

                int W = new_part.size();
                nc::NdArray<double> new_matrix = nc::zeros<double>(K, W);
                for (int i = 0; i < W; i++) {
                    for (int j = 0; j < K; j++) {
                        new_matrix(i, j) = new_part[j](i, 0);
                    }
                }

                Sigma_v.push_back(nc::diag(nc::var(new_matrix)));
            }
        } else {
            Sigma_v.push_back(Sigma_v_init);
        }
    }

    // Prepare outputs (the most important ones)
    std::unordered_map<std::string, nc::NdArray<double>> s_KF_predict_unordered_map;
    for (int i = 0; i < K; i++) {
        nc::NdArray<double> row = nc::zeros<double>(1, s_KF_predict.size());
        for (size_t j = 0; j < s_KF_predict.size(); j++) {
            row(0, j) = s_KF_predict[j](i, 0);
        }
        s_KF_predict_unordered_map[players[i]] = row;
    }

    std::unordered_map<std::string, nc::NdArray<double>> alpha_KF_predict_unordered_map;
    for (int i = 0; i < P; i++) {
        nc::NdArray<double> row = nc::zeros<double>(1, alpha_KF_predict.size());
        for (size_t j = 0; j < alpha_KF_predict.size(); j++) {
            row(0, j) = alpha_KF_predict[j](i, 0);
        }
        alpha_KF_predict_unordered_map[pairs[i]] = row;
    }

    return {s_KF_predict_unordered_map, alpha_KF_update, log_l};
}

/*
 * CORRELATION AND STANDARD DEVIATION TO COVARIANCE MATRIX
 */

nc::NdArray<double> cor2cov(const nc::NdArray<double>& stdevs, const nc::NdArray<double>& V_cor) {
    nc::NdArray<double> V = nc::zeros<double>(V_cor.shape());
    for (int i = 0; i < V.numRows(); i++) {
        for (int j = 0; j < V.numCols(); j++) {
            V(i, j) = V_cor(i, j)*stdevs[i]*stdevs[j];
        }
    }

    return V;
}

/*
 * COVARIANCE TO CORRELATION AND STANDARD DEVIATION MATRIX
 */

std::tuple<nc::NdArray<double>, nc::NdArray<double>> cov2cor(const nc::NdArray<double>& V) {
    nc::NdArray<double> stdevs = nc::sqrt(nc::diag(V));
    nc::NdArray<double> V_cor = nc::zeros<double>(V.shape());
    for (int i = 0; i < V.numRows(); i++) {
        for (int j = 0; j < V.numCols(); j++) {
            V_cor(i, j) = V(i, j)/(stdevs[i]*stdevs[j]);
        }
    }
    return {stdevs, V_cor};
}

/*
 * PROCESS ENSEMBLE VIRTUOSO DATA
 */

// Commentted out because EnsembleFunctions.hpp
// struct VirtuosoRow {
//     std::string condition;
//     std::string leader;
//     int repetition;
//     std::vector<double> ioi_values;
// };

std::tuple<std::unordered_map<std::string, nc::NdArray<double>>, // r_dp,
           std::unordered_map<std::string, nc::NdArray<double>>, // r_nr
           std::unordered_map<std::string, nc::NdArray<double>>, // r_sp
           std::unordered_map<std::string, nc::NdArray<double>>, // s_dp_win_arr
           std::unordered_map<std::string, nc::NdArray<double>>, // s_nr_win_arr
           std::unordered_map<std::string, nc::NdArray<double>>, // s_sp_win_arr
           std::unordered_map<std::string, nc::NdArray<double>>, // A_dp
           std::unordered_map<std::string, nc::NdArray<double>>, // A_nr
           std::unordered_map<std::string, nc::NdArray<double>>, // A_sp
           std::unordered_map<std::string, nc::NdArray<double>>, // t_dp
           std::unordered_map<std::string, nc::NdArray<double>>, // t_nr
           std::unordered_map<std::string, nc::NdArray<double>>> // t_sp
           process_ensemble_data(std::string leader, int rep, int w) { // w = 5

    rapidcsv::Document virtuoso_doc("virtuoso.csv");
    std::vector<std::string> condition_col = (virtuoso_doc.GetColumn<std::string>("condition"));
    std::vector<std::string> leader_col = (virtuoso_doc.GetColumn<std::string>("leader"));
    std::vector<int> repetition_col = (virtuoso_doc.GetColumn<int>("repetition"));
    // std::vector<std::vector<double>> ioi0_col = (virtuoso_doc.GetColumn<std::vector<double>>("ioi0"));
 
    std::vector<VirtuosoRow> virtuoso;

    for (int i = 0; i < virtuoso_doc.GetRowCount(); ++i) {
        VirtuosoRow row;
        row.condition = condition_col[i];
        row.leader = leader_col[i];
        row.repetition = repetition_col[i];

        for (int j = 0; j < 46; ++j) {
            std::string colName = "ioi" + std::to_string(j);
            row.ioi_values.push_back(virtuoso_doc.GetCell<double>(colName, i));
        }


        virtuoso.push_back(row);
    }

    int N = 46;
    int K = 4;

    std::vector<std::string> ioi_cols(N);
    for (int i = 0; i < N; i++) {
        ioi_cols[i] = "ioi" + std::to_string(i);
    }

    std::vector<std::string> players;
    for (int i = 1; i <= K; i++) {
        players.push_back(std::to_string(i));
    }
    std::vector<std::string> pairs;
    for (int i = 1; i <= K; i++) {
        for (int j = 1; j <= K; j++) {
            if (i == j) continue;
            pairs.push_back({std::to_string(i) + std::to_string(j)});
        }
    }

    std::vector<VirtuosoRow> r_dp_raw;
    std::vector<VirtuosoRow> r_nr_raw;
    std::vector<VirtuosoRow> r_sp_raw;
    for (const auto& v : virtuoso) {
        if (v.condition == "DP" && v.leader == leader && v.repetition == rep) {
            r_dp_raw.push_back(v);
        }
        if (v.condition == "NR" && v.leader == leader && v.repetition == rep) {
            r_nr_raw.push_back(v);
        }
        if (v.condition == "SP" && v.leader == leader && v.repetition == rep) {
            r_sp_raw.push_back(v);
        }
    }

    std::unordered_map<std::string, nc::NdArray<double>> r_dp;
    for (int i = 0; i < r_dp_raw.size(); i++) {
        std::vector<double> temp;
        // Replaced aux with this
        temp.push_back(nc::constants::nan);
        temp.push_back(nc::constants::nan);
        for (const auto& val : r_dp_raw[i].ioi_values) {
            temp.push_back(val);
        }
        nc::NdArray<double> temp_arr (temp);
        r_dp[players[i]] = temp_arr;
    }

    std::unordered_map<std::string, nc::NdArray<double>> r_nr;
    for (int i = 0; i < r_nr_raw.size(); i++) {
        std::vector<double> temp;
        // Replaced aux with this
        temp.push_back(nc::constants::nan);
        temp.push_back(nc::constants::nan);
        for (const auto& val : r_nr_raw[i].ioi_values) {
            temp.push_back(val);
        }
        nc::NdArray<double> temp_arr (temp);
        r_nr[players[i]] = temp_arr;
    }

    std::unordered_map<std::string, nc::NdArray<double>> r_sp;
    for (int i = 0; i < r_sp_raw.size(); i++) {
        std::vector<double> temp;
        // Replaced aux with this
        temp.push_back(nc::constants::nan);
        temp.push_back(nc::constants::nan);
        for (const auto& val : r_sp_raw[i].ioi_values) {
            temp.push_back(val);
        }
        nc::NdArray<double> temp_arr (temp);
        r_sp[players[i]] = temp_arr;
    }

    std::unordered_map<std::string, nc::NdArray<double>> t_dp;
    for (const auto& player : players) {
        t_dp[player] = nc::zeros<double>(1, r_dp.at(player).size());
        for (int i = 0; i < r_dp.size(); i++) {
            if (i == 0) t_dp[player][0] = nc::constants::nan;
            else if (i == 1) t_dp[player][1] = 0;
            else {
                t_dp[player][i] = t_dp[player][i-1] + r_dp[player][i];
            }
        }
    }
    std::unordered_map<std::string, nc::NdArray<double>> t_nr;
    for (const auto& player : players) {
        t_nr[player] = nc::zeros<double>(1, r_nr.at(player).size());
        for (int i = 0; i < r_nr.size(); i++) {
            if (i == 0) t_nr[player][0] = nc::constants::nan;
            else if (i == 1) t_nr[player][1] = 0;
            else {
                t_nr[player][i] = t_nr[player][i-1] + r_nr[player][i];
            }
        }
    }
    std::unordered_map<std::string, nc::NdArray<double>> t_sp;
    for (const auto& player : players) {
        t_sp[player] = nc::zeros<double>(1, r_sp.at(player).size());
        for (int i = 0; i < r_sp.size(); i++) {
            if (i == 0) t_sp[player][0] = nc::constants::nan;
            else if (i == 1) t_sp[player][1] = 0;
            else {
                t_sp[player][i] = t_sp[player][i-1] + r_sp[player][i];
            }
        }
    }

    std::unordered_map<std::string, nc::NdArray<double>> A_dp;
    std::unordered_map<std::string, nc::NdArray<double>> A_nr;
    std::unordered_map<std::string, nc::NdArray<double>> A_sp;
    for (const auto& pair : pairs) {
        std::string p1 = pair.substr(0, 1);
        std::string p2 = pair.substr(1, 1);

        A_dp[pair] = t_dp[p1] - t_dp[p2];
        A_nr[pair] = t_nr[p1] - t_nr[p2];
        A_sp[pair] = t_sp[p1] - t_sp[p2];
    }

    // Compute s - NaN for n = 0 and n = 1
    std::unordered_map<std::string, std::vector<double>> s_dp_win;
    std::unordered_map<std::string, std::vector<double>> s_nr_win;
    std::unordered_map<std::string, std::vector<double>> s_sp_win;
    for (const auto& player : players) {
        s_dp_win[player] = {nc::constants::nan, nc::constants::nan};
        s_nr_win[player] = {nc::constants::nan, nc::constants::nan};
        s_sp_win[player] = {nc::constants::nan, nc::constants::nan};
    }

    for (int n = 2; n < N+2; n++) {
        for (const auto& player : players) {
            // adjust window / no window here
            int bound;
            if (n <= w) bound = 2;
            else bound = n - w + 1;

            double sum_dp = std::accumulate(r_dp[player].begin(), r_dp[player].end(), 0.0);
            double mean_dp = sum_dp / r_dp[player].size();
            double sum_nr = std::accumulate(r_nr[player].begin(), r_nr[player].end(), 0.0);
            double mean_nr = sum_nr / r_nr[player].size();
            double sum_sp = std::accumulate(r_sp[player].begin(), r_sp[player].end(), 0.0);
            double mean_sp = sum_sp / r_sp[player].size();

            s_dp_win[player].push_back(mean_dp);
            s_nr_win[player].push_back(mean_nr);
            s_sp_win[player].push_back(mean_sp);
        }
    }

    std::unordered_map<std::string, nc::NdArray<double>> s_dp_win_arr;
    std::unordered_map<std::string, nc::NdArray<double>> s_nr_win_arr;
    std::unordered_map<std::string, nc::NdArray<double>> s_sp_win_arr;
    for (const auto& player : players) {
        s_dp_win_arr[player] = nc::NdArray<double> (s_dp_win[player]);
        s_nr_win_arr[player] = nc::NdArray<double> (s_nr_win[player]);
        s_sp_win_arr[player] = nc::NdArray<double> (s_sp_win[player]);
    }

    return {r_dp, r_nr, r_sp, s_dp_win_arr, s_nr_win_arr, s_sp_win_arr, A_dp, A_nr, A_sp, t_dp, t_nr, t_sp};
}

/*
 * SOME METRICS (ensemble)
 */

std::unordered_map<std::string, std::unordered_map<std::string, nc::NdArray<double>>> metrics_ensemble(
    const std::unordered_map<std::string, nc::NdArray<double>>& s_pred,
    const std::unordered_map<std::string, nc::NdArray<double>>& s_ref) {
    
    std::unordered_map<std::string, std::unordered_map<std::string, nc::NdArray<double>>> metrics;
    for (const auto& [player, val] : s_pred) {
        metrics[player]["corr"] = nc::constants::nan;
        metrics[player]["std"] = nc::constants::nan;
    }

    std::unordered_map<std::string, nc::NdArray<double>> filtered_s_pred;
    std::unordered_map<std::string, nc::NdArray<double>> filtered_s_ref;
    for (const auto& [player, val] : s_pred) {
        std::vector<double> temp_pred;
        std::vector<double> temp_ref;
        for (int i = 0; i < val.size(); i++) {
            if (std::isnan(val[i]) || std::isnan(s_ref.at(player)[i])) continue;
            temp_pred.push_back(val[i]);
            temp_ref.push_back(s_ref.at(player)[i]);
        }
        filtered_s_pred[player] = nc::NdArray<double>(temp_pred);
        filtered_s_ref[player] = nc::NdArray<double>(temp_ref);
    }

    for (const auto& [player, val] : s_pred) {
        metrics[player]["corr"] = nc::corrcoef(nc::vstack({filtered_s_pred.at(player), filtered_s_ref.at(player)})).round(3);
        metrics[player]["std"] = nc::sqrt(nc::nanvar(s_pred.at(player) - s_ref.at(player))).round(3);
    }

    return metrics;
}

/*
 * OBTAIN TRACKING OF s FROM alpha ESTIMATE FROM bGLS (ensemble)
 */

std::unordered_map<std::string, nc::NdArray<double>> s_from_bGLS_ensemble(const std::unordered_map<std::string, double>& alpha_est,
                                                                          const std::unordered_map<std::string, nc::NdArray<double>>& A) {
    int N = A.at("12").size() - 1;

    std::vector<std::string> players;
    // Extract unique players from the pair keys
    for (const auto& [pair, val] : A) {
        std::string p = pair.substr(0, 1);
        if (std::find(players.begin(), players.end(), p) == players.end()) {
            players.push_back(p);
        }
    }
    std::sort(players.begin(), players.end());

    std::unordered_map<std::string, std::vector<double>> s_est;
    for (const auto& player : players) {
        s_est[player] = {nc::constants::nan, nc::constants::nan}; // n = 0 and n = 1
    }

    for (int n = 2; n <= N; n++) {
        for (const auto& player : players) {
            double aux = 0;
            for (const auto& player2 : players) {
                if (player == player2) continue; // Changed break to continue
                aux += alpha_est.at(player + player2) * A.at(player + player2)[n - 1];
            }
            s_est[player].push_back(-aux);
        }
    }

    std::unordered_map<std::string, nc::NdArray<double>> s_est_arr;
    for (const auto& player : players) {
        s_est_arr[player] = nc::NdArray<double> (s_est[player]);
    }

    return s_est_arr;
}

/*
 * RECONSTRUCT r FROM s_estimated (ensemble)
 */

std::unordered_map<std::string, nc::NdArray<double>> r_from_s_ensemble(const std::unordered_map<std::string, nc::NdArray<double>>& s_est,
                                                                       const std::unordered_map<std::string, nc::NdArray<double>>& r,
                                                                       int w) { // w = 5

    int N = r.at("1").size() - 1;
    
    std::vector<std::string> players;
    for (const auto& [player, val] : r) {
        players.push_back(player);
    }

    std::unordered_map<std::string, std::vector<double>> r_est = {};

    for (const auto& player : players) {
        r_est[player] = {nc::constants::nan, nc::constants::nan}; // n = 0 and n = 1
    }

    for (int n = 2; n < w + 1; n++) {
        for (const auto& player : players) {
            r_est[player].push_back(nc::constants::nan);
        }
    }
    
    for (int n = w + 1; n <= N; n++) {
        for (const auto& player : players) {
            // Fix: Calculate rolling mean using iterators
            auto start_it = r.at(player).begin() + n - w + 1;
            auto end_it = r.at(player).begin() + n + 1;
            double sum = std::accumulate(start_it, end_it, 0.0);
            double mean = sum / w;

            r_est[player].push_back(s_est.at(player)[n] + mean);
        }
    }

    std::unordered_map<std::string, nc::NdArray<double>> r_est_arr;
    for (const auto& player : players) {
        r_est_arr[player] = nc::NdArray<double> (r_est[player]); // Fix: Actually return r_est
    }
    
    return r_est_arr;
}

/*
 * ESTIMATE (ensemble)
 */

// struct EstBGLS {
//     std::unordered_map<std::string, double> alpha;
//     double sigma_v;
//     double sigma_m;
//     std::unordered_map<std::string, nc::NdArray<double>> s;
//     std::unordered_map<std::string, nc::NdArray<double>> r;
// };

// struct EstKF {
//     std::unordered_map<std::string, nc::NdArray<double>> alpha_pred;
//     std::vector<nc::NdArray<double>> sigma2_alpha_pred;
//     std::unordered_map<std::string, nc::NdArray<double>> s_pred;
//     std::vector<nc::NdArray<double>> sigma2_s_pred;
//     std::vector<nc::NdArray<double>> alpha_update;
//     std::vector<nc::NdArray<double>> sigma2_alpha_update;
//     std::vector<nc::NdArray<double>> gain;
//     std::vector<nc::NdArray<double>> sigma_v;
//     std::unordered_map<std::string, nc::NdArray<double>> r;
// };

// struct Estimates {
//     EstBGLS bGLS;
//     EstKF KF;
// };

Estimates estimate_ensemble(const std::unordered_map<std::string, nc::NdArray<double>>& s,
                            const std::unordered_map<std::string, nc::NdArray<double>>& r,
                            std::unordered_map<std::string, std::vector<double>> A,
                            const std::unordered_map<std::string, std::vector<double>>& t,
                            double w,
                            const nc::NdArray<double>& Sigma_v_init,
                            double Sigma_w,
                            double alpha_KF_init,
                            double Sigma_alpha_init,
                            bool est_Sigma_v,
                            double w_KF) {
    int N = s.at("1").numRows() - 1;

    std::vector<std::vector<double>> t_vals;
    for (const auto& [player, val] : t) {
        t_vals.push_back(val);
    }
    nc::NdArray<double> t_vals_arr (t_vals);

    // Estimate
    nc::NdArray<double> array = t_vals_arr.transpose();
    nc::NdArray<double> sliced_array = array(nc::Slice(1, array.numRows()), array.cSlice());
    std::tuple<std::unordered_map<std::string, double>, double, double> bGLS_out = bGLS_ensemble(array);
    std::unordered_map<std::string, double> alpha_bGLS = std::get<0>(bGLS_out);
    double sigma_m_bGLS = std::get<1>(bGLS_out);
    double sigma_v_bGLS = std::get<2>(bGLS_out);


    // Convert scalars back to matrices for the C++ internal call
    int K = s.size();
    int P = K * (K - 1);
    nc::NdArray<double> alpha_KF_init_arr = nc::full<double>(nc::Shape(P, 1), alpha_KF_init);
    nc::NdArray<double> Sigma_alpha_init_arr = nc::eye<double>(P) * Sigma_alpha_init;

    // Hoisted A_arr creation up here so both KF and bGLS can use it
    std::unordered_map<std::string, nc::NdArray<double>> A_arr;
    for (const auto& [key, val] : A) {
        A_arr[key] = nc::NdArray<double>(A.at(key));
    }

    auto KF_ensemble_out = KF_ensemble(s,
                                       A_arr, 
                                       Sigma_v_init,
                                       Sigma_w,
                                       alpha_KF_init_arr,
                                       Sigma_alpha_init_arr,
                                       est_Sigma_v,
                                       w_KF);

    // Unpack using the newly swapped tuple order
    std::vector<nc::NdArray<double>> alpha_KF_update = std::get<0>(KF_ensemble_out);
    std::vector<nc::NdArray<double>> sigma2_alpha_KF_predict = std::get<1>(KF_ensemble_out);
    std::unordered_map<std::string, nc::NdArray<double>> s_KF_predict = std::get<2>(KF_ensemble_out);
    std::vector<nc::NdArray<double>> Sigma_s_KF_predict = std::get<3>(KF_ensemble_out);
    std::unordered_map<std::string, nc::NdArray<double>> alpha_KF_predict = std::get<4>(KF_ensemble_out); 
    std::vector<nc::NdArray<double>> Sigma_alpha_KF_update = std::get<5>(KF_ensemble_out);
    std::vector<nc::NdArray<double>> gain_KF = std::get<6>(KF_ensemble_out);
    std::vector<nc::NdArray<double>> Sigma_v = std::get<7>(KF_ensemble_out);
    
    // Reconstruct s
    std::unordered_map<std::string, nc::NdArray<double>> s_bGLS = s_from_bGLS_ensemble(alpha_bGLS, A_arr);

    // Reconstruct r
    std::unordered_map<std::string, nc::NdArray<double>> r_arr;
    for (const auto& [key, val] : r) {
        r_arr[key] = nc::NdArray<double>(r.at(key));
    }
    std::unordered_map<std::string, nc::NdArray<double>> r_bGLS = r_from_s_ensemble(s_bGLS, r_arr, w);

    std::unordered_map<std::string, nc::NdArray<double>> r_KF = r_from_s_ensemble(s_KF_predict, r_arr, w);


    EstBGLS est_bGLS;
    est_bGLS.alpha = alpha_bGLS;
    est_bGLS.sigma_v = sigma_v_bGLS;
    est_bGLS.sigma_m = sigma_m_bGLS;
    est_bGLS.s = s_bGLS;
    est_bGLS.r = r_bGLS;

    EstKF est_KF;
    est_KF.alpha_pred = alpha_KF_predict;
    est_KF.sigma2_alpha_pred = sigma2_alpha_KF_predict;
    est_KF.s_pred = s_KF_predict;
    est_KF.sigma2_s_pred = Sigma_s_KF_predict;
    est_KF.alpha_update = alpha_KF_update;
    est_KF.sigma2_alpha_update = Sigma_alpha_KF_update;
    est_KF.gain = gain_KF;
    est_KF.sigma_v = Sigma_v;
    est_KF.r = r_KF;

    Estimates ests;
    ests.bGLS = est_bGLS;
    ests.KF = est_KF;

    return ests;
}

/*
 * ALPHAS AVERAGE (ensemble)
 */

std::tuple<std::unordered_map<std::string, std::vector<nc::NdArray<double>>>,
           std::unordered_map<std::string, nc::NdArray<double>>>             make_avg_alpha_ensemble(std::vector<std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>>> estimates) {
    std::vector<std::string> methods;
    for (const auto& [key, val] : estimates[0]) {
        methods.push_back(key);
    }
    std::unordered_map<std::string, std::vector<nc::NdArray<double>>> all_alpha;
    std::unordered_map<std::string, nc::NdArray<double>> avg_alpha;

    for (auto& estimate : estimates) {
        for (const auto& method : methods) {
            if (method == "KF") {
                all_alpha[method].push_back(nc::NdArray<double>(estimate.at(method).at("alpha_update")));
            }
            else {
                all_alpha[method].push_back(nc::NdArray<double>(estimate.at(method).at("alpha")));
            }
        }
    }

    for (const auto& method : methods) {
        auto stacked = nc::vstack(all_alpha.at(method));
  
        avg_alpha[method] = nc::mean(stacked, nc::Axis::ROW);
    }

    return {all_alpha, avg_alpha};
}
