############################
## GENERATE ENSEMBLE DATA ##
############################

def generate_ensemble_data(N, Tmkp, alpha, sigma_v, w):
    K = len(Tmkp) # Number of players

    players = [str(k) for k in range(1, K + 1)]
    pairs = [str(k1) + str(k2) for k1 in range(1, K + 1) for k2 in range(1, K + 1) if k1 != k2]

    # Define variables -- with NaN value at n = 0
    T = {player: [np.nan] for player in players}
    t = {player: [np.nan] for player in players}
    A = {pair: [np.nan] for pair in pairs}

    # Initialise values at n = 1
    for player in players:
        T[player].append(0)
        t[player].append(np.random.normal(loc = 0, scale = sigma_v[player]))
    
    for pair in pairs:
        A[pair].append(t[pair[0]][1] - t[pair[1]][1])

    # Simulate onsets
    for n in range(2, N + 1):
        for player in players:
            T[player].append(T[player][n - 1] + Tmkp[player][n])
            
            aux = 0
            for player2 in players:
                if player2 != player:
                    aux += alpha[player + player2][n] * A[player + player2][n - 1]
            t[player].append(t[player][n - 1] + Tmkp[player][n] - aux + np.random.normal(loc = 0, scale = sigma_v[player]))
        
        for pair in pairs:
            A[pair].append(t[pair[0]][n] - t[pair[1]][n])

    # Convert to numpy array
    T = {player: np.array(T[player]) for player in players}
    t = {player: np.array(t[player]) for player in players}
    A = {pair: np.array(A[pair]) for pair in pairs}

    # Compute r and s - NaN for n = 0 and n = 1
    r = {player: 2*[np.nan] for player in players}
    s_true = {player: 2*[np.nan] for player in players}
    s_win = {player: 2*[np.nan] for player in players}

    for n in range (2, N + 1):
        for player in players:
            r[player].append(t[player][n] - t[player][n - 1])
            s_true[player].append(r[player][n] - Tmkp[player][n])
            if n <= w:
                s_win[player].append(r[player][n] - np.mean(r[player][2:n + 1]))
            else:
                s_win[player].append(r[player][n] - np.mean(r[player][(n - w + 1):(n + 1)]))
    
    # Convert to numpy array
    r = {player: np.array(r[player]) for player in players}
    s_true = {player: np.array(s_true[player]) for player in players}
    s_win = {player: np.array(s_win[player]) for player in players}

    return s_true, s_win, r, A, t, T

##################################
## ENSEMBLE bGLS (Peter's code) ##
##################################

def bGLS_ensemble(o_rm):

    # Initialise matrix of Asynchronies
    async_x = o_rm.shape[0] - 1 # height of asynchrony array
    async_y = o_rm.shape[1] # width of asynchrony array
    am = np.zeros((async_x,async_y,async_y))

    for i in range(0,async_x):
        for k in range(0,async_y):
            for j in range(0,async_y):
                am[i,j,k] = o_rm[i + 1,k] - o_rm[i + 1,j] # filling array

    # Calculate Inter-Onset Intervals
    I = o_rm.shape[0] - 1 # number of onsets
    ioi_x = o_rm.shape[0] - 1 # IOI array height
    ioi_y = o_rm.shape[1] # IOI array width

    Rm = np.zeros((ioi_x,ioi_y))
    for i in range(0,ioi_y):
        for j in range(0,ioi_x):
            Rm[j][i] = o_rm[j + 1][i] - o_rm[j][i] # filling array


    number_of_players = o_rm.shape[1]
    alphas = np.zeros((number_of_players, number_of_players)) # Initialise alpha matrix

    o_rm = o_rm[1::]

    for sub in range (0,number_of_players): # iterates through subjects
        R = Rm[:, sub]
        As = np.zeros(((np.size(R, 0)), (number_of_players-1)),  dtype=float)
        others = np.setdiff1d(range(number_of_players), sub)
        for k in range (0,np.size(am,0)):
            for l in range (0,(number_of_players-1)):
                As[k,l] = (am[k, sub, others[l]])
        meanA = np.zeros((1,others.size))
        for i in range(0,others.size):
            meanA[0,i] = np.mean(As[:, i]) # average asynchrony for subject
        meanR = np.mean(R)

        # bGLS!
        iterations = 20
        thresh = 0.001
        N = np.size(R,0)-1
        P = number_of_players-1
        for p in range (0,P):
            As[:,p] = As[:,p] - meanA[0][p]
        b3 = np.vstack(R[1:]-meanR)
        a3 = As[:-1, :]

        k11 = 1
        k12 = 0

        zold = np.zeros((P,1), dtype=float)-9999
        for iteration in range(0,iterations):
            cc = np.diag(k11 * np.ones(N, dtype=float), 0) + np.diag(k12 * np.ones((N - 1), dtype=float), 1) + np.diag(k12 * np.ones((N - 1), dtype=float), -1)
            ic = np.linalg.inv(cc)
            z = np.linalg.pinv(a3.T @ ic @ a3) @ (a3.T @ ic @ b3) # GLS
            d = (a3 @ z - b3).T
            k = np.cov(np.vstack(d[0, :-1]), np.vstack(d[0, 1:]) , rowvar=0) # estimate residual acvf
            k11 = (k[0,0]+k[1,1])/2
            k12 = k[0,1]
          # apply bounds
            if k12>0:
                k12 = 0
            if k11 < ((-3)*k12):
                k11 = ((-3)*k12)
            if sum(abs(z-zold))<thresh:
                break
            zold = z
        finalz = np.insert(z.T, sub, 0)
        alphas[sub] = finalz # insert alpha row into alpha matrix
    sm = np.sqrt(-k12) # Motor noise calculation
    st = np.sqrt(k11-2*(sm**2)) # Timekeeper noise calculation

    alphas_dict = {}

    for player1 in range(number_of_players):
        for player2 in range(number_of_players):
            if player2 != player1:
                alphas_dict[str(player1 + 1) + str(player2 + 1)] = alphas[player1][player2]

    return alphas_dict, sm, st

############################
## ENSEMBLE KALMAN FILTER ##
############################

def KF_ensemble(s, A, Sigma_v_init, Sigma_w = 0.1, alpha_KF_init = 0.25, Sigma_alpha_init = 0.3, est_Sigma_v = False, w = 5):
    K = len(s) # Number of players

    players = [str(k) for k in range(1, K + 1)]
    pairs = [str(k1) + str(k2) for k1 in range(1, K + 1) for k2 in range(1, K + 1) if k1 != k2]
    
    N = s['1'].shape[0] - 1

    # Initialise predictions for alpha (with NaN for n = 0 and n = 1)
    alpha_KF_predict = 2*[np.nan]
    Sigma_alpha_KF_predict = 2*[np.nan]

    # Initialise predictions for s (with NaN for n = 0 and n = 1)
    s_KF_predict = 2*[K*[np.nan]]
    Sigma_s_KF_predict = 2*[np.nan]

    # Initialise updates for alpha (with n = 0 and n = 1)
    gain_KF = 2*[np.nan]
    alpha_KF_update = [K*(K - 1)*[np.nan], alpha_KF_init]
    Sigma_alpha_KF_update = [np.nan, Sigma_alpha_init]

    # Initialise dynamic estimation of sigma_v [TESTING] (with n = 0 and n = 1)
    Sigma_v = [np.nan, Sigma_v_init]

    # Initialise matrix F
    F = [np.nan, np.nan]

    for n in range(2, N + 1):
        # Build matrix F_n
        F_list = []
        for player1 in players:
            A_i = []
            for player2 in players:
                if player2 != player1:
                    A_i.append(-A[player1 + player2][n - 1])
            F_list.append(A_i)
        F.append(block_diag(*F_list))

        # Make vector with s at time n
        s_n_vec = np.array([s[player][n] for player in players])

        # Predict alpha
        alpha_KF_predict.append(alpha_KF_update[n - 1])
        Sigma_alpha_KF_predict.append(Sigma_alpha_KF_update[n - 1] + Sigma_w)

        # Predict s
        s_KF_predict.append(F[n] @ alpha_KF_predict[n])
        Sigma_s_KF_predict.append(F[n] @ Sigma_alpha_KF_predict[n] @ F[n].T + Sigma_v[n - 1])

        # Update alpha
        gain_KF.append(Sigma_alpha_KF_predict[n] @ F[n].T @ np.linalg.inv(Sigma_s_KF_predict[n]))
        alpha_KF_update.append(alpha_KF_predict[n] + gain_KF[n] @ (s_n_vec - s_KF_predict[n]))
        Sigma_alpha_KF_update.append(Sigma_alpha_KF_predict[n] - gain_KF[n] @ F[n] @ Sigma_alpha_KF_predict[n])

        if est_Sigma_v:
            if n <= w:
                Sigma_v.append(Sigma_v_init)
            else:
                # Sigma_v.append(np.sqrt(np.var(s_KF_predict[n - w - 1 + 2:n + 1] - s[n - w - 1 + 2:n + 1]))) ## MAKE!!!!
                Sigma_v.append(np.diag(np.var(np.array(s_KF_predict[n - w - 1 + 2:n + 1]).T - np.array([s[player][n - w - 1 + 2:n + 1] for player in players]), axis = 1)))
        else:
            Sigma_v.append(Sigma_v_init)
    
    # Prepare outputs (the most important ones)
    s_KF_predict = dict(zip(players, np.array(s_KF_predict).T))
    alpha_KF_update = dict(zip(pairs, np.array(alpha_KF_update).T))
    
    return alpha_KF_predict, Sigma_alpha_KF_predict, s_KF_predict, Sigma_s_KF_predict, alpha_KF_update, Sigma_alpha_KF_update, gain_KF, Sigma_v

#############################################################
## CORRELATION AND STANDARD DEVIATION TO COVARIANCE MATRIX ##
#############################################################

def cor2cov(stdevs, V_cor):
    V = np.zeros(V_cor.shape)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i][j] = V_cor[i][j]*stdevs[i]*stdevs[j]
    
    return V

#############################################################
## COVARIANCE TO CORRELATION AND STANDARD DEVIATION MATRIX ##
#############################################################

def cov2cor(V):
    stdevs = np.sqrt(np.diag(V))
    V_cor = np.zeros(V.shape)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_cor[i][j] = V[i][j]/(stdevs[i]*stdevs[j])
    
    return stdevs, V_cor

####################################
## PROCESS ENSEMBLE VIRTUOSO DATA ##
####################################

def process_ensemble_data(leader, rep, w = 5):
    virtuoso = pd.read_csv('virtuoso.csv')

    N = 46
    K = 4

    ioi_cols = ['ioi' + str(k) for k in range(N)]

    players = [str(k) for k in range(1, K + 1)]
    pairs = [str(k1) + str(k2) for k1 in range(1, K + 1) for k2 in range(1, K + 1) if k1 != k2]

    r_dp = virtuoso[(virtuoso['condition'] == 'DP') & (virtuoso['leader'] == leader) & (virtuoso['repetition'] == rep)][ioi_cols].values
    r_nr = virtuoso[(virtuoso['condition'] == 'NR') & (virtuoso['leader'] == leader) & (virtuoso['repetition'] == rep)][ioi_cols].values
    r_sp = virtuoso[(virtuoso['condition'] == 'SP') & (virtuoso['leader'] == leader) & (virtuoso['repetition'] == rep)][ioi_cols].values

    aux = np.empty((4, 2))
    aux.fill(np.nan)
    r_dp = dict(zip(players, np.hstack((aux, r_dp))))
    r_nr = dict(zip(players, np.hstack((aux, r_nr))))
    r_sp = dict(zip(players, np.hstack((aux, r_sp))))

    t_dp = {player: np.hstack((np.nan, 0, r_dp[player][2:].cumsum())) for player in players}
    t_nr = {player: np.hstack((np.nan, 0, r_nr[player][2:].cumsum())) for player in players}
    t_sp = {player: np.hstack((np.nan, 0, r_sp[player][2:].cumsum())) for player in players}

    A_dp = {pair: None for pair in pairs}
    A_nr = {pair: None for pair in pairs}
    A_sp = {pair: None for pair in pairs}
    for pair in pairs:
        A_dp[pair] = t_dp[pair[0]] - t_dp[pair[1]]
        A_nr[pair] = t_nr[pair[0]] - t_nr[pair[1]]
        A_sp[pair] = t_sp[pair[0]] - t_sp[pair[1]]

    # Compute s - NaN for n = 0 and n = 1
    s_dp_win = {player: 2*[np.nan] for player in players}
    s_nr_win = {player: 2*[np.nan] for player in players}
    s_sp_win = {player: 2*[np.nan] for player in players}

    for n in range (2, N + 2):
        for player in players:
            if n <= w:
                s_dp_win[player].append(r_dp[player][n] - np.mean(r_dp[player][2:n + 1]))
                s_nr_win[player].append(r_nr[player][n] - np.mean(r_nr[player][2:n + 1]))
                s_sp_win[player].append(r_sp[player][n] - np.mean(r_sp[player][2:n + 1]))
            else:
                s_dp_win[player].append(r_dp[player][n] - np.mean(r_dp[player][(n - w + 1):(n + 1)]))
                s_nr_win[player].append(r_nr[player][n] - np.mean(r_nr[player][(n - w + 1):(n + 1)]))
                s_sp_win[player].append(r_sp[player][n] - np.mean(r_sp[player][(n - w + 1):(n + 1)]))

    # Convert to numpy array
    s_dp_win = {player: np.array(s_dp_win[player]) for player in players}
    s_nr_win = {player: np.array(s_nr_win[player]) for player in players}
    s_sp_win = {player: np.array(s_sp_win[player]) for player in players}

    return r_dp, r_nr, r_sp, s_dp_win, s_nr_win, s_sp_win, A_dp, A_nr, A_sp, t_dp, t_nr, t_sp

#############################
## SOME METRICS (ensemble) ##
#############################

def metrics_ensemble(s_pred, s_ref):
    metrics = {player: {'corr': None, 'std': None} for player in s_pred.keys()}

    for player in s_pred.keys():
        metrics[player]['corr'] = np.corrcoef(s_pred[player][~np.isnan(s_pred[player])], s_ref[player][~np.isnan(s_pred[player])])[0, 1].round(3)
        metrics[player]['std'] = np.sqrt(np.nanvar(s_pred[player] - s_ref[player])).round(3)
    
    return metrics

###################################################################
## OBTAIN TRACKING OF s FROM alpha ESTIMATE FROM bGLS (ensemble) ##
###################################################################

def s_from_bGLS_ensemble(alpha_est, A):
    N = A['12'].shape[0] - 1

    players = sorted(list({x[0] for x in A.keys()}))

    s_est = {}
    for player in players:
        s_est[player] = 2*[np.nan] # n = 0 and n = 1

    for n in range(2, N + 1):
        for player in players:
            aux = 0
            for player2 in players:
                if player2 != player:
                    aux += alpha_est[player + player2] * A[player + player2][n - 1]
            s_est[player].append(-aux)
    
    for player in players:
        s_est[player] = np.array(s_est[player])
    
    return s_est

###############################################
## RECONSTRUCT r FROM s_estimated (ensemble) ##
###############################################

def r_from_s_ensemble(s_est, r, w = 5):
    N = r['1'].shape[0] - 1
    
    players = r.keys()

    r_est = {}

    for player in players:
        r_est[player] = 2*[np.nan] # n = 0 and n = 1

    for n in range(2, w + 1):
        for player in players:
            r_est[player].append(np.nan)
    
    for n in range(w + 1, N + 1):
        for player in players:
            r_est[player].append(s_est[player][n] + np.mean(r[player][(n - w - 1 + 2):(n - 1 + 2)]))
    
    for player in players:
        r_est[player] = np.array(r_est[player])
    
    return r_est

#########################
## ESTIMATE (ensemble) ##
#########################

def estimate_ensemble(s, r, A, t, w, Sigma_v_init, Sigma_w, alpha_KF_init, Sigma_alpha_init, est_Sigma_v, w_KF):
    N = s['1'].shape[0] - 1

    # Estimate
    alpha_bGLS, sigma_m_bGLS, sigma_v_bGLS = bGLS_ensemble(np.array(list(t.values())).T[1:, ])
    alpha_KF_predict, sigma2_alpha_KF_predict, s_KF_predict, Sigma_s_KF_predict, alpha_KF_update, Sigma_alpha_KF_update, gain_KF, Sigma_v = KF_ensemble(s, A,
                                                                                                                                                        Sigma_v_init = Sigma_v_init,
                                                                                                                                                        Sigma_w = Sigma_w,
                                                                                                                                                        alpha_KF_init = alpha_KF_init,
                                                                                                                                                        Sigma_alpha_init = Sigma_alpha_init,
                                                                                                                                                        est_Sigma_v = est_Sigma_v,
                                                                                                                                                        w = w_KF)
    
    # Reconstruct s
    s_bGLS = s_from_bGLS_ensemble(alpha_bGLS, A)

    # Reconstruct r
    r_bGLS = r_from_s_ensemble(s_bGLS, r, w = 5)

    r_KF = r_from_s_ensemble(s_KF_predict, r, w = w)


    est_bGLS = {'alpha': alpha_bGLS,
                'sigma_v': sigma_v_bGLS,
                'sigma_m': sigma_m_bGLS,
                's': s_bGLS,
                'r': r_bGLS}
    
    est_KF = {'alpha_pred': alpha_KF_predict,
              'sigma2_alpha_pred': sigma2_alpha_KF_predict,
              's_pred': s_KF_predict,
              'sigma2_s_pred': Sigma_s_KF_predict,
              'alpha_update': alpha_KF_update,
              'sigma2_alpha_update': Sigma_alpha_KF_update,
              'gain': gain_KF,
              'sigma_v': Sigma_v,
              'r': r_KF}
    
    ests = {'bGLS': est_bGLS,
            'KF': est_KF}
    
    return ests

###############################
## ALPHAS AVERAGE (ensemble) ##
###############################

def make_avg_alpha_ensemble(estimates):
    methods = estimates[0].keys()
    all_alpha = {key: [] for key in methods}
    avg_alpha = {key: None for key in methods}

    for estimate in estimates:
        for method in methods:
            if method == 'KF':
                all_alpha[method].append(np.array(estimate[method]['alpha_update']))
            else:
                all_alpha[method].append(np.array(estimate[method]['alpha']))
    
    for method in methods:
        avg_alpha[method] = np.array(all_alpha[method]).mean(axis = 0)
    
    return all_alpha, avg_alpha
