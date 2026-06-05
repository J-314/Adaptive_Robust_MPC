import numpy as np
import cvxpy as cp

def compute_unfalsified_polytope(x_t, x_t1, u_t, Pi_w, pi_w, A0, B0, Ai, Bi):
    """
    Computes P_t and q_t defining the unfalsified parameter set Delta_t:
    Delta_t = { theta : P_t @ theta <= q_t }
    Based on disturbance bounds W = { w : Pi_w @ w <= pi_w }
    """
    n_x = x_t.shape[0]
    n_p = len(Ai)
    
    # D_{t-1} matrix where column p is (A_p * x_t + B_p * u_t)
    D_t_minus_1 = np.zeros((n_x, n_p))
    for p in range(n_p):
        D_t_minus_1[:, p] = Ai[p] @ x_t + Bi[p] @ u_t
        
    # d_{t-1} vector
    d_t_minus_1 = A0 @ x_t + B0 @ u_t
    
    # Unfalsified set matrices
    P_t = -Pi_w @ D_t_minus_1
    q_t = pi_w - Pi_w @ (x_t1 - d_t_minus_1)
    
    return P_t, q_t

def update_parameter_polytope(M_theta, mu_prev, P_history, q_history):
    """
    Updates the size vector mu_t for the fixed-complexity polytope Theta_t:
    Theta_t = { theta : M_theta @ theta <= mu_t }
    Solves an LP for each facet to find the minimal mu_t containing the intersection.
    """
    r = M_theta.shape[0]
    mu_t = np.zeros(r)
    
    # H_all @ theta <= h_all represents intersection of Theta_{t-1} and recent Delta_j
    H_all = [M_theta] + P_history
    h_all = [mu_prev] + q_history
    
    H_mat = np.vstack(H_all)
    h_vec = np.concatenate(h_all)
    
    # Solve dual LP for each row of M_theta to bound the intersection
    for i in range(r):
        Lambda = cp.Variable(H_mat.shape[0], nonneg=True)
        objective = cp.Minimize(Lambda @ h_vec)
        constraints = [Lambda @ H_mat == M_theta[i, :]]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='MOSEK')
        
        if prob.status in ["optimal", "optimal_inaccurate"]:
            mu_t[i] = prob.value
        else:
            # Fallback in case of numerical issues
            mu_t[i] = mu_prev[i]
            
    return mu_t

def project_nominal_parameter(theta_bar_prev, M_theta, mu_t):
    """
    Projects the previous nominal parameter vector onto the new Theta_t.
    """
    theta = cp.Variable(M_theta.shape[1])
    objective = cp.Minimize(cp.sum_squares(theta - theta_bar_prev))
    constraints = [M_theta @ theta <= mu_t]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK')
    
    return theta.value