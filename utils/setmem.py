import numpy as np
import scipy.optimize as opt
from scipy.special import gamma


def update_ellipsoid(P_t, theta_t, h, phi, Q):
    """
    Computes the a posteriori ellipsoid $\Theta_{t+1}$ following Theorem 1 
    and verifying the bypass condition (Eq. 26) for $\rho = 0$.
    """
    h = np.atleast_1d(h).flatten()
    theta_t = np.atleast_1d(theta_t).flatten()
    n = len(theta_t)
    
    # 1. Computation of innovation epsilon (Eq. 22)
    epsilon = h - phi.dot(theta_t)
    
    # Inverse of Q (noise shape matrix v(k))
    Q_inv = np.linalg.inv(Q)
    
    # 2. Verification of the condition in Equation 26
    # n(1 - epsilon^T Q^-1 epsilon) - tr(P_t phi^T Q^-1 phi) > 0
    eps_term = np.dot(epsilon, Q_inv.dot(epsilon))
    trace_term = np.trace(P_t.dot(phi.T).dot(Q_inv).dot(phi))
    
    condition_eq26 = n * (1.0 - eps_term) - trace_term
    
    if condition_eq26 > 0:
        # The observation does not reduce uncertainty, skip optimization
        rho_opt = 0.0
    else:
        # 3. Root-finding (Equation 25)
        # Pre-compute eigenvalues lambda_i of the matrix: P_t * phi^T * Q^-1 * phi
        eigenvalue_matrix = P_t.dot(phi.T).dot(Q_inv).dot(phi)
        lambdas = np.real(np.linalg.eigvals(eigenvalue_matrix))
        
        def optimality_equation(rho):
            if rho <= 0:
                return -np.inf
            try:
                # Recurrent inverse term: [rho^-1 Q + phi P_t phi^T]^-1
                inv_term = np.linalg.inv((1.0 / rho) * Q + phi.dot(P_t).dot(phi.T))
                
                # Computation of beta(rho) (Eq. 24)
                beta = 1.0 + rho - epsilon.T @ inv_term @ epsilon
                if beta <= 1e-10:
                    return -np.inf
                    
                # Computation of derivative beta'(rho)
                term_mid = (1.0 / (rho**2)) * Q
                beta_prime = 1.0 - np.dot(epsilon, inv_term.dot(term_mid).dot(inv_term).dot(epsilon))
                
                # Equation 25 (lhs - rhs = 0)
                lhs = np.sum(lambdas / (1.0 + rho * lambdas))
                rhs = n * (beta_prime / beta)
                return lhs - rhs
            except np.linalg.LinAlgError:
                return -np.inf

        try:
            # Find the root using the secant method
            res = opt.root_scalar(optimality_equation, x0=0.1, x1=1.0, method='secant', maxiter=50)
            rho_opt = res.root if res.converged and res.root > 0 else 0.0
        except ValueError:
            rho_opt = 0.0

    # 4. Update the a posteriori ellipsoid
    if rho_opt == 0.0:
        # If rho is 0, the intersection is entirely dominated 
        # by the a priori ellipsoid. Keep P_t and theta_t unchanged.
        updated = False
        return P_t, theta_t, updated
        
    else:
        # Computation of the gain K (Eq. 21)
        inv_term_K = np.linalg.inv(phi.dot(P_t).dot(phi.T) + (1.0 / rho_opt) * Q)
        K = P_t.dot(phi.T).dot(inv_term_K)
        
        # Update of the center theta (Eq. 20)
        theta_t1 = theta_t + K.dot(epsilon)
        
        # Recalculation of beta(rho) for updating P
        beta = 1.0 + rho_opt - np.dot(epsilon, inv_term_K.dot(epsilon))
        
        # Update of the shape matrix P (Eq. 20)
        I_Kphi = np.eye(n) - K.dot(phi)
        P_t1 = beta * (I_Kphi.dot(P_t).dot(I_Kphi.T) + (1.0 / rho_opt) * K.dot(Q).dot(K.T))
        
        # Protective symmetrization
        P_t1 = (P_t1 + P_t1.T) / 2.0
        
        updated = True
        return P_t1, theta_t1, updated


def compute_feasible_ellipsoid(x_t, x_t1, u_t, R, A0, B0, Ai, Bi, C=None):
    """
    Computes the terms h(t+1), phi(t), and Q required for the update,
    and returns the parameters (P_h, a_h) of the observation ellipsoid Delta.
    """
    x_t = np.atleast_1d(x_t).flatten()
    x_t1 = np.atleast_1d(x_t1).flatten()
    u_t = np.atleast_1d(u_t).flatten()
    n_states = x_t.shape[0]
    n_inputs = u_t.shape[0]
    
    if C is None:
        C = np.eye(n_states)
    else:
        C = np.atleast_2d(C)
        
    A0 = np.atleast_2d(A0)
    B0 = np.atleast_2d(B0) if np.atleast_1d(B0).ndim > 1 else np.atleast_1d(B0).reshape(n_states, n_inputs)

    y_t1 = C.dot(x_t1)
    
    # 1. Computation of h(k+1)
    h = y_t1 - C.dot(A0).dot(x_t) - C.dot(B0).dot(u_t)
    
    # 2. Construction of matrix phi(k)
    phi_cols = []
    # Assume Ai and Bi have the same length (equal to the number of theta parameters)
    for Aii, Bii in zip(Ai, Bi):
        Aii = np.atleast_2d(Aii)
        Bii = np.atleast_2d(Bii) if np.atleast_1d(Bii).ndim > 1 else np.atleast_1d(Bii).reshape(n_states, n_inputs)
        col_i = C.dot(Aii).dot(x_t) + C.dot(Bii).dot(u_t)
        phi_cols.append(col_i)
        
    phi = np.column_stack(phi_cols)
    
    # 3. Computation of Q
    Q = C.dot(R).dot(C.T)
    
    # 4. Computation of parameters for the observation ellipsoid Delta (P_h, a_h)
    # P_h = (phi_inv) * Q * (phi_inv)^T  |  a_h = phi_inv * h
    phi_pinv = np.linalg.pinv(phi)
    a_h = phi_pinv.dot(h)
    P_h = phi_pinv.dot(Q).dot(phi_pinv.T)
    
    # Protective symmetrization for Q and P_h
    Q = (Q + Q.T) / 2.0
    P_h = (P_h + P_h.T) / 2.0
    
    return P_h, a_h, h, phi, Q 


def volume_ellipsoid(P):
    """
    Computes the volume of an n-dimensional ellipsoid given its shape matrix P.
    Ultra-stable version based on log-determinant.
    """
    n = P.shape[0]
    
    # 1. Volume of the unit sphere in n-dimensions
    unit_sphere_volume = (np.pi ** (n / 2.0)) / gamma(n / 2.0 + 1.0)
    
    # 2. Ultra-stable computation of the log-determinant
    # slogdet returns a tuple: (sign, log_determinant)
    sign, log_det = np.linalg.slogdet(P)
    
    # Protection: if the sign is <= 0, the ellipsoid is degenerate (zero volume)
    if sign <= 0:
        return 0.0
        
    # 3. Computation of the volume: V = V_n * sqrt(det) = V_n * exp(0.5 * log_det)
    volume = unit_sphere_volume * np.exp(0.5 * log_det)
    
    return volume