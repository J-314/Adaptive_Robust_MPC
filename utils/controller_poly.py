import numpy as np
import cvxpy as cp
import scipy as sp
from dataclasses import dataclass
from .controller import compute_Qf_theta # Re-use your existing Lyapunov solver

@dataclass
class MPC_POLY_PARAMETERS():
    N: int
    T: np.ndarray
    U_list: list
    K: np.ndarray
    Qx: np.ndarray
    Qu: np.ndarray
    M_theta: np.ndarray  # Fixed complexity directions (r x p)
    mu_t: np.ndarray     # Dynamic polytope sizes (r)
    Pi_w: np.ndarray     # Disturbance bounding box matrix
    pi_w: np.ndarray     # Disturbance bounding box vector
    theta_bar: np.ndarray # Nominal parameter
    x0: np.ndarray
    F: np.ndarray        # State constraint matrix
    G: np.ndarray        # Input constraint matrix

class MPC_POLY():
    def __init__(self, affine_lti, param: 'MPC_POLY_PARAMETERS'):
        self.system = affine_lti
        self.param = param
        self._define_problem()   

    def _define_problem(self):
        param = self.param
        system = self.system
        
        A0, Ai = system.A0, system.Ai
        B0, Bi = system.B0, system.Bi
        N, T, U_list, K = param.N, param.T, param.U_list, param.K
        Qx, Qu, theta_bar = param.Qx, param.Qu, param.theta_bar
        F, G = param.F, param.G
        
        n_x = A0.shape[0]
        n_u = B0.shape[1]
        n_p = len(Ai)
        n_T = T.shape[0]
        n_vert = len(U_list)
        r = param.M_theta.shape[0]
        
        # Calculate maximum disturbance in each direction of T
        # w_bar[i] = max_{Pi_w * w <= pi_w} T[i] * w
        self.w_bar = np.zeros(n_T)
        for i in range(n_T):
            w_var = cp.Variable(n_x)
            prob_w = cp.Problem(cp.Maximize(T[i] @ w_var), [param.Pi_w @ w_var <= param.pi_w])
            prob_w.solve(solver='MOSEK')
            self.w_bar[i] = prob_w.value

        Qf = compute_Qf_theta([A0] + Ai, [B0] + Bi, K, Qx, Qu, theta_bar)
        
        self.x_k = cp.Variable((N+1, n_x))
        self.v_k = cp.Variable((N, n_u))
        self.alpha_k = cp.Variable((N+1, n_T))

        # Setup parameters for warm-starting and efficient resolving
        self.Qf_sqrt = cp.Parameter((n_x, n_x), value=np.linalg.cholesky(Qf).T)
        self.xt = cp.Parameter(n_x, value=param.x0)
        self.mu_t = cp.Parameter(r, value=param.mu_t)
        self.M_theta = param.M_theta
        
        self.A_theta = cp.Parameter(A0.shape, value=system.A_theta(theta_bar))
        self.B_theta = cp.Parameter(B0.shape, value=system.B_theta(theta_bar))

        # Cost Formulation
        J = 0.
        for k in range(N):
            J += cp.quad_form(self.x_k[k], Qx) + cp.quad_form(K @ self.x_k[k] + self.v_k[k], Qu)
        J += cp.sum_squares(self.Qf_sqrt @ self.x_k[N])

        constraints = []
        
        # 1. Nominal Dynamics
        for i in range(N):
            constraints.append(self.x_k[i+1] == (self.A_theta + self.B_theta @ K) @ self.x_k[i] + self.B_theta @ self.v_k[i])

        # 2. Initial Condition & Tube initialization
        constraints.append(self.x_k[0] == self.xt)
        constraints.append(T @ self.xt <= self.alpha_k[0])

        # 3. State and Input Constraints (Vectorized implementation)
        for j in range(n_vert):
            Mat_alpha = (F + G @ K) @ U_list[j]
            Mat_v = G
            
            # Constraints over the horizon k = 0, ..., N-1
            constraints.append(self.alpha_k[:-1] @ Mat_alpha.T + self.v_k @ Mat_v.T <= 1.)
            
            # Terminal constraint for k = N
            constraints.append(Mat_alpha @ self.alpha_k[N] <= 1.)

        # 4. Robust Polytopic Tube Constraints using Dual Variables (Farkas' Lemma)
        for j in range(n_vert):
            h_alpha = T @ (A0 + B0 @ K) @ U_list[j]
            h_v = T @ B0
            
            for k in range(N):
                Lambda_kj = cp.Variable((n_T, r), nonneg=True)
                rhs_cols = []
                for p in range(n_p):
                    H_alpha_p = T @ (Ai[p] + Bi[p] @ K) @ U_list[j]
                    H_v_p = T @ Bi[p]
                    rhs_cols.append(H_alpha_p @ self.alpha_k[k] + H_v_p @ self.v_k[k])
                
                RHS = cp.hstack([cp.reshape(c, (n_T, 1), order='C') for c in rhs_cols])
                
                constraints.append(Lambda_kj @ self.M_theta == RHS)
                constraints.append(Lambda_kj @ self.mu_t <= self.alpha_k[k+1] - h_alpha @ self.alpha_k[k] - h_v @ self.v_k[k] - self.w_bar)
            
            # Terminal Tube Constraint (k = N)
            Lambda_Nj = cp.Variable((n_T, r), nonneg=True)
            rhs_cols_N = []
            for p in range(n_p):
                H_alpha_p = T @ (Ai[p] + Bi[p] @ K) @ U_list[j]
                rhs_cols_N.append(H_alpha_p @ self.alpha_k[N])
                
            RHS_N = cp.hstack([cp.reshape(c, (n_T, 1), order='C') for c in rhs_cols_N])
            constraints.append(Lambda_Nj @ self.M_theta == RHS_N)
            constraints.append(Lambda_Nj @ self.mu_t <= self.alpha_k[N] - h_alpha @ self.alpha_k[N] - self.w_bar)

        self.problem = cp.Problem(cp.Minimize(J), constraints)

    def update_problem(self, xt=None, mu_t=None, theta_bar=None):
        if xt is not None:
            self.xt.value = xt
        if mu_t is not None:
            self.mu_t.value = mu_t
        if theta_bar is not None:
            self.A_theta.value = self.system.A_theta(theta_bar)
            self.B_theta.value = self.system.B_theta(theta_bar)
            A_list = [self.system.A0] + self.system.Ai
            B_list = [self.system.B0] + self.system.Bi
            Qf = compute_Qf_theta(A_list, B_list, self.param.K, self.param.Qx, self.param.Qu, theta_bar)
            self.Qf_sqrt.value = np.linalg.cholesky(Qf).T

    def solve_problem(self):
        self.problem.solve(solver='MOSEK', warm_start=False)
        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
        #     raise ValueError(f"Polytopic optimization failed. Status: {self.problem.status}")
            pass

    def optimal_input_sequence(self):
        return self.x_k.value[:self.param.N] @ self.param.K.T + self.v_k.value


def compute_K_poly_contraction(A_list, B_list, T, alpha, vertices, M_theta, mu_0, verbose=False):
    """
    Computes the robust feedback gain K that minimizes the contraction factor lambda
    over a polytopic parameter uncertainty set defined by M_theta @ theta <= mu_0.
    """
    n_T = T.shape[0]
    n_p = len(A_list) - 1
    n_x = A_list[0].shape[0]
    n_u = B_list[0].shape[1]
    n_vert = vertices.shape[0]
    r = M_theta.shape[0]

    lam = cp.Variable()
    K = cp.Variable((n_u, n_x))
    constraints = []
    
    # We want T(A(theta) + B(theta)K) v_j <= lam * 1
    # for all j, and for all theta such that M_theta * theta <= mu_0
    for j in range(n_vert):
        v_j = vertices[j]
        
        # Nominal vector term: h_0 = T(A_0 + B_0 K) v_j
        h_0 = T @ (A_list[0] + B_list[0] @ K) @ v_j
        
        # Uncertain matrix term: H_theta[:, p-1] = T(A_p + B_p K) v_j
        H_theta_cols = []
        for p in range(1, n_p + 1):
            col = T @ (A_list[p] + B_list[p] @ K) @ v_j
            H_theta_cols.append(cp.reshape(col, (n_T, 1),order='C'))
            
        H_theta = cp.hstack(H_theta_cols)
        
        # Farkas' lemma dual variables (one matrix for each vertex)
        Lambda_j = cp.Variable((n_T, r), nonneg=True)
        
        # Duality constraints ensuring robustness over the whole parameter polytope
        constraints.append(Lambda_j @ M_theta == H_theta)
        constraints.append(h_0 + Lambda_j @ mu_0 <= lam *alpha)
    # Solve the problem
    objective = cp.Minimize(lam)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='MOSEK')
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        # raise ValueError(f"Failed to find K. Optimization status: {problem.status}")
        pass
    if verbose:
        print('Contractive factor lambda = ', lam.value)
        print('Feedback gain K = \n', K.value)
        
    return K.value, lam.value

def compute_K_poly_LMI(A_list, B_list, vertices, verbose=False):
    """Computes a robust stabilizing feedback gain K for a linear dynamic system

    over a polytopic uncertainty set defined by its vertices.

    The system is given by:
        x+ = A(theta)x + B(theta)u
    where A(theta) and B(theta) are affine functions of theta.

    Parameters:
    - A_list: List of numpy arrays [A_0, A_1, ..., A_p]
    - B_list: List of numpy arrays [B_0, B_1, ..., B_p]
    - vertices: Numpy array of shape (n_vertices, p) containing the polytope vertices
    - verbose: Boolean flag for printing solver progress and results

    Returns:
    - K: Robust feedback gain matrix of shape (n_u, n_x), or None if infeasible.
    - P: solution of the lyapunov inequality (A + BK)' P (A + BK) - P < 0  
    """
    n_x = A_list[0].shape[0]
    n_u = B_list[0].shape[1]
    n_p = len(A_list) - 1
    n_vert = vertices.shape[0]

    # Decision variables for the LMI
    # Q = P^-1 (Symmetric Positive Definite)
    # Y = K * Q
    Q = cp.Variable((n_x, n_x), PSD=True)
    Y = cp.Variable((n_u, n_x))

    # Constraint list starting with numerical positive definiteness of Q
    constraints = [Q >> 1e-6 * np.eye(n_x)]

    # Impose the stability LMI at each vertex of the parameter polytope
    for j in range(n_vert):
        theta_v = vertices[j]

        # Construct vertex system matrices: A(v_j) and B(v_j)
        A_v = A_list[0] + sum(
            A_list[i + 1] * theta_v[i] for i in range(n_p)
        )
        B_v = B_list[0] + sum(
            B_list[i + 1] * theta_v[i] for i in range(n_p)
        )

        # Closed-loop matrix term multiplied by Q: A_cl(v_j) * Q = A_v * Q + B_v * Y
        A_v_Q_Y = A_v @ Q + B_v @ Y

        # Schur complement formulation for discrete-time Lyapunov stability:
        # [ Q           (A_cl * Q)^T ]
        # [ A_cl * Q        Q        ] > 0
        lmi = cp.bmat([[Q, A_v_Q_Y.T], [A_v_Q_Y, Q]])

        # Enforce strict positive definiteness of the LMI block
        constraints.append(lmi >> 1e-6 * np.eye(2 * n_x))

    # Define and solve the pure feasibility problem
    problem = cp.Problem(cp.Maximize(0), constraints)

    try:
        problem.solve(solver='MOSEK')
    except Exception as e:
        if verbose:
            print(f"Solver encountered an exception: {e}")
        return None, None

    # Check if a valid solution was successfully found
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        if verbose:
            print(
                f"Failed to find a stabilizing K. Solver status: {problem.status}"
            )
        return None, None

    # Recover the feedback gain K = Y * Q^-1
    Q_val = Q.value
    Y_val = Y.value
    P_val = np.linalg.inv(Q_val)
    K_val = Y_val @ P_val

    if verbose:
        print(
            f"Robust stabilizing gain K successfully found over {n_vert} vertices."
        )

    return K_val, P_val

def outerbounding_polytope(P_ellipsoid):
    P = (P_ellipsoid + P_ellipsoid.T)/2
    eig, M = np.linalg.eigh(P)
    T = np.concatenate([M,-M])
    eig_sqrt = np.sqrt(eig)
    alpha = np.concatenate([eig_sqrt, eig_sqrt])
    
    return T, alpha

def build_cvxpy_block_diag(blocks):
    """
    Helper function to build a block diagonal matrix using cp.bmat.
    
    Parameters:
    - blocks: List of square CVXPY expressions or numpy arrays.
    
    Returns:
    - A CVXPY block matrix object.
    """
    n_blocks = len(blocks)
    bmat_grid = []
    
    for i in range(n_blocks):
        row = []
        for j in range(n_blocks):
            if i == j:
                row.append(blocks[i])
            else:
                # Add a zero matrix of the correct shape for off-diagonal blocks
                shape_i = blocks[i].shape[0]
                shape_j = blocks[j].shape[1]
                row.append(np.zeros((shape_i, shape_j)))
        bmat_grid.append(row)
        
    return cp.bmat(bmat_grid)


def compute_K_poly_LMI_quasiconvex(A_list, B_list, vertices, verbose=False):
    """Computes a robust stabilizing feedback gain K for a linear dynamic system
    over a polytopic uncertainty set defined by its vertices, while minimizing
    the contraction rate lambda.

    The system is given by:
        x+ = A(theta)x + B(theta)u
    where A(theta) and B(theta) are affine functions of theta.

    Parameters:
    - A_list: List of numpy arrays [A_0, A_1, ..., A_p]
    - B_list: List of numpy arrays [B_0, B_1, ..., B_p]
    - vertices: Numpy array of shape (n_vertices, p) containing the polytope vertices
    - verbose: Boolean flag for printing solver progress and results

    Returns:
    - K_val: Robust feedback gain matrix of shape (n_u, n_x), or None if infeasible.
    - opt_lambda: The optimized contraction rate (lambda < 1), or None if infeasible.
    """
    n_x = A_list[0].shape[0]
    n_u = B_list[0].shape[1]
    n_p = len(A_list) - 1
    n_vert = vertices.shape[0]

    # Decision variables for the LMI
    # Q = P^-1 (Symmetric Positive Definite)
    # Y = K * Q
    Q = cp.Variable((n_x, n_x), symmetric=True)
    Y = cp.Variable((n_u, n_x))

    # Constraint list starting with numerical positive definiteness of Q
    constraints = [Q >> 1e-6 * np.eye(n_x)]

    A_blocks = []
    B_blocks = []
    zero_mat = np.zeros((n_x, n_x))

    # Impose the stability LMI at each vertex of the parameter polytope
    for j in range(n_vert):
        theta_v = vertices[j]

        # Construct vertex system matrices: A(v_j) and B(v_j)
        A_v = A_list[0] + sum(
            A_list[i + 1] * theta_v[i] for i in range(n_p)
        )
        B_v = B_list[0] + sum(
            B_list[i + 1] * theta_v[i] for i in range(n_p)
        )

        # Closed-loop matrix term multiplied by Q: A_cl(v_j) * Q = A_v * Q + B_v * Y
        X = A_v @ Q + B_v @ Y

        # Block matrix A_mat = [0, X^T; X, 0]
        A_mat_raw = cp.bmat([
            [zero_mat, X.T],
            [X,        zero_mat]
        ])
        A_mat = 0.5 * (A_mat_raw + A_mat_raw.T)  # Enforce symmetry

        # Block matrix B_mat = [Q, 0; 0, Q]
        B_mat_raw = cp.bmat([
            [Q,        zero_mat],
            [zero_mat, Q       ]
        ])
        B_mat = 0.5 * (B_mat_raw + B_mat_raw.T)  # Enforce symmetry
        
        A_blocks.append(A_mat)
        B_blocks.append(B_mat)

    # Combine all vertices into large block matrices using the helper
        A_big_raw = build_cvxpy_block_diag(A_blocks)
        B_big_raw = build_cvxpy_block_diag(B_blocks)

        # CRITICAL FIX: Enforce exact symmetry on the massive matrices
        # otherwise CVXPY/MOSEK numerical precision might fail the GEVP requirements
        A_big = 0.5 * (A_big_raw + A_big_raw.T)
        B_big = 0.5 * (B_big_raw + B_big_raw.T)

        # GEVP objective
        lambda_expr = cp.gen_lambda_max(A_big, B_big)
        objective = cp.Minimize(lambda_expr)

        # Relax the bound slightly. If it works with 1.0, you can tighten it later.
        constraints.append(lambda_expr <= 1.0)

        # Define the DQCP problem
        problem = cp.Problem(objective, constraints)

    if verbose:
        print(f"Is the problem DQCP? {problem.is_dqcp()}")

    try:
        # qcp=True is strictly required for gen_lambda_max
        # MOSEK is preferred; SCS can be used as an open-source fallback
        problem.solve(qcp=True, solver='MOSEK')
    except Exception as e:
        if verbose:
            print(f"Solver encountered an exception: {e}")
        return None, None

    # Check if a valid solution was successfully found
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        if verbose:
            print(f"Failed to find a stabilizing K. Solver status: {problem.status}")
        return None, None

    # Recover the feedback gain K = Y * Q^-1
    Q_val = Q.value
    Y_val = Y.value
    K_val = Y_val @ np.linalg.inv(Q_val)
    opt_lambda = problem.value

    if verbose:
        print(f"Robust stabilizing gain K successfully found over {n_vert} vertices.")
        print(f"Optimal contraction rate (lambda): {opt_lambda:.4f}")

    return K_val, opt_lambda


def compute_K_poly_LMI_lambda(A_list, B_list, vertices, lambda_val=1, verbose=False):
    """
    Computes a robust stabilizing feedback gain K for a linear dynamic system
    over a polytopic uncertainty set, guaranteeing a FIXED contraction rate lambda.

    The system is given by:
        x+ = A(theta)x + B(theta)u

    Parameters:
    - A_list: List of numpy arrays [A_0, A_1, ..., A_p]
    - B_list: List of numpy arrays [B_0, B_1, ..., B_p]
    - vertices: Numpy array of shape (n_vertices, p) containing the polytope vertices
    - lambda_val: Float (0, 1], the desired fixed contraction rate
    - verbose: Boolean flag for printing solver progress and results

    Returns:
    - K_val: Robust feedback gain matrix of shape (n_u, n_x), or None if infeasible.
    """
    n_x = A_list[0].shape[0]
    n_u = B_list[0].shape[1]
    n_p = len(A_list) - 1
    n_vert = vertices.shape[0]

    # Pre-compute the squared lambda scalar
    lambda_sq = lambda_val ** 2

    # Decision variables for the LMI
    Q = cp.Variable((n_x, n_x), symmetric=True)
    Y = cp.Variable((n_u, n_x))

    # We enforce strict positive definiteness with a small numerical margin
    constraints = [Q >> 1e-6 * np.eye(n_x)]

    # Impose the stability LMI at each vertex of the parameter polytope
    for j in range(n_vert):
        theta_v = vertices[j]

        # Construct vertex system matrices: A(v_j) and B(v_j)
        A_v = A_list[0] + sum(
            A_list[i + 1] * theta_v[i] for i in range(n_p)
        )
        B_v = B_list[0] + sum(
            B_list[i + 1] * theta_v[i] for i in range(n_p)
        )

        # Closed-loop matrix term multiplied by Q: X = A_v * Q + B_v * Y
        X = A_v @ Q + B_v @ Y

        # Schur complement formulation for the fixed lambda:
        # [ lambda^2 * Q      X^T ]
        # [ X                 Q   ] >= 0
        top_left = lambda_sq * Q
        top_right = X.T
        bottom_left = X
        bottom_right = Q
        
        lmi = cp.bmat([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])

        # Enforce the LMI to be positive semi-definite
        constraints.append(lmi >> 0)

    # Define the pure feasibility problem (Maximize 0)
    problem = cp.Problem(cp.Maximize(0), constraints)

    try:
        # MOSEK is highly recommended, SCS can be used as fallback
        problem.solve(solver=cp.MOSEK, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"Solver encountered an exception: {e}")
        return None, None

    # Check if a valid solution was successfully found
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        if verbose:
            print(f"Failed to find a stabilizing K for lambda={lambda_val}. Status: {problem.status}")
        return None, None

    # Recover the feedback gain K = Y * Q^-1
    Q_val = Q.value
    Y_val = Y.value
    P_val = np.linalg.inv(Q_val)
    K_val = Y_val @ P_val

    if verbose:
        print(f"Success! Found a robust stabilizing K for lambda = {lambda_val:.4f}")

    return K_val, P_val