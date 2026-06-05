import numpy as np
import utils
import cvxpy as cp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import joblib

# ------------------------------------------------------------------
# 1. UTILITIES FOR JOBLIB & TQDM
# ------------------------------------------------------------------
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

def is_mpc_feasible(mpc, x0):
    """
    Checks if the MPC problem is feasible for a given initial state x0.
    Returns:
        1 : Feasible (Optimal solution found)
        0 : Mathematically Infeasible
       -1 : Numerical crash (SolverError due to extreme values)
    """
    try:
        mpc.update_problem(xt=x0)
        mpc.solve_problem()
        status = mpc.problem.status
        
        if status in [cp.OPTIMAL]:
            return 1
        elif status in [cp.INFEASIBLE]:
            return 0
        else:
            return -1
            
    except cp.error.SolverError:
        return -1
    except Exception:
        return -1

# ------------------------------------------------------------------
# 2. SYSTEM DEFINITION & POLYTOPIC BASELINE SETUP
# ------------------------------------------------------------------
A0 = np.array([[0.5, 0.2], [-0.1, 0.6]])
A1 = np.array([[0.042, 0], [0.072, 0.03]])
A2 = np.array([[0.015, 0.019], [0.009, 0.035]])
A3 = np.array([[0., 0], [0., 0.]])
A_list = [A0, A1, A2, A3]

B0 = np.array([[0], [0.5]])
B1 = np.array([[0.], [0.]])
B2 = np.array([[0.], [0.]])
B3 = np.array([[0.397], [0.059]])
B_list = [B0, B1, B2, B3]

theta_true = np.array([.8, .2, -.5])

# Constraints Fx + Gu <= 1 (x_2 >= -0.3 and u <= 1)
F = np.array([[0, -1/.3], [0, 0]])
G = np.array([[0], [1.]])

# Polytopic Parameter Set constraints (M_theta * theta <= mu_theta)
M_theta = np.vstack([np.eye(3), -np.eye(3)]) 
mu_theta = np.ones(6)
vertices_parameter = utils.polytope_vertices(M_theta, mu_theta)

# Polytopic Disturbance constraints (PI_W * w <= pi_w)
w_inf_max = 0.05
PI_W = np.vstack([np.eye(2), -np.eye(2)])
pi_w = np.ones(4) * w_inf_max

Horizon = 30
Qx = np.eye(2)
Qu = np.array([[1]])

# Compute baseline Polytopic parameters
K_poly, P_lyap_poly = utils.compute_K_poly_LMI_lambda(A_list, B_list, vertices=vertices_parameter)
T_poly, alpha_poly = utils.outerbounding_polytope(P_lyap_poly)
U_list_poly = utils.polytope_U(T_poly, alpha_poly)

param_poly = utils.MPC_POLY_PARAMETERS(
    N=Horizon, T=T_poly, U_list=U_list_poly, K=K_poly, 
    Qx=Qx, Qu=Qu, M_theta=M_theta, mu_t=mu_theta, 
    Pi_w=PI_W, pi_w=pi_w, theta_bar=np.zeros(3), 
    x0=np.zeros(2), F=F, G=G
)

system_poly = utils.LTI_AFFINE(A_list, B_list, theta_true, np.zeros(2))
mpc_poly = utils.MPC_POLY(system_poly, param_poly)

# ------------------------------------------------------------------
# 2.5. PREPARE HYBRID MPC (BASE CASE) FOR BOUNDING BOX
# ------------------------------------------------------------------
print("\n--- PREPARING HYBRID MPC (BASE) FOR BOUNDING BOX ---")

def solve_lmi_for_P0(P_0_matrix, theta_0, label):
    """Solves the LMI once for a unique P_0 matrix to extract K."""
    print(f"Solving LMI for {label}...")
    try:
        P_0_sqrt = np.linalg.cholesky(P_0_matrix).T
        K_hybrid, P_lyap_hybrid = utils.compute_K_LMI_lambda(A_list, B_list, P_0_sqrt, theta_0)
        
        if K_hybrid is not None:
            T_hybrid, alpha_hybrid = utils.outerbounding_polytope(P_lyap_hybrid)
            U_list_hybrid = utils.polytope_U(T_hybrid, alpha_hybrid)
            return K_hybrid, T_hybrid, U_list_hybrid
            
    except Exception as e:
        print(f"Failed to compute K for {label}: {e}")
        
    return None, None, None

def pack_hybrid_params(K, T, U_list, Pt, theta_t, R_mat):
    """Safely assembles the MPC parameter object for the hybrid cases."""
    if K is None: 
        return None
    return utils.MPC_PARAMETERS(
        N=Horizon, T=T, U_list=U_list, K=K, 
        Qx=Qx, Qu=Qu, Pt=Pt, R=R_mat, F=F, G=G, 
        theta_hat=theta_t, x0=np.zeros(2)
    )

R_base = np.eye(2) * (0.05**2)
P_0_base = np.eye(3)
theta_0_base = np.zeros(3)

K_base, T_base, U_base = solve_lmi_for_P0(P_0_base, theta_0_base, "P_0 Base (for Bounding Box)")
param_hybrid_base = pack_hybrid_params(K_base, T_base, U_base, P_0_base, theta_0_base, R_base)

system_hybrid_base = utils.LTI_AFFINE(A_list, B_list, theta_true, np.zeros(2))
mpc_hybrid_base = utils.MPC(system_hybrid_base, param_hybrid_base)

# ------------------------------------------------------------------
# 3. PHASE 1: DYNAMIC BOUNDING BOX (MULTIPLICATIVE + BISECTION)
# ------------------------------------------------------------------
def check_edge_feasibility(mpc, constant_axis, constant_val, range_min, range_max, num_points=15):
    """Checks if a specific edge of the testing box touches any strictly feasible points."""
    test_values = np.linspace(range_min, range_max, num_points)
    
    for val in test_values:
        pt = np.array([constant_val, val]) if constant_axis == 'x' else np.array([val, constant_val])
        if is_mpc_feasible(mpc, pt) == 1:
            return True
    return False

def find_boundary(mpc, axis, direction, start_val, other_min, other_max, factor=1.5, tol=0.05):
    """Finds the exact boundary using Multiplicative Expansion followed by Bisection."""
    val = start_val
    prev_val = 0.0 
    
    while check_edge_feasibility(mpc, axis, val, other_min, other_max):
        prev_val = val
        val *= factor
        if val == 0: 
            val = direction * 0.1 

    low = min(prev_val, val)
    high = max(prev_val, val)
    
    while (high - low) > tol:
        mid = (low + high) / 2.0
        
        if check_edge_feasibility(mpc, axis, mid, other_min, other_max):
            if direction == 1: low = mid  
            else: high = mid              
        else:
            if direction == 1: high = mid 
            else: low = mid               
            
    return low if direction == 1 else high

print("\n--- PHASE 1: Estimating Bounding Box with Hybrid MPC (Base) ---")
x_min, x_max = -0.1, 0.1
y_min, y_max = -0.1, 0.1

for iteration in range(2): 
    x_max = find_boundary(mpc_hybrid_base, 'x',  1, x_max, y_min, y_max)
    x_min = find_boundary(mpc_hybrid_base, 'x', -1, x_min, y_min, y_max)
    y_max = find_boundary(mpc_hybrid_base, 'y',  1, y_max, x_min, x_max)
    y_min = find_boundary(mpc_hybrid_base, 'y', -1, y_min, x_min, x_max)

width = x_max - x_min
height = y_max - y_min
center_x = (x_max + x_min) / 2.0
center_y = (y_max + y_min) / 2.0

print(f"Tight Boundaries Found: X:[{x_min:.3f}, {x_max:.3f}], Y:[{y_min:.3f}, {y_max:.3f}]")

# ------------------------------------------------------------------
# 4. PHASE 2: HIGH-RESOLUTION GRID EVALUATION
# ------------------------------------------------------------------
print("\n--- PHASE 2: Pre-calculating remaining K matrices ---")

vol_factor_R = 4 / np.pi
vol_factor_P = (6 / np.pi)**(2/3)

R_scaled = np.eye(2) * vol_factor_R * (0.05**2)
P_0_scaled = np.eye(3) * vol_factor_P 
theta_0_scaled = np.zeros(3)

precomputed_params = {0: param_poly} 

K_scaled, T_scaled, U_scaled = solve_lmi_for_P0(P_0_scaled, theta_0_scaled, "P_0 Scaled (Cases 2 & 4)")

precomputed_params[1] = param_hybrid_base 
precomputed_params[2] = pack_hybrid_params(K_scaled, T_scaled, U_scaled, P_0_scaled, theta_0_scaled, R_base)
precomputed_params[3] = pack_hybrid_params(K_base, T_base, U_base, P_0_base, theta_0_base, R_scaled)
precomputed_params[4] = pack_hybrid_params(K_scaled, T_scaled, U_scaled, P_0_scaled, theta_0_scaled, R_scaled)

# CASE 5: Ablation study. Using Scaled matrices, but forcing the Polytopic K and Tube
precomputed_params[5] = pack_hybrid_params(
    K=K_poly, 
    T=T_poly, 
    U_list=U_list_poly, 
    Pt=P_0_scaled, 
    theta_t=theta_0_scaled, 
    R_mat=R_scaled
)

def evaluate_point(x0, case_idx, params_dict):
    """Parallel worker function."""
    case_params = params_dict[case_idx]
    
    if case_params is None:
        return 0 
        
    sys_worker = utils.LTI_AFFINE(A_list, B_list, theta_true, x0)
    
    if case_idx == 0:
        mpc_worker = utils.MPC_POLY(sys_worker, case_params)
    else:
        mpc_worker = utils.MPC(sys_worker, case_params)
    
    return is_mpc_feasible(mpc_worker, x0)

print("\n--- PHASE 2: High Resolution Parallel Evaluation ---")
grid_resolution_x = 100
grid_resolution_y = int((grid_resolution_x) * (height / width)) 
grid_resolution_y = max(10, grid_resolution_y) 

grid_x = np.linspace(x_min, x_max, grid_resolution_x)
grid_y = np.linspace(y_min, y_max, grid_resolution_y)

X1, X2 = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([X1.ravel(), X2.ravel()]).T

# Added the 6th configuration to the results dictionary
results = {
    'Polytopic Baseline': [],
    'Case 1 (Same Radii)': [],
    'Case 2 (Radius R, Vol P)': [],
    'Case 3 (Vol R, Radius P)': [],
    'Case 4 (Same Volumes)': [],
    'Case 5 (Polytopic K, Scaled R&P)': []
}

for case_idx, case_name in enumerate(results.keys()):
    print(f"\nEvaluating: {case_name}")
    
    if precomputed_params[case_idx] is None:
        print(f"Skipping {case_name} grid evaluation (LMI was globally infeasible).")
        feasibility_array = [0] * len(grid_points)
    else:
        tasks = [delayed(evaluate_point)(pt, case_idx, precomputed_params) for pt in grid_points]
        
        with tqdm_joblib(tqdm(desc="Evaluating Grid", total=len(tasks), smoothing=0.1)):
            feasibility_array = Parallel(n_jobs=-2)(tasks)
        
    results[case_name] = np.array(feasibility_array).reshape(X1.shape)

# ------------------------------------------------------------------
# 5. VISUALIZATION
# ------------------------------------------------------------------
print("\n--- Plotting Results ---")
# Increased the number of subplots to 6 and widened the figure slightly
fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
fig.suptitle('Feasible Regions of Initial States (Polytopic vs Hybrid Approaches)', fontsize=16)

# Added a 6th color map for the new case
colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'PuBu']

for ax, (case_name, feasibility_grid), cmap in zip(axes, results.items(), colors):
    grid_int = feasibility_grid.astype(int)
    
    levels = [-1.5, -0.5, 0.5, 1.5]
    plot_colors = ['#555555', 'white', plt.get_cmap(cmap)(0.6)]
    
    cf = ax.contourf(X1, X2, grid_int, levels=levels, colors=plot_colors, alpha=0.8)
    ax.contour(X1, X2, grid_int, levels=[0.5], colors='black', linewidths=1.5)
    
    ax.set_title(case_name, fontsize=10)
    ax.set_xlabel('$x_1$')
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].set_ylabel('$x_2$')
plt.tight_layout()
plt.show()