import utils
import numpy as np
import pickle
import time


# Load specific results
with open(r'results\experiment_n6_m6_unc01.pkl', 'rb') as file:
    data_loaded = pickle.load(file)

A_list = data_loaded['A_list']
B_list = data_loaded['B_list']
theta_star = data_loaded['theta_star'] # || ||_inf and || ||_2 <= 1 
lambda_val = data_loaded['lambda_LMI']

# Common Parameters
n = A_list[0].shape[0]
m = B_list[0].shape[1]

N_sim = 30
Horizon = 30
Qx = np.eye(n)
Qu = np.eye(m)
F = np.zeros((1, n))
G = np.zeros((1, m))
theta_bar_0 = np.zeros(3)

# Polytopic MPC 
PI_W = np.vstack([np.eye(n), -np.eye(n)])
pi_w = np.ones(2 * n) * 0.05
M_theta = np.vstack([np.eye(3), -np.eye(3)]) 
mu_0 = np.ones(6)
vertices_parameter = utils.polytope_vertices(M_theta, mu_0)

K_poly, P_poly = utils.compute_K_poly_LMI_lambda(A_list, B_list, vertices=vertices_parameter, lambda_val=lambda_val)
T_poly, alpha = utils.outerbounding_polytope(P_poly)
U_list_poly = utils.polytope_U(T_poly, alpha)
N_u = 2

# Generate a single random initial condition
x0 = np.random.randn(n) * 5

# Initialize disturbances and system
W = utils.random_polytopic_disturbance(PI_W, pi_w, N_sim=N_sim, random=False)
system = utils.LTI_AFFINE(A_list, B_list, theta_star, x0)

param_poly = utils.MPC_POLY_PARAMETERS(
    N=Horizon,
    T=T_poly,
    U_list=U_list_poly,
    K=K_poly,
    Qx=Qx,
    Qu=Qu,
    M_theta=M_theta,
    mu_t=mu_0,
    Pi_w=PI_W,
    pi_w=pi_w,
    theta_bar=theta_bar_0,
    x0=x0,
    F=F,
    G=G,
)

mpc_polytopic = utils.MPC_POLY(system, param_poly)

print('Simulating closed loop with fully polytopic MPC...\n')
X_poly, U_poly, Theta_estimator_poly, times_poly = utils.simulate_closed_loop_poly(
    system=system,
    controller=mpc_polytopic,
    N_sim=N_sim,
    W=W,
    N_u=N_u,
    report=True
)

Theta_plot = [Theta_estimator_poly[i] for i in range(min([N_sim,5]))]

utils.plot_parameter_polytopes_3d(
    theta_list=Theta_plot, 
    M_theta=M_theta, 
    interactive=False, 
    true_theta=theta_star
)

# Reset system state for Hybrid MPC comparison
system.set_state(x0)

# Hybrid MPC setup
w_inf_max = 0.05
P_0 = np.eye(3)
P_0_sqrt = np.linalg.cholesky(P_0).T
R = np.eye(n) * w_inf_max**2
W_hybrid = utils.random_ellipsoidal_disturbance(R, N_sim, random=False)

param_hybrid = utils.MPC_PARAMETERS(
    N=Horizon,
    T=T_poly,
    U_list=U_list_poly,
    K=K_poly,
    Qx=Qx,
    Qu=Qu,
    Pt=P_0,
    R=R,
    F=F,
    G=G,
    theta_hat=theta_bar_0,
    x0=x0
)

mpc_hybrid = utils.MPC(system, param_hybrid)

print('Simulating closed loop with hybrid MPC...')
X_hybrid, U_hybrid, ALPHA_hybrid, Theta_estimate_hybrid, Theta_control_hybrid, times_hybrid = utils.simulate_closed_loop(
    system, 
    mpc_hybrid, 
    N_sim, 
    W_hybrid, 
    random=True, 
    report=True
)

utils.plot_ellipsoids(
    ellipsoids_list=Theta_estimate_hybrid,
    interactive=False,
)

utils.plot_ellipsoids(
    ellipsoids_list=Theta_control_hybrid,
    interactive=False,
)

print('-' * 50)
print('FULLY POLYTOPIC')
print(f'Total time: {times_poly["total"]}\nTime for update: {times_poly["update"]}\nTime for solving the MPC: {times_poly["solve"]}')

print('\nHYBRID')
print(f'Total time: {times_hybrid["total"]}\nTime for update: {times_hybrid["update"]}\nTime for solving the MPC: {times_hybrid["solve"]}')

entry_log = {
    'X_hybrid': X_hybrid,
    'U_hybrid': U_hybrid,
    'X_poly': X_poly, 
    'U_poly': U_poly, 
    'Theta_poly': Theta_estimator_poly, 
    'Theta_hybrid': Theta_estimate_hybrid, 
    'times_poly': times_poly, 
    'times_hybrid': times_hybrid
}

# Wrap in a list to keep the output structure consistent with the previous parallel version
data_to_save = [entry_log]

with open(r'results\comparison_data.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

print('\nData saved successfully')