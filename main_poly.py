import numpy as np
import utils
from utils.setmem_poly import compute_unfalsified_polytope, update_parameter_polytope, project_nominal_parameter
from utils.controller_poly import MPC_POLY_PARAMETERS, MPC_POLY
import time

# --- System Definition (Identical to main.py) ---
A0 = np.array([[0.5, 0.2], [-0.1, 0.6]])
A1 = np.array([[0.042, 0], [0.072, 0.03]])
A2 = np.array([[0.015, 0.019], [0.009, 0.035]])
A3 = np.array([[0., 0], [0., 0.]])
A_list = [A0, A1, A2, A3]

B0 = np.array([[0], [0.5]])
B1 = np.array([0., 0]).reshape(2, 1)
B2 = np.array([0., 0]).reshape(2, 1)
B3 = np.array([0.397, 0.059]).reshape(2, 1)
B_list = [B0, B1, B2, B3]

theta_true = np.array([.8, .2, -.5])
x0 = np.array([-5, 3])
system = utils.LTI_AFFINE(A_list, B_list, theta_true, x0)

# --- Polytopic Uncertainty Setup ---
# Equivalent to ||w||_inf <= 0.05
Pi_w = np.vstack([np.eye(2), -np.eye(2)])
pi_w = np.ones(4) * 0.05

# Initial Parameter Polytope Theta_0: ||theta||_inf <= 1
M_theta = np.vstack([np.eye(3), -np.eye(3)])
mu_0 = np.ones(6)
theta_bar_0 = np.zeros(3)
vertices_polytope = utils.polytope_vertices(M_theta,mu_0)
# --- Controller Initialization ---
T_width, T_height = 2., 2.
T = np.vstack([np.diag([2/T_width, 2/T_height]), -np.diag([2/T_width, 2/T_height])])
alpha = np.ones(4)
vertices = utils.polytope_vertices(T, alpha)
U_list = utils.polytope_U(T, alpha)

#K, lam = utils.compute_K_poly(A_list, B_list, T, alpha, vertices,M_theta, mu_0, True)
#if lam >=1:
#    raise ValueError('It is not possible to find a K that makes the system lambda-contractive')

lambda_val = 0.99
K,P = utils.compute_K_poly_LMI_lambda(A_list,B_list,vertices_polytope,lambda_val,verbose=False)
if K is None:
    raise ValueError(f'It is not possible to find a stabilizing K with a value of lambda = {lambda_val}')
param = MPC_POLY_PARAMETERS(
    N=10, T=T, U_list=U_list, K=K,
    Qx=np.eye(2), Qu=0.01*np.eye(1),
    M_theta=M_theta, mu_t=mu_0.copy(),
    Pi_w=Pi_w, pi_w=pi_w,
    theta_bar=theta_bar_0.copy(),
    x0=x0,
    F = np.array( #x_2 >= -0.3
            [
                [0, -1/0.3],
                [0, 0]
            ]
        ),
    G = np.array(
        [
            [0],
            [1]
        ]
    ), # Fx + Gu <= 1
)

mpc = MPC_POLY(system, param)

# --- Closed-Loop Simulation ---
N_sim = 50
N_u = 2 #HYPERPARAMETER USED IN CANNON'S WORK  Window size for updating the polytope 

W = utils.random_polytopic_disturbance(Pi_w, pi_w, N_sim=N_sim, random=True)

X, U, Theta_estimator, times = utils.simulate_closed_loop_poly(
    system=system,
    controller=mpc,
    N_sim=N_sim,
    W=W,
    N_u=N_u,
    report=True
)


Theta_plot = [Theta_estimator[i] for i in range(5)]

utils.plot_parameter_polytopes_3d(
    theta_list=Theta_plot, 
    M_theta=M_theta, 
    interactive=False, 
    true_theta=theta_true
)