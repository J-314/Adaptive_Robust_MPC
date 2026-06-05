import numpy as np
import utils

# system definition
# x_t+1 = A_0 x_t + B_0 u_t + sum theta_i ( A_i x_t + B_i u_t )
A0 = np.array([[0.5, 0.2],[-0.1, 0.6]])
A1 = np.array([[0.042,0],[0.072,0.03]])
A2 = np.array([[0.015,0.019],[0.009,0.035]])
A3 = np.array([[0.,0],[0.,0.]])
A_list = [A0, A1, A2, A3]

B0 = np.array([[0],[0.5]])
B1 = np.array([0.,0]).reshape(2,1)
B2 = np.array([0.,0]).reshape(2,1)
B3 = np.array([0.397,0.059]).reshape(2,1)
B_list = [B0, B1, B2, B3]

theta_true = np.array([.8, .2, -.5])

x0 = np.array([-5,3])
system = utils.LTI_AFFINE(A_list,B_list,theta_true,x0)

#uncertainty parameters
w_max = 0.05
P = np.eye(3)
P_sqrt = np.linalg.cholesky(P).T
theta_0 = np.zeros(3)
R = np.eye(2)*w_max**2


#rectangular polytope
T_width = 2.
T_height = 1.
T = np.vstack([np.diag(np.array([2/T_width,2/T_height])),-np.diag(np.array([2/T_width, 2/T_height]))])
alpha = np.ones(4) #minimal general rectangle to well-define the Uj matrices 
n_T = T.shape[0]

vertices = utils.polytope_vertices(T,alpha)
U_list = utils.polytope_U(T,alpha)
n_vert = vertices.shape[0]


#compute the feedback gain K that minimizes the contraction factor lambda
K,lam = utils.compute_K_contractive(A_list,B_list,T,vertices,P_sqrt,theta_0,True)

#MPC parameters
parameters = utils.MPC_PARAMETERS(
    N = 10,
    T = T,
    U_list = U_list,
    K = K,
    Qx = np.eye(2),
    Qu = 0.01*np.eye(1),
    Pt = P,
    R = R,
    F = np.array( #x_2 >= -0.3
            [
                [0, -1/0.3],
                [0, 0]
            ]
        ),
    G = np.array( #u <= 1
        [
            [0],
            [1]
        ]
    ), # Fx+Gu <= 1
 
    theta_hat= theta_0,

    x0 = x0
)

mpc_controller = utils.MPC(system, parameters)

N_sim = 50

W = utils.random_ellipsoidal_disturbance(R,N_sim,random=True)
X,U, ALPHA, Theta_estimate, Theta_control , times = utils.simulate_closed_loop(system,mpc_controller, N_sim, W , excitation= None, random = True, report=True)

L_e = len(Theta_estimate)
L_c = len(Theta_control)
volumes_est = np.zeros(L_e)
volumes_ctr = np.zeros(L_c)

for i in range(L_e):
    volumes_est[i] = utils.volume_ellipsoid(Theta_estimate[i][0])
for i in range(L_c):
    volumes_ctr[i] = utils.volume_ellipsoid(Theta_control[i][0])


utils.plot_ellipsoids(Theta_estimate,interactive = False)
print(f'')
print(f'Conditional update happened {len(Theta_control)-1} times')
print(f'Simulation took {times['total']} seconds')
print(f'Conditional update took {times['update']} seconds')
print(f'Solving the problems took {times['solve']} seconds')

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(volumes_est, color='tab:blue', label='Volumes of estimated sets')
axs[1].plot(volumes_ctr, color='tab:orange', label='Volumes of sets used for control')
plt.tight_layout()
plt.show()


# # CONTROL WITH ONLY FEEDBACK GAIN
# X_proportional_feedback = np.zeros_like(X)
# system.set_state(x0)
# X_proportional_feedback[0,:] = x0
# for t in range(N_sim):
#     x = X_proportional_feedback[t,:]
#     system.step(K@x,W[t])
#     X_proportional_feedback[t+1,:] = system.get_state()

# utils.plot_trajectory(X, X_proportional_feedback)



