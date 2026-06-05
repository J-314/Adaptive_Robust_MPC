import numpy as np
from time import time
import cvxpy as cp
from utils import (
    MPC,
    LTI_AFFINE,
    update_ellipsoid,
    compute_feasible_ellipsoid
)

class MPCSolverError(Exception):
    """Custom exception to carry the CVXPY solver status back to the main loop."""
    def __init__(self, status, time_step = None):
        self.status = status
        self.time_step = time_step
        super().__init__(f"MPC failed with status: {status}")

def random_ellipsoidal_disturbance(R, N=None, w0=None, random=True):
    n = R.shape[0]
    L = np.linalg.cholesky(R)
    
    if N is None:
        N = 1
    if w0 is None:
        w0 = np.zeros(n)
        
    if not random:
        rng = np.random.default_rng(4242)
    else:
        rng = np.random.default_rng()

    versor = rng.standard_normal((N, n))
    versor = versor / np.linalg.norm(versor, ord=2, axis=1, keepdims=True)
    radius = rng.random((N, 1)) ** (1/n)

    z = radius * versor
    w = np.matmul(z, L.T) + w0
    
    return w.squeeze()

def simulate_closed_loop(system: 'LTI_AFFINE', controller: 'MPC', N_sim, W=None, excitation=None, report=False, random=True):
    A0 = system.A0
    B0 = system.B0
    Ai = system.Ai
    Bi = system.Bi
    R = controller.param.R

    Theta_estimator = [(controller.param.Pt, controller.param.theta_hat)]
    Theta_control = [(controller.param.Pt, controller.param.theta_hat)]
    X = np.zeros((N_sim + 1, A0.shape[0]))
    U = np.zeros((N_sim, B0.shape[1]))
    ALPHA = np.zeros((N_sim, controller.param.N + 1, controller.param.T.shape[0]))

    if W is None:
        W = random_ellipsoidal_disturbance(controller.param.R, N_sim, random=random)
    
    xt = system.x0
    X[0, :] = xt
    t_start = time()
    time_feasibility = 0
    time_solve = 0
    
    for t in range(N_sim):
        # 1. Solve optimization problem
        t_s_start = time()
        try:
            controller.solve_problem()
            if controller.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise MPCSolverError(controller.problem.status,t)
        except cp.error.SolverError:
            raise MPCSolverError("SOLVER_CRASHED",t)
        except Exception as e:
            if isinstance(e, MPCSolverError):
                raise e
            else:
                raise MPCSolverError(f"UNKNOWN_ERROR: {type(e).__name__}",t)
                
        t_s_stop = time()
        time_solve += t_s_stop - t_s_start

        # Extract optimal input
        u_opt = controller.optimal_input_sequence()
        if excitation is None:
            ut = u_opt[0]
        else:
            ut = u_opt[0] + excitation * np.random.randn(u_opt.shape[1])
            
        U[t, :] = ut
        ALPHA[t, :, :] = controller.optimal_alpha_sequence()

        # 2. Control the system
        system.step(ut, W[t])
        xt1 = system.get_state()
        X[t + 1, :] = xt1

        # 3. Update the ellipsoid
        Ph, ah, h, phi, Q = compute_feasible_ellipsoid(xt, xt1, ut, R, A0, B0, Ai, Bi)
        Pt1, theta_hat1, updated = update_ellipsoid(Theta_estimator[-1][0], Theta_estimator[-1][1], h, phi, Q)
        
        if updated:
            Theta_estimator.append((Pt1, theta_hat1))

            # Test for conditional update
            t_c_start = time()
            feasible = controller.check_and_update(Pt1, theta_hat1)
            t_c_stop = time()
            time_feasibility += t_c_stop - t_c_start
            if feasible:
                Theta_control.append((Pt1, theta_hat1))

        # 4. Update the MPC controller parameters
        controller.update_problem(xt1)
        xt = xt1
        
    t_stop = time()
    time_total = t_stop - t_start
    
    out_time = {'total': time_total, 'update': time_feasibility, 'solve': time_solve} 
    
    if report:
        output = (X, U, ALPHA, Theta_estimator, Theta_control, out_time)
    else:
        output = (X, U)
        
    return output