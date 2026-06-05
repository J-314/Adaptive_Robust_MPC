import numpy as np
from time import time
import cvxpy as cp
from .setmem_poly import compute_unfalsified_polytope, update_parameter_polytope, project_nominal_parameter
from .poly import polytope_vertices


class MPCSolverError(Exception):
    """Custom exception to carry the CVXPY solver status back to the main loop."""
    def __init__(self, status, time_step = None):
        self.status = status
        self.time_step = time_step
        super().__init__(f"MPC failed with status: {status}")


def random_polytopic_disturbance(Pi_w, pi_w, N_sim=1, random=True):
    """
    Generates random disturbances uniformly distributed within a polytope 
    defined by Pi_w * w <= pi_w using rejection sampling.
    """
    if not random:
        rng = np.random.default_rng(4242) 
    else:
        rng = np.random.default_rng()
        
    vertices = polytope_vertices(Pi_w, pi_w)
    if vertices is None or len(vertices) == 0:
        raise ValueError("The provided disturbance polytope is empty or unbounded.")
        
    w_min = np.min(vertices, axis=0)
    w_max = np.max(vertices, axis=0)
    n_w = w_min.shape[0]
    
    W = np.zeros((N_sim, n_w))
    valid_count = 0
    batch_size = max(N_sim * 10, 100)
    
    while valid_count < N_sim:
        samples = rng.uniform(low=w_min, high=w_max, size=(batch_size, n_w))
        residuals = Pi_w @ samples.T - pi_w.reshape(-1, 1)
        valid_mask = np.all(residuals <= 1e-8, axis=0)
        valid_samples = samples[valid_mask]
        num_valid = valid_samples.shape[0]
        
        if num_valid > 0:
            needed = N_sim - valid_count
            take = min(needed, num_valid)
            W[valid_count:valid_count+take, :] = valid_samples[:take, :]
            valid_count += take
            
    if N_sim == 1:
        return W.squeeze()
        
    return W


def simulate_closed_loop_poly(system, controller, N_sim, W=None, N_u=2, excitation=None, report=False, random=True):
    """
    Simulates the closed-loop system using the Polytopic Robust Adaptive MPC.
    """
    A0, Ai = system.A0, system.Ai
    B0, Bi = system.B0, system.Bi
    Pi_w, pi_w = controller.param.Pi_w, controller.param.pi_w
    M_theta = controller.param.M_theta

    Theta_estimator = [(controller.param.mu_t.copy(), controller.param.theta_bar.copy())]
    X = np.zeros((N_sim + 1, A0.shape[0]))
    U = np.zeros((N_sim, B0.shape[1]))
    
    if W is None:
        W = random_polytopic_disturbance(Pi_w, pi_w, N_sim, random=random)

    xt = system.x0
    X[0, :] = xt
    
    mu_t = controller.param.mu_t.copy()
    theta_bar = controller.param.theta_bar.copy()
    
    P_history = []
    q_history = []

    time_solve = 0
    time_update = 0
    t_start = time()

    for t in range(N_sim):
        # 1. Solve optimization problem
        t_s_start = time()
        
        try:
            controller.solve_problem()
            # If the problem is not OPTIMAL, it's a mathematical infeasibility
            if controller.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise MPCSolverError(controller.problem.status, t)
                
        except cp.error.SolverError:
            # If CVXPY throws this, MOSEK crashed numerically
            raise MPCSolverError("SOLVER_CRASHED", t)
            
        except Exception as e:
            if isinstance(e, MPCSolverError):
                raise e
            else:
                raise MPCSolverError(f"UNKNOWN_ERROR: {type(e).__name__}", t)
                
        t_s_stop = time()
        time_solve += t_s_stop - t_s_start

        # Extract optimal input
        u_opt = controller.optimal_input_sequence()
        if excitation is None:
            ut = u_opt[0]
        else:
            ut = u_opt[0] + excitation * np.random.randn(u_opt.shape[1])
            
        U[t, :] = ut

        # 2. Control the physical system
        system.step(ut, W[t])
        xt1 = system.get_state()
        X[t+1, :] = xt1

        # 3. Update the polytopic parameter set
        t_u_start = time()
        P_t, q_t = compute_unfalsified_polytope(xt, xt1, ut, Pi_w, pi_w, A0, B0, Ai, Bi)
        
        P_history.append(P_t)
        q_history.append(q_t)
        if len(P_history) > N_u:
            P_history.pop(0)
            q_history.pop(0)
            
        mu_t_new = update_parameter_polytope(M_theta, mu_t, P_history, q_history)
        theta_bar_new = project_nominal_parameter(theta_bar, M_theta, mu_t_new)
        
        t_u_stop = time()
        time_update += t_u_stop - t_u_start

        mu_t = mu_t_new
        theta_bar = theta_bar_new
        Theta_estimator.append((mu_t.copy(), theta_bar.copy()))

        # 4. Update the MPC controller parameters for the next step
        controller.update_problem(xt=xt1, mu_t=mu_t, theta_bar=theta_bar)
        xt = xt1

    t_stop = time()
    time_total = t_stop - t_start
    
    out_time = {'total': time_total, 'update': time_update, 'solve': time_solve}
    
    if report:
        output = (X, U, Theta_estimator, out_time)
    else:
        output = (X, U)
        
    return output