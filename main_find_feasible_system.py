import numpy as np
import itertools
import pickle
import utils
import os

from joblib import Parallel, delayed

import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to let joblib the tqdm bar communicate."""
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

class InfeasibilityKContractive(Exception):
    pass

class InfeasibilityKLMI(Exception):
    pass


def generate_system(n, m, p, delta_norm, stable=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    vertices_theta = list(itertools.product([-1, 1], repeat=p))
    theta_star = rng.uniform(-1, 1, p)
    eigenvalues = rng.uniform(-1, 1, n)

    if not stable:
        for i in range(n // 2):
            eigenvalues[i] = rng.uniform(1, 1.5) * rng.choice([-1, 1])
                
    D = np.diag(eigenvalues)
    singular_T = True
    
    while True:
        T = rng.standard_normal((n, n))
        if np.absolute(np.linalg.det(T)) > 0.1:
            break
            
    T_inv = np.linalg.inv(T)
    A_nom = T @ D @ T_inv
    B_nom = rng.standard_normal((n, m))
    
    norm_A_nom = np.linalg.norm(A_nom, ord=2)
    norm_B_nom = np.linalg.norm(B_nom, ord=2)
    
    A_i_list = []
    B_i_list = []

    for i in range(p):
        Ai_rand = rng.standard_normal((n, n))
        Bi_rand = rng.standard_normal((n, m))
        
        Ai = Ai_rand * (delta_norm * norm_A_nom / np.linalg.norm(Ai_rand, ord=2))
        Bi = Bi_rand * (delta_norm * norm_B_nom / np.linalg.norm(Bi_rand, ord=2))
        
        A_i_list.append(Ai)
        B_i_list.append(Bi)

    A_0 = np.copy(A_nom)
    B_0 = np.copy(B_nom)
    
    for i in range(p):
        A_0 -= A_i_list[i] * theta_star[i]
        B_0 -= B_i_list[i] * theta_star[i]

    A_list = [A_0] + A_i_list
    B_list = [B_0] + B_i_list

    return A_list, B_list, theta_star
    
def outerbounding_polytope(P_ellipsoid):
    P = (P_ellipsoid + P_ellipsoid.T)/2
    eig, M = np.linalg.eigh(P)
    T = np.concatenate([M,-M])
    eig_sqrt = np.sqrt(eig)
    alpha = np.concatenate([eig_sqrt, eig_sqrt])
    
    return T, alpha


def run_single_attempt(n, m, d, delta_val, attempt, A_list, B_list, theta_star, stable_system, W, Pi_w, pi_w, M_theta, mu_0, vertices_parameter, Horizon, N_sim, N_u, F, G):
    x0 = np.zeros(n)
    system = utils.LTI_AFFINE(A_list, B_list, theta_star, x0)
    
    log_entry = {
        'n_states': n, 'm_inputs': m, 'delta_norm': delta_val, 'attempt': attempt,
        'x0': x0.copy(), 'W': W.copy(), 'A_list': A_list, 'B_list': B_list, 'theta_star': theta_star, 'stable': stable_system,
        'K_con': None, 'K_con_feasible': False, 'MPC_con_feasible': False, 'Status_con': 'UNKNOWN',
        'K_LMI': None, 'lambda_LMI': None, 'K_LMI_feasible': True, 'MPC_LMI_feasible': False, 'Status_LMI': 'UNKNOWN',
        'Infeasibility_timeStep_con': None, 'Infeasibility_timeStep_LMI': None,'exclusive_feasibility': 'NONE', 'T':None, 'alpha':None, 
        'N_sim': N_sim, 'Horizon': Horizon, 'PI_W': Pi_w.copy(), 'pi_w':pi_w.copy(), 'M_theta':M_theta.copy(), 'mu_theta':mu_0.copy(), 'theta_bar':theta_bar_0.copy(), 'F':F.copy(), 'G':G.copy()
    }
    
    # --- LMI METHOD ---
    lambda_LMI = 1.0
    while not log_entry['MPC_LMI_feasible'] and log_entry['K_LMI_feasible'] and lambda_LMI > 0:
        try:
            K_LMI, P_LMI = utils.compute_K_poly_LMI_lambda(A_list, B_list, vertices=vertices_parameter, lambda_val=lambda_LMI, verbose=False)
            if K_LMI is None:
                raise InfeasibilityKLMI
            
            log_entry['K_LMI'] = K_LMI
            log_entry['lambda_LMI'] = lambda_LMI
            
            T, alpha = outerbounding_polytope(P_LMI)
            vertices_state = utils.polytope_vertices(T, alpha)
            U_list = utils.polytope_U(T, alpha)

            param = utils.MPC_POLY_PARAMETERS(
                N=Horizon, T=T, U_list=U_list, K=K_LMI, Qx=np.eye(n), Qu=np.eye(m),
                M_theta=M_theta, mu_t=mu_0, Pi_w=Pi_w, pi_w=pi_w, theta_bar=theta_bar_0, x0=x0, F=F, G=G
            )
            
            mpc_poly = utils.MPC_POLY(system, param)
            X, U = utils.simulate_closed_loop_poly(system=system, controller=mpc_poly, N_sim=N_sim, W=W, N_u=N_u, report=False)
            log_entry['MPC_LMI_feasible'] = True
            log_entry['Status_LMI'] = 'OPTIMAL'
            
        except InfeasibilityKLMI:
            if lambda_LMI == 1.0:
                log_entry['K_LMI_feasible'] = False
                log_entry['Status_LMI'] = 'K_INFEASIBLE'
            break
        except utils.sim_poly.MPCSolverError as e:
            log_entry['Status_LMI'] = e.status
            log_entry['Infeasibility_timeStep_LMI'] = e.time_step
        system.set_state(x0)
        if not log_entry['MPC_LMI_feasible'] and log_entry['K_LMI_feasible']:
            lambda_LMI = round(lambda_LMI - 0.05, 3)

    # --- CONTRACTIVE METHOD ---
    if log_entry['K_LMI_feasible']:
        log_entry['T'] = T
        log_entry['alpha'] = alpha
        try:
            K_con, lam = utils.compute_K_poly_contraction(A_list, B_list, T, alpha, vertices_state, M_theta, mu_0, verbose=False)
            if lam > 1:
                raise InfeasibilityKContractive
                
            log_entry['K_con'] = K_con
            log_entry['K_con_feasible'] = True
            
            param = utils.MPC_POLY_PARAMETERS(
                N=Horizon, T=T, U_list=U_list, K=K_con, Qx=np.eye(n), Qu=np.eye(m),
                M_theta=M_theta, mu_t=mu_0, Pi_w=Pi_w, pi_w=pi_w, theta_bar=theta_bar_0, x0=x0, F=F, G=G
            )
            
            mpc_poly = utils.MPC_POLY(system, param)
            X, U = utils.simulate_closed_loop_poly(system=system, controller=mpc_poly, N_sim=N_sim, W=W, N_u=N_u, report=False)
            log_entry['MPC_con_feasible'] = True
            log_entry['Status_con'] = 'OPTIMAL'
            
        except InfeasibilityKContractive:
            log_entry['Status_con'] = 'K_INFEASIBLE_LAMBDA_GEQ_1'
        except utils.sim_poly.MPCSolverError as e:
            log_entry['Status_con'] = e.status
            log_entry['Infeasibility_timeStep_con'] = e.time_step
        system.set_state(x0)
    else:
        log_entry['Status_con'] = 'NO_POLYTOPE'

    # --- EXCLUSIVE FEASIBILITY ---
    if log_entry['MPC_con_feasible'] and log_entry['MPC_LMI_feasible']: log_entry['exclusive_feasibility'] = 'BOTH'
    elif log_entry['MPC_con_feasible'] and not log_entry['MPC_LMI_feasible']: log_entry['exclusive_feasibility'] = 'CONTRACTIVE_ONLY'
    elif not log_entry['MPC_con_feasible'] and log_entry['MPC_LMI_feasible']: log_entry['exclusive_feasibility'] = 'LMI_ONLY'
    else: log_entry['exclusive_feasibility'] = 'NONE'

    return log_entry


# =====================================================================
# MAIN EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    Horizon = 10; p = 3; w_inf_max = 0.05; attempts_per_parameters = 200; N_sim = 2; N_u = 2
    rng = np.random.default_rng(42)
    M_theta = np.vstack([np.eye(p), -np.eye(p)]); mu_theta = np.ones(2 * p); theta_bar_0 = np.zeros(p)
    vertices_parameter = utils.polytope_vertices(M_theta, mu_0)

    n_min = 5; n_max = 6; m_min = 3; m_max = 6; delta_norm_relative = [0.1]; stable_system = False

    # Generate first all the possible combinations of parameters (task generator)
    tasks = []
    for n in range(n_min, n_max+1):
        Pi_w = np.vstack([np.eye(n), -np.eye(n)]); pi_w = np.ones(2 * n) * w_inf_max
        W = utils.random_polytopic_disturbance(Pi_w, pi_w, N_sim=N_sim, random=True)
        F = np.zeros((1, n))
        
        for m in range(m_min, min([m_max,n])+1):
            G = np.zeros((1, m))
            for d in range(len(delta_norm_relative)):
                for attempt in range(attempts_per_parameters):
                    A_list, B_list, theta_star = generate_system(n, m, p, delta_norm_relative[d], stable=stable_system, rng=rng)
                    tasks.append(
                        delayed(run_single_attempt)(
                            n, m, d, delta_norm_relative[d], attempt, A_list, B_list, theta_star, stable_system, 
                            W, Pi_w, pi_w, M_theta, mu_theta, vertices_parameter, Horizon, N_sim, N_u, F, G
                        )
                    )

    print(f"Pre-calculation completed. Launching {len(tasks)} tasks on all but 1 available CPU cores...")
    
    with tqdm_joblib(tqdm(desc="Parallel Simulations", total=len(tasks), smoothing = 0.01)):
        simulation_logs = Parallel(n_jobs=-2)(tasks)

    # Saving
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_folder = os.path.join(parent_dir, "results")
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "simulation_data.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(simulation_logs, f)

    print(f"Simulation finished. Data saved to '{file_path}'")