import os
import pickle
import pandas as pd
import utils
import cvxpy as cp
from utils.controller_poly import MPC_POLY_PARAMETERS, MPC_POLY
import numpy as np

def load_data(filename="simulation_data.pkl", dirname = "polytopic_results"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    filepath = os.path.join(parent_dir, dirname, filename)
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded {len(data)} simulation records from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: file not found. I was looking in: \n{filepath}")
        return None

def run_statistical_analysis(logs):
    print("\n" + "="*50)
    print("--- ANALYSIS ---")
    print("="*50)
    
    df = pd.DataFrame(logs)
    
    # 1. Overall Success by State Dimension
    print("\n1. FEASIBILITY RATE BY STATE DIMENSION (n_states)")
    print("-" * 50)
    success_rates = df.groupby('n_states')[['MPC_con_feasible', 'MPC_LMI_feasible']].mean() * 100
    success_rates.columns = ['Contractive Success (%)', 'LMI Success (%)']
    print(success_rates.round(2))
    
    # 2. Exclusive Feasibility Breakdown
    print("\n2. EXCLUSIVE FEASIBILITY (Who solved what?)")
    print("-" * 50)
    exclusive_counts = df['exclusive_feasibility'].value_counts()
    print(exclusive_counts)
    
    # 3. Failure Analysis: Contractive Method
    print("\n3. FAILURE REASONS: CONTRACTIVE METHOD")
    print("-" * 50)
    failed_con = df[~df['MPC_con_feasible']]
    if not failed_con.empty:
        reasons_con = failed_con['Status_con'].value_counts()
        print(reasons_con)
    else:
        print("No failures recorded for Contractive Method.")
        
    # 4. Failure Analysis: LMI Method
    print("\n4. FAILURE REASONS: LMI METHOD")
    print("-" * 50)
    failed_lmi = df[~df['MPC_LMI_feasible']]
    if not failed_lmi.empty:
        reasons_lmi = failed_lmi['Status_LMI'].value_counts()
        print(reasons_lmi)
    else:
        print("No failures recorded for LMI Method.")

    # 5. Failure Time Step Distribution
    print("\n5. FAILURE TIME STEP DISTRIBUTION (When did the MPC fail?)")
    print("-" * 50)
    
    # Check Contractive Method Times
    if 'Infeasibility_timeStep_con' in df.columns:
        con_times = df[~df['MPC_con_feasible']]['Infeasibility_timeStep_con'].dropna()
        if not con_times.empty:
            print("Contractive Method Failures by Time Step:")
            print(con_times.value_counts().sort_index())
        else:
            print("No time-step data for Contractive Method failures.")
    else:
        print("Contractive time step data not found in this dataset.")
            
    # Check LMI Method Times
    if 'Infeasibility_timeStep_LMI' in df.columns:
        lmi_times = df[~df['MPC_LMI_feasible']]['Infeasibility_timeStep_LMI'].dropna()
        if not lmi_times.empty:
            print("\nLMI Method Failures by Time Step:")
            print(lmi_times.value_counts().sort_index())
        else:
            print("\nNo time-step data for LMI Method failures.")
    else:
        print("LMI time step data not found in this dataset.")
        
    print("="*50 + "\n")


def replay_experiment(log_entry, method_to_test):
    print(f"\nReplaying Experiment: {method_to_test} Method")
    print(f"Parameters: n={log_entry['n_states']}, m={log_entry['m_inputs']}, delta={log_entry['delta_norm']}")
    
    n = log_entry['n_states']
    m = log_entry['m_inputs']
    x0 = log_entry['x0']
    W = log_entry['W']
    system = utils.LTI_AFFINE(log_entry['A_list'], log_entry['B_list'], log_entry['theta_star'], x0)
    T = log_entry['T']
    alpha = log_entry['alpha']
    
    Horizon = log_entry['Horizon']
    Pi_w = log_entry['PI_W']
    pi_w = log_entry['pi_w']
    M_theta = log_entry['M_theta']
    mu_0 = log_entry['mu_theta']
    theta_bar_0 = log_entry['theta_bar']
    F = log_entry['F']
    G = log_entry['G']
    N_sim = log_entry['N_sim']
    
    U_list = utils.polytope_U(T, alpha)
    

    K_matrix = log_entry['K_con'] if method_to_test == 'CONTRACTIVE' else log_entry['K_LMI']
    
    if K_matrix is None:
        print(f"Cannot replay MPC: The {method_to_test} method failed at the K matrix computation step.")
        return

    param = MPC_POLY_PARAMETERS(
        N=Horizon, T=T, U_list=U_list, K=K_matrix, Qx=np.eye(n), Qu=np.eye(m),
        M_theta=M_theta, mu_t=mu_0, Pi_w=Pi_w, pi_w=pi_w, theta_bar=theta_bar_0, x0=x0, F=F, G=G
    )
    
    mpc_poly = MPC_POLY(system, param)
    
    original_cp_solve = cp.Problem.solve
    def forced_verbose_solve(self, *args, **kwargs):
        kwargs['verbose'] = True
        return original_cp_solve(self, *args, **kwargs)

    cp.Problem.solve = forced_verbose_solve

    print("\n--- STARTING VERBOSE SIMULATION ---")
    try:
        utils.simulate_closed_loop_poly(
            system=system, controller=mpc_poly, N_sim=N_sim, W=W, N_u=2, report=False
        )
        print("Wait, the simulation succeeded this time?!")
    except Exception as e:
        print(f"\n--- SIMULATION FAILED AS EXPECTED ---")
        import traceback
        traceback.print_exc()
    finally:
        cp.Problem.solve = original_cp_solve


def interactive_debug_menu(logs):

    while True:
        print("\n" + "="*50)
        print("--- DEBUGGER MENU ---")
        print("="*50)
        print("1. Replay a successfull experiment")
        print("2. Replay a failed experiment")
        print("3. Exit Debugger")
        
        choice = input("\nEnter your choice (1/2/3): ")
    
        if choice == '3':
            return
        elif choice == '1':
            successful = True
        elif choice == '2':
            successful = False
        else:
            print('Invalid choice.')
            continue
    

        mpc_failures = [
            log for log in logs 
            if log['Status_con'] not in ['OPTIMAL', 'UNKNOWN', 'K_INFEASIBLE_LAMBDA_GEQ_1'] 
            or log['Status_LMI'] not in ['OPTIMAL', 'UNKNOWN', 'K_INFEASIBLE']
        ]

        mpc_successes = [
            log for log in logs
            if log['Status_con'] == 'OPTIMAL'
            or log['Status_LMI'] == 'OPTIMAL'
        ]

        if not mpc_failures and not successful:
            print("\nNo MPC-stage failures found in the dataset! All recorded failures happened at the K matrix stage.")
            continue
        elif not mpc_successes and successful:
            print("\nNo MPC-stage successes found in the dataset!")
            continue
        else: 
            break


    while True:
        candidates = []
        if not successful:
            print("\n" + "-"*50)
            print("1. Replay a Mathematical Infeasibility (e.g., 'infeasible')")
            print("2. Replay a Solver Crash (e.g., 'SOLVER_CRASHED')")
            print("3. Replay an Unexpected Error (e.g., 'ValueError', 'UNKNOWN_ERROR')")
            print("4. Return")
            
            choice = input("\nEnter your choice (1/2/3/4): ")
            
            if choice == '4':
                break
                
            if choice not in ['1', '2', '3']:
                print("Invalid choice.")
                continue

            known_math_fails = ['infeasible', 'unbounded']
            known_crashes = ['SOLVER_CRASHED']
            ignored_con_states = ['OPTIMAL', 'UNKNOWN', 'K_INFEASIBLE_LAMBDA_GEQ_1']
            ignored_lmi_states = ['OPTIMAL', 'UNKNOWN', 'K_INFEASIBLE']

            
            for log in mpc_failures:
                s_con = log['Status_con']
                if choice == '1' and s_con in known_math_fails: candidates.append((log, 'CONTRACTIVE'))
                elif choice == '2' and s_con in known_crashes: candidates.append((log, 'CONTRACTIVE'))
                elif choice == '3' and s_con not in known_math_fails and s_con not in known_crashes and s_con not in ignored_con_states:
                    candidates.append((log, 'CONTRACTIVE'))
                    
                s_lmi = log['Status_LMI']
                if choice == '1' and s_lmi in known_math_fails: candidates.append((log, 'LMI'))
                elif choice == '2' and s_lmi in known_crashes: candidates.append((log, 'LMI'))
                elif choice == '3' and s_lmi not in known_math_fails and s_lmi not in known_crashes and s_lmi not in ignored_lmi_states:
                    candidates.append((log, 'LMI'))
        else:
            for log in mpc_successes:
                s_con = log['Status_con']
                if s_con == 'OPTIMAL':
                    candidates.append((log,'CONTRACTIVE'))
                s_lmi = log['Status_LMI']
                if s_lmi == 'OPTIMAL':
                    candidates.append((log,'LMI'))

        if not candidates:
            print(f"\nNo failures found matching that category.")
            continue
            
        print(f"\nFound {len(candidates)} matching experiments.")
        
        # Filter by N
        available_ns = sorted(list(set([c[0]['n_states'] for c in candidates])))
        print(f"\nAvailable State Dimensions (n): {available_ns}")
        try:
            sel_n = int(input("Select 'n' to filter by: "))
            if sel_n not in available_ns: raise ValueError
        except ValueError:
            print("Invalid selection.")
            continue
        candidates = [c for c in candidates if c[0]['n_states'] == sel_n]
        
        # Filter by M
        available_ms = sorted(list(set([c[0]['m_inputs'] for c in candidates])))
        print(f"\nAvailable Input Dimensions (m): {available_ms}")
        try:
            sel_m = int(input("Select 'm' to filter by: "))
            if sel_m not in available_ms: raise ValueError
        except ValueError:
            print("Invalid selection.")
            continue
        candidates = [c for c in candidates if c[0]['m_inputs'] == sel_m]
        
        # Filter by Delta
        available_deltas = sorted(list(set([c[0]['delta_norm'] for c in candidates])))
        print(f"\nAvailable Delta Norms: {available_deltas}")
        try:
            sel_delta_input = input("Select 'delta' to filter by (type exact number): ")
            sel_delta = float(sel_delta_input)
            candidates = [c for c in candidates if np.isclose(c[0]['delta_norm'], sel_delta, atol=1e-8)]
            if not candidates: raise ValueError
        except ValueError:
            print("Invalid selection.")
            continue
            
        # Filter by Method
        available_methods = sorted(list(set([c[1] for c in candidates])))
        if len(available_methods) > 1:
            print(f"\nAvailable Methods: {available_methods}")
            sel_method = input("Select Method (CONTRACTIVE or LMI): ").strip().upper()
            candidates = [c for c in candidates if c[1] == sel_method]
        else:
            sel_method = available_methods[0]
            print(f"\nOnly {sel_method} method failures available for these parameters.")
            
        # Filter by Time Step
        if not successful:
            if sel_method == 'CONTRACTIVE':
                time_key = 'Infeasibility_timeStep_con'
            else:
                time_key = 'Infeasibility_timeStep_LMI'
                
            available_times = sorted(list(set([c[0].get(time_key) for c in candidates if c[0].get(time_key) is not None])))
            
            if available_times:
                print(f"\nAvailable Failure Time Steps ({sel_method}): {available_times}")
                try:
                    sel_time_input = input("Select time step to filter by (or press Enter to skip): ")
                    if sel_time_input.strip():
                        sel_time = float(sel_time_input) # Use float to safely handle numerical matching
                        candidates = [c for c in candidates if c[0].get(time_key) is not None and np.isclose(c[0].get(time_key), sel_time)]
                        if not candidates: raise ValueError
                except ValueError:
                    print("Invalid selection. Returning to main menu.")
                    continue

        if not candidates:
            print("No experiments left after filtering.")
            continue
            
        print(f"\nFound {len(candidates)} attempts matching these criteria. Replaying the first one...")
        selected_log, selected_method = candidates[0]
        replay_experiment(selected_log, selected_method)

if __name__ == "__main__":
    choice = '0'
    while True:
        filename = input('\nInsert file name (without .pkl):\n')

        filename = filename + '.pkl'
    
        data = load_data(filename,'results')
        if data:
            break

    if data:
        while True:
            print("\nMAIN MENU:")
            print("1. Run Analysis")
            print("2. Replay Specific Experiment")
            print("3. Quit")
            main_choice = input("Select an option (1/2/3): ")
            
            if main_choice == '1': run_statistical_analysis(data)
            elif main_choice == '2': interactive_debug_menu(data)
            elif main_choice == '3': break
            else: print("Invalid choice.")