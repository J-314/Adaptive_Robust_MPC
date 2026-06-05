import numpy as np
import cvxpy as cp
import scipy as sp
from scipy.linalg import solve_discrete_lyapunov
from dataclasses import dataclass

@dataclass
class MPC_PARAMETERS():
    N: int
    T: np.ndarray
    U_list: list
    K: np.ndarray
    Qx: np.ndarray
    Qu: np.ndarray
    Pt: np.ndarray
    R: np.ndarray
    F: np.ndarray
    G: np.ndarray
    theta_hat: np.ndarray
    x0 : np.ndarray

class MPC():
    def __init__(self,affine_lti, param:'MPC_PARAMETERS'):
        self.system = affine_lti
        self.param = param
        self._define_problem()   

    def _define_problem(self):
        param = self.param
        system = self.system
        A_list = [self.system.A0] + self.system.Ai
        B_list = [self.system.B0] + self.system.Bi
        N = param.N
        n_x = system.A0.shape[0]
        n_u = system.B0.shape[1]
        n_p = len(system.Ai)
        T = param.T
        n_T = T.shape[0]
        U_list =param.U_list
        n_vert = len(U_list)
        K = param.K
        Qx = param.Qx
        Qu = param.Qu
        theta_hat = param.theta_hat
        Qf = compute_Qf_theta(A_list,B_list,K,Qx,Qu,theta_hat)
        xt = param.x0
        Pt = param.Pt
        R = param.R
        F = param.F
        G = param.G
        R_sqrt = np.linalg.cholesky(R).T

        A0 = system.A0
        Ai = system.Ai
        B0 = system.B0
        Bi = system.Bi
        #optimization variables definition

        self.x_k = cp.Variable((N+1,n_x))
        self.v_k = cp.Variable((N,n_u))
        self.alpha_k = cp.Variable((N+1,n_T))

        #parameters definition

        self.Qf_sqrt = cp.Parameter((n_x,n_x),value=np.linalg.cholesky(Qf).T)
        self.xt = cp.Parameter(n_x,value = xt)
        self.Pt_sqrt = cp.Parameter(Pt.shape, value=np.linalg.cholesky(Pt).T)
        self.theta_hat = cp.Parameter(Pt.shape[0],value=theta_hat)

        self.A_theta = cp.Parameter(A0.shape,value = system.A_theta(theta_hat))
        self.B_theta = cp.Parameter(B0.shape, value = system.B_theta(theta_hat))

        #objective definition
        J = 0.
        for k in range(N):
            J += cp.quad_form(self.x_k[k],Qx) + cp.quad_form(K@self.x_k[k] + self.v_k[k],Qu)
        J += cp.sum_squares(self.Qf_sqrt@self.x_k[N])

        # #constraints definition
        # constraints = []
        # #nominal dynamics constraint
        # for i in range(N):
        #     constraints.append( self.x_k[i+1] == (self.A_theta + self.B_theta@K)@self.x_k[i] + self.B_theta@self.v_k[i] )

        # #initial conditions
        # constraints.append(self.x_k[0] == self.xt)

        # #first tube constraint
        # constraints.append(T@self.xt <= self.alpha_k[0])

        # #state and input constraints
        # for j in range(n_vert):
        #     for i in range(N):
        #         constraints.append((F + G@K)@U_list[j]@self.alpha_k[i] + G@self.v_k[i] <=1.)
        #     constraints.append((F + G@K)@U_list[j]@self.alpha_k[N] <= 1. )

        # #robust tube constraints
        # for i in range(n_T):
        #     h_v = T[i]@B0
        #     H_v = np.zeros((n_p,n_u))
        #     z4 = np.linalg.norm(R_sqrt@T[i],2)
        #     for p in range(n_p):
        #         H_v[p,:] = T[i]@Bi[p]
        #     for j in range(n_vert):
        #         h_alpha = T[i]@(A0 + B0@K)@U_list[j]
        #         H_alpha = np.zeros((n_p,n_T))
        #         for p in range(n_p):
        #             H_alpha[p,:] = T[i]@(Ai[p] + Bi[p]@K)@U_list[j]
        #         for k in range(N):
        #             z1 = self.Pt_sqrt@(H_alpha@self.alpha_k[k] + H_v@self.v_k[k])
        #             z2 = self.alpha_k[k+1,i]
        #             z3 = h_alpha@self.alpha_k[k] + h_v@self.v_k[k] + self.theta_hat@(H_alpha@self.alpha_k[k] + H_v@self.v_k[k])
        #             constraints.append(cp.norm(z1,2) <= z2 - z3 - z4)
        #         z1_N = self.Pt_sqrt@H_alpha@self.alpha_k[N]
        #         z2_N = self.alpha_k[N,i]
        #         z3_N = h_alpha@self.alpha_k[N] + self.theta_hat@H_alpha@self.alpha_k[N]
        #         constraints.append(cp.norm(z1_N,2) <= z2_N - z3_N - z4)
        # constraints definition
        
        constraints = []
        
        # 1. nominal dynamics constraint (Questo lo lasciamo iterativo essendo solo N vincoli lineari)
        for i in range(N):
            constraints.append( self.x_k[i+1] == (self.A_theta + self.B_theta@K)@self.x_k[i] + self.B_theta@self.v_k[i] )

        # 2. initial conditions
        constraints.append(self.x_k[0] == self.xt)

        # 3. first tube constraint
        constraints.append(T@self.xt <= self.alpha_k[0])

        # ==========================================
        # 4. state and input constraints (VETTORIZZATI)
        # ==========================================
        for j in range(n_vert):
            Mat_alpha = (F + G@K)@U_list[j]
            Mat_v = G
            
            # Vettorizzato sull'orizzonte N:
            # self.alpha_k[:-1] ha shape (N, n_T), Mat_alpha.T ha shape (n_T, n_c)
            constraints.append(self.alpha_k[:-1] @ Mat_alpha.T + self.v_k @ Mat_v.T <= 1.)
            
            # Step terminale N
            constraints.append(Mat_alpha @ self.alpha_k[N] <= 1.)

        # ==========================================
        # 5. robust tube constraints (VETTORIZZATI)
        # ==========================================
        for i in range(n_T):
            h_v = T[i]@B0
            H_v = np.zeros((n_p,n_u))
            z4 = np.linalg.norm(R_sqrt@T[i],2)
            
            for p in range(n_p):
                H_v[p,:] = T[i]@Bi[p]
                
            for j in range(n_vert):
                h_alpha = T[i]@(A0 + B0@K)@U_list[j]
                H_alpha = np.zeros((n_p,n_T))
                
                for p in range(n_p):
                    H_alpha[p,:] = T[i]@(Ai[p] + Bi[p]@K)@U_list[j]
                
                # Vettorizzazione su N per evitare il loop 'for k in range(N)'
                # Expr_k ha shape (N, n_p)
                Expr_k = self.alpha_k[:-1] @ H_alpha.T + self.v_k @ H_v.T
                
                # Z1 ha shape (N, n_p) - moltiplichiamo per il parametro trasposto
                Z1 = Expr_k @ self.Pt_sqrt.T
                
                # Z2 ha shape (N,)
                Z2 = self.alpha_k[1:, i]
                
                # Z3 ha shape (N,)
                Z3 = self.alpha_k[:-1] @ h_alpha + self.v_k @ h_v + Expr_k @ self.theta_hat
                
                # Aggiungiamo UN SOLO vincolo SOCP per l'intero orizzonte N usando axis=1
                constraints.append(cp.norm(Z1, 2, axis=1) <= Z2 - Z3 - z4)
                
                # Vincolo separato solo per l'istante terminale N
                z1_N = self.Pt_sqrt @ H_alpha @ self.alpha_k[N]
                z2_N = self.alpha_k[N, i]
                z3_N = h_alpha @ self.alpha_k[N] + self.theta_hat @ H_alpha @ self.alpha_k[N]
                constraints.append(cp.norm(z1_N, 2) <= z2_N - z3_N - z4)

        objective = cp.Minimize(J)
        self.problem = cp.Problem(objective,constraints)

    def update_problem(self, xt = None, Pt = None, theta_hat = None):
        if xt is not None:
            self.xt.value = xt
        if  Pt is not None:
            self.param.Pt = Pt.copy()
            self.Pt_sqrt.value = np.linalg.cholesky(Pt).T
        if theta_hat is not None:
            self.param.theta_hat = theta_hat.copy()
            self.theta_hat.value = theta_hat
            self.A_theta.value = self.system.A_theta(theta_hat)
            self.B_theta.value = self.system.B_theta(theta_hat)
            A_list = [self.system.A0] + self.system.Ai
            B_list = [self.system.B0] + self.system.Bi
            Qf = compute_Qf_theta(A_list,B_list,self.param.K,self.param.Qx,self.param.Qu,theta_hat)
            self.param.Qf = Qf
            self.Qf_sqrt.value = np.linalg.cholesky(Qf).T

    def solve_problem(self):
        self.problem.solve(solver = 'MOSEK', warm_start = False) 
        # if self.problem.status not in ["optimal", "optimal_inaccurate"]:
        #     raise ValueError(f"Failed optimization. Status: {self.problem.status}")
    
    def optimal_input_sequence(self):
        x_k = self.x_k.value[:self.param.N]
        K = self.param.K
        v_k = self.v_k.value
        return x_k@K.T + v_k

    def optimal_alpha_sequence(self):
        return self.alpha_k.value

    def check_recursive_feasibility(self):
        N = self.param.N
        n_vert = len(self.param.U_list)
        alpha_k = self.alpha_k.value
        alpha_candidate = np.concatenate([alpha_k[1:], alpha_k[N:N+1]])
        v_k = self.v_k.value
        n_u = self.system.B0.shape[1]
        v_candidate = np.concatenate([v_k[1:], np.zeros((1,n_u))])
        
        self.v_k.value = v_candidate
        self.alpha_k.value = alpha_candidate

        feasible = True
        for i, con in enumerate(self.problem.constraints):
            if i < N + 2 + 2*n_vert: #i <= N + 2 + (1+N)*n_vert:
                continue
            violation = con.violation()
            if violation is not None and np.max(violation) > 1e-6:
                feasible = False
                break
        return feasible

    def check_and_update(self,Pt,theta_hat):
        Pt_sqrt_original = self.Pt_sqrt.value
        theta_hat_original = self.theta_hat.value

        self.Pt_sqrt.value = np.linalg.cholesky(Pt).T
        self.theta_hat.value = theta_hat

        feasible =  self.check_recursive_feasibility()
        if not feasible:
            self.Pt_sqrt.value = Pt_sqrt_original
            self.theta_hat.value = theta_hat_original

        return feasible



def compute_K_contractive(A_list, B_list, T, vertices, P0_sqrt, theta0, verbose = False):
    n_T = T.shape[0]
    n_p = len(A_list)-1
    n_x = A_list[0].shape[0]
    n_u = B_list[0].shape[1]
    n_vert = vertices.shape[0]

    lam = cp.Variable()
    K = cp.Variable((n_u,n_x))
    constraints = []
    for j in range(n_vert):
        for i in range(n_T):
            v = []
            for p in range(n_p+1):#from 0 to n_p 
                vi = T[i].reshape(-1,n_x)@(A_list[p]+B_list[p]@K)@vertices[j].reshape(n_x,1)
                v.append(vi)
            eta = cp.vstack(v[1:])
            constraints.append(v[0] + cp.norm(P0_sqrt@eta) + eta.T@theta0 <= lam)

    objective = cp.Minimize(lam)
    problem = cp.Problem(objective,constraints)
    problem.solve()
    if verbose:
        print('Contractive factor lambda = ', lam.value)
        print('Feeback gain K = ', K.value)
    K = np.array(K.value)
    lam = np.array(lam.value)
    return K,lam


def compute_K_LMI_lambda(A_list, B_list, P_0_sqrt, theta_0, lambda_val=1.0, verbose=False):
    """
    Computes a robust stabilizing feedback gain K for an ellipsoidal uncertainty set
    using Petersen's Lemma and Linear Matrix Inequalities (LMIs).
    """
    n = A_list[0].shape[0]
    m = B_list[0].shape[1]
    p = len(A_list) - 1
    
    Q = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    tau = cp.Variable(pos=True)
    
    # 1. Nominal system Phi_0 = A(theta_0)Q + B(theta_0)Y
    A_nom = A_list[0] + sum(A_list[i+1] * theta_0[i] for i in range(p))
    B_nom = B_list[0] + sum(B_list[i+1] * theta_0[i] for i in range(p))
    Phi_0 = A_nom @ Q + B_nom @ Y
    
    # 2. Build Structural Perturbation Matrix E(Q,Y)
    # E is constructed by stacking Psi_i vertically
    Psi = []
    for i in range(p):
        Psi_i = 0
        for j in range(p):
            # P_0_sqrt corresponds to Sigma^{1/2}
            weight = P_0_sqrt[j, i]
            if weight != 0:
                Psi_i += weight * (A_list[j+1] @ Q + B_list[j+1] @ Y)
                
        # If all weights were zero, we must still append a zero matrix of correct size
        if isinstance(Psi_i, int) and Psi_i == 0:
            Psi_i = np.zeros((n, n))
            
        Psi.append(Psi_i)
        
    E = cp.vstack(Psi) # Shape: (p*n, n)
    
    # 3. Build Constant Matrix H
    H = np.hstack([np.eye(n) for _ in range(p)]) # Shape: (n, p*n)
    
    # Zero blocks for the LMI
    Z_n_pn = np.zeros((n, p * n))
    Z_pn_n = np.zeros((p * n, n))
    Z_pn_pn = np.zeros((p * n, p * n))
    I_pn = np.eye(p * n)
    
    # 4. Final LMI Formulation
    LMI = cp.bmat([
        [lambda_val**2 * Q,   Phi_0.T,       E.T,            Z_n_pn],
        [Phi_0,            Q,             Z_n_pn,         tau * H],
        [E,                Z_pn_n,        tau * I_pn,     Z_pn_pn],
        [Z_pn_n,           tau * H.T,     Z_pn_pn,        tau * I_pn]
    ])
    
    constraints = [
        LMI >> 0,
        Q >> 1e-8* np.eye(n),
        tau >= 1e-8
    ]
    
    prob = cp.Problem(cp.Maximize(0), constraints)
    
    try:
        prob.solve(solver=cp.MOSEK, verbose=verbose)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Recover K = Y * Q^-1
            P_val = np.linalg.inv(Q.value)
            K_val = Y.value @ P_val
            return K_val, P_val
        else:
            return None, None
    except cp.error.SolverError:
        return None, None


def compute_Qf_theta(A_list, B_list, K, Q_x, Q_u, theta):
    """
    Calcola la matrice terminale Qf per un parametro theta specifico
    risolvendo l'equazione di Lyapunov a tempo discreto.
    """
    # 1. Costruiamo A(theta) e B(theta)
    A_theta = np.copy(A_list[0])
    B_theta = np.copy(B_list[0])
    
    # Ricorda che A_list[0] è la nominale, le perturbazioni partono da indice 1
    for i, th in enumerate(theta):
        A_theta += A_list[i+1] * th
        B_theta += B_list[i+1] * th
        
    # 2. Calcolo della matrice in anello chiuso A_cl
    A_cl = A_theta + B_theta @ K
    
    # --- REALITY CHECK ---
    # L'equazione di Lyapunov ammette una soluzione definita positiva
    # SOLO SE la matrice in anello chiuso è strettamente stabile.
    autovalori = np.linalg.eigvals(A_cl)
    raggio_spettrale = np.max(np.abs(autovalori))
    if raggio_spettrale >= 1.0:
        raise ValueError(f"Impossibile risolvere Lyapunov: il K fornito non stabilizza "
                         f"il sistema in questo specifico theta. (Raggio spettrale = {raggio_spettrale:.4f})")

    # 3. Costruzione del termine noto: Q_step = Q_x + K^T * Q_u * K
    Q_step = Q_x + K.T @ Q_u @ K
    
    # 4. Risoluzione tramite SciPy
    # solve_discrete_lyapunov(M, Q) risolve M * X * M^T - X + Q = 0.
    # Siccome la nostra equazione è A_cl^T * Qf * A_cl - Qf + Q_step = 0,
    # passiamo la matrice trasposta (A_cl.T) come primo argomento.
    Qf = solve_discrete_lyapunov(A_cl.T, Q_step)
    Qf = (Qf + Qf.T)/2
    return Qf
