import numpy as np

class LTI_AFFINE():
    def __init__(self, A_list, B_list, theta_true, x0):
        self.A0 = A_list[0]
        self.Ai = A_list[1:]
        self.B0 = B_list[0]
        self.Bi = B_list[1:]
        self.theta_true = theta_true
        self.A_true = self.A_theta(theta_true)
        self.B_true = self.B_theta(theta_true)
        self.x0 = x0.copy()
    def A_theta(self,theta):
        A = self.A0.copy()
        for i in range(len(self.Ai)):
            A += self.Ai[i]*theta[i]
        return A
    def B_theta(self,theta):
        B = self.B0.copy()
        for i in range(len(self.Bi)):
            B += self.Bi[i]*theta[i]
        return B
    def step(self,u,w):
        x0 = self.x0
        if self.B_true.ndim == 1:
            x1 = np.matmul(self.A_true,x0) + self.B_true*u + w
        else:    
            x1 = np.matmul(self.A_true,x0) + np.matmul(self.B_true,u) + w
        self.x0 = x1
        
    def set_state(self,x0):
        self.x0 = x0.copy()

    def get_state(self):
        return self.x0.copy()