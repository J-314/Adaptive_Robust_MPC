import numpy as np
import scipy as sp


def polytope_vertices(A,b,interior_point = None):
    if b.ndim == 1:
        b = b.reshape(b.size,-1)
    halfspaces = np.hstack([A,-b])
    if interior_point is None:
        n = A.shape[1]
        interior_point = np.zeros(n)
    hs = sp.spatial.HalfspaceIntersection(halfspaces,interior_point)
    vertices = hs.intersections
    return vertices

def active_contraints(A,b,point,tol =None):
    if tol is None:
        tol = 1e-8
    residues = np.matmul(A,point) - b
    active_indeces = np.abs(residues) <= tol
    return active_indeces

def polytope_U(A,b,interior_point=None,tol=None):
    vertices = polytope_vertices(A,b,interior_point)
    n = vertices.shape[1]
    N = vertices.shape[0]
    U_list = []
    for j in range(N):
        active_idx_j = active_contraints(A,b,vertices[j],tol)
        A_active_j = A[active_idx_j]
        U_reduced_j = np.linalg.inv(A_active_j)
        U_j = np.zeros((n,A.shape[0]))
        U_j.T[active_idx_j] = U_reduced_j.T
        U_list.append(U_j)
    return U_list

