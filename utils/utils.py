import torch
import scipy
import numpy as np
import scipy.linalg
from networkx import adjacency_matrix
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def calculate_Atilde(A, M, alpha):
    
    
    """
    A: adjacent matrix, numpy array, [N, N]
    M: number of power iterations, scalar
    alpha: jump probability, scalar
    """
    
    alpha = 1 - alpha
    
    # Number of nodes in this graph
    N = A.shape[0]
    
    # Add a self loop
    A = A + np.identity(N)
    
    # Update the degree matrix (Because the added self-loops)
    D = np.diag(np.sum(A, axis=1))
    
    # Calculate A_hat and (D^{1/2})^{-1}
    D_sqrt_inv = scipy.linalg.inv(scipy.linalg.sqrtm(D))
    A_hat = D_sqrt_inv @ A @ D_sqrt_inv
    
    
    # Power iteration: A_tilde = (1-\alpha)(\sum_{i=0}^{M-1} \alpha^{i}\hat{A}^{i}) + \alpha^{M}\hat{A}^{M}

    A_tilde = np.zeros((N,N))
    A_hat_i = np.identity(N)
    alpha_i = 1
    for i in range(0, M):
        A_tilde = A_tilde + alpha_i*A_hat_i
        alpha_i = alpha_i*alpha
        A_hat_i = A_hat_i @ A_hat
    A_tilde = (1-alpha)*A_tilde
    A_tilde = A_tilde + alpha_i*A_hat_i
    
    # A_tilde: [N, N], 2-d float tensor
    return torch.tensor(A_tilde).type(torch.FloatTensor)



def pygdata_to_frameformat(data):
    
    N = data.x.shape[0]
    Xs = data.x.view(N, 1, -1).type(torch.FloatTensor)
    ys = data.y.view(N, 1).type(torch.LongTensor)
    G = to_networkx(data, to_undirected=True)
    A = np.array(adjacency_matrix(G).todense())
    return G, Xs, ys, A
    
    
    

