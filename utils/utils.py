import torch
import scipy
import numpy as np
import scipy.linalg
from networkx import adjacency_matrix
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 


def calculate_Atilde(A, K, alpha):
    
    
    """
    A: adjacent matrix, numpy array, [N, N]
    K: number of power iterations, scalar
    alpha: jump probability, scalar
    """
    
    
    # Number of nodes in this graph
    N = A.shape[0]
    
    # Add a self loop
    A = A + np.identity(N)
    
    # Update the degree matrix (Because the added self-loops)
    D = np.diag(np.sum(A, axis=1))
    
    # Calculate A_hat and (D^{1/2})^{-1}
    D_sqrt_inv = scipy.linalg.inv(scipy.linalg.sqrtm(D))
    A_hat = D_sqrt_inv @ A @ D_sqrt_inv
    
    
    # Power iteration: A_tilde = (1-\alpha)(\sum_{i=0}^{K} \alpha^{i}\hat{A}^{i})

    A_tilde = np.zeros((N,N))
    A_hat_i = np.identity(N)
    alpha_i = 1
    for i in range(0, K+1):
        A_tilde = A_tilde + alpha_i*A_hat_i
        alpha_i = alpha_i*alpha
        A_hat_i = A_hat_i @ A_hat
    A_tilde = (1-alpha)*A_tilde
    
    # A_tilde: [N, N], 2-d float tensor
    return torch.tensor(A_tilde).type(torch.FloatTensor)



def pygdata_to_frameformat(data):
    
    N = data.x.shape[0]
    Xs = data.x.view(N, 1, -1).type(torch.FloatTensor)
    ys = data.y.view(N, 1).type(torch.LongTensor)
    G = to_networkx(data, to_undirected=True)
    A = np.array(adjacency_matrix(G).todense())
    return G, Xs, ys, A
    
    
    

