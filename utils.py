import torch
import scipy
import numpy as np
import scipy.linalg
from networkx import adjacency_matrix
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


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

    """
    A_tilde = np.zeros((N,N))
    for i in range(0, K+1):
        A_tilde += (alpha**i)*np.linalg.matrix_power(A_hat, i)
    A_tilde = (1-alpha)*A_tilde
    """
    
    
    # A_tilde: [N, N], 2-d float tensor
    return torch.tensor(A_tilde).type(torch.FloatTensor)


def sparse_split_graphdata(random_seed, labels, train_size, num_class):
    N = labels.shape[0]
    class_ids = {}
    for c in np.unique(labels):
        class_ids[c] = np.where(labels==c)[0]
    train_ids = []
    np.random.seed(random_seed)
    for c in range(num_class):
        train_ids.extend(np.random.choice(class_ids[c], size=train_size, replace=False))
    remain_ids = list(set(np.arange(N)) - set(train_ids))
    np.random.shuffle(remain_ids)
    val_ids = remain_ids[0:train_size*num_class]
    test_ids = remain_ids[train_size*num_class:]
    
    return train_ids, val_ids, test_ids

def csbm_to_pygdata(csbm, train_ids, val_ids, test_ids):
    
    X = torch.cat(csbm.Xs, dim=0)
    y = torch.cat(csbm.ys, dim=0)
    
    edge_index = []
    N = csbm.A.shape[0]
    for i in range(N):
        for j in range(N):
            if (i != j):
                if (csbm.A[i,j] == 1):
                    edge_index.append([i,j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=X, y=y, edge_index=edge_index, train_mask=train_ids, val_mask=val_ids, test_mask=test_ids)



def pygdata_to_frameformat(data):
    
    N = data.x.shape[0]
    Xs = data.x.view(N, 1, -1).type(torch.FloatTensor)
    ys = data.y.view(N, 1).type(torch.LongTensor)
    G = to_networkx(data, to_undirected=True)
    A = np.array(adjacency_matrix(G).todense())
    return G, Xs, ys, A
    
    
    

