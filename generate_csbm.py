import numpy as np
import torch

class cSBM:
    
    def __init__(self, N, p, d, mu, l):
        
        """
        N: number of nodes
        p: dimension of feature vector 
        d: average degree
        l: lambda, hyperparameter
        mu: mu, hyperparameter
        """
        
        
        # Generate class from {-1, 1} for each node
        v = np.random.choice(a = [-1, 1],
                             size = N,
                             replace = True,
                             p = [0.5, 0.5])
        
        
        # Mask -1 to 0 and store the result in v_mask
        v_mask = np.copy(v)
        v_mask[v==-1] = 0
        
        # calculate c_in and c_out
        c_in = d + np.sqrt(d)*l
        c_out = d - np.sqrt(d)*l
        
        #u = np.random.normal(loc=0, scale=1/np.sqrt(p), size=p)
        
        # Generate the adjacent matrix without self-loop
        A = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1, N):
                if (v[i] == v[j]):
                    if (np.random.choice(a = [1,0], p = [c_in/N, 1-c_in/N])):
                        A[i,j] = 1.0
                    else:
                        A[i,j] = 0.0
                else:
                    if (np.random.choice(a = [1,0],p = [c_out/N, 1-c_out/N])):
                        A[i,j] = 1.0
                    else:
                        A[i,j] = 0.0
        A = A + A.T
        
        # Save all the necessary parameters
        self.v = v
        self.v_mask = v_mask
        self.A = A
        self.p = p
        self.N = N
        self.mu = mu
        self.d = d
        self.l = l
        xi = N/p
        self.phi = np.arctan((l*np.sqrt(xi))/mu)*(2/np.pi)
        self.threshold = l**2 + (mu**2)/(N/p)
        
        
    def generate_node_parameters(self):
        
        v = self.v
        p = self.p
        mu = self.mu
        N = self.N
        b = []
        
        u = np.random.normal(loc=0, scale=1/np.sqrt(p), size=p)
        self.u = u
        
        for i in range(N):
            b_i = np.sqrt(mu/N)*v[i]*u + np.random.normal(loc=0, scale=1, size=p)/np.sqrt(p)
            b.append(b_i)
        b = np.array(b)
        self.b = b
        
    def generate_node_features(self, n_local, method):
        
        # n_local: number of local data points for each node. 
        
        v = self.v
        p = self.p
        mu = self.mu
        u = self.u
        N = self.N
        b = self.b
        
        if method == "DNC":
            feature_dim = p
        else:
            raise ValueError("wrong method!")

        
        self.Xs = torch.zeros(N, n_local, feature_dim).type(torch.FloatTensor)
        self.ys = torch.zeros(N, n_local).type(torch.LongTensor)
                
        if (n_local == 1 and method == "DNC"):
            for i in range(N):
                self.Xs[i] = torch.from_numpy(b[i].reshape(1, -1)).type(torch.FloatTensor)
                self.ys[i] = torch.tensor(self.v_mask[i]).view(-1).type(torch.LongTensor)
                
        else:
            raise ValueError("wrong combination between method and n_local!")
