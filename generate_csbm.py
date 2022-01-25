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
        
    def generate_node_data(self, n_local, method):
        
        # n_local: number of local data points for each node. 
        
        v = self.v
        p = self.p
        mu = self.mu
        u = self.u
        N = self.N
        b = self.b
        
        if method == "DNC":
            feature_dim = p
            
        elif method == "SNC":
            feature_dim = p
            
        elif method == "GC":
            feature_dim = int((p-1)/4)
            
        else:
            raise ValueError("wrong method!")

        
        self.Xs = torch.zeros(N, n_local, feature_dim).type(torch.FloatTensor)
        self.ys = torch.zeros(N, n_local).type(torch.LongTensor)
                
        if (n_local == 1 and method == "DNC"):
            for i in range(N):
                self.Xs[i] = torch.from_numpy(b[i].reshape(1, -1)).type(torch.FloatTensor)
                self.ys[i] = torch.tensor(self.v_mask[i]).view(-1).type(torch.LongTensor)
                
                
        elif (method == "SNC"):
            
            for i in range(N):
                
                cov_vec = np.exp(b[i])
                
                X = np.random.multivariate_normal(mean=b[i],
                                                  cov=np.diag(cov_vec), size=n_local)
                
                self.Xs[i] = torch.from_numpy(X).type(torch.FloatTensor)
                self.ys[i] = torch.tensor([self.v_mask[i]]*n_local).type(torch.LongTensor)
                
                
        elif (method == "GC"):
            
            
            base_var = 0.1
            if not ((p-1) % 4 == 0):
                raise ValueError("For this generation method to work, p-1 needs to be divisble by 4")
                
            bp = b[:,0]
            bp = (bp-np.min(bp))/(np.max(bp)-np.min(bp)).reshape(-1)
            self.bp = bp

            mu0 = b[:, 1:int((p-1)/4)+1]
            sigma0 = b[:,int((p-1)/4)+1:int((p-1)/2)+1]
            sigma0 = sigma0 - np.amin(sigma0, axis=0) + base_var

            mu1 = b[:,int((p-1)/2)+1:int(3*(p-1)/4)+1]
            sigma1 = b[:,int(3*(p-1)/4)+1:]
            sigma1 = sigma1 - np.amin(sigma1, axis=0) + base_var
            
            for i in range(N):
                y = np.random.binomial(n=1, p=bp[i], size=n_local)
                self.ys[i] = torch.tensor(y).type(torch.LongTensor)
                               
                X = []
                
                for j in range(n_local):
                               
                    if (y[j] == 0):
                        X.append(np.random.multivariate_normal(mean=mu0[i],
                                                                 cov=np.diag(sigma0[i])))
                    else:
                        X.append(np.random.multivariate_normal(mean=mu1[i],
                                                     cov=np.diag(sigma1[i])))
                               
                X = np.array(X)
                self.Xs[i] = torch.from_numpy(X).type(torch.FloatTensor)
                
        else:
            raise ValueError("wrong combination between method and n_local!")
