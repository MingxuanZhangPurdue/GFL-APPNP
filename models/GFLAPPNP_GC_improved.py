import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device, grad_device


def general_nll_loss(log_yhat, yhot):
    
    """
    yhat: [N, num_class]
    yhot: target hot tensor, [N, num_class]
    """
    
    y_bool = yhot.type(torch.BoolTensor) if device == "cpu" else yhot.type(torch.cuda.BoolTensor)
    
    N, _ = yhot.shape
    
    nll_loss = (-1*log_yhat.view(-1))[y_bool.view(-1)].sum()/N
    
    return nll_loss

class Node:
    
    def __init__(self, local_model, N, node_idx, X, y, n_train, n_val, num_classes=2):
        
        
        self.model = copy.deepcopy(local_model).to(device)
        self.idx = node_idx
        self.n_local = X.shape[0]
        self.train_dataloader = None
        self.train_batchsize = 1
        self.ids_mask = np.ones(N, dtype=bool)
        self.ids_mask[node_idx] = False
        
        ids = np.arange(self.n_local)
        train_ids = ids[0:n_train]
        val_ids = ids[n_train:n_train+n_val]
        test_ids = ids[n_train+n_val:]
        
        self.X, self.y = X.to(device), y.to(device)
        self.X_train, self.y_train = self.X[train_ids], self.y[train_ids]
        self.X_val, self.y_val = self.X[val_ids], self.y[val_ids]
        self.X_test, self.y_test = self.X[test_ids], self.y[test_ids]
        
        self.tnc = num_classes
        self.train_class_ids = {}
        train_y_np = self.y_train.cpu().numpy()
        for c in np.unique(train_y_np):
            self.train_class_ids[c] = np.where(train_y_np==c)[0]
        
        
    def receieve_central_parameters(self, cmodel):
        
        with torch.no_grad():
            for pname, param in self.model.named_parameters():
                param.copy_(cmodel.state_dict()[pname])
                
                           
    def upload_information(self, gradient, noise):
        
        dh_class = [] if gradient else None
        h_class = []
        
        for c in range(self.tnc):
            if c in self.train_class_ids:
                x = self.X_train[self.train_class_ids[c],:]
                h, dh = self.get_rep_and_grad(x, gradient, noise)
            else:
                x = torch.zeros(1, self.X_train[0,:].shape[0]).to(device)
                with torch.no_grad():
                    h = self.model(x)
                if (gradient):
                    dh = {}
                    for pname, param in self.model.named_parameters():
                        dh[pname] = torch.zeros((self.tnc,)+param.shape)

            h_class.append(h)
            if gradient:
                dh_class.append(dh)
        return h_class, dh_class
    
    
    
    
    def get_rep_and_grad(self, x, gradient, noise):
            
        if gradient:
            
            self.model.zero_grad()
        
            h = self.model(x)
            
            h = torch.mean(h, dim=0, keepdims=True)
            
            num_class = self.tnc

            dh = {}

            for i in range(num_class):

                h[0, i].backward(retain_graph=True)

                for pname, param in self.model.named_parameters():

                    if pname in dh:
                        dh[pname].append(param.grad.data.detach().cpu().clone())
                    else:
                        dh[pname] = []
                        dh[pname].append(param.grad.data.detach().cpu().clone())

                    if (i == num_class-1):
                        
                        param_shape = dh[pname][0].shape
                        dh[pname] = torch.cat(dh[pname], dim=0).view((num_class,)+param_shape).to(grad_device)
                        
                        if noise == True:
                            dh[pname] += torch.randn(dh[pname].shape).to(grad_device)

                self.model.zero_grad()

            return h.detach(), dh
        
        else:
            with torch.no_grad():
                h = self.model(x)
                
            return h, None
    
    
    def local_update(self, A_tilde_k_d, A_tilde_k_gd, Ck_class, dH_class, 
                              batch_size, learning_rate, I, gradient):
        
        f not (batch_size <= self.n_local):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=
        
        k = self.idx
        
        N = A_tilde_k.shape[0]
        
        num_class = self.tnc
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                optimizer.zero_grad()
                
                for c in range(self.tnc):
                                                                
                    H_c = self.model(X_batch)
                    Z_c = A_tilde_k_d[k]*H_c + Ck_class[c]
                    y_hat_c = F.softmax(Z_c, dim=1)
                                                                
                    y_onehot_c = torch.zeros(y_batch.shape[0], num_class).to(device)
                    y_onehot_c[:,c] = 1                                            
                                                                
                    
                    if (gradient == True and dH_class != None):
                        loss_c = general_nll_loss(torch.log(y_hat_c), y_onehot_c)
                        loss_c.backward()
                        Errs_c = (y_hat_c - y_onehot_c).to(grad_device)
                        for pname, param in self.model.named_parameters():
                            p_grad_c = torch.einsum("i,ibcd->ibcd", A_tilde_k_gd[self.ids_mask], dH_class[c][pname][self.ids_mask])
                            p_grad_c = torch.einsum("ab,cbef->ef", Errs_c, p_grad_c)/X.shape[0]
                            param.grad.data += p_grad_c.to(device)
                    else:
                        loss_c = general_nll_loss(torch.log(y_hat_c), y_onehot_c)
                        loss_c.backward()
                
                optimizer.step()
    
    
class Central_Server:
    
    def __init__(self, model, node_list, A_tilde):

        self.A_tilde = A_tilde.to(device)
        self.A_tilde_gdevice = A_tilde.to(grad_device)
        
        self.node_list = node_list
        self.N = len(node_list)
        
        self.cmodel = copy.deepcopy(model).to(device)
        
        self.best_cmodel = None
        self.best_valloss = np.inf
        self.best_valacc = 0
        
        
    def broadcast_central_parameters(self):
        
        for node in self.node_list:
            node.receieve_central_parameters(self.cmodel)
        
    def collect_node_information(self, gradient, noise):
        
        H = []
        
        if gradient:
            
            dH = {}
            for pname in self.cmodel.state_dict().keys():
                dH[pname] = []
                
            for i in range(self.N):
                h_i, dh_i = self.node_list[i].upload_information(gradient, noise)
                H.append(h_i)
                for pname in self.cmodel.state_dict().keys():
                    dH[pname].append(dh_i[pname])

            # H: [N, num_class]
            H = torch.cat(H, dim=0)
            
            for pname in self.cmodel.state_dict().keys():
                shape = dH[pname][0].shape
                dH[pname] = torch.cat(dH[pname], dim=0).view((self.N,)+shape) 
            
            # dH: a list of gradient tensors for each parameter
            # dH[pname]: [N, num_class, *pname.shape]
            return H, dH
        
        else:
            for i in range(self.N):
                h_i, _ = self.node_list[i].upload_information(gradient, noise)
                H.append(h_i)

            # H: [N, num_class]
            H = torch.cat(H, dim=0)
            
            return H, None
            
            
        
    def communication(self, 
                      batch_size, learning_rate, I, 
                      gradient=True, noise=False):
          
        self.broadcast_central_parameters()
        
        # H: [N, num_class]
        H, dH = self.collect_node_information(gradient, noise)
        
        # C: [N, num_class]
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
        
        for k in range(self.N):
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            self.node_list[k].local_update(self.A_tilde[k,:], self.A_tilde_gdevice[k,:], 
                                           C_k, dH, 
                                           batch_size, learning_rate, 
                                           I, 
                                           gradient)
        
        self.mean_aggregation()

        train_loss, train_acc, val_loss, val_acc = self.eval_train_val()
        
        if (val_loss < self.best_valloss):
            self.best_valloss = val_loss
            self.best_cmodel = copy.deepcopy(self.cmodel)
            self.best_valacc = val_acc
        
        return train_loss, train_acc, val_loss, val_acc
            
    
    
    def mean_aggregation(self):
    
        with torch.no_grad():

            for pname, param in self.cmodel.named_parameters():

                p = self.node_list[0].model.state_dict()[pname]

                for i in range(1, self.N):

                    p = p + self.node_list[i].model.state_dict()[pname]

                p = p/self.N

                param.copy_(p)