import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device, grad_device

class Node:
    
    def __init__(self, local_model, N, node_idx, X, y, n_train, n_val):
        
        
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
    
    
    def receieve_central_parameters(self, cmodel):
        
        with torch.no_grad():
            for pname, param in self.model.named_parameters():
                param.copy_(cmodel.state_dict()[pname])
                
    def upload_information(self, gradient, noise):
        
        x = self.X
            
        if gradient:
            
            self.model.zero_grad()
        
            h = torch.mean(self.model(x), dim=0, keepdim=True)
            
            num_class = h.shape[-1]

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
                h = torch.mean(self.model(x), dim=0, keepdim=True)
                
            return h, None
    

    
    def local_update(self, A_tilde_k_d, A_tilde_k_gd, C_k, dH, batch_size,
                     learning_rate, I, 
                     gradient):
        
        if not (batch_size <= self.n_local):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=self.train_batchsize, 
                                                                shuffle=True, drop_last=True)
        
        k = self.idx
        
        N = A_tilde_k_d.shape[0]
        
        num_class = C_k.shape[-1]
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X, y in self.train_dataloader: 
                
                optimizer.zero_grad()
                H = self.model(X)
                Z = A_tilde_k_d[k]*H + C_k
                y_hat = F.softmax(Z, dim=1)

                if (gradient == True and dH != None):
                    local_loss = F.nll_loss(torch.log(y_hat), y)
                    local_loss.backward()
                    with torch.no_grad():
                        y_onehot = torch.zeros(y.shape[0], num_class).to(device)
                        y_onehot[np.arange(y.shape[0]), y] = 1
                        Errs = (y_hat - y_onehot).to(grad_device)
                        for pname, param in self.model.named_parameters():
                            p_grad = torch.einsum("i,ibcd->ibcd", A_tilde_k_gd[self.ids_mask], dH[pname][self.ids_mask])
                            p_grad = torch.einsum("ab,cbef->ef", Errs, p_grad)/X.shape[0]
                            param.grad.data += p_grad.to(device)
                        
                else:
                    local_loss = F.nll_loss(torch.log(y_hat), y)
                    local_loss.backward()
                    
                optimizer.step()
            
            
    def cmodel_eval(self, cmodel, A_tilde_k, C_k, mode="train"):
        
        if mode == "train":
            X, y = self.X_train, self.y_train
        elif mode == "val":
            X, y = self.X_val, self.y_val
        elif mode == "test":
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("mode should be either train, val, or test!")
            
        k = self.idx
        with torch.no_grad():
            H = cmodel(X)
            Z = A_tilde_k[k]*H + C_k
            y_hat = F.softmax(Z, dim=1)
            y_pred = torch.max(y_hat, dim=1)[1]
            acc = (y_pred == y).sum().item()/y.shape[0]
            loss = F.nll_loss(torch.log(y_hat), y)
        return loss, acc
    
    
    def cmodel_collect(self, cmodel):
        
        x = self.X_train
        with torch.no_grad():  
            h = torch.mean(cmodel(x), dim=0, keepdims=True)
        return h
    
    
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
 
 

    def eval_train_val(self):
        
        avg_trainloss = 0
        avg_trainacc = 0
        avg_valloss = 0
        avg_valacc = 0
        
        H = []
        for i in range(self.N):
            H.append(self.node_list[i].cmodel_collect(self.cmodel))
        H = torch.cat(H, dim=0)
            
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
            
        for k in range(self.N):
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            tloss, tacc = self.node_list[k].cmodel_eval(self.cmodel, self.A_tilde[k,:], C_k, mode="train")
            vloss, vacc = self.node_list[k].cmodel_eval(self.cmodel, self.A_tilde[k,:], C_k, mode="val")
            avg_trainloss += tloss.item()
            avg_trainacc += tacc
            avg_valloss += vloss.item()
            avg_valacc += vacc
        avg_trainloss = avg_trainloss/self.N
        avg_trainacc = avg_trainacc/self.N
        avg_valloss = avg_valloss/self.N
        avg_valacc = avg_valacc/self.N
        
        return  avg_trainloss, avg_trainacc, avg_valloss, avg_valacc
    
    
    
    def eval_test(self):
        
        avg_testloss = 0
        avg_testacc = 0
        
        H = []
        for i in range(self.N):
            H.append(self.node_list[i].cmodel_collect(self.best_cmodel))
        H = torch.cat(H, dim=0)
            
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
            
        for k in range(self.N):
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            tloss, tacc = self.node_list[k].cmodel_eval(self.best_cmodel, self.A_tilde[k,:], C_k, mode="test")
            avg_testloss += tloss.item()
            avg_testacc += tacc
        avg_testloss = avg_testloss/self.N
        avg_testacc = avg_testacc/self.N
        
        
        return  avg_testloss, avg_testacc
        
            
            
            
            
        
        