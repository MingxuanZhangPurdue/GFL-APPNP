import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device, grad_device
import torch.utils.data



class Node:
    
    def __init__(self, local_model, N, node_idx, X, y):
        
        self.model = local_model.to(device)
        self.idx = node_idx
        self.X = X
        self.y = y
        self.n_local = self.X.shape[0]
        self.data_loader = None
        self.ids_mask = np.ones(N, dtype=bool)
        self.ids_mask[node_idx] = False
        
    
    def receieve_central_parameters(self, cmodel):
        
        with torch.no_grad():
            for pname, param in self.model.named_parameters():
                param.copy_(cmodel.state_dict()[pname])
                
                
    def upload_information(self, gradient, 
                           hidden_noise, gradient_noise,
                           hn_std, gn_std):
        
        x = self.X
        
        #x = x.to(device)
            
        if gradient:
            
            self.model.zero_grad()
        
            h = torch.mean(self.model(x), dim=0, keepdim=True)
            
            if hidden_noise:
                h += hn_std*torch.randn(h.shape).to(device)
            
            num_class = h.shape[-1]

            dh = {}

            for i in range(num_class):

                h[0, i].backward(retain_graph=True)

                for pname, param in self.model.named_parameters():

                    if pname in dh:
                        dh[pname].append(param.grad.data.detach().clone())
                        
                    else:
                        dh[pname] = []
                        dh[pname].append(param.grad.data.detach().clone())

                    if (i == num_class-1):
                        
                        param_shape = dh[pname][0].shape
                        dh[pname] = torch.cat(dh[pname], dim=0).view((num_class,)+param_shape).to(grad_device)
                        
                        if gradient_noise == True:
                            dh[pname] += gn_std*torch.randn(dh[pname].shape).to(grad_device)

                self.model.zero_grad()

            return h.detach(), dh
        
        else:
            with torch.no_grad():
                h = torch.mean(self.model(x), dim=0, keepdim=True)
                if hidden_noise:
                    h += hn_std*torch.randn(h.shape).to(device)
                
            return h, None
    

    
    def local_update(self, A_tilde_k_d, A_tilde_k_gd, C_k, dH, batch_size,
                     learning_rate, I):
        
        if (batch_size > self.n_local):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.data_loader == None):
            dataset = torch.utils.data.TensorDataset(self.X, self.y)
            self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        k = self.idx
        
        N = A_tilde_k_d.shape[0]
        
        num_class = C_k.shape[-1]
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X, y in self.data_loader: 
                
                optimizer.zero_grad()
                #X, y = X.to(device), y.to(device)
                H = self.model(X)
                Z = A_tilde_k_d[k]*H + C_k
                y_hat = F.softmax(Z, dim=1)

                if dH != None:
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
                
    def majority_eval(self, cmodel, A_tilde_k, C_k):
        k = self.idx
        with torch.no_grad():
            X, y = self.X, self.y
            H = cmodel(X)
            Z = A_tilde_k[k]*H + C_k
            y_hat = F.softmax(Z, dim=1)
            y_pred = torch.max(y_hat, dim=1)[1]
            labels, label_counts = torch.unique(y_pred, sorted=True, return_counts=True)
            major_pred = labels[torch.argmax(label_counts)]
            acc = (major_pred == y).sum().item()/y.shape[0]
            loss = F.nll_loss(torch.log(y_hat), y)
        return loss, acc
        
            
            
    def cmodel_eval(self, cmodel, A_tilde_k, C_k):
        k = self.idx
        with torch.no_grad():
            X, y = self.X, self.y
            H = cmodel(X)
            Z = A_tilde_k[k]*H + C_k
            y_hat = F.softmax(Z, dim=1)
            y_pred = torch.max(y_hat, dim=1)[1]
            acc = (y_pred == y).sum().item()/y.shape[0]
            loss = F.nll_loss(torch.log(y_hat), y)
        return loss, acc
    
    
    def cmodel_collect(self, cmodel, hidden_noise, hn_std):
        
        x = self.X
        #x = x.to(device)
        with torch.no_grad():  
            h = torch.mean(cmodel(x), dim=0, keepdims=True)
            if hidden_noise:
                h += hn_std*torch.randn(h.shape).to(device)
        return h
        
            
            
            
class Central_Server:
    
    def __init__(self, init_model, 
                 node_list, A_tilde, 
                 train_indices, valid_indices, test_indices,
                 gradient=True,
                 hidden_noise=False, gradient_noise=False,
                 hn_std=0.1, gn_std=0.1):

        self.A_tilde = A_tilde.to(device)
        self.A_tilde_gdevice = A_tilde.to(grad_device)
        self.node_list = node_list
        self.N = len(node_list)
        self.cmodel = copy.deepcopy(init_model).to(device)
        self.train_ids = train_indices
        self.val_ids = valid_indices
        self.test_ids = test_indices
        self.best_cmodel = None
        self.best_valloss = np.inf
        self.best_valacc = 0
        self.gradient = gradient
        self.hidden_noise = hidden_noise
        self.gradient_noise = gradient_noise
        self.hn_std = hn_std
        self.gn_std = gn_std
        
        '''
        print (self.gradient,
        self.hidden_noise,
        self.gradient_noise,
        self.hn_std,
        self.gn_std)
        '''
        
        
    def broadcast_central_parameters(self):
        
        if self.cmodel == None:
            raise ValueError("Central model is None, Please initilalize it first.")
        
        for node in self.node_list:
            node.receieve_central_parameters(self.cmodel)
        
    def collect_node_information(self):
        
        H = []
        
        if self.gradient:
            
            dH = {}
            for pname in self.cmodel.state_dict().keys():
                dH[pname] = []
                
            for i in range(self.N):
                h_i, dh_i = self.node_list[i].upload_information(self.gradient,
                                                                 self.hidden_noise, self.gradient_noise,
                                                                 self.hn_std, self.gn_std)
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
                h_i, _ = self.node_list[i].upload_information(self.gradient,
                                                              self.hidden_noise, self.gradient_noise,
                                                              self.hn_std, self.gn_std)
                H.append(h_i)

            # H: [N, num_class]
            H = torch.cat(H, dim=0)
            
            return H, None
            
            
        
    def communication(self, 
                      batch_size, learning_rate, I):
          
        self.broadcast_central_parameters()
        
        # H: [N, num_class]
        H, dH = self.collect_node_information()
        
        # C: [N, num_class]
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
        
        for k in self.train_ids:
            #print (k)
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            self.node_list[k].local_update(self.A_tilde[k,:], self.A_tilde_gdevice[k,:], C_k, dH, batch_size, learning_rate, I)
        
        self.mean_aggregation_train()

        train_loss, train_acc, val_loss, val_acc = self.eval_train_val()
        
        if (val_loss < self.best_valloss):
            self.best_valloss = val_loss
            self.best_cmodel = copy.deepcopy(self.cmodel)
            self.best_valacc = val_acc
        
        return train_loss, train_acc, val_loss, val_acc
            
    
    
    def mean_aggregation_train(self):
        
        num_train = len(self.train_ids)
    
        with torch.no_grad():

            for pname, param in self.cmodel.named_parameters():

                p = self.node_list[self.train_ids[0]].model.state_dict()[pname]

                for i in range(1, num_train):

                    p = p + self.node_list[self.train_ids[i]].model.state_dict()[pname]

                p = p/num_train

                param.copy_(p)
 
 

    def eval_train_val(self):
        
        avg_trainloss = 0
        avg_trainacc = 0
        avg_valloss = 0
        avg_valacc = 0
        
        H = []
        for i in range(self.N):
            H.append(self.node_list[i].cmodel_collect(self.cmodel, self.hidden_noise, self.hn_std))
        H = torch.cat(H, dim=0)
            
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
            
        for k in self.train_ids:
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            tloss, tacc = self.node_list[k].cmodel_eval(self.cmodel, self.A_tilde[k,:], C_k)
            avg_trainloss += tloss.item()
            avg_trainacc += tacc
        avg_trainloss = avg_trainloss/len(self.train_ids)
        avg_trainacc = avg_trainacc/len(self.train_ids)
        

        for k in self.val_ids:
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            vloss, vacc = self.node_list[k].cmodel_eval(self.cmodel, self.A_tilde[k,:], C_k)
            avg_valloss += vloss.item()
            avg_valacc += vacc
        avg_valloss = avg_valloss/len(self.val_ids)
        avg_valacc = avg_valacc/len(self.val_ids)
        
        return  avg_trainloss, avg_trainacc, avg_valloss, avg_valacc
    
    
    
    def eval_test(self):
        
        avg_testloss = 0
        avg_testacc = 0
        
        H = []
        for i in range(self.N):
            H.append(self.node_list[i].cmodel_collect(self.best_cmodel, self.hidden_noise, self.hn_std))
        H = torch.cat(H, dim=0)
            
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
            
        for k in self.test_ids:
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
            tloss, tacc = self.node_list[k].cmodel_eval(self.best_cmodel, self.A_tilde[k,:], C_k)
            avg_testloss += tloss.item()
            avg_testacc += tacc
        avg_testloss = avg_testloss/len(self.test_ids)
        avg_testacc = avg_testacc/len(self.test_ids)
        
        
        return  avg_testloss, avg_testacc