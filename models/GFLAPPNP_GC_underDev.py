import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device, grad_device


def general_nll_loss(log_yhat, yhot, coef=-1):
    
    """
    yhat: [N, num_class]
    yhot: target hot tensor, [N, num_class]
    """
    
    y_bool = yhot.type(torch.BoolTensor) if device == "cpu" else yhot.type(torch.cuda.BoolTensor)
    
    nll_loss = (coef*log_yhat.view(-1))[y_bool.view(-1)].sum()
    
    return nll_loss

class Node:
    
    
    # Check done
    def __init__(self, local_model, N, node_idx, X, y, n_train, n_val, tnc=2):
        
        
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
        
        self.tnc = tnc
        self.N = N
        self.output_dim = tnc
        self.train_class_ids = {}
        train_y_np = self.y_train.cpu().numpy()
        avail_train_class = np.unique(train_y_np)
        for c in avail_train_class:
            self.train_class_ids[c] = np.where(train_y_np==c)[0]
    
    
    # Check done   
    def receieve_central_parameters(self, cmodel):
        
        with torch.no_grad():
            for pname, param in self.model.named_parameters():
                param.copy_(cmodel.state_dict()[pname])
                
    # Check done                       
    def upload_information(self, gradient, noise):
        
        dh_class = {} if gradient else None
        h_class = {}
        
        for c in range(self.tnc):
            if c in self.train_class_ids:
                x = self.X_train[self.train_class_ids[c]]
                h, dh = self.get_rep_and_grad(x, gradient, noise)
            else:
                h = torch.zeros(1, self.output_dim).to(device)
                if (gradient):
                    dh = {}
                    for pname, param in self.model.named_parameters():
                        dh[pname] = torch.zeros((self.output_dim,)+param.shape)

            h_class[c] = h
            if gradient:
                dh_class[c] = dh
        return h_class, dh_class
    
    
    
    # Check done
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
        
    # Check done
    def adversarial_local_update(self, A_tilde_k_d, A_tilde_k_gd, Ck_class, dH_class, 
                                 batch_size, learning_rate, I, gradient):
        
        if (batch_size > self.X_train.shape[0]):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=batch_size)
        
        k = self.idx
        
        N = self.N
        
        num_class = self.tnc
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                batch_size = X_batch.shape[0]
                optimizer.zero_grad()
                ids = torch.argsort(y_batch)
                X_batch, y_batch = X_batch[ids,:], y_batch[ids]
                avail_class, class_count = torch.unique(y_batch, sorted=True, return_counts=True)
                start = 0
                
                for i in range(avail_class.shape[0]):
                    
                    c_true = avail_class[i].item()
                    X, y = X_batch[start:start+class_count[i]], y_batch[start:start+class_count[i]]
                    start = class_count[i]
                    
                    for c in range(self.tnc):

                        H = self.model(X)
                        Z_c = A_tilde_k_d[k]*H + Ck_class[c]
                        y_hat_c = F.softmax(Z_c, dim=1)
                        y_hot_c = torch.zeros(y.shape[0], num_class).to(device)
                        y_hot_c[np.arange(y.shape[0]), c] = 1
                        
                        
                        if c == c_true:
                            coef = -1
                        else:
                            coef = 1
                            
                        if (gradient == True and dH_class != None):
                            loss_c = general_nll_loss(torch.log(y_hat_c), y_hot_c, coef)/(batch_size)
                            loss_c.backward()
                            with torch.no_grad():
                                Errs_c = -1*coef*(y_hat_c - y_hot_c).to(grad_device)
                                for pname, param in self.model.named_parameters():
                                    p_grad_c = torch.einsum("i,ibcd->ibcd", A_tilde_k_gd[self.ids_mask], 
                                                            dH_class[c][pname][self.ids_mask])

                                    p_grad_c = torch.einsum("ab,cbef->ef", Errs_c, p_grad_c)/(batch_size)
                                    param.grad.data += p_grad_c.to(device)
                        else:
                            loss_c = general_nll_loss(torch.log(y_hat_c), y_hot_c, coef)/(batch_size)
                            loss_c.backward()

                optimizer.step()
        
        
    # Check done
    def weighted_local_update(self, A_tilde_k_d, A_tilde_k_gd, Ck_class, dH_class, 
                           batch_size, learning_rate, I, gradient):
        
        if (batch_size > self.X_train.shape[0]):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=batch_size)
        
        k = self.idx
        
        N = self.N
        
        num_class = self.tnc
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                batch_size = X_batch.shape[0]
                optimizer.zero_grad()
                ids = torch.argsort(y_batch)
                X_batch, y_batch = X_batch[ids,:], y_batch[ids]
                avail_class, class_count = torch.unique(y_batch, sorted=True, return_counts=True)
                start = 0
                
                for i in range(avail_class.shape[0]):
                    
                    c_true = avail_class[i].item()
                    X, y = X_batch[start:start+class_count[i]], y_batch[start:start+class_count[i]]
                    start = class_count[i]
                    
                    for c in range(self.tnc):

                        H = self.model(X)
                        Z_c = A_tilde_k_d[k]*H + Ck_class[c]
                        y_hat_c = F.softmax(Z_c, dim=1)
                        
                        if c == c_true:
                            y_hot_c = torch.zeros(y.shape[0], num_class).to(device)
                            y_hot_c[np.arange(y.shape[0]), c_true] = 1
                            weight = 1
                        else:
                            y_hot_c = torch.ones(y.shape[0], num_class).to(device)
                            y_hot_c[np.arange(y.shape[0]),c] = 0
                            weight = (num_class-1)
                            
                        if (gradient == True and dH_class != None):
                            loss_c = general_nll_loss(torch.log(y_hat_c), y_hot_c)/(weight*batch_size)
                            loss_c.backward()
                            with torch.no_grad():
                                Errs_c = (y_hat_c - y_hot_c).to(grad_device)
                                for pname, param in self.model.named_parameters():
                                    p_grad_c = torch.einsum("i,ibcd->ibcd", A_tilde_k_gd[self.ids_mask], 
                                                            dH_class[c][pname][self.ids_mask])

                                    p_grad_c = torch.einsum("ab,cbef->ef", Errs_c, p_grad_c)/(weight*batch_size)
                                    param.grad.data += p_grad_c.to(device)
                        else:
                            loss_c = general_nll_loss(torch.log(y_hat_c), y_hot_c)/(weight*batch_size)
                            loss_c.backward()

                optimizer.step()
        
        
    # Check done
    def greedy_local_update(self, A_tilde_k_d, A_tilde_k_gd, Ck_class, dH_class, 
                           batch_size, learning_rate, I, gradient):
        
        if (batch_size > self.X_train.shape[0]):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=batch_size)
        
        k = self.idx
        
        N = self.N
        
        num_class = self.tnc
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                ids = torch.argsort(y_batch)
                X_batch, y_batch = X_batch[ids,:], y_batch[ids]
                avail_class, class_count = torch.unique(y_batch, sorted=True, return_counts=True)
                start = 0
                optimizer.zero_grad()
                
                for i in range(avail_class.shape[0]):
                    
                    c = avail_class[i].item()
                    X, y = X_batch[start:start+class_count[i]], y_batch[start:start+class_count[i]]
                    start = class_count[i]
                    
                    
                    H = self.model(X)
                    Z = A_tilde_k_d[k]*H + Ck_class[c]
                    y_hat = F.softmax(Z, dim=1)
                    
                    y_onehot = torch.zeros(y.shape[0], num_class).to(device)
                    y_onehot[np.arange(y.shape[0]), y] = 1
                            
                    if (gradient == True and dH_class != None):
                        loss = F.nll_loss(torch.log(y_hat), y)
                        loss.backward()
                        with torch.no_grad():
                            Errs = (y_hat - y_onehot).to(grad_device)
                            for pname, param in self.model.named_parameters():
                                p_grad = torch.einsum("i,ibcd->ibcd", A_tilde_k_gd[self.ids_mask], 
                                                        dH_class[c][pname][self.ids_mask])
                                
                                p_grad = torch.einsum("ab,cbef->ef", Errs, p_grad)/X_batch.shape[0]
                                param.grad.data += p_grad.to(device)
                    else:
                        loss = F.nll_loss(torch.log(y_hat), y)
                        loss.backward()
                
                optimizer.step()
    
    
    # Check done
    def local_update(self, A_tilde_k_d, A_tilde_k_gd, Ck_class, dH_class, 
                           batch_size, learning_rate, I, gradient):
        
        if (batch_size > self.X_train.shape[0]):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=batch_size)
        
        k = self.idx
        
        N = self.N
        
        num_class = self.tnc
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                
                optimizer.zero_grad()
                batch_size = X_batch.shape[0]
                
                for c in range(self.tnc):
                                                                
                    H = self.model(X_batch)
                    Z_c = A_tilde_k_d[k]*H + Ck_class[c]
                    y_hat_c = F.softmax(Z_c, dim=1)
                    
                    batch_size = X_batch.shape[0]
                    y_hot_c = torch.zeros(y_batch.shape[0], num_class).to(device)
                    
                    
                    for j in range(batch_size):
                        if (y_batch[j] == c):
                            hot_j = torch.zeros(num_class)
                            hot_j[c] = 1
                        else:
                            hot_j = torch.ones(num_class)
                            hot_j[c] = 0
                        y_hot_c[j,:] = hot_j
                            
                    if (gradient == True and dH_class != None):
                        loss_c = general_nll_loss(torch.log(y_hat_c), y_hot_c)/batch_size
                        loss_c.backward()
                        with torch.no_grad():
                            Errs_c = (y_hat_c - y_hot_c).to(grad_device)
                            for pname, param in self.model.named_parameters():
                                p_grad_c = torch.einsum("i,ibcd->ibcd", A_tilde_k_gd[self.ids_mask], 
                                                        dH_class[c][pname][self.ids_mask])
                                
                                p_grad_c = torch.einsum("ab,cbef->ef", Errs_c, p_grad_c)/batch_size
                                param.grad.data += p_grad_c.to(device)
                    else:
                        loss_c = general_nll_loss(torch.log(y_hat_c), y_hot_c)/batch_size
                        loss_c.backward()
                
                optimizer.step()
                
    # Check done            
    def cmodel_eval(self, cmodel, A_tilde_k, Ck_class, mode="train"):
        
        if mode == "train":
            X, y = self.X_train, self.y_train
        elif mode == "valid":
            X, y = self.X_val, self.y_val
        elif mode == "test":
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("mode should be either train, val, or test!")
            
        k = self.idx
        
        with torch.no_grad():
            H = cmodel(X)
            y_hat = []
            for c in range(self.tnc):
                Z_c = A_tilde_k[k]*H + Ck_class[c]
                y_hat_c = F.softmax(Z_c, dim=1)[:,c].view(-1, 1)
                y_hat.append(y_hat_c)
            y_hat = torch.cat(y_hat, dim=1)    
            y_pred = torch.max(y_hat, dim=1)[1]
            acc = (y_pred == y).sum().item()/y.shape[0]
            loss = F.nll_loss(torch.log(y_hat), y)
        return loss, acc
    
    # Check done
    def cmodel_collect(self, cmodel):
        
        h_class = {}
        for c in range(self.tnc):
            if c in self.train_class_ids:
                x = self.X_train[self.train_class_ids[c],:]
                with torch.no_grad():
                    h = torch.mean(cmodel(x), dim=0, keepdim=True)
            else:
                h = torch.zeros(1, self.output_dim).to(device)
            h_class[c] = h
            
        return h_class
    
    
class Central_Server:
    
    # Check done
    def __init__(self, model, node_list, A_tilde, tnc):

        self.A_tilde = A_tilde.to(device)
        self.A_tilde_gdevice = A_tilde.to(grad_device)
        
        self.node_list = node_list
        self.N = len(node_list)
        self.tnc = tnc
        
        self.cmodel = copy.deepcopy(model).to(device)
        
        self.best_cmodel = None
        self.best_valloss = np.inf
        self.best_valacc = 0
        
    # Check done    
    def broadcast_central_parameters(self):
        
        for node in self.node_list:
            node.receieve_central_parameters(self.cmodel)
    
    # Check done
    def collect_node_information(self, gradient, noise):
        
        H_class = {}
        
        if gradient:
            
            dH_class = {}
            for c in range(self.tnc):
                dH_class[c] = {}
                H_class[c] = []
                for pname in self.cmodel.state_dict().keys():
                    dH_class[c][pname] = []
                
            for i in range(self.N):
                h_i, dh_i = self.node_list[i].upload_information(gradient, noise)
                for c in range(self.tnc):
                    H_class[c].append(h_i[c])
                    for pname in self.cmodel.state_dict().keys():
                        dH_class[c][pname].append(dh_i[c][pname])

            # H: [N, num_class]
            for c in range(self.tnc):
                H_class[c] = torch.cat(H_class[c], dim=0)
                for pname in self.cmodel.state_dict().keys():
                    shape = dH_class[c][pname][0].shape
                    dH_class[c][pname] = torch.cat(dH_class[c][pname], dim=0).view((self.N,)+shape) 
            
            # dH: a list of gradient tensors for each parameter
            # dH[pname]: [N, num_class, *pname.shape]
            return H_class, dH_class
        
        else:
            
            for c in range(self.tnc):
                H_class[c] = []
            
            for i in range(self.N):
                h_i, _ = self.node_list[i].upload_information(gradient, noise)
                for c in range(self.tnc):
                    H_class[c].append(h_i[c])

            # H: [N, num_class]
            for c in range(self.tnc):
                H_class[c] = torch.cat(H_class[c], dim=0)
            
            return H_class, None
            
            
    # Check done    
    def communication(self, 
                      batch_size, learning_rate, I, 
                      gradient=True, noise=False):
          
        self.broadcast_central_parameters()
        
        H_class, dH_class = self.collect_node_information(gradient, noise)
        
        C_class = {}
        with torch.no_grad():
            for c in range(self.tnc):
                C_class[c] = torch.matmul(self.A_tilde, H_class[c])
        
        for k in range(self.N):
            Ck_class = {}
            for c in range(self.tnc):
                with torch.no_grad():
                    Ck_c = C_class[c][k,:] - self.A_tilde[k,k]*H_class[c][k,:]
                    Ck_class[c] = Ck_c
            self.node_list[k].adversarial_local_update(self.A_tilde[k,:], self.A_tilde_gdevice[k,:], 
                                           Ck_class, dH_class, 
                                           batch_size, learning_rate, I, gradient)
        
        self.mean_aggregation()

        train_loss, train_acc, val_loss, val_acc = self.eval_train_val()
        
        if (val_loss < self.best_valloss):
            self.best_valloss = val_loss
            self.best_cmodel = copy.deepcopy(self.cmodel)
            self.best_valacc = val_acc
        
        return train_loss, train_acc, val_loss, val_acc
            
    
    # Check done
    def mean_aggregation(self):
    
        with torch.no_grad():

            for pname, param in self.cmodel.named_parameters():

                p = self.node_list[0].model.state_dict()[pname]

                for i in range(1, self.N):

                    p = p + self.node_list[i].model.state_dict()[pname]

                p = p/self.N

                param.copy_(p)
                
                
    # Check done            
    def eval_train_val(self):
        
        avg_trainloss = 0
        avg_trainacc = 0
        avg_valloss = 0
        avg_valacc = 0
        
        """
        """
        H_class = {}
        
        for i in range(self.N):
            hi_class = self.node_list[i].cmodel_collect(self.cmodel)
            for c in range(self.tnc):
                if (i == 0):
                    H_class[c] = []
                H_class[c].append(hi_class[c])
                if (i == self.N-1):
                    H_class[c] = torch.cat(H_class[c], dim=0)
                    
        C_class = {}
        with torch.no_grad():
            for c in range(self.tnc):
                C_class[c] = torch.matmul(self.A_tilde, H_class[c])
        
        for k in range(self.N):
            Ck_class = {}
            for c in range(self.tnc):
                with torch.no_grad():
                    Ck_c = C_class[c][k,:] - self.A_tilde[k,k]*H_class[c][k,:]
                    Ck_class[c] = Ck_c
            tloss, tacc = self.node_list[k].cmodel_eval(self.cmodel, self.A_tilde[k,:], Ck_class, "train")
            vloss, vacc = self.node_list[k].cmodel_eval(self.cmodel, self.A_tilde[k,:], Ck_class, "valid")
            avg_valloss += vloss
            avg_valacc += vacc
            avg_trainloss += tloss
            avg_trainacc += tacc
            
        avg_trainloss = avg_trainloss/self.N
        avg_trainacc = avg_trainacc/self.N
        avg_valloss = avg_valloss/self.N
        avg_valacc = avg_valacc/self.N
            
        
        
        return  avg_trainloss, avg_trainacc, avg_valloss, avg_valacc
    
    
    # Check done
    def eval_test(self):
        
        avg_testloss = 0
        avg_testacc = 0
        
        testlosses = []
        testaccs = []
        

        H_class = {}
        
        for i in range(self.N):
            hi_class = self.node_list[i].cmodel_collect(self.best_cmodel)
            for c in range(self.tnc):
                if (i == 0):
                    H_class[c] = []
                H_class[c].append(hi_class[c])
                if (i == self.N-1):
                    H_class[c] = torch.cat(H_class[c], dim=0)
                    
        C_class = {}
        with torch.no_grad():
            for c in range(self.tnc):
                C_class[c] = torch.matmul(self.A_tilde, H_class[c])
        
        
        for k in range(self.N):
            Ck_class = {}
            for c in range(self.tnc):
                with torch.no_grad():
                    Ck_c = C_class[c][k,:] - self.A_tilde[k,k]*H_class[c][k,:]
                    Ck_class[c] = Ck_c
            tloss, tacc = self.node_list[k].cmodel_eval(self.best_cmodel, self.A_tilde[k,:], Ck_class, "test")
            testlosses.append(tloss)
            testaccs.append(tacc)
            avg_testloss += tloss
            avg_testacc += tacc
            
        avg_testloss = avg_testloss/self.N
        avg_testacc = avg_testacc/self.N
            
        
        
        return  avg_testloss, avg_testacc, testaccs, testlosses