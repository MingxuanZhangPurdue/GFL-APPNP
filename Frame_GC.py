import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device
import torch.utils.data

class Node:
    
    def __init__(self, local_model, node_idx, X, y, total_n_class, n_train, n_val, bp_i, rs=None):
        
        
        self.model = local_model.to(device)
        self.idx = node_idx
        self.n_local = X.shape[0]
        self.tnc = total_n_class
        self.train_dataloader = None
        self.true_bp = bp_i
        
        
        ids = np.arange(self.n_local)
        if (rs != None):
            np.random.seed(rs)
        np.random.shuffle(ids)
        train_ids = ids[0:n_train]
        val_ids = ids[n_train:n_train+n_val]
        test_ids = ids[n_train+n_val:]
        
        self.X_train, self.y_train = X[train_ids,:], y[train_ids]
        self.X_val, self.y_val = X[val_ids,:], y[val_ids]
        self.X_test, self.y_test = X[test_ids,:], y[test_ids]
        
        self.train_class_ids = {}
        train_y_np = self.y_train.numpy()
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
        
        x = x.to(device)
            
        if gradient:
            
            self.model.zero_grad()
        
            h = self.model(x)
            
            h = torch.mean(h, dim=0, keepdims=True)
            
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
                        d1, d2 = dh[pname][0].shape
                        dh[pname] = torch.cat(dh[pname], dim=0).view(num_class, d1, d2)
                        if (noise):
                            dh[pname] += torch.randn(dh[pname].shape)

                self.model.zero_grad()

            return h.detach(), dh
        
        else:
            with torch.no_grad():
                h = self.model(x)
                
            return h, None
    

    
    def local_update(self, A_tilde_k, Ck_class, dH_class, 
                     batch_size, learning_rate, I, gradient):
        
        if not (batch_size <= self.n_local):
            raise AssertionError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None):
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        k = self.idx
        
        N = A_tilde_k.shape[0]
        
        num_class = Ck_class[0].shape[-1]
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                ids = torch.argsort(y_batch)
                X_batch, y_batch = X_batch[ids,:], y_batch[ids]
                avail_class, class_count = torch.unique(y_batch, sorted=True, return_counts=True)
                start = 0
                optimizer.zero_grad()
                for i in range(avail_class.shape[0]):
                    
                    c = avail_class[i]
                    X, y = X_batch[start:start+class_count[i],:], y_batch[start:start+class_count[i]]
                    start = class_count[i]
                    
                    
                    X, y = X.to(device), y.to(device)
                    H = self.model(X)
                    Z = A_tilde_k[k]*H + Ck_class[c]
                    y_hat = F.softmax(Z, dim=1)

                    if (gradient == True and dH_class != None):
                        local_loss = F.nll_loss(torch.log(y_hat), y)/avail_class.shape[0]
                        local_loss.backward()
                        with torch.no_grad():
                            y_onehot = torch.zeros(y.shape[0], num_class).to(device)
                            y_onehot[np.arange(y.shape[0]), y] = 1
                            Errs = y_hat - y_onehot
                            for pname, param in self.model.named_parameters():
                                n_grad = torch.zeros(param.grad.data.shape).to(device)
                                for i in range(N):
                                    if (i != k):
                                        n_grad += A_tilde_k[i]*torch.tensordot(Errs, dH_class[c][i][pname].to(device), 
                                                                               dims=1).sum(dim=0)/X.shape[0]
                                param.grad.data += n_grad/avail_class.shape[0]

                    else:
                        local_loss = F.nll_loss(torch.log(y_hat), y)
                        local_loss.backward()
                    
                optimizer.step()
                    
                    
    def weighted_local_update(self, A_tilde_k, Ck_class, dH_class, 
                              batch_size, learning_rate, I, gradient):
        
        if not (batch_size <= self.n_local):
            raise AssertionError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None):
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        k = self.idx
        
        N = A_tilde_k.shape[0]
        
        num_class = self.tnc
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X_batch, y_batch in self.train_dataloader:
                
                optimizer.zero_grad()
                
                for c in range(self.tnc):
                    H_c = self.model(X_batch)
                    Z_c = A_tilde_k[k]*H_c + Ck_class[c]
                    y_hat_c = F.softmax(Z_c, dim=1)
                    coef = self.true_bp if c == 1 else 1-self.true_bp
                    
                    if (gradient == True and dH_class != None):
                        loss_c = F.nll_loss(torch.log(y_hat_c), y_batch)*coef
                        loss_c.backward()

                        with torch.no_grad():
                            y_onehot_c = torch.zeros(y_batch.shape[0], num_class).to(device)
                            y_onehot_c[np.arange(y_batch.shape[0]), y_batch] = 1
                            Errs_c = y_hat_c - y_onehot_c
                            for pname, param in self.model.named_parameters():
                                n_grad = torch.zeros(param.grad.data.shape).to(device)
                                for i in range(N):
                                    if (i != k):
                                        n_grad += A_tilde_k[i]*torch.tensordot(Errs_c, dH_class[c][i][pname].to(device), 
                                                                               dims=1).sum(dim=0)
                                param.grad.data += (coef*n_grad)/X_batch.shape[0]
                    else:
                        loss_c = F.nll_loss(torch.log(y_hat_c), y_batch)*coef
                        loss_c.backward()
                
                optimizer.step()
                
    
        
 
    def weighted_eval(self, mode, cmodel, A_tilde_k, Ck_class):
        k = self.idx
        
        if mode == "train":
            X, y = self.X_train, self.y_train
  
        elif mode == "valid":
            X, y = self.X_val, self.y_val

        elif mode == "test":
            X, y = self.X_test, self.y_test
   
        else:
            print ("Unspecified mode, will use train set!")
            X, y = self.X_train, self.y_train
        
        X_batch, y_batch = X.to(device), y.to(device)
        y_pred = []
        y_hats = []
        with torch.no_grad():
            H = cmodel(X_batch)
            for c in range(self.tnc):
                Z_c = A_tilde_k[k]*H + Ck_class[c]
                y_hat_c = F.softmax(Z_c, dim=1)
                y_hats.append(y_hat_c)
                coef = self.true_bp if c == 1 else 1-self.true_bp
                y_pred.append(y_hat_c[:,c].view(-1,1)*coef)
            
            y_pred = torch.cat(y_pred, dim=1)
            y_pred = torch.max(y_pred, dim=1)[1]
            acc = (y_pred == y_batch).sum().item()/y_batch.shape[-1]
            
            y_hat = []
            for i in range(y_batch.shape[-1]):
                y_hat.append(y_hats[y_pred[i]][i].view(1,-1))
                
            y_hat = torch.cat(y_hat, dim=0)
            loss = F.nll_loss(torch.log(y_hat), y_batch).item()
            
        return loss, acc  
    
    
    def cmodel_collect(self, cmodel):

        h_class = []
        for c in range(self.tnc):
            if c in self.train_class_ids:
                x = self.X_train[self.train_class_ids[c],:]
                x = x.to(device)
                with torch.no_grad():
                    h = torch.mean(cmodel(x), dim=0, keepdim=True)
            else:
                x = torch.zeros(1, self.X_train[0,:].shape[0]).to(device)
                with torch.no_grad():
                    h = cmodel(x)
            h_class.append(h)
            
        return h_class
    
    
class Central_Server:
    
    def __init__(self, node_list, A_tilde, total_n_class):
        
        
        self.A_tilde = A_tilde.to(device)
        self.node_list = node_list
        self.N = len(node_list)
        self.tnc = total_n_class
        self.cmodel = None
        self.best_cmodel = None
        self.best_valloss = np.inf
        
    def init_central_parameters(self, model):
        
        self.cmodel = copy.deepcopy(model)
        self.cmodel = self.cmodel.to(device)
        
        
    def broadcast_central_parameters(self):
        
        if self.cmodel == None:
            raise ValueError("Central model is None, Please initilalize it first.")
        
        for node in self.node_list:
            node.receieve_central_parameters(self.cmodel)
        
    def collect_node_information(self, gradient, noise):
        
        H_class = []
        dH_class = []
        
        for i in range(self.N):
            h_class, dh_class = self.node_list[i].upload_information(gradient, noise)
            for c in range(self.tnc):
                if (i == 0):
                    H_class.append([])
                    if gradient:
                        dH_class.append([])
                H_class[c].append(h_class[c])
                if gradient:
                    dH_class[c].append(dh_class[c])
                if (i == self.N-1):
                    H_class[c] = torch.cat(H_class[c], dim=0)
                
        if (gradient):        
            return H_class, dH_class 
        else:
            return H_class, None
            
        
    def communication(self, 
                      batch_size, learning_rate=0.01, I=10, 
                      gradient=True, noise=False, 
                      update_method="weighted"):
            
        self.broadcast_central_parameters()
        
        
        H_class, dH_class = self.collect_node_information(gradient, noise)
        
        C_class = []
        with torch.no_grad():
            for c in range(self.tnc):
                C_class.append(torch.matmul(self.A_tilde, H_class[c]))
        
        
        for k in range(self.N):
            Ck_class = []
            for c in range(self.tnc):
                with torch.no_grad():
                    C_k = C_class[c][k,:] - self.A_tilde[k,k]*H_class[c][k,:]
                    Ck_class.append(C_k)
            if update_method == "weighted":
                self.node_list[k].weighted_local_update(self.A_tilde[k,:], Ck_class, dH_class, batch_size,
                         learning_rate, I, gradient)
            else:
                self.node_list[k].local_update(self.A_tilde[k,:], Ck_class, dH_class, batch_size,
                         learning_rate, I, gradient)
                
            
        self.mean_aggregation()
        
        
        train_loss, train_acc, val_loss, val_acc = self.eval_train_val()
        
        if (val_loss < self.best_valloss):
            self.best_valloss = val_loss
            self.best_cmodel = copy.deepcopy(self.cmodel)
        
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
        
        """
        """
        H_class = []
        
        for i in range(self.N):
            h_class = self.node_list[i].cmodel_collect(self.cmodel)
            for c in range(self.tnc):
                if (i == 0):
                    H_class.append([])
                H_class[c].append(h_class[c])
                if (i == self.N-1):
                    H_class[c] = torch.cat(H_class[c], dim=0)
                    
        C_class = []
        with torch.no_grad():
            for c in range(self.tnc):
                C_class.append(torch.matmul(self.A_tilde, H_class[c]))
        
        for k in range(self.N):
            Ck_class = []
            for c in range(self.tnc):
                with torch.no_grad():
                    C_k = C_class[c][k,:] - self.A_tilde[k,k]*H_class[c][k,:]
                    Ck_class.append(C_k)
            tloss, tacc = self.node_list[k].weighted_eval("train", self.cmodel, self.A_tilde[k,:], Ck_class)
            vloss, vacc = self.node_list[k].weighted_eval("valid", self.cmodel, self.A_tilde[k,:], Ck_class)
            avg_valloss += vloss
            avg_valacc += vacc
            avg_trainloss += tloss
            avg_trainacc += tacc
            
        avg_trainloss = avg_trainloss/self.N
        avg_trainacc = avg_trainacc/self.N
        avg_valloss = avg_valloss/self.N
        avg_valacc = avg_valacc/self.N
            
        
        
        return  avg_trainloss, avg_trainacc, avg_valloss, avg_valacc
    
    
    
    def eval_test(self):
        
        avg_testloss = 0
        avg_testacc = 0
        

        H_class = []
        
        for i in range(self.N):
            h_class = self.node_list[i].cmodel_collect(self.best_cmodel)
            for c in range(self.tnc):
                if (i == 0):
                    H_class.append([])
                H_class[c].append(h_class[c])
                if (i == self.N-1):
                    H_class[c] = torch.cat(H_class[c], dim=0)
                    
        C_class = []
        with torch.no_grad():
            for c in range(self.tnc):
                C_class.append(torch.matmul(self.A_tilde, H_class[c]))
        
        for k in range(self.N):
            Ck_class = []
            for c in range(self.tnc):
                with torch.no_grad():
                    C_k = C_class[c][k,:] - self.A_tilde[k,k]*H_class[c][k,:]
                    Ck_class.append(C_k)
            tloss, tacc = self.node_list[k].weighted_eval("test", self.best_cmodel, self.A_tilde[k,:], Ck_class)
            avg_testloss += tloss
            avg_testacc += tacc
            
        avg_testloss = avg_testloss/self.N
        avg_testacc = avg_testacc/self.N
            
        
        
        return  avg_testloss, avg_testacc