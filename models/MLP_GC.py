import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device, grad_device

class Node:
    
    def __init__(self, local_model, X, y, n_train, n_val):
        
        
        self.model = copy.deepcopy(local_model).to(device)
        self.n_local = X.shape[0]
        self.train_dataloader = None
        self.train_batchsize = 1
        
        ids = np.arange(self.n_local)
        train_ids = ids[0:n_train]
        val_ids = ids[n_train:n_train+n_val]
        test_ids = ids[n_train+n_val:]
        
        self.X, self.y = X.to(device), y.to(device)
        self.X_train, self.y_train = self.X[train_ids], self.y[train_ids]
        self.X_val, self.y_val = self.X[val_ids], self.y[val_ids]
        self.X_test, self.y_test = self.X[test_ids], self.y[test_ids]
        
        self.best_valloss = np.inf
        self.best_local_model = None
        self.best_valacc = 0
        self.optimizer = None
                
    
    def local_update(self, batch_size, learning_rate):
        
        if not (batch_size <= self.n_local):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.train_dataloader == None or self.train_batchsize != batch_size):
            
            self.train_batchsize = batch_size
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                                batch_size=self.train_batchsize, 
                                                                shuffle=True, drop_last=True)
        
        if self.optimizer == None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for X, y in self.train_dataloader: 

            self.optimizer.zero_grad()
            Z = self.model(X)
            y_hat = F.softmax(Z, dim=1)
            local_loss = F.nll_loss(torch.log(y_hat), y)
            local_loss.backward()
            self.optimizer.step()
            
            
    def local_eval(self, mode="train"):
        
        if mode == "train":
            X, y = self.X_train, self.y_train
            model = self.model
        elif mode == "val":
            X, y = self.X_val, self.y_val
            model = self.model
        elif mode == "test":
            X, y = self.X_test, self.y_test
            model = self.best_model
        else:
            raise ValueError("mode should be either train, val, or test!")
            
        with torch.no_grad():
            Z = model(X)
            y_hat = F.softmax(Z, dim=1)
            y_pred = torch.max(y_hat, dim=1)[1]
            acc = (y_pred == y).sum().item()/y.shape[0]
            loss = F.nll_loss(torch.log(y_hat), y)
            
            if mode == "val" and loss < self.best_valloss:
                self.best_valloss = loss.item()
                self.best_valacc = acc
                self.best_model = copy.deepcopy(self.model)
                
        return loss, acc
    
    
class Central_Server:
    
    def __init__(self, node_list):


        
        self.node_list = node_list
        self.N = len(node_list)
       
            
        
    def train_nodes(self, batch_size, learning_rate, num_epoch, Print=False):
          
        for epoch in range(num_epoch):
            
            for k in range(self.N):

                self.node_list[k].local_update(batch_size, learning_rate)

            train_loss, train_acc, val_loss, val_acc = self.eval_train_val()
            
            if (Print):
                print (f"epoch:", epoch, 
                        "Average train loss:", "{:.5f}".format(train_loss), "Average train accuracy:", "{:.3f}".format(train_acc),
                        "Average val loss:", "{:.5f}".format(val_loss), "Average val accuracy:", "{:.3f}".format(val_acc), flush=True)
 
 

    def eval_train_val(self):
        
        avg_trainloss = 0
        avg_trainacc = 0
        avg_valloss = 0
        avg_valacc = 0
            
        for k in range(self.N):
            tloss, tacc = self.node_list[k].local_eval(mode="train")
            vloss, vacc = self.node_list[k].local_eval(mode="val")
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
        
        for k in range(self.N):
            tloss, tacc = self.node_list[k].local_eval(mode="test")
            avg_testloss += tloss.item()
            avg_testacc += tacc
        avg_testloss = avg_testloss/self.N
        avg_testacc = avg_testacc/self.N
        
        
        return  avg_testloss, avg_testacc
        
            
            
            
            
        
        