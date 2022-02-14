import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import device, grad_device
import torch.utils.data



class Node:
    
    def __init__(self, local_model, X, y):
        
        self.model = local_model.to(device)
        self.X = X
        self.y = y
        self.n_local = self.X.shape[0]
        self.data_loader = None
    
    def receieve_central_parameters(self, cmodel):
        
        with torch.no_grad():
            for pname, param in self.model.named_parameters():
                param.copy_(cmodel.state_dict()[pname])
                
                

    def local_update(self, batch_size, learning_rate, I):
        
        if not (batch_size <= self.n_local):
            raise ValueError("batch size should be less or equal to the number of local data points")
        
        if (self.data_loader == None):
            dataset = torch.utils.data.TensorDataset(self.X, self.y)
            self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        for ith_update in range(I):
            
            for X, y in self.data_loader: 
                
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                Z = self.model(X)
                y_hat = F.softmax(Z, dim=1)
                local_loss = F.nll_loss(torch.log(y_hat), y)
                local_loss.backward()
                optimizer.step()
            
            
    def cmodel_eval(self, cmodel):
        with torch.no_grad():
            X, y = self.X.to(device), self.y.to(device)
            Z = cmodel(X)
            y_hat = F.softmax(Z, dim=1)
            y_pred = torch.max(y_hat, dim=1)[1]
            acc = (y_pred == y).sum().item()/y.shape[0]
            loss = F.nll_loss(torch.log(y_hat), y)
        return loss, acc
            
            
            
class Central_Server:
    
    def __init__(self, node_list, train_indices, valid_indices, test_indices):


        self.node_list = node_list
        self.N = len(node_list)
        self.cmodel = None
        
        self.train_ids = train_indices
        self.val_ids = valid_indices
        self.test_ids = test_indices
        
        self.best_cmodel = None
        self.best_valloss = np.inf
        self.best_valacc = 0
        
    def init_central_parameters(self, model):
        
        self.cmodel = copy.deepcopy(model)
        self.cmodel = self.cmodel.to(device)
        
        
    def broadcast_central_parameters(self):
        
        if self.cmodel == None:
            raise ValueError("Central model is None, Please initilalize it first.")
        
        for node in self.node_list:
            node.receieve_central_parameters(self.cmodel)
            
        
    def communication(self, batch_size, learning_rate, I):
          
        self.broadcast_central_parameters()
        
        for k in self.train_ids:
            self.node_list[k].local_update(batch_size, learning_rate, I)
        
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
            
        for k in self.train_ids:
            tloss, tacc = self.node_list[k].cmodel_eval(self.cmodel)
            avg_trainloss += tloss.item()
            avg_trainacc += tacc
        avg_trainloss = avg_trainloss/len(self.train_ids)
        avg_trainacc = avg_trainacc/len(self.train_ids)
        

        for k in self.val_ids:
            vloss, vacc = self.node_list[k].cmodel_eval(self.cmodel)
            avg_valloss += vloss.item()
            avg_valacc += vacc
        avg_valloss = avg_valloss/len(self.val_ids)
        avg_valacc = avg_valacc/len(self.val_ids)
        
        return  avg_trainloss, avg_trainacc, avg_valloss, avg_valacc
    
    
    
    def eval_test(self):
        
        avg_testloss = 0
        avg_testacc = 0
        
        for k in self.test_ids:
            tloss, tacc = self.node_list[k].cmodel_eval(self.best_cmodel)
            avg_testloss += tloss.item()
            avg_testacc += tacc
        avg_testloss = avg_testloss/len(self.test_ids)
        avg_testacc = avg_testacc/len(self.test_ids)
        
        return  avg_testloss, avg_testacc
        
            
            
            
            
        
        