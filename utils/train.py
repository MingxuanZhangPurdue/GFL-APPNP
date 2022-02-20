import torch
import numpy as np
import copy
import torch.optim as optim
import torch.nn.functional as F


def train_FedMLP(server, num_communication, 
                 batch_size, learning_rate=0.1, I=10, 
                 Print=False, 
                 checkpoint=None, tl=None, ta=None, vl=None, va=None):
    
    if checkpoint != None:
        server.cmodel.load_state_dict(checkpoint["model_state_dict"])
        server.best_cmodel.load_state_dict(checkpoint["best_model_state_dict"])
        server.best_valloss = checkpoint["best_valloss"]
        server.best_valacc = checkpoint["best_valacc"]
        train_losses, val_losses = tl.tolist(), vl.tolist()
        train_accs, val_accs = ta.tolist(), va.tolist()
        pre_n_communication = len(ta)
    
    else:
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        pre_n_communication = 0
    
    for ith in range(num_communication):
        
        train_loss, train_acc, val_loss, val_acc = server.communication(batch_size, learning_rate, I)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (Print):
            print (f"Communication:", ith+1+pre_n_communication, 
                    "Average train loss:", "{:.5f}".format(train_loss), "Average train accuracy:", "{:.3f}".format(train_acc),
                    "Average val loss:", "{:.5f}".format(val_loss), "Average val accuracy:", "{:.3f}".format(val_acc), flush=True)
            
    return train_losses, train_accs, val_losses, val_accs


def train_GC(server, num_communication, 
             batch_size, learning_rate=0.1, I=10,
             gradient=True, noise=False, 
             Print=False, print_time=1,
             checkpoint=None, tl=None, ta=None, vl=None, va=None):
    
    
    if checkpoint != None:
        server.cmodel.load_state_dict(checkpoint["model_state_dict"])
        server.best_cmodel.load_state_dict(checkpoint["best_model_state_dict"])
        server.best_valloss = checkpoint["best_valloss"]
        server.best_valacc = checkpoint["best_valacc"]
        train_losses, val_losses = tl.tolist(), vl.tolist()
        train_accs, val_accs = ta.tolist(), va.tolist()
        pre_n_communication = len(ta)
    
    else:
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        pre_n_communication = 0
    
    for ith in range(num_communication):
        
        train_loss, train_acc, val_loss, val_acc = server.communication(batch_size, learning_rate, I, gradient, noise)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (Print and ith % print_time == 0):
            print (f"Communication:", ith+1+pre_n_communication, 
                    "Average train loss:", "{:.5f}".format(train_loss), "Average train accuracy:", "{:.3f}".format(train_acc),
                    "Average val loss:", "{:.5f}".format(val_loss), "Average val accuracy:", "{:.3f}".format(val_acc), flush=True)
            
    return train_losses, train_accs, val_losses, val_accs

def train_NC(server, num_communication, 
             batch_size, learning_rate=0.1, I=10,
             gradient=True, noise=False, 
             Print=False, print_time=1,
             checkpoint=None, tl=None, ta=None, vl=None, va=None):
    
    if checkpoint != None:
        server.cmodel.load_state_dict(checkpoint["model_state_dict"])
        server.best_cmodel.load_state_dict(checkpoint["best_model_state_dict"])
        server.best_valloss = checkpoint["best_valloss"]
        server.best_valacc = checkpoint["best_valacc"]
        train_losses, val_losses = tl.tolist(), vl.tolist()
        train_accs, val_accs = ta.tolist(), va.tolist()
        pre_n_communication = len(ta)
    
    else:
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        pre_n_communication = 0
    
    for ith in range(num_communication):
        
        train_loss, train_acc, val_loss, val_acc = server.communication(batch_size, learning_rate, I, gradient, noise)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (Print and ith % print_time==0):
            print (f"Communication:", ith+1+pre_n_communication, 
                    "Average train loss:", "{:.5f}".format(train_loss), "Average train accuracy:", "{:.3f}".format(train_acc),
                    "Average val loss:", "{:.5f}".format(val_loss), "Average val accuracy:", "{:.3f}".format(val_acc), flush=True)
            
    return train_losses, train_accs, val_losses, val_accs


def train_pyg_model(pyg_data, 
                    model, 
                    num_epoch, optimizer, 
                    mask=True, Print=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = pyg_data.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_model = None
    best_valloss = np.inf
    best_valacc = 0
    num_train = data.train_mask.sum() if mask else len(data.train_mask)
    num_val = data.val_mask.sum() if mask else len(data.val_mask)
    num_test = data.test_mask.sum() if mask else len(data.test_mask)
    
    model.train()
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            out = model(data)
            train_losses.append(F.nll_loss(out[data.train_mask], data.y[data.train_mask]).item())
            val_losses.append(F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item())
            
            pred = out.argmax(dim=1)
            train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
            train_acc = int(train_correct) / int(num_train)
            
            val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            val_acc = int(val_correct) / int(num_val)
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
           
                          
            """
            if val_acc > best_valacc:
                best_valloss = val_losses[epoch]
                best_model = copy.deepcopy(model)
                best_valacc = val_acc
                
            elif val_acc == best_valacc:
            """
            if val_losses[epoch] < best_valloss:
                    best_valloss = val_losses[epoch]
                    best_model = copy.deepcopy(model)
                    best_valacc = val_acc
              
            if (Print):
                print (f"Epoch:", epoch+1, 
                        "Average train loss:", "{:.5f}".format(train_losses[epoch]), "Average train accuracy:", "{:.3f}".format(train_acc),
                        "Average val loss:", "{:.5f}".format(val_losses[epoch]), "Average val accuracy:", "{:.3f}".format(val_acc), flush=True)
            model.train()
            
            """
            if (epoch >= 100 and epoch >= num_epoch/2):
                if val_losses[epoch] >= np.mean(val_losses[epoch-100:epoch]):
                    with torch.no_grad():
                        best_model.eval()
                        out = best_model(data)
                        pred = out.argmax(dim=1)
                        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                        test_acc = int(test_correct) / int(num_test)
                    return train_losses, train_accs, val_losses, val_accs, test_acc, best_model
            """
   

    with torch.no_grad():
        best_model.eval()
        out = best_model(data)
        pred = out.argmax(dim=1)
        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(test_correct) / int(num_test)
    return train_losses, train_accs, val_losses, val_accs, test_acc, best_model
                
            


def train_APPNP(X, y, init_mlp, A_tilde, 
               train_ids, val_ids, test_ids,
               num_update=1000,
               learning_rate=0.1, Print=False):
    
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    test_acc = 0
    
    A_tilde = A_tilde.to(device)
    model = copy.deepcopy(init_mlp).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    X, y = X.to(device), y.to(device)
    
    best_model = None
    best_valloss = np.inf
    best_valacc = 0
    
    num_train = len(train_ids)
    num_val = len(val_ids)
    num_test = len(test_ids)
        
    for update in range(num_update):
        
        optimizer.zero_grad()
        H = model(X)
        Z = torch.matmul(A_tilde, H)
        train_Z = Z[train_ids]
        train_y = y[train_ids]
        train_loss = F.cross_entropy(train_Z, train_y, reduction="mean")
        train_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            H = model(X)
            Z = torch.matmul(A_tilde, H)
            
            train_loss = F.cross_entropy(Z[train_ids], y[train_ids], reduction="mean").item()
            val_loss = F.cross_entropy(Z[val_ids], y[val_ids], reduction="mean").item()
            
            pred = Z.argmax(dim=1)
            train_acc = int((pred[train_ids] == y[train_ids]).sum())/num_train
            val_acc = int((pred[val_ids] == y[val_ids]).sum())/num_val
            
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_valloss:
                    best_valloss = val_loss
                    best_model = copy.deepcopy(model)
                    best_valacc = val_acc
              
            if (Print):
                print (f"update:", update+1, 
                        "Average train loss:", "{:.5f}".format(train_loss), "Average train accuracy:", "{:.3f}".format(train_acc),
                        "Average val loss:", "{:.5f}".format(val_loss), "Average val accuracy:", "{:.3f}".format(val_acc), flush=True)
            model.train()
            
            
    with torch.no_grad():
        best_model.eval()
        H = best_model(X)
        Z = torch.matmul(A_tilde, H)
        pred = Z.argmax(dim=1)
        test_acc = (pred[test_ids] == y[test_ids]).sum()/num_test
        
    return train_losses, train_accs, val_losses, val_accs, test_acc, best_model
            
            
            
            
            
            
            
            
            
            
            