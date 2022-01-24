import copy
import Frame_GC
import Frame_NC
import Frame_FedMLP


def set_up_FedMLP(Xs, ys, initial_model, train_ids, val_ids, test_ids):
    
    N = len(Xs)
    
    node_list = []
    
    for i in range(N):
            
        model_i = copy.deepcopy(initial_model)
        node_i = Frame_FedMLP.Node(local_model=model_i, node_idx=i, X=Xs[i], y=ys[i])
        node_list.append(node_i)
        
    server = Frame_FedMLP.Central_Server(node_list, train_ids, val_ids, test_ids)
    
    server.init_central_parameters(initial_model)
    
    return server


def set_up_GC(Xs, ys, bp, initial_model, A_tilde, ith=None, tnc=2, ntrain=30, nval=10):
    
    N = A_tilde.shape[0]
    
    node_list = []
    
    for i in range(N):
         
        if ith != None:
            random_seed = i+ith*1000
        else:
            random_seed = None
            
        model_i = copy.deepcopy(initial_model)
        
        node_i = Frame_GC.Node(local_model=model_i, node_idx=i, X=Xs[i], y=ys[i], 
                               total_n_class=tnc, n_train=ntrain, n_val=nval, bp_i=bp[i], rs=random_seed)
        
        node_list.append(node_i)
        
    server = Frame_GC.Central_Server(node_list, A_tilde, total_n_class=tnc)
    
    server.init_central_parameters(initial_model)
    
    return server

def set_up_NC(Xs, ys, initial_model, A_tilde, train_ids, val_ids, test_ids):
    
    N = A_tilde.shape[0]
    
    node_list = []
    
    for i in range(N):
        
        model_i = copy.deepcopy(initial_model)
        
        node_i = Frame_NC.Node(local_model=model_i, N=N, node_idx=i, X=Xs[i], y=ys[i])
        
        node_list.append(node_i)
        
    server = Frame_NC.Central_Server(node_list, A_tilde, train_ids, val_ids, test_ids)
    
    server.init_central_parameters(initial_model)
    
    return server