import copy
from models import GFLAPPNP_SC
from models import GFLAPPNP_NC
from models import FedMLP_SC
from models import MLP_SC


def set_up_MLP_SC(Xs, ys, initial_model, n_train, n_val):
    
    N = Xs.shape[0]
    
    node_list = []
    
    for i in range(N):
        
        model_i = copy.deepcopy(initial_model)
        
        node_i = MLP_SC.Node(local_model=model_i, X=Xs[i], y=ys[i], n_train=n_train, n_val=n_val)
        
        node_list.append(node_i)
        
    server = MLP_SC.Central_Server(node_list)
    
    return server


def set_up_FedMLP_SC(Xs, ys, initial_model, n_train, n_val):
    
    N = Xs.shape[0]
    
    node_list = []
    
    for i in range(N):
        
        model_i = copy.deepcopy(initial_model)
        
        node_i = FedMLP_SC.Node(local_model=model_i, X=Xs[i], y=ys[i], n_train=n_train, n_val=n_val)
        
        node_list.append(node_i)
        
    server = FedMLP_SC.Central_Server(initial_model, node_list)
    
    return server


def set_up_NC(Xs, ys, initial_model, A_tilde, 
              train_ids, val_ids, test_ids,
              gradient=True,
              hidden_noise=False, gradient_noise=False,
              hn_std=0.1, gn_std=0.01):
    
    N = A_tilde.shape[0]
    
    node_list = []
    
    for i in range(N):
        
        model_i = copy.deepcopy(initial_model)
        
        node_i = GFLAPPNP_NC.Node(local_model=model_i, N=N, node_idx=i, X=Xs[i], y=ys[i])
        
        node_list.append(node_i)
        
    server = GFLAPPNP_NC.Central_Server(initial_model, 
                                        node_list, A_tilde, 
                                        train_ids, val_ids, test_ids,
                                        gradient,
                                        hidden_noise, gradient_noise,
                                        hn_std, gn_std)
    
    return server


def set_up_SC(Xs, ys, initial_model, A_tilde, n_train, n_val, tnc=2):
    
    N = A_tilde.shape[0]
    
    node_list = []
    
    
    for i in range(N):

        node_i = GFLAPPNP_SC.Node(local_model=initial_model, 
                                  N=N, node_idx=i, 
                                  X=Xs[i], y=ys[i], 
                                  n_train=n_train, n_val=n_val)

        node_list.append(node_i)

    server = GFLAPPNP_SC.Central_Server(initial_model, node_list, A_tilde)
        
    
    return server