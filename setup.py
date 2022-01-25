import copy
import GFLAPPNP_GC
import GFLAPPNP_NC


def set_up_NC(Xs, ys, initial_model, A_tilde, train_ids, val_ids, test_ids):
    
    N = A_tilde.shape[0]
    
    node_list = []
    
    for i in range(N):
        
        model_i = copy.deepcopy(initial_model)
        
        node_i = GFLAPPNP_NC.Node(local_model=model_i, N=N, node_idx=i, X=Xs[i], y=ys[i])
        
        node_list.append(node_i)
        
    server = GFLAPPNP_NC.Central_Server(node_list, A_tilde, train_ids, val_ids, test_ids)
    
    server.init_central_parameters(initial_model)
    
    return server


def set_up_GC(Xs, ys, initial_model, A_tilde, n_train, n_val):
    
    N = A_tilde.shape[0]
    
    node_list = []
    
    for i in range(N):
        
        node_i = GFLAPPNP_GC.Node(local_model=initial_model, 
                                  N=N, node_idx=i, 
                                  X=Xs[i], y=ys[i], 
                                  n_train=n_train, n_val=n_val)
        
        node_list.append(node_i)
        
    server = GFLAPPNP_GC.Central_Server(initial_model, node_list, A_tilde)
    
    return server