import copy
import Frame_NC


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