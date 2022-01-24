from Frame_NC import *
from config import *
from utils import *
from generate_csbm import *
from models import *
from train import *
from setup import *

import torch 
import copy
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from os.path import exists

final_seeds = [101993, 124709, 196252,  95930,  68222, 101539,  22989,  45367,
       166831, 189085,  12237, 242044,  40182,  84405, 234468, 233451,
       154898,  81745,  70716,  39777, 248183, 109371, 112311, 229323,
         2160, 219137, 221729,  98972, 238056, 265088,  90081, 271232,
       260735,  96076, 121375,  11447]

final_cc_ids = [1, 0, 2, 3, 2, 1, 2, 1, 2, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 3, 0,
       0, 0, 1, 2, 1, 0, 1, 1, 0, 0, 3, 0, 0, 1]

def generate_subcora(rs, ith_cc, G, cora_data, Xs, ys, A, subgraph_size = 300):
    torch.manual_seed(rs)
    ids = torch.randperm(cora_data.y.shape[0]).numpy()
    sub_ids = ids[0:800]
    subcora = G.subgraph(ids[0:800])
    connected_components = sorted(nx.connected_components(subcora), key=len, reverse=True)
    train_nodes = np.array(list(connected_components[ith_cc]))
    all_reachable_nodes = []
    for train_node in train_nodes:
        for reachable_node in nx.bfs_tree(G,source=train_node, depth_limit=2):
            all_reachable_nodes.append(reachable_node)
    avail_nodes = np.array(list(set(all_reachable_nodes) - set(train_nodes)))
    num_train = len(train_nodes)
    num_val = num_train
    val_nodes = avail_nodes[:num_train]
    test_nodes = avail_nodes[num_train:subgraph_size-num_train]
    all_ids = np.concatenate([train_nodes,val_nodes,test_nodes], axis=0)
    
    #print (len(max(nx.connected_components(G.subgraph(all_ids)), key=len)))
    
    sub_Xs = Xs[all_ids]
    sub_ys = ys[all_ids]
    sub_A = A[all_ids,][:,all_ids]
    
    train_mask = np.arange(num_train)
    val_mask = np.arange(num_train, num_train+num_val)
    test_mask = np.arange(num_train+num_val,subgraph_size)
    
    X = sub_Xs.view(subgraph_size, -1)
    y = sub_ys.view(subgraph_size)

    edge_index = []
    N = sub_A.shape[0]
    for i in range(N):
        for j in range(N):
            if (i != j):
                if (sub_A[i,j] == 1):
                    edge_index.append([i,j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    subcora_data = Data(x=X, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    
    return subcora_data, sub_Xs, sub_ys, sub_A, calculate_Atilde(sub_A, 10, 0.9), train_mask, val_mask, test_mask


def rs_exp(method, train_ids, val_ids, test_ids,
           ith, folder_path, Xs, ys, A_tilde,
           num_communication=500, batch_size=1,
           learning_rate=0.01, I=10, gradient=True, noise=False,
           Print=False, resume=False,
           bias=True,
           save=True):
    

    torch.manual_seed(0)
    init_mlp = MLP(Xs[0].shape[1], 64, 7, bias)
    
    if method == "NC":
        m = "NC/"
        server = set_up_NC(Xs, ys, init_mlp, A_tilde, train_ids, val_ids, test_ids)
    else:
        ValueError("method should be NC/!")
        
    if (gradient == False):
        grad = "no_grad/"
    elif (gradient == True and noise==False):
        grad = "biased_grad/"
    elif (gradient == True and noise==True):
        grad = "noisy_grad/"
        
    if resume:
        checkpoint = torch.load(folder_path + m + grad + "I" + str(I) + "/model_" + str(ith))
        tl = np.load(folder_path + m + grad + "I" + str(I) + "/tl_" + str(ith)+".npy")
        ta = np.load(folder_path + m + grad + "I" + str(I) + "/ta_" + str(ith)+".npy")
        vl = np.load(folder_path + m + grad + "I" + str(I) + "/vl_" + str(ith)+".npy")
        va = np.load(folder_path + m + grad + "I" + str(I) + "/va_" + str(ith)+".npy")
    
        tl, ta, vl, va = train_NC(server, num_communication, batch_size, learning_rate, I,
                                  gradient, noise, 
                                  Print, 
                                  checkpoint, tl, ta, vl, va)
        
    else:
        tl, ta, vl, va = train_NC(server, num_communication, batch_size, learning_rate, I,
                                  gradient, noise, 
                                  Print)
    if save:    
        np.save(folder_path + m + grad + "I" + str(I) + "/tl_" + str(ith), tl)
        np.save(folder_path + m + grad + "I" + str(I) + "/ta_" + str(ith), ta)
        np.save(folder_path + m + grad + "I" + str(I) + "/vl_" + str(ith), vl)
        np.save(folder_path + m + grad + "I" + str(I) + "/va_" + str(ith), va)

        PATH = folder_path + m + grad + "I" + str(I) + "/model_" + str(ith)
        torch.save({
                'best_model_state_dict': server.best_cmodel.state_dict(),
                'learning_rate': learning_rate,
                'test_acc': server.eval_test()[1],
                'model_state_dict': server.cmodel.state_dict(),
                'best_valloss': server.best_valloss,
                'best_valacc': server.best_valacc,
                }, PATH)
        
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

dataset = Planetoid(root='/tmp/Cora', name='Cora')

cora_data = dataset[0]

G, Xs, ys, A = pygdata_to_frameformat(cora_data)


for ith in range(20):
    for I in [1]:
        subcora_data, sub_Xs, sub_ys, sub_A, sub_Atilde, train_ids, val_ids, test_ids = generate_subcora(final_seeds[ith],
                                                                                                 final_cc_ids[ith],
                                                                                                 G,
                                                                                                 cora_data, 
                                                                                                 Xs, ys, 
                                                                                                 A,
                                                                                                 subgraph_size=300)
        nc = int(4000/I)
        rs_exp("NC", train_ids, val_ids, test_ids,
               ith, folder_path="./cora/result/", 
               Xs=sub_Xs, ys=sub_ys, A_tilde=sub_Atilde,
               num_communication=nc, batch_size=1,
               learning_rate=0.02, I=I, gradient=True, noise=True,
               Print=False, resume=False, bias=False, save=True)
        print (I, ith, "done")