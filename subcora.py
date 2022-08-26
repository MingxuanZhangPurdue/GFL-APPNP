import torch
import numpy as np
from utils.utils import *
from utils.train_helpers import *
from models.models import *
from models.setup import *
import networkx as nx
import pickle
 
import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from os.path import exists
import argparse
import os

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


parser = argparse.ArgumentParser(description='SubCora experiments.')
parser.add_argument('--seed_idx', default=0, type=int, help='Which seed to use to generate subCora data.')
parser.add_argument('--Print', default=True, type=bool, help='Whether to print training stats during training.')
parser.add_argument('--print_time', default=20, type=int, help='Print stat for each print_time communications.')

parser.add_argument('--gradient', default=True, type=bool, help='Share gradient of hidden representation.')
parser.add_argument('--hidden_noise', default=True, type=bool, help='Add random noise to hidden representation.')
parser.add_argument('--gradient_noise', default=False, type=bool, help='Add random noise to gradient.')
parser.add_argument('--hn_std', default=0.1, type=float, help='Standard deviation for hidden noise.')
parser.add_argument('--gn_std', default=0.01, type=float, help='Standard deviation for gradient noise.')
parser.add_argument('--bias', default=False, type=bool, help='Bias in MLP.')


parser.add_argument('--lr', default=0.01, type=float, help='Learning rate.')
parser.add_argument('--I', default=10, type=int, help='Number of local updates.')
parser.add_argument('--nc', default=300, type=int, help='Number of communications.')


args = parser.parse_args()



final_seeds = [101993, 124709, 196252,  95930,  68222, 101539,  22989,  45367,
       166831, 189085,  12237, 242044,  40182,  84405, 234468, 233451,
       154898,  81745,  70716,  39777, 248183, 109371, 112311, 229323,
         2160, 219137, 221729,  98972, 238056, 265088,  90081, 271232,
       260735,  96076, 121375,  11447]
final_cc_ids = [1, 0, 2, 3, 2, 1, 2, 1, 2, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 3, 0,
       0, 0, 1, 2, 1, 0, 1, 1, 0, 0, 3, 0, 0, 1]

dataset = Planetoid(root='/tmp/Cora', name='Cora')
cora_data = dataset[0]
G, Xs, ys, A = pygdata_to_frameformat(cora_data)

def generate_subcora(rs, ith_cc, 
                     G, cora_data, Xs, ys, A, 
                     subgraph_size = 300):
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


if args.hidden_noise and args.gradient_noise:
    noise_type = "hg_noise_"+str(args.hn_std)+"_"+str(args.gn_std)+"/"
elif args.hidden_noise and not args.gradient_noise:
    noise_type = "h_noise_"+str(args.hn_std)+"/"
elif not args.hidden_noise and args.gradient_noise:
    noise_type = "g_noise_"+str(args.gn_std)+"/"
else:
    raise ValueError("Please specify noise type!")


if not os.path.exists('experiments/cora/result/GFLAPPNP/'+noise_type+"I"+str(args.I)+"/"):
    os.makedirs('experiments/cora/result/GFLAPPNP/'+noise_type+"I"+str(args.I)+"/")
    
folder_path = 'experiments/cora/result/GFLAPPNP/'+noise_type+"I"+str(args.I)+"/"

subcora_data, sub_Xs, sub_ys, sub_A, sub_Atilde, train_ids, val_ids, test_ids = generate_subcora(final_seeds[args.seed_idx],
                                                                                                 final_cc_ids[args.seed_idx],
                                                                                                 G,
                                                                                                 cora_data, 
                                                                                                 Xs, ys, 
                                                                                                 A,
                                                                                                 subgraph_size=300)
torch.manual_seed(0)
init_mlp = MLP(sub_Xs[0].shape[1], 64, 7, bias=args.bias)

server = set_up_NC(sub_Xs, sub_ys, init_mlp, sub_Atilde, 
                   train_ids, val_ids, test_ids,
                   args.gradient,
                   args.hidden_noise, args.gradient_noise,
                   args.hn_std, args.gn_std)

tl, ta, vl, va = train_NC(server, args.nc, 1, args.lr, args.I, args.Print, args.print_time)
   
np.save(folder_path + "/tl_" + str(args.seed_idx), tl)
np.save(folder_path + "/ta_" + str(args.seed_idx), ta)
np.save(folder_path + "/vl_" + str(args.seed_idx), vl)
np.save(folder_path + "/va_" + str(args.seed_idx), va)

PATH = folder_path + "/model_" + str(args.seed_idx)

torch.save({
        'best_model_state_dict': server.best_cmodel.state_dict(),
        'learning_rate': args.lr,
        'test_acc': server.eval_test()[1],
        'model_state_dict': server.cmodel.state_dict(),
        'best_valloss': server.best_valloss,
        'best_valacc': server.best_valacc,
        }, PATH)



