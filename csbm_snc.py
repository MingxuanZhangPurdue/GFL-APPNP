import torch
import numpy as np
from utils.utils import *
from utils.train_helpers import train_NC
from models.models import MLP
from models.setup import set_up_NC
import networkx as nx
import pickle
 
import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from os.path import exists
import argparse
import os


parser = argparse.ArgumentParser(description='cSBM SNC experiments.')
parser.add_argument('--Print', default=False, type=bool, help='Whether to print training stats during training.')
parser.add_argument('--print_time', default=10, type=int, help='Print stat for each print_time communications.')

parser.add_argument('--batchsize', default=40, type=int, help='Batch size for local updates.')
parser.add_argument('--hn_std', default=0.1, type=float, help='Standard deviation for hidden noise.')
parser.add_argument('--gn_std', default=0.1, type=float, help='Standard deviation for gradient noise.')
parser.add_argument('--bias', action='store_true', default=False, help='Bias in MLP.')
parser.add_argument('--gradient', action='store_true', default=False, help='Share gradients of hidden representation.')
parser.add_argument('--hidden_noise', action='store_true', default=False, help='Add random noise to hidden representation.')
parser.add_argument('--gradient_noise', action='store_true', default=False, help='Add random noise to gradient.')


parser.add_argument('--lr', default=0.2, type=float, help='Learning rate.')
parser.add_argument('--I', default=10, type=int, help='Number of local updates.')
parser.add_argument('--nc', default=500, type=int, help='Number of communications.')


args = parser.parse_args()

if args.hidden_noise and args.gradient_noise:
    noise_type = "hg_noise_"+str(args.hn_std)+"_"+str(args.gn_std)+"/"
elif args.hidden_noise and not args.gradient_noise:
    noise_type = "h_noise_"+str(args.hn_std)+"/"
elif not args.hidden_noise and args.gradient_noise:
    noise_type = "g_noise_"+str(args.gn_std)+"/"
else:
    noise_type = "test/"


if not os.path.exists('experiments/SNC/0.78phi/result/GFLAPPNP/'+noise_type+"I"+str(args.I)+"/"):
    os.makedirs('experiments/SNC/0.78phi/result/GFLAPPNP/'+noise_type+"I"+str(args.I)+"/")
    
folder_path = 'experiments/SNC/0.78phi/result/GFLAPPNP/'+noise_type+"I"+str(args.I)+"/"

import sys
sys.path.append('utils/')
train_ids = np.load("experiments/SNC/0.78phi/data/train_ids.npy")
val_ids = np.load("experiments/SNC/0.78phi/data/val_ids.npy")
test_ids = np.load("experiments/SNC/0.78phi/data/test_ids.npy")


for i in range(0, 20):  
    
    file_to_open= open("experiments/SNC/0.78phi/data/"+"csbm_"+str(i)+".pickle", "rb")
    csbm = pickle.load(file_to_open)
    file_to_open.close()
    A_tilde = calculate_Atilde(csbm.A, M=10, alpha=0.1)
    
    torch.manual_seed(0)
    init_mlp = MLP(csbm.Xs[0].shape[1], 64, 2, bias=args.bias)
    
    print ("ith:", i)    

    server = set_up_NC(csbm.Xs, csbm.ys, init_mlp, A_tilde, 
                       train_ids, val_ids, test_ids,
                       args.gradient,
                       args.hidden_noise, args.gradient_noise,
                       args.hn_std, args.gn_std)

    tl, ta, vl, va = train_NC(server, args.nc, args.batchsize, args.lr, args.I, args.Print, args.print_time)

    np.save(folder_path + "/tl_" + str(i), tl)
    np.save(folder_path + "/ta_" + str(i), ta)
    np.save(folder_path + "/vl_" + str(i), vl)
    np.save(folder_path + "/va_" + str(i), va)

    PATH = folder_path + "/model_" + str(i)
    
    print (server.eval_test()[1], server.best_valacc)

    torch.save({
            'best_model_state_dict': server.best_cmodel.state_dict(),
            'learning_rate': args.lr,
            'test_acc': server.eval_test()[1],
            'model_state_dict': server.cmodel.state_dict(),
            'best_valloss': server.best_valloss,
            'best_valacc': server.best_valacc,
            }, PATH)



