# Import necessary libraries
import torch
import numpy as np
from utils.generate_csbm import *
from utils.train_helpers import *
from utils.utils import *
from models.models import *
from models.setup import *



"""
Deterministic Node Classification:
Here we give an example on how to train our method for deterministic node classification task using synthetic graphs.
"""

"""
Refer to appendix of our paper for a breif recap for csbm and the meaning of corresponding hyperparameters. Here "l" represents the \lambda hyperparmeter.
"""

# Setup graph structure, i.e. adjacency matrix and node labels
synthetic_graph = cSBM(N=100, p=100, d=10, mu=1, l=2)

# Setup parameters for data generation, for example, the gaussian vector "u" is drawn by this function.
synthetic_graph.generate_node_parameters()

# Generate data for each node, since we are considering deterministic node classification tasks, we specify the parameter "method” to be DNC.
# And set n_local to 1.
synthetic_graph.generate_node_data(method="DNC", n_local=1)

# Here for simplicity, we use the first 5 nodes as train set, next 5 nodes as valid set, and the rest as test set. According to our theory,
# you want the subgraph induced by the train set to be connected, however our method works in practice for non-connected train set. And, due to the nature of GNN, you may also want to achieve a good class balance for train set.
train_ids = np.arange(10)
val_ids = np.arange(10,20)
test_ids = np.arange(20,100)

# We need to get the A_tilde matrix as the graph aggerator for APPNP, refer to appendix for details. Here A represents the adjacency matrix.
A_tilde = calculate_Atilde(A = synthetic_graph.A, M=10, alpha=0.1)

# A base MLP model is needed for APPNP (set bias to False is required).
init_mlp = MLP(input_dim=100, hidden_dim=64, output_dim=2, bias=False)




# Set up the "server" system.
# To do so, you need an initial MLP model, A_tilde Matrix, test, val, train indices, and two tensors:
# Xs: a 3d Float tensor for graph feature vectors with dimensions: [number_of_nodes, number_of_local_data, feeature_vector_dimension]
# ys: a Long tensor for node labels, with dimensions: [number_of_nodes, number_of_local_data]
# Note that, for this sample deterministic node classification task, the dimension for Xs is [50, 1, 100]
# For ys, it is [50, 1]
# You can easily get these two tensors as follow,
# Note that, to use other graph data, just provide data based on the following format to set up a server.
server_dnc = set_up_NC(Xs = synthetic_graph.Xs, ys=synthetic_graph.ys, 
                       initial_model = init_mlp, 
                       A_tilde=A_tilde, 
                       train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                       gradient=True, 
                       gradient_noise=True, hidden_noise=True,
                       gn_std=0.01, hn_std=0.01)




# Once the sever is setup, you can train our model as following:
# We use the train function provided in train_helper.py.
# You need to specify:
"""
num_communications: number of communications you want to run.

batch size: Use for mini-batch training for local updates. Note that, for deterministic node classification task, 
            this parameter has to be 1, since each node has only one data point.

learning_rate: learning rate for SGD

I: Number of local updates before next communcation. Total number of updates =  num_communications * I.

gradient: Whether to upload hidden representaion gradient to central server, if set False, only hidden representation will be uploaded.

noise: Whether to upload noisy hidden gradient to central sever, only vaild, if gradient is set to True.


Print: Whether to print out average tain, valid loss and accuacy for each communication.

print_time: If Print is set to True, only print when at communication round t, such that (t mod print_time) = 0.
"""
# This function will return train loss, train accuracy, val loss, val accuracy calculated at each communication round as list.
tl_dnc, ta_dnc, vl_dnc, va_dnc = train_NC(server=server_dnc, num_communication=200, 
                                          batch_size=1, learning_rate=0.2, I=10,
                                          Print=True, print_time=10)


# After training is completed, call this function to evaluate test loss and accuracy using the model with lowest valid loss.
test_loss_dnc, test_accuracy_dnc = server_dnc.eval_test()
print ("For example on stochastic node classification, the test accuracy is", test_accuracy_dnc)



'''
"""
Stochastic Node Classification:
Here we give an example on how to train our method for stochastic node classification task using synthetic graphs.

We will not write explaination for steps that are the same as the one we explained already
"""


synthetic_graph = cSBM(N=50, p=100, d=10, mu=1, l=2)


synthetic_graph.generate_node_parameters()

# Generate data for each node, since we are considering stochastic node classification tasks, we specify the parameter "method” to be SNC.
# And set n_local to some number that can be larger than 1.
synthetic_graph.generate_node_data(method="SNC", n_local=20)


train_ids = np.arange(5)
val_ids = np.arange(5,10)
test_ids = np.arange(10,40)


A_tilde = calculate_Atilde(A = synthetic_graph.A, K=10, alpha=0.1)


init_mlp = MLP(input_dim=100, hidden_dim=64, output_dim=2, bias=False)


# Xs: [50, 20, 100]
# ys: [50, 20]
server_snc = set_up_NC(Xs = synthetic_graph.Xs, ys=synthetic_graph.ys, 
                       initial_model = init_mlp, 
                       A_tilde=A_tilde, 
                       train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)




# Once the sever is setup, you can train our model as following:
# We use the train function provided in train_helper.py.
# You need to specify:
# Now you can set a batch size that is larget than 1

tl_snc, ta_snc, vl_snc, va_snc = train_NC(server=server_snc, num_communication=10, 
                                          batch_size=20, learning_rate=0.2, I=10,
                                          gradient=True, noise=False, 
                                          Print=True, print_time=1)


test_loss_snc, test_accuracy_snc = server_snc.eval_test()
print ("For example on stochastic node classification, the test accuracy is", test_accuracy_snc)




"""
Supervised Classification:
Here we give an example on how to train our method for supervised classification task using synthetic graphs.

We will not write explaination for steps that are the same as the one we explained already
"""


synthetic_graph = cSBM(N=40, p=100, d=8, mu=1, l=2)

synthetic_graph.generate_node_parameters()

# Generate data for each node, since we are considering stochastic node classification tasks, we specify the parameter "method” to be SC.
synthetic_graph.generate_node_data(method="SC", n_local=120)

# Note that, under supervised classification task, we use all nodes in the graph, instead we split the local data into train, valid, test set. Here for simplicity, we use 10 as train, 10 as valid, 100 as test for each node.
# And we want to the entire graph as connected (Set a higher d can achieve this easily), again in practice, this does not have to be true.
num_train = 10
num_valid = 10
# we do not need num_test, since num_test = 120 - num_train - num_valid = 100


A_tilde = calculate_Atilde(A = synthetic_graph.A, K=10, alpha=0.1)


init_mlp = MLP(input_dim=100, hidden_dim=64, output_dim=2, bias=False)



# Xs: [30, 120, 100]
# ys: [30, 120]
# An extra parameter tnc which stands for total number of classes, in our case it is 2
# Note that, this set up function will use the first n_train of local data as train set, next n_val sa val set, the rest as test set
# If you want a specific set as train set, arrange your Xs such that the first n_train is the set you want to use as trainn set.
server_sc = set_up_SC(Xs = synthetic_graph.Xs, 
                      ys = synthetic_graph.ys, 
                      initial_model = init_mlp, 
                      A_tilde = A_tilde, 
                      n_train=num_train, n_val=num_valid, 
                      tnc=2)




# Use a baatch size of 5.
tl_sc, ta_sc, vl_sc, va_sc = train_SC(server=server_sc, num_communication=10, 
                                      batch_size=5, learning_rate=0.2, I=10,
                                      gradient=True, noise=False, 
                                      Print=True, print_time=1)


test_loss_sc, test_accuracy_sc = server_sc.eval_test()
print ("For example on supervised classification, the test accuracy is", test_accuracy_sc)


# Thanks for using our code!
'''
















