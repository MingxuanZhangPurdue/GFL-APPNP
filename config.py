import torch
device = "cpu"
grad_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')