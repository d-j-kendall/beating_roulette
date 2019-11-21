import anode
import torch
from torch.utils.data import dataloader
from predict.Data import StateData
from predict.Models import AnodePrediction

torch.set_default_tensor_type('torch.cuda.FloatTensor')
data_set = StateData('../training_data2.txt')

data_loader = dataloader.DataLoader(data_set, batch_size=10, shuffle=True, num_workers=0)

model = AnodePrediction('first_run.pd')

model.set_data_loader(data_loader)

model.train(100)




