import anode
import torch
from torch.utils.data import dataloader
from predict.Data import StateData, StateData2
from predict.Models import AnodePrediction, ConvAnodePrediction, ReducedAnodePrediction

torch.set_default_tensor_type('torch.cuda.FloatTensor')
data_set = StateData2('../training_data.txt')

data_loader = dataloader.DataLoader(data_set, batch_size=120, shuffle=True, num_workers=0)

model = ReducedAnodePrediction('reduced.pd')

model.set_data_loader(data_loader)

model.train(300)




