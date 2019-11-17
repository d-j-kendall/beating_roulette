import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import matplotlib.pyplot as plt
import torchvision
import csv
import glob
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import glob
from anode.models import *
from anode.training import *
from predict.Data import *
import torch.utils.data as data

class Prediction(nn.Module):


    def __init__(self):
        super(Prediction,self).__init__()

        self.linear = nn.Linear(8, 30, True)
        self.linear2 = nn.Linear(30, 30, True)
        self.linear3 = nn.Linear(30, 1, True)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = nn.LogSigmoid()(x)
        return x


class AnodePrediction:

    def __init__(self, trained_params=""):
        self.ode = ODENet(device=torch.device('cuda'),
                          data_dim=8,
                          hidden_dim=10,
                          output_dim=1,
                          augment_dim=1,
                          non_linearity='relu',
                          adjoint=True)

        self.optimizer = torch.optim.Adam(self.ode.parameters(), lr=1e-3)
        self.trained_params = trained_params
        self.trainer = Trainer(self.ode,self.optimizer, torch.device('cuda'))
        self.data_loader = None

        if len(trained_params)>0:
            self.ode.load_state_dict(torch.load(self.trained_params))

    def train(self, epochs=10):
        self.trainer.train(self.data_loader, epochs)
        torch.save(self.ode, self.trained_params)

    def set_data_loader(self, training_set):
        self.data_loader = data.DataLoader(StateData(training_set), batch_size=20, shuffle=True, num_workers=20)






