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
from anode.conv_models import *
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
                          adjoint=False,
                          )

        self.optimizer = torch.optim.Adam(self.ode.parameters(), lr=1e-3)
        self.trained_params = trained_params
        self.trainer = Trainer(self.ode, self.optimizer, torch.device('cuda'))
        self.data_loader = None

        if len(trained_params)>0:
            try:
                self.ode = torch.load(self.trained_params)
            except FileNotFoundError:
                print('File Not Found')

    def __call__(self, *args, **kwargs):
        return self.ode(args[0])

    def train(self, epochs=10):
        self.trainer.train(self.data_loader, epochs)
        torch.save(self.ode, self.trained_params)

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

class ConvAnodePrediction:
    def __init__(self, trained_params=""):
        self.ode = ConvODENet(torch.device('cuda'), (1, 1, 8), 4, 1, 1, non_linearity='sigmoid')

        self.optimizer = torch.optim.Adam(self.ode.parameters(), lr=1e-3)
        self.trained_params = trained_params
        self.trainer = Trainer(self.ode, self.optimizer, torch.device('cuda'))
        self.data_loader = None

        if len(trained_params)>0:
            try:
                self.ode = torch.load(self.trained_params)
            except FileNotFoundError:
                print('File Not Found')

    def __call__(self, *args, **kwargs):
        return self.ode(args[0])

    def train(self, epochs=10):
        self.trainer.train(self.data_loader, epochs)
        torch.save(self.ode, self.trained_params)

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader


class CustomPrediction(nn.Module):

    def __init__(self, trained_params=''):
        super(CustomPrediction, self).__init__()

        self.conv11 = nn.Conv1d(1, 6, 3, 1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.trained_params = trained_params
        self.data_loader = None

        if len(trained_params) > 0:
            try:
                self.ode = torch.load(self.trained_params)
            except FileNotFoundError:
                print('File Not Found')

    def __call__(self, *args, **kwargs):
        return self.ode(args[0])

    def train(self, epochs=10):
        self.trainer.train(self.data_loader, epochs)
        torch.save(self.ode, self.trained_params)

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    def train(self, epochs=10):
        self.trainer.train(self.data_loader, epochs)
        torch.save(self.ode, self.trained_params)

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader


class ReducedAnodePrediction:

    def __init__(self, trained_params=""):
        self.ode = ODENet(device=torch.device('cuda'),
                          data_dim=4,
                          hidden_dim=10,
                          output_dim=1,
                          augment_dim=1,
                          non_linearity='softplus',
                          adjoint=False,

                          )
        self.optimizer = optim.Adam(self.ode.parameters(), lr=0.01)
        self.trained_params = trained_params
        self.trainer = Trainer(self.ode, self.optimizer, torch.device('cuda'))
        self.data_loader = None

        if len(trained_params)>0:
            try:
                self.ode = torch.load(self.trained_params)
            except FileNotFoundError:
                print('File Not Found')

    def __call__(self, *args, **kwargs):
        return self.ode(args[0])

    def train(self, epochs=10):
        self.trainer.train(self.data_loader, epochs)
        torch.save(self.ode, self.trained_params)

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

