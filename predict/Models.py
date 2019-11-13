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
