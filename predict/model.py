
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Predict(torch.nn.Module):
    def __init__(self):
        super(Predict, self).__init__()