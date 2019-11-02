import torch
from anode.conv_models import ConvODENet
from anode.models import *
from anode.discrete_models import *
from anode.training import *

class RCModel:
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 ):
        self.x_ball_fun = ODENet(torch.device('cuda'), data_dim=in_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=1,
                                 time_dependent=False)

        self.y_ball_fun = ODENet(torch.device('cuda'), data_dim=in_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=1,
                                 time_dependent=False)

        self.x_zero_fun = ODENet(torch.device('cuda'),
                                 data_dim=in_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=1,
                                 time_dependent=False)

        self.y_zero_fun = ODENet(torch.device('cuda'),
                                 data_dim=in_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=1,
                                 time_dependent=False)

    def get_models(self):
        return self.x_ball_fun, self.y_ball_fun, self.x_zero_fun, self.y_zero_fun

