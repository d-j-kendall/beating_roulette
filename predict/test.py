
import predict.model as model
import torch
import matplotlib.pyplot as plt
import numpy as np

r_model = model.RCModel(4, 16)
r_model.x_ball_fun.load_state_dict(torch.load('/home/dkendall/mnt/ext4hdd/sdesign_data/video_data/file'))
r_model.x_ball_fun.eval()

xmodel = r_model.x_ball_fun

with torch.no_grad():
    t_vect = np.arange(0, 3, 0.01, dtype=float)
    output = np.empty([len(t_vect), 1])
    for i, val in enumerate(t_vect):
        output[i] = xmodel.forward(torch.tensor([35.0, 900, 0.0, val])).numpy().data[0]

    plt.plot(t_vect, output)
