import torch
import json
from anode.training import Trainer
import predict.model as pm
import torch.utils.data as data
import os

class XYData(data.Dataset):

    def __init__(self, file, axis, num_spins):
        super().__init__()
        self.file = open(file, 'r')
        self.axis = axis
        self.num_spins = num_spins
        self._len = 0
        self.data = []
    def __len__(self):
        if self._len == 0:
            self.load_data()
        return self._len

    def load_data(self):
        last_line = None
        curr_line = None
        for line in self.file:
            if last_line is None:
                last_line = json.loads(line)
            else:
                last_line = curr_line

class PVdataset(data.Dataset):

    def __init__(self, _dir, axis, cls):
        super().__init__()
        self.file_list = os.listdir(_dir)
        self.file_iter = iter(self.file_list)
        self._dir = _dir
        self.current_file = None
        self.set_next_file()
        self.axis = str(axis)
        self.cls = str(cls)
        self.line = None
        self.set_next_line()
        self._len = 0

    def __len__(self) -> int:
        return 100000

    def __getitem__(self, index: int):

        if self.cls == 'ball':
            inst = 0
        elif self.cls =='zero':
            inst = 1
        else:
            assert False, "Wrong class parameter"

        line, next_line = self.get_next_lines()

        detection_0 = None
        detection_1 = None

        while detection_0 is None and len(next_line) > 0:
            for det in line:
                if det['cls'] == inst:
                    detection_0 = det
            if detection_0 is None:
                line, next_line = self.get_next_lines()

        while detection_1 is None and len(next_line) > 0:
            for det in next_line:
                if det['cls'] == inst:
                    detection_1 = det
            if detection_1 is None:
                _, next_line = self.get_next_lines()

        if detection_0 and detection_1 is not None:
            axis_position = detection_0[str(self.axis)]
            axis_velocity = detection_0[str(self.axis+'v')]
            t0 = detection_0['t']
            t1 = detection_1['t']
            axis_pos_target = detection_1[str(self.axis)]

            return torch.tensor([[axis_velocity, axis_position, t0, t1]]), torch.tensor([axis_pos_target])
        else:
            self.set_next_file()

    def set_next_file(self):
        file = next(self.file_iter)
        while not file.endswith('.txt'):
            file = next(self.file_iter)
        self.current_file = open(os.path.join(self._dir, file), 'r')

    def set_next_line(self):
        self.line = self.current_file.readline()
        if len(self.line) <= 0:
            self.line = None
            self.set_next_file()

    def get_next_lines(self):
        if self.line is not None and len(self.line) > 0:
            line = json.loads(self.line)
            self.set_next_line()
        if self.line is not None and len(self.line) > 0:
            next_line = json.loads(self.line)
        else:
            self.set_next_file()
            self.set_next_line()
            return self.get_next_lines()
        return line, next_line

class XYSet(data.Dataset):
    def __init__(self, file, axis):
        super(XYSet , self).__init__()
        self.file = open(file, 'r')
        self.total_tensor = torch.tensor([[]], dtype=torch.float)
        self.instance_tensor = torch.Tensor()
        self.axis = axis
        self.device = torch.device('cuda')
        self.load_data()


    def load_data(self):

        for i, line in enumerate(self.file):
            detection = json.loads(line)
            instance_tensor = torch.tensor([[detection['x'], detection['y'], detection['xv'], detection['yv'], detection['t']]], dtype=torch.float)

            if i == 0:
                self.total_tensor = instance_tensor
            else:
                self.total_tensor = torch.cat((self.total_tensor, instance_tensor), 0)

    def __len__(self) -> int:
        return int(self.total_tensor.shape[0])

    def __getitem__(self, index: int):
        if self.axis == 'x':
            return torch.cat((self.total_tensor[0, [0, 2, 4]], self.total_tensor[index, [4]]), 0).to(device=self.device), self.total_tensor[index,[0,2]].to(device=self.device)

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = torch.device('cpu')
    else:
        dev = torch.device('cpu')

    roulette_model = pm.RCModel(4, 16)

    x_ball_optimizer = torch.optim.Adam(roulette_model.x_ball_fun.parameters(), lr=1e-3)
    dataset = XYSet('/home/dkendall/mnt/ext4hdd/sdesign_data/video_data/go_pro_detected_output2/ball-GOPR0037.MP4.txt', 'x')
    dataloader = data.DataLoader(dataset, 1752, True)

    trainer = Trainer(roulette_model.x_ball_fun, x_ball_optimizer, dev)
    trainer.train(dataloader, 200)
    torch.save(roulette_model.x_ball_fun.state_dict(), '/home/dkendall/mnt/ext4hdd/sdesign_data/video_data/file')
    # y_ball_optimizer = torch.optim.Adam(roulette_model.y_ball_fun.parameters(), lr=1e-3)
    # x_zero_optimizer = torch.optim.Adam(roulette_model.x_zero_fun.parameters(), lr=1e-3)
    # y_zero_optimizer = torch.optim.Adam(roulette_model.y_zero_fun.parameters(), lr=1e-3)





