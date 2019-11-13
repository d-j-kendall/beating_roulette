import torch.utils.data as data

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


class StateData(data.Dataset):

    def __init__(self, train_file):
        super(data.Dataset, self)
        self.tensor = None


