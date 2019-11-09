import os
from pv2sv import StateVector as sv

def generate_training_file(_dir, file_out):

    for file in os.listdir():
        if file.endswith('.txt'):
            f_data = file.split('.')[0]
            f_data = file.split('_')
            result = int(f_data[-1])
            y_center = int(f_data[-2])
            x_center = int(f_data[-3])
            sv.pv2sv(file, file_out, x_center, y_center, result)