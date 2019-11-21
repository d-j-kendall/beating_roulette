import os
from pv2sv import StateVector as sv
import argparse

def generate_training_file(_dir, file_out):

    for file in os.listdir(_dir):
        if file.endswith('.txt'):
            f_data = file.split('.')[0]
            f_data = f_data.split('_')
            result = int(f_data[-1])
            y_center = int(f_data[-2])
            x_center = int(f_data[-3])
            fps = f_data[-4]
            fps = int(fps[:-3])
            sv.pv2sv(os.path.join(_dir, file), file_out, x_center, y_center, result, 1/fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory to get training data from')
    parser.add_argument('--out', type=str, help='Output file to put data in')

    opts = parser.parse_args()

    generate_training_file(opts.dir, opts.out)
