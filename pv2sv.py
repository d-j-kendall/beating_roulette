import numpy as np
import json
import os
import argparse



class StateVector:

    def __init__(self, pred_model):
        self.pred_model  # Predictive model network

    @staticmethod
    def extract_parametric_initial_conditions(_dir, dir_out, time_step):
        file_list = os.listdir(_dir)
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        for file in file_list:
            if file.endswith('.txt'):
                spin = []
                with open(os.path.join(_dir, file), 'r') as f:
                    last_ball, last_zero = None, None
                    i = 0
                    det = f.readline()
                    while len(det) > 0:
                        frame = []
                        ball, zero = None, None
                        ball = None
                        zero = None
                        if len(det) > 0:
                            json_detection = json.loads(det)

                        if len(json_detection) > 0:
                            for inst in json_detection:
                                if inst["cls"] == 0:
                                    ball = inst
                                elif inst["cls"] == 1:
                                    zero = inst
                                else:
                                    assert False, "Malformed class type in json array, class should be 0 or 1 only"

                        if last_ball and ball:
                            ball["xv"] = ball['x'] - last_ball['x']
                            ball["yv"] = ball['y'] - last_ball['y']
                            ball['t'] = i*time_step
                            frame.append(ball)

                        if last_zero and zero:
                            zero["xv"] = zero['x'] - last_zero['x']
                            zero["yv"] = zero['y'] - last_zero['y']
                            zero['t'] = i*time_step
                            frame.append(zero)

                        last_ball = ball
                        last_zero = zero
                        if len(frame) > 0:
                            spin.append(frame)
                        i=i+1
                        det = f.readline()

                with open(os.path.join(dir_out, file), 'w') as new_f:
                    for frame in spin:
                        new_f.write(json.dumps(frame)+'\n')

            else:
                pass


    @staticmethod
    def extract_polar_state(filein, fileout, x_center, y_center, DT):
        last_ball, last_zero = None, None
        b, z, i = 0, 0, 0
        new_spin = False
        spin = []
        pos_file = open(filein, 'r')
        det = pos_file.readline()
        while det is not None:
            ball_flag = False
            zero_flag = False
            frame_state_vector = []
            if len(det) > 0:
                json_detection = json.loads(det)
            else:
                print("File finished")
                break
            if len(json_detection) > 0:
                for inst in json_detection:
                    if inst["cls"] == 0:
                        ball = inst
                        ball_flag = True
                    elif inst["cls"] == 1:
                        zero = inst
                        zero_flag = True
                    else:
                        assert False, "Malformed class type in json array, class should be 0 or 1 only"
                i = i + 1  # increment total count

                #  Calculate ball speeds and accelerations
                if last_ball is not None and ball_flag:
                    ball = StateVector.to_polar(ball, x_center, y_center)
                    ball['w'] = (StateVector.rad_dist(ball['theta'], last_ball['theta']) / (DT*(i - b)))
                    if last_ball['w'] is not None:
                        ball['a'] = ((ball['w'] - last_ball['w']) / (DT*(i - b)))
                    frame_state_vector.append(ball)

                    last_ball = ball
                    b = b + 1  # increment ball detection count
                elif last_ball is None and ball_flag:
                    last_ball = StateVector.to_polar(ball, x_center, y_center)
                    b = b + 1  # increment ball detection count

                #  Calculate zero speeds and accelerations
                if last_zero is not None and zero_flag:
                    zero = StateVector.to_polar(zero, x_center, y_center)
                    zero['w'] = ((StateVector.rad_dist(zero['theta'], last_zero['theta'])) / (DT*(i - z)))
                    if last_zero['w'] is not None:
                        last_zero['a'] = ((zero['w'] - last_zero['w']) / ((i - z)*DT))
                    frame_state_vector.append(zero)
                    last_zero = zero
                    z = z + 1
                elif last_zero is None and zero_flag:
                    last_zero = StateVector.to_polar(zero, x_center, y_center)
                    z = z + 1
            else:
                i = i + 1
            end_spin_threshold = 0.1
            # try:
            #     omegas_exist = zero['w'] is not None and ball['w'] is not None
            # except KeyError:
            #     omegas_exist = False
            try:
                ball_zero_diff = abs(zero['w'] - ball['w'])
            except TypeError:
                ball_zero_diff = 1 + end_spin_threshold

            if ball_zero_diff > 100 * end_spin_threshold:
                new_spin = True
            if new_spin:
                spin.append(frame_state_vector)
                if ball_flag and zero_flag and ball_zero_diff < end_spin_threshold:
                    spin_file = open(f"{fileout}-{ball['theta']}-{zero['theta']}.txt", 'w')
                    for vector in spin:
                        spin_file.write(json.dumps(vector) + "\n")
                    spin.clear()
                    new_spin = False
            det = pos_file.readline()

    @staticmethod
    def to_polar(det_object, x_center, y_center):
        #  Calculate midpoint
        try:
            x1 = det_object.pop('x1')
            x2 = det_object.pop('x2')
            y1 = det_object.pop('y1')
            y2 = det_object.pop('y2')
            x = (x1 + x2) / 2 - x_center
            y = (y1 + y2) / 2 - y_center
            r = (x ** 2 + y ** 2) ** (1 / 2)
            if x > 0:
                if y >= 0:
                    theta = np.degrees(np.arctan(y / x))
                else:
                    theta = np.degrees(np.arctan(y / x)) + 360
            elif x < 0:
                theta = np.degrees(np.arctan(y / x)) + 180
            elif x == 0:
                if y > 0:
                    theta = 90
                elif y < 0:
                    theta = 270

            det_object['r'] = r
            det_object['theta'] = theta
            det_object['w'] = None
            det_object['a'] = None
            return det_object

        except KeyError:
            return det_object

    @staticmethod
    def rad_dist(b, t):  # Calculates radial distance between two points
        a = b - t
        a = (a + 180) % 360 - 180
        return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--polar', '-p', action='store_true', help='Weather to do the file or directory in polar or parametric form')
    parser.add_argument('--dir-in', '-d', type=str, default='./', help='Specify in the input directory')
    parser.add_argument('--dir-out','-o', type=str, default="output", help='specify the output directory')
    parser.add_argument('-x', '--x-center', type=int, default=700, help='specify x center for polar form')
    parser.add_argument('-y','--y-center', type=int, default=400, help='specify y center for polar form')
    parser.add_argument('-t','--time-step', type=float, default=0.01, help='specify time step for data set')

    args = parser.parse_args()

    if args.polar:
        pass
    else:
        StateVector.extract_parametric_initial_conditions(args.dir_in, args.dir_out, args.time_step)