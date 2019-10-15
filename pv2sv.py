import numpy as np
import json


class StateVector:

    def __init__(self, pred_model):
        self.pred_model  # Predictive model network

    @staticmethod
    def pv2sv(filein, fileout, x_center, y_center, DT):
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
