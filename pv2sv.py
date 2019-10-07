import numpy as np
import json


class StateVector:

    def __init__(self, pred_model):
        self.pred_model  # Predictive model network

    @classmethod
    def pv2sv(filein, fileout, x_center, y_center):
        last_ball, last_zero = None
        b, z, i = 0
        pos_file = open(filein, 'r')
        det = pos_file.readline()
        while det != None:
            frame_state_vector = []
            json_detection = json.loads(det)
            if (len(json_detection) > 0):
                ball_flag = False
                zero_flag = False
                for inst in det:
                    if inst["cls"] == 0:
                        ball = inst
                        ball_flag = True
                    elif inst["cls"] == 1:
                        zero = inst
                        zero_flag = True
                    else:
                        assert False, ("Malformed classtype in json array, class should be 0 or 1 only")
                i = i + 1  # increment total count

                #  Calculate ball speeds and accelerations
                if last_ball != None and ball_flag:
                    ball = StateVector.toPolar(ball)
                    ball['w'] = ((ball['theta'] - last_ball['theta']) / (i - b))
                    if last_ball['w'] != None:
                        ball['a'] = ((ball['w'] - last_ball['w']) / (i - b))
                    frame_state_vector.append(ball)
                    last_ball = ball
                    b = b + 1  # increment ball detection count
                elif last_ball == None and ball_flag:
                    last_ball = ball
                    b = b + 1  # increment ball detection count

                #  Calculate zero speeds and accelerations
                if last_zero != None and zero_flag:
                    zero = StateVector.toPolar(zero)
                    zero['w'] = ((ball.get['theta'] - last_ball['theta']) / (i - b))
                    if last_ball['w'] !=None:
                        last_ball['a'] = ((zero['w'] - last_zero['w']) / (i - b))
                    frame_state_vector.append(zero)
                    last_zero = zero
                    z = z + 1
                elif last_zero == None and zero_flag:
                    last_zero = zero
                    z = z+1
            else:
                i = i + 1

            if




    @classmethod
    def toPolar(self, object):
        #  Calculate midpoint
        x1 = object.pop('x1')
        x2 = object.pop('x2')
        y1 = object.pop('y1')
        y2 = object.pop('y2')
        x = (x1+x2)/2
        y = (y1+y2)/2
        r = (x**2+y**2)**(1/2)
        if x>0:
            theta = np.degrees(np.arctan(y/x))
        elif x<0:
            theta = np.degrees(np.arctan(y/x))+180
        elif x==0:
            if y>0:
                theta = 90
            elif y<0:
                theta = -90


        object['r'] = r
        object['theta'] = theta
        object['w'] = None
        object['a'] = None
