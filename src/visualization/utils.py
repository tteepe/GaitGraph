# ------------------------------------------------------------------------------
# Modified from:
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import numpy as np


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))


keypoints = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

# Xiaochu Style
# (R,G,B)
color1 = [(179, 0, 0), (228, 26, 28), (255, 255, 51),
          (49, 163, 84), (0, 109, 45), (255, 255, 51),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
          (217, 95, 14), (254, 153, 41), (255, 255, 51),
          (44, 127, 184), (0, 0, 255)]

link_pairs1 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
]

point_color1 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 51), (255, 255, 51),
                (254, 153, 41), (44, 127, 184),
                (217, 95, 14), (0, 0, 255),
                (255, 255, 51), (255, 255, 51), (228, 26, 28),
                (49, 163, 84), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

xiaochu_style = ColorStyle(color1, link_pairs1, point_color1)

# Chunhua Style
# (R,G,B)
color2 = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
          (0, 176, 240), (0, 176, 240), (0, 176, 240),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
          (255, 255, 0), (255, 255, 0), (169, 209, 142),
          (169, 209, 142), (169, 209, 142)]

link_pairs2 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6], [6, 8], [8, 10],
]

point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict
