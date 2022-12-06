#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Taxi Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator(**kwargs):
    return gym.make("Taxi-v3")