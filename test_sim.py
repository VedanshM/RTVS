#import generate_data as H
import hae as H
from array import array

import numpy as np
env = H.HabitatEnv()

poses = np.loadtxt('pose.txt')

for i in range(len(poses)):
    env.apply_pose(poses[i])

env.end_sim()
