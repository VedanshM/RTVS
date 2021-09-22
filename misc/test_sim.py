#import generate_data as H
import habitatenv as H
from array import array

import numpy as np
env = H.HabitatEnv()

vel = [0, 0, 0, 30, 30, 30] #1x6
env.example(vel, 0)
