#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1] + "/logs/log.txt") as f:
    data = f.readlines()

errors = [ line.split()[-1][:-4] for line in data ]
errors = list(map(np.float, errors))

plt.plot(errors)
plt.savefig(sys.argv[1] + "/logs/p_error.png")
os.system(f"ln -sf {sys.argv[1] + '/logs/p_error.png'} p_error.png")
