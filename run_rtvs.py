# Abhinav and Nomaan, October 22, 2020
# This runs with FLOW DEPTH!!
import habitat_sim
from habitat_utils import take_step, sim_settings, make_cfg
import warnings
from utils.photo_error import mse_
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from PIL import Image

from rtvs import Rtvs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)

warnings.filterwarnings("ignore")


def main():
    folder = sys.argv[1]

    # Create folder for image results
    if not os.path.exists(folder+'/results'):
        os.makedirs(folder+'/results')

    scene_glb = folder + "/" + os.path.basename(folder).capitalize() + ".glb"
    sim_settings['scene'] = scene_glb
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    if folder == 'Baseline/ballou' or folder == 'Baseline/roane':
        pos = np.array([0.0,  0.0, 0.0])
        state = habitat_sim.agent.AgentState(position=pos)
        sim._default_agent.set_state(state)
        V = np.array([[0, 0, 0, 0, 0, 0]])
    elif folder == "Baseline/test":
        sim._default_agent.set_state(habitat_sim.agent.AgentState(
            position=[-1.7926959,  0.11083889, 19.255245]))
        V = np.array([[0, 0, 0, 0, 0, 0]])
    else:
        V = np.array([[0, 0, 0, 0, 0, 0]])

    observations = take_step(V, sim)
    print("Pose : ", sim._default_agent.get_state(
    ).sensor_states['color_sensor'].position)

    # Create folder for results
    if not os.path.exists(folder+'/logs'):
        os.makedirs(folder+'/logs')

    # Open files to log stuff out
    f = open(folder + "/logs/log.txt", "w+")
    f_pe = open(folder + "/logs/photo_error.txt", "w+")
    f_pose = open(folder + "/logs/pose.txt", "w+")
    img_goal_path = folder + "/des.png"

    color_obs = observations['color_sensor']
    img_src = Image.fromarray(color_obs, mode="RGBA")
    img_src.save(folder + "/results/test.rgba.%05d.%05d.png" % (0, 0))
    img_src = color_obs[:, :, :3]
    pre_img_src = img_src
    img_goal = np.asarray(Image.open(img_goal_path).convert("RGB"))

    photo_error_val = mse_(img_src, img_goal)
    perrors = [photo_error_val]
    print("Initial Photometric Error: ")
    print(mse_(img_src, img_goal))
    start_time = time.time()
    step = 1
    print(sim._default_agent.state)
    rtvs = Rtvs(img_goal)

    while photo_error_val > 500 and step < 5000:
        stime = time.time()
        vel = rtvs.get_vel(img_src, pre_img_src)
        algo_time = time.time() - stime
        photo_error_val = mse_(img_src, img_goal)
        perrors.append(photo_error_val)

        observations = take_step([vel], sim)

        print("Step Number: ", step)
        print("Velocity : ", vel.round(8))
        print("Photometric Error : ", photo_error_val)
        print("algo time: ", algo_time)

        pre_img_src = img_src
        img_src = observations["color_sensor"][:, :, :3]
        Image.fromarray(observations['color_sensor'], mode="RGBA").save(
            folder + "/results/test.rgba.%05d.%05d.png" % (0, step))

        f.write("Photometric error = " + str(photo_error_val) + "\n")
        f_pe.write(str(photo_error_val) + "\n")
        f_pose.write(
            "Step : " + str(sim._default_agent.get_state().sensor_states['color_sensor'].position) + "\n")
        f.flush()
        f_pe.flush()
        f_pose.flush()

        step = step + 1

    time_taken = time.time() - start_time
    f.write("Time Taken: " + str(time_taken) + "secs \n")
    # Cleanup
    f.close()
    f_pe.close()

    plt.plot(perrors)
    plt.savefig(folder + "a.png")


if __name__ == '__main__':
    main()
