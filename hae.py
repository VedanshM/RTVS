import os
import random
import math
import pickle

import transformations_tf as tft

import numpy as np
import magnum as mn
from PIL import Image
from settings import default_sim_settings, make_cfg
from habitat_sim.scene import SceneNode

from utils.frame_utils import read_gen
import matplotlib.pyplot as plt

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_magnum,
    quat_to_magnum,
)

class HabitatEnv():
    def __init__(self):
        self.scene = '/scratch/Baseline_LSTM/roane/'
        self._cfg = make_cfg(self.scene + 'Roane.glb')
        self.init_common(np.array([0,0,0,1,0,0,0]))
        agent_node = self._sim.agents[0].scene_node
        self.agent_object_id = self._sim.add_object(1, agent_node)
        self._sim.set_object_motion_type(
            habitat_sim.physics.MotionType.KINEMATIC, self.agent_object_id
        )
        assert (
        self._sim.get_object_motion_type(self.agent_object_id)
        == habitat_sim.physics.MotionType.KINEMATIC
        )
        # Saving Start Frame
        observations = self._sim.get_sensor_observations()
        self.save_color_observation(observations, 0, 0)
        self.step = 0
        #self.save_depth_observation(observations, 0, 0, self.scene)
       
    def init_common(self, init_state):
        self._sim = habitat_sim.Simulator(self._cfg)
        random.seed(default_sim_settings["seed"])
        self._sim.seed(default_sim_settings["seed"])
        start_state = self.init_agent_state(default_sim_settings["default_agent"], init_state)
        return start_state
        
    def init_agent_state(self, agent_id, init_state):
        start_state = habitat_sim.agent.AgentState()
        x, y, z, w, p, q, r = init_state
        start_state.position = np.array([x, y, z]).astype('float32')
        start_state.rotation = np.quaternion(w,p,q,r)
        agent = self._sim.initialize_agent(agent_id, start_state)
        start_state = agent.get_state()
        return start_state
    
    def get_agent_pose(self):
        agent = self._sim._default_agent
        state = agent.get_state()
        position = state.position
        rotation = state.rotation
        pose = [position[0], position[1], position[2], rotation.w, rotation.x, rotation.y, rotation.z]
        return pose

    def save_color_observation(self, obs, frame, step):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save(self.scene + "icra20/test.rgba.%05d.%05d.png" % (frame, step))
        #color_img = read_gen(folder + "icra20/test.rgba.%05d.%05d.png" % (frame, step))
        return color_img
    
    def save_depth_observation(self, obs, frame, step, folder):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save(folder + "/results/test.depth.%05d.%05d.png" % (frame, step))
        depth_img = plt.imread(folder + "/results/test.depth.%05d.%05d.png" % (frame, step))
        return depth_img
    
    def apply_pose(self, pose):
        self.step += 1
        state = habitat_sim.agent.AgentState()
        state.position = np.array([pose[0], pose[1], pose[2]])
        state.rotation = np.quaternion(pose[3], pose[4], pose[5], pose[6])
        agent = self._sim._default_agent
        agent.set_state(state)
        obs = self._sim.get_sensor_observations()
        self.save_color_observation(obs, 0, self.step)

    def end_sim(self):
        self._sim.close()
        del self._sim
