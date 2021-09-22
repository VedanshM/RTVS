import os
import random

import cv2
import numpy as np
import magnum as mn
from PIL import Image
from settings import default_sim_settings, make_cfg
from habitat_sim.scene import SceneNode

from utils.frame_utils import read_gen
import matplotlib.pyplot as plt

import habitat_sim
import habitat_sim.agent

class HabitatEnvG():
    def __init__(self):
        self._cfg = make_cfg()
        self.init_common()
       
    def init_common(self):
        self._sim = habitat_sim.Simulator(self._cfg)
        random.seed(default_sim_settings["seed"])
        self._sim.seed(default_sim_settings["seed"])
        start_state = self.init_agent_state(default_sim_settings["default_agent"])
        return start_state
        
    def init_agent_state(self, agent_id):
        start_state = habitat_sim.agent.AgentState()
        start_state.position = np.array([-1, 0, 8]).astype('float32')
        start_state.rotation = np.quaternion(1,0,0,0)
        agent = self._sim.initialize_agent(agent_id, start_state)
        start_state = agent.get_state()
        return start_state
    
    def save_color_observation(self, obs, key, frame):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save("castle/" + key + "/test.rgba.%05d.png" % frame)
        return color_img
    
    def save_depth_observation(self, obs, key, frame):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save("castle/" + key + "/test.depth.%05d.png" % frame)
        return depth_img

    def agent_controller(self, agent, velocity):
        #_X_AXIS = 0
        #_Y_AXIS = 1
        #_Z_AXIS = 2
        axises = [0,1,2]
        for axis in axises:
            # Translate
            ax = agent.scene_node.transformation[axis].xyz
            agent.scene_node.translate_local(ax * velocity[axis])
            # Rotate
            _rotate_local_fns = [
                SceneNode.rotate_x_local,
                SceneNode.rotate_y_local,
                SceneNode.rotate_z_local,
            ]
            _rotate_local_fns[axis](agent.scene_node, mn.Deg(velocity[axis]))
            agent.scene_node.rotation = agent.scene_node.rotation.normalized()

    def example(self):
        '''
        vel : n x 6 velocity vector
        '''

        agent_id = default_sim_settings["default_agent"]
        agent = self._sim._default_agent

        Agent_state = agent.get_state()

        '''
        folder_name = {'test_gen_L_F_U':[1,4,6],
                'test_gen_L_F_D':[4,6,5],
                'test_gen_L_B_U':[1,2,4],
                'test_gen_L_B_D':[2,4,5],
                'test_gen_R_F_U':[1,3,6],
                'test_gen_R_F_D':[3,5,6],
                'test_gen_R_B_U':[1,2,3],
                'test_gen_R_B_D':[2,3,5],
                'test_gen_RL_RL_RL':[]}
        '''
        folder_name = {
            'test_gen_RR' : [7],
            'test_gen_RL' : [8],
            'test_gen_RU' : [9],
            'test_gen_RD' : [10],
            'test_gen_RF' : [11],
            'test_gen_RB' : [12]
        }

        for key, value in folder_name.items():
            if not os.path.isdir("castle/" + key):
                os.makedirs("./castle/" + key)
            frame = 0

            observations = self._sim.get_sensor_observations()
            self.save_color_observation(observations, key, frame)
            self.save_depth_observation(observations, key, frame)
            frame += 1
            
            for j in value:
                if(j==1):
                    action = "move_up"
                elif(j==2):
                    action = "move_backward"
                elif(j==3):
                    action = "move_right"
                elif(j==4):
                    action = "move_left"
                elif(j==5):
                    action = "move_down"
                elif(j==6):
                    action = "move_forward"
                elif(j==7):
                    action = "rotate_right"
                elif(j==8):
                    action = "rotate_left"
                elif(j==9):
                    action = "rotate_up"
                elif(j==10):
                    action = "rotate_down"
                elif(j==11):
                    action = "rotate_forward"
                elif(j==12):
                    action = "rotate_backward"
                
                state = agent.get_state()
                agent.act(action)
                print("poto save")
                frame += 1

                observations = self._sim.get_sensor_observations()
                self.save_color_observation(observations, key, frame)
                self.save_depth_observation(observations, key, frame)

                agent.set_state(Agent_state)

    def end_sim(self):
        self._sim.close()
        del self._sim