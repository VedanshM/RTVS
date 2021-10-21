#======================================================================================#
# Importing Important Libraries #
#======================================================================================#
import habitat_sim
import habitat
import random
import matplotlib.pyplot as plt

import numpy as np
import cv2

from habitat_sim import registry

import magnum as mn
#======================================================================================#


#======================================================================================#
# Defining navigation keys #
#======================================================================================#
FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="u"
DOWN_KEY = "s"

U_KEY = "8"
D_KEY = "2"
ML_KEY = "4"
MR_KEY = "6"
FINISH="f"

action_dict ={
    "w" : "move_forward",
    "a" : "look_left",
    "d" : "look_right",
    "u" : "look_up",
    "s" : "look_down",
    "n" : "rotate_sensor_anti_clockwise",
    "c" : "rotate_sensor_clockwise",
    "8" : "move_up",
    "2" : "move_down",
    "4" : "move_left",
    "6" : "move_right"
}
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]
#======================================================================================#

#======================================================================================#
# Defining simulator settings #
#======================================================================================#
test_scene = "./gibson/Pablo.glb"

sim_settings = {
    "width": 512,  # Spatial resolution of the observations
    "height": 384,
    "scene": "./Baseline/skokloster-castle/Skokloster-castle.glb",  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "hfov": 90,
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": False,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}
#======================================================================================#

#======================================================================================#
# Defining Simulator Configuration #
#======================================================================================#
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sensor_specs = []
    def create_camera_spec(**kw_args):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [settings["height"], settings["width"]]
        camera_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(camera_sensor_spec, k, kw_args[k])
        return camera_sensor_spec

    # Note: all sensors must have the same resolution
    if settings["color_sensor"]:
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = create_camera_spec(
            uuid="depth_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
         "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        
        "look_left": habitat_sim.agent.ActionSpec(
            "look_left", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "look_right": habitat_sim.agent.ActionSpec(
            "look_right", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        
        "look_anti": habitat_sim.agent.ActionSpec(
            "rotate_sensor_anti_clockwise", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "look_clock": habitat_sim.agent.ActionSpec(
            "rotate_sensor_clockwise", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        
        
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        
        "move_up": habitat_sim.agent.ActionSpec(
            "move_up", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

if __name__ == '__main__':
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)


#======================================================================================#


#======================================================================================#
# Changing the amount of actuation #
#======================================================================================#
def take_step(V, sim):
    V = V[0]
    print("Velocity : ", V)
    factor = 0.00416
    rf = 180/np.pi
    v1, v2, v3= 1, 1, 1
    sim.config.agents[0].action_space['move_right'].actuation.amount = v1*factor*V[0]
    sim.config.agents[0].action_space['move_up'].actuation.amount = -v2*factor*V[1]
    sim.config.agents[0].action_space['move_backward'].actuation.amount = -v3*factor*V[2]
    sim.config.agents[0].action_space['look_left'].actuation.amount = -rf*factor*V[4]
    sim.config.agents[0].action_space['look_up'].actuation.amount = rf*factor*V[3]
    sim.config.agents[0].action_space['look_anti'].actuation.amount = rf*factor*V[5]
    sim.step("move_right")
    sim.step("move_backward")
    sim.step("move_up")
    sim.step("look_up")
    sim.step("look_left")
    observations = sim.step("look_anti")
    return observations

    
#======================================================================================#


#======================================================================================#
# Setting gravity #
#======================================================================================#
# print(habitat_sim.geo.GRAVITY)
# sim.set_gravity(habitat_sim.geo.GRAVITY, sim.config.SIMULATOR.SCENE)
# print(sim.scene_id)
#======================================================================================#

#======================================================================================#
# Action names #
#======================================================================================#
# action_names = list(
#     cfg.agents[
#         sim_settings["default_agent"]
#     ].action_space.keys()
# )
# total_frames = 0
# max_frames = 1
# while total_frames < max_frames:
#     action = random.choice(action_names)
#     print("action", action)
#     observations = sim.step('look_up')
#     rgb = observations["color_sensor"]
#     semantic = observations["semantic_sensor"]
#     depth = observations["depth_sensor"]
#
#     # display_sample(rgb, semantic, depth)
#     cv2.imshow('test', rgb)
#     cv2.waitKey(0)
#     total_frames += 1
#
# print(action_names)
#======================================================================================#


#======================================================================================#
# Commenting using keys #
#======================================================================================#
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():

    
    observations = sim.step('move_forward')
    cv2.imshow("RGB", observations["color_sensor"])

    print("Agent stepping around inside environment.")

    count_steps = 0
    while True:
        keystroke = cv2.waitKey(0)
        if(keystroke == "f"):
            return
        print("Key Stroke : ", chr(keystroke))
        action = action_dict[chr(keystroke)]
        print("Action is : ", action)

        # if keystroke == ord(FORWARD_KEY):
        #     action = HabitatSimActions.MOVE_FORWARD
        #     print("action: FORWARD")
        # elif keystroke == ord(LEFT_KEY):
        #     action = HabitatSimActions.TURN_LEFT
        #     print("action: LEFT")
        # elif keystroke == ord(RIGHT_KEY):
        #     action = HabitatSimActions.TURN_RIGHT
        #     print("action: RIGHT")
        # elif keystroke == ord(UP_KEY):
        #     action = HabitatSimActions.LOOK_UP
        #     print("action: RIGHT")
        # elif keystroke == ord(FINISH):
        #     action = HabitatSimActions.STOP
        #     print("action: FINISH")
        # else:
        #     print("INVALID KEY")
        #     continue

        observations = sim.step(action)
        print(observations.keys())
        count_steps += 1

        print(observations["color_sensor"].shape)
        cv2.imshow("RGB", observations["color_sensor"])
        cv2.imwrite('test.png', observations["color_sensor"])

if __name__ == '__main__':
    example()
#======================================================================================#


# [AgentConfiguration(height=1.5, radius=0.1, mass=32.0, linear_acceleration=20.0, angular_acceleration=12.566370614359172, linear_friction=0.5, angular_friction=1.0, coefficient_of_restitution=0.0, sensor_specifications=[<habitat_sim._ext.habitat_sim_bindings.SensorSpec object at 0x1239deaf0>, <habitat_sim._ext.habitat_sim_bindings.SensorSpec object at 0x1239deb48>, <habitat_sim._ext.habitat_sim_bindings.SensorSpec object at 0x1239deba0>], action_space={'move_forward': ActionSpec(name='move_forward', actuation=ActuationSpec(amount=1.0, constraint=None)), 'move_down': ActionSpec(name='move_down', actuation=ActuationSpec(amount=1.0, constraint=None)), 'turn_left': ActionSpec(name='turn_left', actuation=ActuationSpec(amount=1.0, constraint=None)), 'turn_right': ActionSpec(name='turn_right', actuation=ActuationSpec(amount=1.0, constraint=None)), 'look_up': ActionSpec(name='look_up', actuation=ActuationSpec(amount=1.0, constraint=None)), 'look_down': ActionSpec(name='look_down', actuation=ActuationSpec(amount=1.0, constraint=None)), 'move_up': ActionSpec(name='move_up', actuation=ActuationSpec(amount=1.0, constraint=None)), 'move_left': ActionSpec(name='move_left', actuation=ActuationSpec(amount=1.0, constraint=None)), 'move_right': ActionSpec(name='move_right', actuation=ActuationSpec(amount=1.0, constraint=None))}, body_type='cylinder')]
