# Abhinav and Nomaan, October 22, 2020
## This runs with FLOW DEPTH!! 
import os
import sys
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

from utils.frame_utils import read_gen, flow_to_image
from utils.photo_error import mse_
from calculate_flow import FlowNet2Utils
import warnings
warnings.filterwarnings("ignore")

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

# from single_lstm_model import Model
from origina_dcem_model import Model 
from interactionmatrix_cuda import InteractionMatrix

from scipy.spatial.transform import Rotation as R
from final import update_v
from final import sim_settings, make_cfg
import habitat_sim

LR = 0.005  # Learning Rate
horizon = 10  # Horizon

flow_utils = FlowNet2Utils()
intermat = InteractionMatrix()
vs_lstm = Model().to(device="cuda:0")
optimiser = torch.optim.Adam(
    vs_lstm.parameters(), lr=LR, betas=(0.93, 0.999))
loss_fn = torch.nn.MSELoss(size_average=False)



def take_action(sim, V):
    sim.step("move_left")
    sim.step("move_right")
    sim.step("move_forward")
    sim.step("move_backward")
    sim.step("move_up")
    sim.step("move_down")
    sim.step("look_up")
    sim.step("look_down")
    sim.step("look_left")
    sim.step("look_right")
    sim.step("look_clock")
    observations = sim.step("look_anti")
    return observations, sim


class Get_v:
    def get_vel(self, img_goal, img_src, pre_img_src=None):
        global horizon
        photo_error_val = mse_(img_src, img_goal)

        if photo_error_val < 6000 and photo_error_val > 3600:
            horizon = 10*(photo_error_val/6000)
        elif photo_error_val < 3000:
            horizon = 6
        ct = 20
        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        flow_depth_proxy = flow_utils.flow_calculate(img_src, pre_img_src).astype('float64')
            #itm = time.time()
        Cy, Cx = flow_depth_proxy.shape[1]/2, flow_depth_proxy.shape[0]/2
        flow_depth=np.linalg.norm(flow_depth_proxy[::ct, ::ct],axis=2)
        flow_depth=flow_depth.astype('float64')
        vel, Lsx, Lsy = intermat.getData(0.6*(1/(1+np.exp(-1/flow_depth)) - 0.5), ct, Cy, Cx)

        Lsx = torch.tensor(Lsx, dtype = torch.float32).to(device="cuda:0")
        Lsy = torch.tensor(Lsy, dtype = torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype = torch.float32).to(device="cuda:0")
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))

        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        vs_lstm.zero_grad()
        f_hat = vs_lstm.forward(vel, Lsx, Lsy, horizon, f12)
        loss = loss_fn(f_hat, f12)

        print("MSE:", str(np.sqrt(loss.item())))
        loss.backward(retain_graph=True)
        optimiser.step()

        #Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        f_hat = vs_lstm.forward(vel, Lsx, Lsy, -horizon,
                                f12.to(torch.device('cuda:0')))
        return vs_lstm.v_interm[0], photo_error_val



def main():
    folder = sys.argv[1]
    getv = Get_v()


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
    
    sim = update_v(V, sim)
    observations, sim = take_action(sim, V)
    print("Pose 3 : ", sim._default_agent.get_state().sensor_states['color_sensor'].position)

    # Create folder for results
    if not os.path.exists(folder+'/logs'):
        os.makedirs(folder+'/logs')

    # Open files to log stuff out 
    f = open(folder + "/logs/log.txt","w+")
    f_pe = open(folder + "/logs/photo_error.txt", "w+")
    f_pose = open(folder + "/logs/pose.txt", "w+")
    img_goal_path = folder + "/des.png"
    
    color_obs = observations['color_sensor']
    img_src = Image.fromarray(color_obs, mode="RGBA")
    img_src.save(folder + "/results/test.rgba.%05d.%05d.png" % (0, 0))
    img_src =  color_obs[:, :, :3]
    pre_img_src = img_src
    img_goal = np.asarray(Image.open(img_goal_path).convert("RGB"))    

    photo_error_val=mse_(img_src,img_goal)
    perrors = [photo_error_val]
    print("Initial Photometric Error: ")
    print(mse_(img_src, img_goal))
    start_time = time.time() 
    step = 1
    print(sim._default_agent.state)

    while photo_error_val > 500 and step < 5000:
        print("Step Number: ", step)
        print("Photometric Error : ", photo_error_val)
        vel, photo_error_val = getv.get_vel(img_goal, img_src, pre_img_src)
        
        perrors.append(photo_error_val)

        sim = update_v([vel], sim)
        observations, sim = take_action(sim, vel)

        pre_img_src = img_src
        img_src = observations["color_sensor"][:,:,:3]
        Image.fromarray(observations['color_sensor'], mode="RGBA").save(folder + "/results/test.rgba.%05d.%05d.png" % (0, step))
        
        f.write("Photometric error = " + str(photo_error_val) + "\n")
        f_pe.write(str(photo_error_val) + "\n")
        f_pose.write("Step : "+ str(sim._default_agent.get_state().sensor_states['color_sensor'].position)  + "\n")
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
    plt.savefig(folder+ "a.png")

    # save indvidial image and gif
    # onlyfiles = [f for f in listdir(folder + "/results") if f.endswith(".png")]
    # onlyfiles.sort()
    # images = []
    # for filename in onlyfiles:
    #     images.append(imageio.imread(folder + '/results/' + filename))
    # imageio.mimsave(folder + '/results/output.gif', images, fps=4)

    

if __name__ == '__main__':
    main()
