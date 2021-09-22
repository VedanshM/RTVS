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
import habitatenv as hs
import imageio
from os import listdir
from os.path import isfile, join

from scipy.spatial.transform import Rotation as R
from final import update_v
from final import sim_settings, make_cfg
import habitat_sim
import cv2 

# Early Stopping
from early_stopping.pytorchtools import EarlyStopping

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

def main():
    folder = sys.argv[1]
    vel_init = int(sys.argv[2])
    rnn_type = int(sys.argv[3])
    depth_type = int(sys.argv[4])

    ### Hyper Parameters, which need to be tuned 
    ITERS = 1 # iterations per MPC step 
    SEQ_LEN = 52
    LR = 0.005 # Learning Rate
    horizon = 10 # Horizon 
    level = 0
    
    if vel_init == 1:
        vel_init_type = 'RANDOM'
    else:
        vel_init_type = 'IBVS'

    if rnn_type == 1:
        rnn_type = 'LSTM'
    else:
        rnn_type = 'GRU'

    if depth_type == 1:
        depth_type = 'TRUE'
    else:
        depth_type = 'FLOW'

    # Create folder for image results
    if not os.path.exists(folder+'/results'):
        os.makedirs(folder+'/results')

    flow_utils = FlowNet2Utils()
    intermat = InteractionMatrix()
    
    scene_glb = folder + "/" + os.path.basename(folder).capitalize() + ".glb"
    sim_settings['scene'] = scene_glb
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    
    if folder == 'Baseline/ballou' or folder == 'Baseline/roane':
        pos = np.array([0.0,  0.0, 0.0])
        state = habitat_sim.agent.AgentState(position=pos)
        sim._default_agent.set_state(state)
        V = np.array([[0, 0, 0, 0, 0, 0]])
    else :
        V = np.array([[0, 0, 0, 0, 0, 0]])
    
    sim = update_v(V, sim)
    observations, sim = take_action(sim, V)
    print("Pose 3 : ", sim._default_agent.get_state().sensor_states['color_sensor'].position)
    loss_fn = torch.nn.MSELoss(size_average=False)

    # Create folder for results
    if not os.path.exists(folder+'/logs'):
        os.makedirs(folder+'/logs')

    # Open files to log stuff out 
    f = open(folder + "/logs/log.txt","w+")
    f_pe = open(folder + "/logs/photo_error.txt", "w+")
    f_pose = open(folder + "/logs/pose.txt", "w+")
    f_var = open(folder +'/logs/variance.txt', "w+")
    f_mean = open(folder +'/logs/mean.txt', "w+")
    f_time = open(folder + '/logs/time.txt', "w+")
    f_vel = open(folder +'/logs/velocity.txt', "w+")
    f_pose_total = open(folder + "/logs/pose_total.txt", "w+")  #NQ

    img_source_path = folder + "/results/" + "test.rgba.00000.00000.png"
    img_goal_path = folder + "/des.png"
    
    color_obs = observations['color_sensor']
    img_src = Image.fromarray(color_obs, mode="RGBA")
    img_src.save(folder + "/results/test.rgba.%05d.%05d.png" % (0, 0))
    img_src = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (0, 0))
    img_src =  color_obs[:, :, :3]
    pre_img_src = img_src
    img_goal = read_gen(img_goal_path)
    print(img_goal.shape, color_obs.shape)
    d1 = observations["depth_sensor"]
    depth_img = Image.fromarray(
             (d1 / 10 * 255).astype(np.uint8), mode="L"
             )
    depth_img.save(folder + "/results/test.depth.%05d.%05d.png" % (0, 0))
    depth_img = plt.imread(folder + "/results/test.depth.%05d.%05d.png" % (0, 0))
    d1 = depth_img
    
    vs_lstm = Model().to(device="cuda:0")
    optimiser = torch.optim.Adam(vs_lstm.parameters(), lr=LR, betas=(0.93, 0.999))

    photo_error_val=mse_(img_src,img_goal)
    print("Initial Photometric Error: ")
    print(mse_(img_src, img_goal))
    start_time = time.time() 
    step=1 # making 0 to 1 for flow depth

    while photo_error_val > 500 and step < 5000:
        mpc_time = time.time()
        ct = 20
        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        if step == 1:
            Cy, Cx = d1.shape[1]/2, d1.shape[0]/2
            vel, Lsx, Lsy = intermat.getData(d1[::ct, ::ct], ct, Cy, Cx)
        else:
                # Whole else rewritten by NQ.
            flow_depth_proxy = flow_utils.flow_calculate(img_src, pre_img_src).astype('float64')
                #itm = time.time()
            Cy, Cx = flow_depth_proxy.shape[1]/2, flow_depth_proxy.shape[0]/2
            flow_depth=np.linalg.norm(flow_depth_proxy[::ct, ::ct],axis=2)
            flow_depth=flow_depth.astype('float64')
            vel, Lsx, Lsy = intermat.getData(0.6*(1/(1+np.exp(-1/flow_depth)) - 0.5), ct, Cy, Cx)
                #itm = time.time() - itm print("Step 1: ", time.time() - mpc_time)



        #cuda_time = time.time()
        Lsx = torch.tensor(Lsx, dtype = torch.float32).to(device="cuda:0")
        Lsy = torch.tensor(Lsy, dtype = torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype = torch.float32).to(device="cuda:0")
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))
        #flag = 0
        #f.write("Processing Optimization Step: " + str(step) + "\n")
       # ts=time.time()
        print("Iterations : ", step)

        ### ITERS iterations per MPC Step
        for cnt in range(ITERS):
            #t1 = time.time()
            vs_lstm.v_interm = []
            vs_lstm.f_interm = []
            vs_lstm.mean_interm = []
        
            vs_lstm.zero_grad()
            f_hat = vs_lstm.forward(vel, Lsx, Lsy,horizon, f12)
            loss = loss_fn(f_hat, f12) 
            #f.write("Epoch " + str(cnt) + "\n")
            #f.write("MSE: " + str(np.sqrt(loss.item())))
            print("MSE:", str(np.sqrt(loss.item())))
            loss.backward(retain_graph=True)
            optimiser.step()
        
        #print()
        #Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []
        #time_taken_ = time.time() - ts
        
        #print("Optimisation Time : ", time_taken_)
       
        f_hat = vs_lstm.forward(vel, Lsx, Lsy, -horizon, f12.to(torch.device('cuda:0')))
        
        print("Step 3: ", time.time() - mpc_time)
        #f_var.write(str(vs_lstm.f_interm[0][0].view(1,6)) + '\n') 
        #f_vel.write(str(vs_lstm.v_interm[0]) + '\n')
        #f_mean.write(str(vs_lstm.mean_interm[0][0].view(1,6)) + '\n')

        #f_time.write(str(time_taken_) + '\n')
        # f_mean.write(str(vs_lstm.mean_interm[0]))
        
        print("Photometric Error : ", photo_error_val)
        #f.write("Predicted Velocities: \n")
        #f.write(str(vs_lstm.v_interm))
        #f.write("\n")
        ### Saving the images
        sim = update_v([vs_lstm.v_interm[0]], sim)
        observations, sim = take_action(sim, vs_lstm.v_interm[0])
        pre_img_src = img_src
        img_src = observations["color_sensor"][:,:,:3]
        #img_src = Image.fromarray(color_obs, mode="RGBA")
        #img_src.save(folder + "/results/test.rgba.%05d.%05d.png" % (step, step))
        #img_src = read_gen(folder + "/results/test.rgba.%05d.%05d.png" % (step, step))
        
        #d1 = observations["depth_sensor"]
        #depth_img = Image.fromarray(
        #    (d1 / 10 * 255).astype(np.uint8), mode="L"
        #)
        #depth_img.save(folder + "/results/test.depth.%05d.%05d.png" % (step, step))
        #depth_img = plt.imread(folder + "/results/test.depth.%05d.%05d.png" % (step, step))
        #d1 = depth_img
        photo_error_val = mse_(img_src,img_goal)
        
        if photo_error_val < 6000 and photo_error_val > 3600:
            horizon = 10*(photo_error_val/6000)
        elif photo_error_val < 3000:
            horizon = 6
        
        
        f.write("Photometric error = " + str(photo_error_val) + "\n")
        #f.write("Step Number: " + str(step) + "\n")
        f_pe.write(str(photo_error_val) + "\n")
        f_pose.write("Step : "+ str(sim._default_agent.get_state().sensor_states['color_sensor'].position)  + "\n")
        #f_pose_total.write("Step : "+ str(sim._default_agent.get_state())  + "\n")      #NQ
        #f_pose_total.flush()   #NQ
        f.flush()
        f_pe.flush()
        f_pose.flush()
        #if step > 1 : 
        #    print("Interaction matrix time : ", itm)
        #print("ovh time : ", time.time() - ovhT)
        step = step + 1
        print("MPC Iteration time : ", time.time() - mpc_time)
        print()
        #print("time : ", tt)
        
        
    time_taken = time.time() - start_time
    f.write("Time Taken: " + str(time_taken) + "secs \n")
    # Cleanup
    f.close()
    f_pe.close()
    
    del flow_utils
    del intermat
    del env
    del vs_lstm
    del loss_fn
    del optimiser

    # save indvidial image and gif
    onlyfiles = [f for f in listdir(folder + "/results") if f.endswith(".png")]
    onlyfiles.sort()
    images = []
    for filename in onlyfiles:
        images.append(imageio.imread(folder + '/results/' + filename))
    imageio.mimsave(folder + '/results/output.gif', images, fps=4)

    

if __name__ == '__main__':
    main()
