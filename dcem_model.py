import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torch.nn.parameter import Parameter
import numpy as np
from torch.distributions import Normal

class Model(nn.Module):
    def __init__(self, lstm_units=4, seqlen=7.5):
        super(Model, self).__init__()
        self.seqlen = seqlen
        self.lstm_units = lstm_units
        self.batch_size = 1
        self.veldim = 6
        # self.velocities = 6

        ### adding mu and sigma
        # init_mu = 0.
        # init_sigma = 1.
        init_mu = 1
        init_sigma = 1 
        self.n_sample = 8

        ### Initialise Mu and Sigma
        self.mu = init_mu * torch.ones((1,1), requires_grad=True).cuda()
        self.sigma = init_sigma * torch.ones((1,1), requires_grad=True).cuda()
        self.dist = Normal(self.mu, self.sigma)

        self.f_interm= []
        self.v_interm= []
        self.mean_interm = []

        self.block = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 256),
                nn.ReLU(),
                nn.Linear(256, 16),
                nn.ReLU()
                )
        
        ## an unblocking layer, which projects it from 6D to 1D space
        self.unblock = nn.Sequential(
                nn.Linear(6,1),
                nn.ReLU()
        )

        self.linear = nn.Linear(16, 6)
        self.pooling = torch.nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, vel, Lsx, Lsy, horizon, f12):
        vel = self.dist.rsample((self.n_sample,)).transpose(0, 1).cuda()
        #vel = torch.rand(8, 1, device=torch.device('cuda:0'))
        vel = self.block(vel.view(8, 1, 1))
        vel = torch.sigmoid(self.linear(vel)).view(8, 6)
        # 8, 6 Velocity Vector
        self.f_interm.append(torch.var(vel, dim=0))
        # self.mean_interm.append(torch.mean(vel, dim=0))
        
        vel = vel.view(8, 1, 1, 6)*2 - 1
        
        ### Horizon Bit 
        if horizon < 0 :
            flag = 0
        else :
            flag = 1
        
        if flag == 1:
            vels = vel * horizon
        else :
            vels = vel * -horizon

        # print("Vels Shape: ", vels.shape)
        Lsx = Lsx.view(1, 384, 512, 6)
        Lsy = Lsy.view(1, 384, 512, 6)
        # print("Lsx Shape, Lsy Shape: ", Lsx.shape, Lsy.shape)

        f_hat = torch.cat((torch.sum(Lsx*vels,-1).unsqueeze(-1) , \
                torch.sum(Lsy*vels,-1).unsqueeze(-1)),-1)
        
        f_hat = self.pooling(f_hat.permute(0, 3, 1, 2))
        if horizon < 0:
            loss_fn = torch.nn.MSELoss(size_average=False, reduce=False)
            #print("Fat size:", f_hat.size())
            #print("F12 size:", f12.size())
            loss = loss_fn(f_hat, f12)
            loss = torch.mean(loss.view(8, 2*191*255), dim =1)
            inx = torch.argmin(loss) # index corr to velocity with lowest loss 
            vel = vel.view(8, 6)[inx]
            self.v_interm.append(vel)

        # 1 x 6
        dist = self.unblock(vel)
        # 1 x 1
        # variational inferencing.
        # 1 dist : mu, sigma
        # 
        # half cem: one 
        # update sigma
        self.mu = dist # mu, sigma 
        self.sigma = ((dist - self.mu)**2).sqrt()
        #((I * (X - mu.unsqueeze(1))**2).sum(dim=1) / n_elite).sqrt() 
        print(f_hat[::3, ::3].shape)
        return f_hat 


if __name__ == '__main__':
    vs = Model().to("cuda:0")
    ve = torch.zeros(6).to("cuda:0")
    print(list(vs.parameters()))
