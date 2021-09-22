import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torch.nn.parameter import Parameter
import numpy as np
from torch.distributions import Normal
from transformer_layers import TransformerEncoder, TransformerModel, SAP, Mean
from transformer_layers import position_encoding
import torch.nn.functional as F
from argparse import Namespace


class Model(nn.Module):
    def __init__(self, lookahead = 5):
        super(Model, self).__init__()
        device = torch.device('cuda:0')
        self.device = device
        #self.linear = nn.Linear(6, 6)
        #self.pooling_strategy = Mean(6)
        self.seq_len = 5
        self.pooling = torch.nn.AvgPool2d(kernel_size=1, stride=1)
        self.n_sample = 1
        config = {
            "hparams" : {
                 "hidden_size": 6,
                 "num_hidden_layers": 5,
                 "num_attention_heads": 2,
                 "intermediate_size": 6,
                 "hidden_act": "gelu",
                 "hidden_dropout_prob": 0.6,
                 "attention_probs_dropout_prob": 0.6,
                 "initializer_range": 0.02,
                 "layer_norm_eps": 1.0e-12,
                 "share_layer": False,
                 "max_input_length": 0,
                 "pre_layer_norm": False
            }   
        }

        self.transformer = TransformerModel(
            Namespace(**config["hparams"]), 6
        )

    def process_input_data(self, vel):
        batch_size = vel.shape[0]
        seq_len = vel.shape[1]
        pos_enc = position_encoding(seq_len, 6) # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)
        pos_enc = torch.FloatTensor(pos_enc).to(device=vel.device, dtype=torch.float32).expand(vel.size(0), *pos_enc.size()) # (batch_size, seq_len, hidden_size)
        attn_mask = torch.FloatTensor(attn_mask).to(device=vel.device, dtype=torch.float32) # (batch_size, seq_len)
        return vel, pos_enc, attn_mask # (x, pos_enc, attention_mask)

    def forward(self, vel, Lsx, Lsy, f12):
        #vel = self.dist.rsample((self.n_sample,)).to(self.device)
        print(vel.shape)
        vel, pos_enc, attn_mask = self.process_input_data(vel)
        vel = self.transformer(vel, pos_enc, attn_mask)
        print(vel.shape)
        # self attention based pooling
        # torch.Size([64, 512, 6])
        #self.f_interm.append(self.sigma)
        #self.mean_interm.append(self.mu)
        #vel = self.pooling_strategy(vel[0])
        #vel = torch.sigmoid(self.linear(vel))    
        # 8 x 6 velocities sampled from distribution

        vel = vel.view(self.n_sample, 1, 1, 6)*2 - 1
        Lsx = Lsx.view(1, f12.shape[2], f12.shape[3], 6)
        Lsy = Lsy.view(1, f12.shape[2], f12.shape[3], 6)

        f_hat = torch.cat((torch.sum(Lsx*vel,-1).unsqueeze(-1) , \
                torch.sum(Lsy*vel,-1).unsqueeze(-1)),-1)
        
        f_hat = self.pooling(f_hat.permute(0, 3, 1, 2))

        vel = vel.view(6,)
        self.v_interm.append(vel)

        return f_hat

if __name__ == '__main__':
    vs = Model().to("cuda:2")
    v, Lsx, Lsy, f12 = None, None, None, None
    print(vs)
    vs.forward(v, Lsx, Lsy, f12)
