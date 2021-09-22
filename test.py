import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from torch.nn.parameter import Parameter
# R = torch.zeros((24, 6))
# cnt = -1
# for i in range(24):
#     if i%4 == 0:
#         cnt+=1
#     R[i][cnt] = 1
#
# y = torch.tensor([1, 2, 3, 4, 5, 6])
# x = torch.ones((4, 6))
# hiddens = x * y
#
# t1 = time.time()
# torch.mm(R, hiddens.T)
# print("Mat mul time : ", time.time() -t1)


class Model(nn.Module):
    def __init__(self, lstm_units=4, seqlen=5):
        super().__init__()
        self.seqlen = seqlen
        self.lstm_units = lstm_units
        self.batch_size = 1
        self.veldim = 6
        
        self.f_interm= []
        self.v_interm= []

        # Input Gate
        self.Wix = Parameter(torch.Tensor(self.veldim, lstm_units))
        self.Wih = Parameter(torch.Tensor(self.veldim*lstm_units, lstm_units))
        self.bi = Parameter(torch.Tensor(self.veldim, lstm_units))
        
        # Forget Gate
        self.Wfx = Parameter(torch.Tensor(self.veldim, lstm_units))
        self.Wfh = Parameter(torch.Tensor(self.veldim*lstm_units, lstm_units))
        self.bf = Parameter(torch.Tensor(self.veldim, lstm_units))
        
        # Output Gate
        self.Wox = Parameter(torch.Tensor(self.veldim, lstm_units))
        self.Woh = Parameter(torch.Tensor(self.veldim*lstm_units, lstm_units))
        self.bo = Parameter(torch.Tensor(self.veldim, lstm_units))
        
        #Calculating G
        self.Wgx = Parameter(torch.Tensor(self.veldim, lstm_units))
        self.Wgh = Parameter(torch.Tensor(self.veldim*lstm_units, lstm_units))
        self.bg = Parameter(torch.Tensor(self.veldim, lstm_units))
        
        #linear layer
        self.linear = Parameter(torch.Tensor(self.veldim, lstm_units))
        self.bl = Parameter(torch.Tensor(self.veldim))

        #Defining hidden units
        self.h, self.c = self.init_hidden()


        #Reparametrisation Variable
        self.R = torch.zeros((self.veldim*self.lstm_units, self.veldim)).cuda()
        cnt = -1
        for i in range(24):
            if i%4 == 0:
                cnt+=1
            self.R[i][cnt] = 1

    def init_hidden(self):
        cell = torch.randn(self.veldim, self.lstm_units)
        cell = Variable(cell.cuda())
        hidden = torch.randn(self.veldim, self.lstm_units)
        hidden = Variable(hidden.cuda())
        return hidden, cell

    def lstm(self, x, h, c):
        repeat = torch.mm(self.R, h)
        x = x.view(6, 1)

        it = torch.sigmoid((self.Wix*x) +
            torch.sum((self.Wih * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bi)
        ft = torch.sigmoid((self.Wfx*x) +
            torch.sum((self.Wfh * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bf)
        ot = torch.sigmoid((self.Wox*x) +
            torch.sum((self.Woh * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bo)

        cbar = torch.sigmoid((self.Wgx*x) +
            torch.sum((self.Wgh * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bg)
        c = ft * c + it*cbar
        h = ot*torch.sigmoid(c)

        out = torch.tanh(torch.sum(self.linear*h, dim = 1)+ self.bl)
        return out, h, c

    def forward(self, vel, Lsx, Lsy):
        for i in range(self.seqlen):
            if i==0:
                out,  h, c = self.lstm(vel, self.h, self.c)
                vels = out
            else :
                vels  = vels + out
                out,  h, c = self.lstm(out, h, c)
            self.v_interm.append(out.data.cpu().numpy())
        
        f_hat = torch.cat((torch.sum(Lsx0*vels,-1).unsqueeze(-1) , \
                torch.sum(Lsy*velt,-1).unsqueeze(-1)),-1)
        
        return f_hat 


if __name__ == '__main__':
    vs = Model().to("cuda:0")
    ve = torch.zeros(6).to("cuda:0")
    t1 = time.time()
    for i in range(100):
        vs(ve)
    print(t1 - time.time())
