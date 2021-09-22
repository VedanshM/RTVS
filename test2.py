import torch
import torch.nn as nn
from torch.autograd import Variable
import time
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
        # Input Gate
        self.Wix = torch.randn((lstm_units, self.veldim), requires_grad=True)
        self.Wih = torch.randn((self.veldim*lstm_units, lstm_units), requires_grad=True)
        self.bi = torch.zeros((self.veldim, lstm_units), requires_grad=True)
        
        # Forget Gate
        self.Wfx = torch.randn((lstm_units, self.veldim), requires_grad=True)
        self.Wfh = torch.randn((self.veldim*lstm_units, lstm_units), requires_grad=True)
        self.bf = torch.zeros((self.veldim, lstm_units), requires_grad=True)
        
        # Output Gate
        self.Wox = torch.randn((lstm_units, self.veldim), requires_grad=True)
        self.Woh = torch.randn((self.veldim*lstm_units, lstm_units), requires_grad=True)
        self.bo = torch.zeros((self.veldim, lstm_units), requires_grad=True)
        
        #Calculating G
        self.Wgx = torch.randn((lstm_units, self.veldim), requires_grad=True)
        self.Wgh = torch.randn((self.veldim*lstm_units, lstm_units), requires_grad=True)
        self.bg = torch.zeros((self.veldim, lstm_units), requires_grad=True)
        
        #linear layer
        self.linear = torch.randn((self.veldim, lstm_units))
        self.bl = torch.randn(self.veldim)
        
        #Defining hidden units
        self.h, self.c = self.init_hidden()
        
        
        #Reparametrisation Variable
        self.R = torch.zeros((self.veldim*self.lstm_units, self.veldim))
        cnt = -1
        for i in range(24):
            if i%4 == 0:
                cnt+=1
            self.R[i][cnt] = 1

    def init_hidden(self):
        cell = torch.randn(self.veldim, self.lstm_units)
        cell = Variable(cell)
        hidden = torch.randn(self.veldim, self.lstm_units)
        hidden = Variable(hidden)
        return hidden, cell
        
    def lstm(self, x, h, c):
        
        repeat = torch.mm(self.R, h)
        it = torch.sigmoid((self.Wix*x).T +
            torch.sum((self.Wih * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bi)
        ft = torch.sigmoid((self.Wfx*x).T +
            torch.sum((self.Wfh * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bf)
        ot = torch.sigmoid((self.Wox*x).T +
            torch.sum((self.Woh * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bo)
        
        cbar = torch.sigmoid((self.Wgx*x).T +
            torch.sum((self.Wgh * repeat), dim=1).view(self.veldim, self.lstm_units)
            + self.bg)
        c = ft * c + it*cbar
        h = ot*torch.sigmoid(c)
        
        out = torch.tanh(torch.sum(self.linear*h, dim = 1)+self.bl)
        return out, h, c

    def forward(self, vel):
        for i in range(self.seqlen):
            if i==0:
                out,  h, c = self.lstm(vel, self.h, self.c)
            else :
                out,  h, c = self.lstm(out, h, c)
        
        return out
        
        
if __name__ == '__main__':
    vs = Model()
    t1 = time.time()
    for i in range(100):
        vs(torch.zeros(6))
    print(time.time() -t1)
    print(vs.parameters())
    from model import VisualServoingLSTM
    x = VisualServoingLSTM('LSTM')
    print(x.parameters())
