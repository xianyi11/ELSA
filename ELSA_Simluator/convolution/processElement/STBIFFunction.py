import torch
import torch.nn as nn

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

class STBIFNeuron(nn.Module):
    def __init__(self,M,N,pos_max, neg_min, bias, name="ST-BIF", outSpike=False):
        super(STBIFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.M = M
        self.N = N
        self.is_work = False
        self.bias = bias
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.pos_max = pos_max
        self.neg_min = neg_min
        self.outSpike = outSpike
        
        self.eps = 0

        self.T = 32
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.accu2 = []
        self.first = True
        self.name = name

    def __repr__(self):
            return f"STBIFNeuron(pos_max={self.pos_max}, neg_min={self.neg_min}, M={self.M}, N={self.N})"

    def reset(self):
        # print("STBIFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self,input,verbose=False):

        x = (float(self.M))*input

        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            if len(x.shape) == 4:
                self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + (2**(self.N-1)) + (self.bias.reshape(1,-1,1,1) if self.bias is not None else 0.0)*(2**(self.N))
            else:
                self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + (2**(self.N-1)) + (self.bias if self.bias is not None else 0.0)*(2**(self.N))

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - float(2**(self.N)) >= 0) & (self.acc_q < self.pos_max)
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - (2**(self.N))
        self.q[neg_spike_position] = self.q[neg_spike_position] + (2**(self.N))

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False


        return self.cur_output
