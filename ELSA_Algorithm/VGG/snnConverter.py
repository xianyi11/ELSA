import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
import numpy as np
import scipy
from models import *
import glo

class STBIFNeuron(nn.Module):
    def __init__(self,M,N,pos_max, neg_min, bias, name="ST-BIF", outSpike=False, T = 32, Add=False):
        super(STBIFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.Add = Add
        self.M = M

        self.N = N
        self.is_work = False
        self.bias = bias
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.pos_max = pos_max
        self.neg_min = neg_min
        self.outSpike = outSpike
        self.spike = True
        
        self.eps = 0

        self.Add = Add
        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.accu2 = []
        self.first = True
        self.name = name

    def __repr__(self):
        return f"STBIFNeuron(pos_max={self.pos_max}, neg_min={self.neg_min}, M={self.M}, N={self.N}, Add={self.Add})"

    def reset(self):
        # print("STBIFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        self.neg_spike_position = None
        self.accu = []
        self.accu1 = []
        self.accu2 = []
        self.t = 0

    def forward(self,input,verbose=False):
        # self.T = self.T + 1
        # if verbose:
        #     self.accu = self.accu + input
        #     if self.T == 32:
        #         print("SNN input",self.accu.mean())
        self.t = self.t + 1        
        if glo.get_value("record_inout") and self.first and self.outSpike:
            self.accu.append(input[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
        
        if self.Add:
            x = (float(self.M[0]))*input[0] + (float(self.M[1]))*input[1]
        else:
            x = (float(self.M))*input
        # if verbose:
        #     self.accu1 = self.accu1 + x
        #     if self.T == 32:
        #         print("SNN quantized input",self.accu1[0:4,0,0,:])
        #         print("quantized bias", self.bias)
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

        # if verbose:
        #     self.accu2= self.accu2 + self.cur_output
        #     if self.T == 32:
        #         print("quantized output",self.accu2[0:4,0,0,:])
        #         print("quantized output",self.accu2.mean())
        
        # print("self.cur_output",self.cur_output)
        if glo.get_value("record_inout") and self.first and self.outSpike:
            self.accu1.append(self.cur_output[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
        
        return self.cur_output


class SpikeInferAvgPool(t.nn.Module):
    def __init__(self, m:QuanInferAvgPool, name=f"AvgPool", T = 32):
        super(SpikeInferAvgPool,self).__init__()
        self.m = m
        self.M = m.M
        self.N = m.N
        self.thd_pos = m.thd_pos
        self.thd_neg = m.thd_neg
        self.kernel_size = self.m.kernel_size
        self.neuron = STBIFNeuron(self.M,self.N, self.thd_pos, self.thd_neg, None)
        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.first = True 
        self.spike = True
        self.name = name
           
    def forward(self,x):
        self.t = self.t + 1   
        if glo.get_value("record_inout") and self.first:
            self.accu.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                del self.accu

        x = self.m.m.m(x)*self.kernel_size*self.kernel_size
        x = self.neuron(x)

        if glo.get_value("record_inout") and self.first:
            self.accu1.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                del self.accu1
        return x


class SpikeResidualAddNew(t.nn.Module):
    def __init__(self,M,N, quan_out_fn:LsqQuan, name=f"Residual", T = 32):
        super(SpikeResidualAddNew,self).__init__()
        self.neuron = STBIFNeuron(M,N, quan_out_fn.thd_pos, 0, None, Add=True)
        self.accu1 = 0.0
        self.accu2 = 0.0
        self.accu3 = 0.0
        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.accu2 = []
        self.first = True
        self.spike = True
        self.name = name

    def forward(self,input1,input2):

        self.t = self.t + 1        
        if glo.get_value("record_inout") and self.first:
            self.accu.append(input1[0].unsqueeze(0)+0)
            self.accu2.append(input2[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".input1")
                save_input_for_bin_snn(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+".input2")
                # print("===============================Inference Spiking Addition===============================")
                # print("spiking input1:",torch.stack(self.accu).sum(dim=0).abs().mean())
                # print("spiking input2:",torch.stack(self.accu2).sum(dim=0).abs().mean())
                del self.accu
                del self.accu2

        output = self.neuron((input1,input2))

        if glo.get_value("record_inout") and self.first:
            self.accu1.append(output[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                # print("spiking output:",torch.stack(self.accu1).sum(dim=0).abs().mean())
                del self.accu1

        return output


class SpikeResidualAdd(t.nn.Module):
    def __init__(self, quan_out_fn:LsqQuan, name=f"Residual", T = 32):
        super(SpikeResidualAdd,self).__init__()
        self.neuron = STBIFNeuron(1,0, quan_out_fn.thd_pos, quan_out_fn.thd_neg, None)
        self.accu1 = 0.0
        self.accu2 = 0.0
        self.accu3 = 0.0
        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.accu2 = []
        self.first = True
        self.spike = True
        self.name = name

    def forward(self,input1,input2):

        self.t = self.t + 1        
        if glo.get_value("record_inout") and self.first:
            self.accu.append(input1[0].unsqueeze(0)+0)
            self.accu2.append(input2[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".input1")
                save_input_for_bin_snn(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+".input2")
                del self.accu
                del self.accu2

        output = self.neuron(input1+input2)

        if glo.get_value("record_inout") and self.first:
            self.accu1.append(output[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                del self.accu1

        return output

class SpikeInferLinear(t.nn.Linear):
    def __init__(self, m: QuanInferLinear, name=f"act", T = 32, directlyOut = False):
        assert type(m) == QuanInferLinear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)        

        self.m = m  
        self.thd_neg = m.thd_neg
        self.thd_pos = m.thd_pos
        self.spike = True

        self.weight = m.weight
        self.bias = m.bias
            #print(self.bias)
        self.M = m.M
        self.N = m.N
        self.neuron = STBIFNeuron(self.M,self.N, self.thd_pos, self.thd_neg, self.bias)

        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.first = True
        self.name = name
        self.directlyOut = directlyOut

    def forward(self,x):        

        self.t = self.t + 1        
        if glo.get_value("record_inout") and self.first:
            self.accu.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_fc_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                del self.accu
                self.first = False
                
        wx = t.nn.functional.linear(x, self.weight)
        if self.directlyOut:
            x = wx*self.M/(2**self.N)
            if self.t == 1:
                x = x + self.bias
        else:
            x = self.neuron(wx)
        
        # if self.first:
        #     self.accu1.append(x[0].unsqueeze(0)+0)
        #     if self.t == self.T:
        #         self.first = False
        #         output_accu = torch.stack(self.accu1,dim=0)
        #         save_fc_input_for_bin_snn(output_accu, glo.get_value("output_bin_snn_dir"),self.name+".out")
        #         del self.accu1                
        if self.t == self.T:
            self.t = 0
        
        return x

class SpikeInferConv2dFuseBN(t.nn.Conv2d):
    def __init__(self, m: QuanInferConv2dFuseBN, relu = True, name=f"act", T = 32):
        assert type(m) == QuanInferConv2dFuseBN
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=True if m.bias is not None else False,
                    padding_mode=m.padding_mode)

        self.m = m  
        self.thd_neg = m.thd_neg
        self.thd_pos = m.thd_pos
        self.spike = True

        self.weight = m.weight
        self.bias = m.bias
            #print(self.bias)
        self.M = m.M
        self.N = m.N
        self.is_first = m.is_first
        self.first = True
        if relu == True:
            self.neuron = STBIFNeuron(self.M,self.N, self.thd_pos, 0, self.bias)
        else:
            self.neuron = STBIFNeuron(self.M,self.N, self.thd_pos, self.thd_neg, self.bias)
        
        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.name = name
        
    def forward(self,x):
        self.t = self.t + 1
        if glo.get_value("record_inout") and self.first:
            self.accu.append(x[0].unsqueeze(0)+0.0)
            # print(x[0].shape,x[0].abs().sum())
            if self.t == self.T:
                # print("===========================Inference Spiking Conv2d==================================")
                # print("spiking input:",torch.stack(self.accu).sum(dim=0).abs().mean())
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                del self.accu

        wx = self._conv_forward(x, self.weight,bias=None)
        x = self.neuron(wx)

        if glo.get_value("record_inout") and self.first:
            self.accu1.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                # print("spiking output:",torch.stack(self.accu1).sum(dim=0).abs().mean())
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                del self.accu1
        return x




index3 = 0
def spiking_inference_model_fusebn(model, TimeStep):
    global index3
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, BasicBlock) or isinstance(child, BasicBlockCifar) or isinstance(child, BasicBlockResNet12):
            model._modules[name].conv1 = SpikeInferConv2dFuseBN(m=child.conv1, relu=True, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            model._modules[name].conv2 = SpikeInferConv2dFuseBN(m=child.conv2, relu=False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            if isinstance(child.downsample,t.nn.Sequential):
                child.downsample[0] = SpikeInferConv2dFuseBN(m=child.downsample[0], relu=False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
                index3 = index3 + 1
            # else:
            #     child.downsample = STBIFNeuron(child.M,child.N, child.conv2.m.m.quan_out_fn.thd_pos, 0.0, None,outSpike=True,T=TimeStep)
            is_need = True
            model._modules[name].relu1 = nn.Identity()
            model._modules[name].relu2 = nn.Identity()
            model._modules[name].relu3 = nn.Identity()
            model._modules[name].spikeResidual = SpikeResidualAddNew(M=[model._modules[name].ResidualAdd.M1.data,model._modules[name].ResidualAdd.M2.data], \
                                                                     N=model._modules[name].ResidualAdd.N.data, \
                                                                     quan_out_fn=model._modules[name].ResidualAdd.m.quan_a_fn, \
                                                                     name=f"residual_act{index3}",T=TimeStep)
            model._modules[name].ResidualAdd = None
            
        elif isinstance(child, Bottleneck):
            model._modules[name].conv1 = SpikeInferConv2dFuseBN(m=child.conv1, relu=True, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            model._modules[name].conv2 = SpikeInferConv2dFuseBN(m=child.conv2, relu=True, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            model._modules[name].conv3 = SpikeInferConv2dFuseBN(m=child.conv3, relu=False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1

            if isinstance(child.downsample,t.nn.Sequential):
                child.downsample[0] = SpikeInferConv2dFuseBN(m=child.downsample[0], relu=False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
                index3 = index3 + 1

            # else:
            #     child.downsample = STBIFNeuron(child.M,child.N, child.conv3.m.m.quan_out_fn.thd_pos, 0.0, None, outSpike=True,T=TimeStep)

            model._modules[name].relu1 = nn.Identity()
            model._modules[name].relu2 = nn.Identity()
            model._modules[name].relu3 = nn.Identity()
            model._modules[name].relu4 = nn.Identity()
            # model._modules[name].spikeResidual = SpikeResidualAdd(quan_out_fn=child.conv3.m.m.quan_out_fn,name=f"residual_act{index3}",T=TimeStep)
            model._modules[name].spikeResidual = SpikeResidualAddNew(M=[model._modules[name].ResidualAdd.M1.data,model._modules[name].ResidualAdd.M2.data], \
                                                                     N=model._modules[name].ResidualAdd.N.data, \
                                                                     quan_out_fn=model._modules[name].ResidualAdd.m.quan_a_fn, \
                                                                     name=f"residual_act{index3}",T=TimeStep)
            model._modules[name].ResidualAdd = None

            is_need = True
        
        elif isinstance(child, InvertedResidual):
            model._modules[name].conv[0] = SpikeInferConv2dFuseBN(m=child.conv[0], relu=True, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            model._modules[name].conv[3] = SpikeInferConv2dFuseBN(m=child.conv[3], relu=True if len(child.conv) == 8 else False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            model._modules[name].conv[2] = nn.Identity()
            child.downsample = STBIFNeuron(child.M,child.N, child.conv[3].m.m.quan_out_fn.thd_pos, child.conv[3].m.m.quan_out_fn.thd_neg, None, outSpike=True,T=TimeStep)
            model._modules[name].spikeResidual = SpikeResidualAdd(quan_out_fn=child.conv[3].m.m.quan_out_fn,name=f"residual_act{index3}",T=TimeStep)
            if len(child.conv) == 8:
                model._modules[name].conv[6] = SpikeInferConv2dFuseBN(m=child.conv[6], relu=False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
                index3 = index3 + 1
                model._modules[name].conv[5] = nn.Identity()
                child.downsample = STBIFNeuron(child.M,child.N, child.conv[6].m.m.quan_out_fn.thd_pos, child.conv[6].m.m.quan_out_fn.thd_neg, None, outSpike=True,T=TimeStep)
                model._modules[name].spikeResidual = SpikeResidualAdd(quan_out_fn=child.conv[6].m.m.quan_out_fn,name=f"residual_act{index3}",T=TimeStep)

            is_need = True
                        
        elif isinstance(child, QuanInferLinear):
            model._modules[name] = SpikeInferLinear(child, name=f"Linear_act{index3}",T=TimeStep, directlyOut=True)
            index3 = index3 + 1
            is_need = True
        elif isinstance(child, QuanInferAvgPool):
            model._modules[name] = SpikeInferAvgPool(m=child, name=f"AvgPool_act{index3}",T=TimeStep)
            index3 = index3 + 1
            is_need = True
        elif isinstance(child, QuanInferConv2dFuseBN):
            model._modules[name] = SpikeInferConv2dFuseBN(m=child, relu=True, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
            index3 = index3 + 1
            is_need = True
        elif isinstance(child, nn.ReLU):
            model._modules[name] = nn.Identity()
            index3 = index3 + 1
            is_need = True
        if not is_need:
            spiking_inference_model_fusebn(child,TimeStep)


def set_snn_save_name(model, calOrder):
    children = list(model.named_modules())
    for name, child in children:
        if isinstance(child, SpikeInferLinear):
            child.name = name
            calOrder.append(name)
        if isinstance(child, SpikeInferConv2dFuseBN):
            child.name = name
            calOrder.append(name)
        if isinstance(child, SpikeInferAvgPool):
            child.name = name
            calOrder.append(name)
        if isinstance(child, SpikeResidualAdd):
            child.name = name
            calOrder.append(name)
        if name.count("downsample") and isinstance(child, STBIFNeuron):
            child.name = name
            calOrder.append(name)


def reset_function(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, STBIFNeuron):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_function(child)            

def get_subtensors(tensor,sample_grain=255):
    input = tensor.clone()
    for i in range(int(sample_grain)):
        output = input.clone()
        output[output>0] = 1
        output[output<0] = -1
        input = input - output
        output = (output).unsqueeze(0)
        if i == 0:
            accu = output
        else:
            accu = torch.cat((accu,output),dim=0)
    # print("accu.shape",accu.shape)
    return accu

class IntegerSNNWarapper(nn.Module):
    def __init__(self, ANNModel, modelName, bit ,max_timestep=32):
        super(IntegerSNNWarapper,self).__init__()
        self.model = ANNModel
        self.bit = bit
        self.max_timestep = max_timestep
        self.modelName = modelName

        spiking_inference_model_fusebn(self.model,max_timestep)
    
    def reset(self):
        reset_function(self.model)
        
    def forward(self,x):
        accu = None
        # self.midFeatureAccu = 0.0
        # self.midFeatureAccu1 = 0.0
        # self.inputAccu = 0.0
        if self.modelName == 'resnet18' or self.modelName == 'resnet50' or self.modelName == 'resnet34' or self.modelName == 'resnet12' or self.modelName == 'resnet20':
            x = self.model.module.conv1.m.m.quan_a_fn(x)/self.model.module.conv1.m.m.quan_a_fn.s
        elif self.modelName == 'mobilenetv2':
            x = self.model.module.features[0][0].m.m.quan_a_fn(x)/self.model.module.features[0][0].m.m.quan_a_fn.s
        elif self.modelName == 'vgg16':
            x = self.model.module.features[0].m.m.quan_a_fn(x)/self.model.module.features[0].m.m.quan_a_fn.s
        
        # x = get_subtensors(x,sample_grain=2**(self.bit-1))
        for i in range(self.max_timestep):
            if i  == 0:
                input = x + 0.0
            else:
                input = torch.zeros(x.shape).to(x.device)
            
            output = self.model(input)
            # self.inputAccu = self.inputAccu + self.model.module.input
            # self.midFeatureAccu = self.midFeatureAccu + self.model.module.midFeature
            # self.midFeatureAccu1 = self.midFeatureAccu1 + self.model.module.midFeature1
            if i == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if (i+1) % 100 == 0:
                print(f"SNN inference: {i}/{self.max_timestep}")            
        return accu

def save_for_bin_snn(model,dir):
    children = list(model.named_modules())
    for name, child in children:
        if isinstance(child, BasicBlock) or isinstance(child, Bottleneck):
            if child.M is not None:
                M = int(child.M.data.item())
                M_binfile = open(f'{dir}/{name}_M.bin','wb')
                M_binfile.write(M.to_bytes(length=4,byteorder='big',signed =True))
                M_binfile.close()
                            
                N = int(child.N.data.item())
                N_binfile = open(f'{dir}/{name}_N.bin','wb')
                N_binfile.write(N.to_bytes(length=1,byteorder='big',signed =True))
                N_binfile.close()

        if isinstance(child, SpikeInferLinear):
            # print("Wrtie bin: QuanInferLinear")
            has_spike = torch.abs(child.weight).sum()
            assert has_spike != 0, "some errors in input, all the element are 0!!!"

            weight_list = child.weight.data.tolist()
            weight_binfile = open(f'{dir}/{name}_weight_N1={child.weight.shape[0]}_N2={child.weight.shape[1]}.bin','wb')
            for i in range(len(weight_list)):
                for j in range(len(weight_list[0])):
                    weight_binfile.write(int(round(float(weight_list[i][j]))).to_bytes(length=1,byteorder='big',signed =True))
            weight_binfile.close()

            bias_list = child.bias.data.tolist()
            # print("fc:",bias_list)
            bias_binfile = open(f'{dir}/{name}_bias_N={child.bias.shape[0]}.bin','wb')
            for i in range(len(bias_list)):
                bias_binfile.write(int(round(float(bias_list[i]))).to_bytes(length=1,byteorder='big',signed =True))
            bias_binfile.close()
            
            M = int(child.M.data.item())
            M_binfile = open(f'{dir}/{name}_M.bin','wb')
            M_binfile.write(M.to_bytes(length=4,byteorder='big',signed =True))
            M_binfile.close()
                        
            N = int(child.N.data.item())
            N_binfile = open(f'{dir}/{name}_N.bin','wb')
            N_binfile.write(N.to_bytes(length=1,byteorder='big',signed =True))
            N_binfile.close()
        
        if isinstance(child, SpikeInferConv2dFuseBN):
            # print("Wrtie bin: QuanInferConv2d")
            has_spike = torch.abs(child.weight).sum()
            assert has_spike != 0, "some errors in input, all the element are 0!!!"
            weight_list = child.weight.data.tolist()
            weight_binfile = open(f'{dir}/{name}_weight_C1={child.weight.shape[0]}_C2={child.weight.shape[1]}_KH={child.weight.shape[2]}_KW={child.weight.shape[3]}.bin','wb')
            C_out,C_in,KH,KW = child.weight.shape
            for i in range(C_out):
                for j in range(C_in):
                    for n in range(KH):
                        for m in range(KW):
                            weight_binfile.write(int(round(float(weight_list[i][j][n][m]))).to_bytes(length=1,byteorder='big',signed =True))
            weight_binfile.close()

            # has_spike = torch.abs(child.bias).sum()
            # assert has_spike != 0, "some errors in bias, all the element are 0!!!"
            bias_list = child.bias.data.tolist()
            bias_binfile = open(f'{dir}/{name}_bias_N={child.bias.shape[0]}.bin','wb')
            # print("Conv:",bias_list)
            for i in range(len(bias_list)):
                bias_binfile.write(int(round(float(bias_list[i]))).to_bytes(length=1,byteorder='big',signed =True))
            bias_binfile.close()
            
            M = int(child.M.data.item())
            M_binfile = open(f'{dir}/{name}_M.bin','wb')
            M_binfile.write(M.to_bytes(length=4,byteorder='big',signed =True))
            M_binfile.close()
                        
            N = int(child.N.data.item())
            N_binfile = open(f'{dir}/{name}_N.bin','wb')
            N_binfile.write(N.to_bytes(length=1,byteorder='big',signed =True))            
            N_binfile.close()
        
        if isinstance(child, SpikeInferAvgPool) or name.count("downsample") and isinstance(child, STBIFNeuron) and name.count("neuron") == 0:
            
            if child.M is not None:
                M = int(child.M.data.item())
                M_binfile = open(f'{dir}/{name}_M.bin','wb')
                M_binfile.write(M.to_bytes(length=4,byteorder='big',signed =True))
                M_binfile.close()
                            
                N = int(child.N.data.item())
                N_binfile = open(f'{dir}/{name}_N.bin','wb')
                N_binfile.write(N.to_bytes(length=1,byteorder='big',signed =True))            
                N_binfile.close()
            
            

def save_input_for_bin_snn(input,dir,name):
    T,B,C,H,W = input.shape[0],input.shape[1],input.shape[2],input.shape[3],input.shape[4]
    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    local_rank = torch.distributed.get_rank()
    # print(T,B,C,H,W,input.abs().sum())
    # print("max_value",input.max(),"min_value",input.min())
    
    if local_rank == 0:
        input_list = input.tolist()
        input_binfile = open(f'{dir}/act_{name}_T={T}_B={B}_C={C}_H={H}_W={W}.bin','wb')
        for t in range(T):
            for i in range(B):
                for j in range(C):
                    for n in range(H):
                        for m in range(W):
                            input_binfile.write(int(round(float(input_list[t][i][j][n][m]))).to_bytes(length=1,byteorder='big',signed =True))
        input_binfile.close()
    
def save_fc_input_for_bin_snn(input,dir,name):
    T,B,N = input.shape

    has_spike = torch.abs(input).sum()
    assert has_spike != 0, "some errors in input, all the element are 0!!!"
    # print(input.shape)
    local_rank = torch.distributed.get_rank()
    
    if local_rank == 0:
        input_list = input.tolist()
        input_binfile = open(f'{dir}/act_{name}_T={T}_B={B}_N={N}.bin','wb')
        for t in range(T):
            for i in range(B):
                for j in range(N):
                    input_binfile.write(int(round(float(input_list[t][i][j]))).to_bytes(length=1,byteorder='big',signed =True))
        input_binfile.close()