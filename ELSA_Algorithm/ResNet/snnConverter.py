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

# optional Triton acceleration
try:
    import triton
    import triton.language as tl
except ImportError:  # runtime 环境没有 Triton 时自动退回到 PyTorch 实现
    triton = None
    tl = None

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
        self.first = False
        self.name = name

    @staticmethod
    def _can_use_triton(x: torch.Tensor) -> bool:
        return (
            (triton is not None)
            and torch.is_tensor(x)
            and x.is_cuda
        )

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
        if self.first and self.outSpike:
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
            # if len(x.shape) == 4:
            #     self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + (2**(self.N-1)) + (self.bias.reshape(1,-1,1,1) if self.bias is not None else 0.0)*(2**(self.N))
            # else:
            #     self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + (2**(self.N-1)) + (self.bias if self.bias is not None else 0.0)*(2**(self.N))
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + (2**(self.N-1))

        if self.t <= 5:
            if len(x.shape) == 4:
                self.q = self.q + (self.bias.reshape(1,-1,1,1) if self.bias is not None else 0.0)*(2**(self.N))/5
            else:
                # print(x.shape, self.q.shape, self.bias.shape)
                self.q = self.q + (self.bias if self.bias is not None else 0.0)*(2**(self.N))/5
                
        self.is_work = True

        # Triton 加速路径（仅在 CUDA + Triton 可用时启用）
        if self._can_use_triton(x):
            # 注意：根据原实现，self.acc_q 在这里已经是 round 过的整数，
            # 因此 kernel 中可以直接按整数处理 self.acc_q，而无需再做 round。
            q_flat = self.q.view(-1)
            acc_q_flat = self.acc_q.view(-1)
            cur_out_flat = self.cur_output.view(-1)
            x_flat = (x.detach() if torch.is_tensor(x) else x).view(-1)

            n_elements = x_flat.numel()
            BLOCK = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)

            st_bif_step_kernel[grid](
                q_flat,
                acc_q_flat,
                cur_out_flat,
                x_flat,
                n_elements,
                self.pos_max,
                self.neg_min,
                self.eps,
                float(2 ** self.N),
                BLOCK=BLOCK,
            )
        else:
            # 原始 PyTorch 逐元素实现
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
        if self.first and self.outSpike:
            self.accu1.append(self.cur_output[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
        
        return self.cur_output


# Triton kernel：对 STBIFNeuron 的单步状态更新做逐元素并行加速
if triton is not None:

    @triton.jit
    def st_bif_step_kernel(
        q_ptr,           # float32*
        acc_q_ptr,       # float32* （已为整数值）
        cur_out_ptr,     # same shape as q_ptr
        x_ptr,           # 输入 x
        n_elements,      # 参与计算的元素个数
        pos_max,         # 正脉冲累积上界
        neg_min,         # 负脉冲累积下界
        eps,             # eps
        two_power_N,     # 2**N
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        q = tl.load(q_ptr + offs, mask=mask, other=0.0)
        acc_q = tl.load(acc_q_ptr + offs, mask=mask, other=0.0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # q 累加输入
        q = q + x

        # 根据门限产生正负脉冲
        spike_pos = (q - two_power_N >= 0) & (acc_q < pos_max)
        spike_neg = (q < -eps) & (acc_q > neg_min)

        # cur_output：默认 0，有正/负脉冲时设为 ±1
        cur_out = tl.zeros_like(q)
        cur_out = tl.where(spike_pos, 1.0, cur_out)
        cur_out = tl.where(spike_neg, -1.0, cur_out)

        # 更新 acc_q
        acc_q_new = acc_q + cur_out

        # 根据脉冲更新 q
        q = tl.where(spike_pos, q - two_power_N, q)
        q = tl.where(spike_neg, q + two_power_N, q)

        tl.store(q_ptr + offs, q, mask=mask)
        tl.store(acc_q_ptr + offs, acc_q_new, mask=mask)
        tl.store(cur_out_ptr + offs, cur_out, mask=mask)


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
        self.first = False
        self.spike = True
        self.name = name
           
    def forward(self,x):
        self.t = self.t + 1   
        if self.first:
            self.accu.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                del self.accu

        x = self.m.m.m(x)*self.kernel_size*self.kernel_size
        x = self.neuron(x)

        if self.first:
            self.accu1.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                self.first = False
                save_input_for_bin_snn(torch.stack(self.accu1), glo.get_value("output_bin_snn_dir"),self.name+".out")
                del self.accu1
        return x


class SpikeResidualAddNew(t.nn.Module):
    def __init__(self,M,N, quan_out_fn:LsqQuan, name=f"Residual", T = 32, relu=True):
        super(SpikeResidualAddNew,self).__init__()
        self.neuron = STBIFNeuron(M,N, quan_out_fn.thd_pos, (0 if relu else quan_out_fn.thd_neg) , None, Add=True)
        self.accu1 = 0.0
        self.accu2 = 0.0
        self.accu3 = 0.0
        self.T = T
        self.t = 0
        self.accu = []
        self.accu1 = []
        self.accu2 = []
        self.first = False
        self.spike = True
        self.name = name

    def forward(self,input1,input2):

        self.t = self.t + 1        
        if self.first:
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

        if self.first:
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
        self.first = False
        self.spike = True
        self.name = name

    def forward(self,input1,input2):

        self.t = self.t + 1        
        if self.first:
            self.accu.append(input1[0].unsqueeze(0)+0)
            self.accu2.append(input2[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".input1")
                save_input_for_bin_snn(torch.stack(self.accu2), glo.get_value("output_bin_snn_dir"),self.name+".input2")
                del self.accu
                del self.accu2

        output = self.neuron(input1+input2)

        if self.first:
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
        self.first = False
        self.name = name
        self.directlyOut = directlyOut

    def forward(self,x):        

        self.t = self.t + 1        
        if self.first:
            self.accu.append(x[0].unsqueeze(0)+0)
            if self.t == self.T:
                save_fc_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                del self.accu
                self.first = False
                
        wx = t.nn.functional.linear(x, self.weight)
        if self.directlyOut:
            x = wx*self.M/(2**self.N)
            if self.t <= 5:
                x = x + self.bias/5
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
        self.first = False
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
        if self.first:
            self.accu.append(x[0].unsqueeze(0)+0.0)
            # print(x[0].shape,x[0].abs().sum())
            if self.t == self.T:
                # print("===========================Inference Spiking Conv2d==================================")
                # print("spiking input:",torch.stack(self.accu).sum(dim=0).abs().mean())
                save_input_for_bin_snn(torch.stack(self.accu), glo.get_value("output_bin_snn_dir"),self.name+".in")
                del self.accu

        wx = self._conv_forward(x, self.weight,bias=None)
        x = self.neuron(wx)

        if self.first:
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
            model._modules[name].spikeResidual = SpikeResidualAddNew(M=[model._modules[name].ResidualAdd.M1.data,model._modules[name].ResidualAdd.M2.data], \
                                                                     N=model._modules[name].ResidualAdd.N.data, \
                                                                     quan_out_fn=model._modules[name].ResidualAdd.m.quan_a_fn, \
                                                                     name=f"residual_act{index3}",T=TimeStep,relu=False)
            if len(child.conv) == 8:
                model._modules[name].conv[6] = SpikeInferConv2dFuseBN(m=child.conv[6], relu=False, name=f"Conv2dFuseBN_act{index3}",T=TimeStep)
                index3 = index3 + 1
                model._modules[name].conv[5] = nn.Identity()
                model._modules[name].spikeResidual = SpikeResidualAddNew(M=[model._modules[name].ResidualAdd.M1.data,model._modules[name].ResidualAdd.M2.data], \
                                                                     N=model._modules[name].ResidualAdd.N.data, \
                                                                     quan_out_fn=model._modules[name].ResidualAdd.m.quan_a_fn, \
                                                                     name=f"residual_act{index3}",T=TimeStep,relu=False)
            model._modules[name].ResidualAdd = None
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
        
    def forward(self,x,verbose=False):
        accu = None
        # self.midFeatureAccu = 0.0
        # self.midFeatureAccu1 = 0.0
        # self.inputAccu = 0.0
        if self.modelName == 'resnet18' or self.modelName == 'resnet50' or self.modelName == 'resnet101' or self.modelName == 'resnet34' or self.modelName == 'resnet12' or self.modelName == 'resnet20':
            x = self.model.module.conv1.m.m.quan_a_fn(x)/self.model.module.conv1.m.m.quan_a_fn.s
        elif self.modelName == 'mobilenetv2':
            x = self.model.module.features[0][0].m.m.quan_a_fn(x)/self.model.module.features[0][0].m.m.quan_a_fn.s
        elif self.modelName == 'vgg16':
            x = self.model.module.features[0].m.m.quan_a_fn(x)/self.model.module.features[0].m.m.quan_a_fn.s
        
        # x = get_subtensors(x,sample_grain=4)
        accu_list = []
        for i in range(self.max_timestep):
            # if i < x.shape[0]:
            #     input = x[i] + 0.0
            # else:
            #     input = torch.zeros(x.shape).to(x.device)
            if i < 5:
                input = x/5 + 0.0
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
            if verbose:
                accu_list.append(accu)
            if (i+1) % 100 == 0:
                print(f"SNN inference: {i}/{self.max_timestep}")
        if verbose:
            return accu, accu_list
        else:
            return accu


class IntegerSNNWarapperElastic(nn.Module):
    def __init__(self, ANNModel, modelName, bit, max_timestep=32, confidence_thr=0.9, true_stop=False):
        super(IntegerSNNWarapperElastic, self).__init__()
        self.model = ANNModel
        self.bit = bit
        self.max_timestep = max_timestep
        self.modelName = modelName
        self.confidence_thr = confidence_thr
        self.true_stop = true_stop

        # 统计早退预测与最终预测类别不一致的情况
        self.mismatch_num = 0  # 累计不一致样本数
        self.mismatch_den = 0  # 累计早退样本总数

        # 统计平均早退的 time-step（只对首次达到阈值的样本计数）
        self.early_exit_timestep_sum = 0  # 累计早退发生的时间步之和（1-based）
        self.early_exit_timestep_cnt = 0  # 累计早退样本数
        
        self.latency_per_time = [0.9266, 1.4139, 1.7989, 2.0959, 2.3306, 2.5218, 2.6953, 2.8463, 2.9559,
        3.0360, 3.1093, 3.1744, 3.2320, 3.2769, 3.3117, 3.3325, 3.3501, 3.3667,
        3.3810, 3.3921, 3.3997, 3.4062, 3.4114, 3.4131]

        # avg: 2.33 mismatch: 2.67%
        
        acc_list = [
            0.126, 1.886, 10.122, 19.522, 29.893, 40.401, 50.957, 60.137,
            66.819, 71.269, 73.671, 75.116, 75.670, 75.928, 75.958, 75.764,
            75.674, 75.540, 75.364, 75.122, 74.987, 74.897, 74.851, 74.857,
        ]

        self.total_latency = 0.0
        self.sample_count = 0

        spiking_inference_model_fusebn(self.model, max_timestep)

    def reset(self):
        reset_function(self.model)

    def forward(self, x, verbose: bool = False):
        accu = None

        if self.modelName == 'resnet18' or self.modelName == 'resnet101' or self.modelName == 'resnet50' or self.modelName == 'resnet34' or self.modelName == 'resnet12' or self.modelName == 'resnet20':
            x = self.model.module.conv1.m.m.quan_a_fn(x) / self.model.module.conv1.m.m.quan_a_fn.s
        elif self.modelName == 'mobilenetv2':
            x = self.model.module.features[0][0].m.m.quan_a_fn(x) / self.model.module.features[0][0].m.m.quan_a_fn.s
        elif self.modelName == 'vgg16':
            x = self.model.module.features[0].m.m.quan_a_fn(x) / self.model.module.features[0].m.m.quan_a_fn.s

        accu_list = []
        early_pred = None  # 记录第一次达到置信度阈值时的类别预测
        early_step = None  # 记录对应的 time-step（1-based）

        for i in range(self.max_timestep):
            if i < 5:
                input = x / 5 + 0.0
            else:
                input = torch.zeros(x.shape).to(x.device)

            output = self.model(input)

            if i == 0:
                accu = output + 0.0
            else:
                accu = accu + output

            if verbose:
                accu_list.append(accu)

            # 计算当前时间步的置信度，用于早退判定

            probs = F.softmax(accu, dim=1)

            conf_vals, pred_cls = probs.max(dim=1)

            # 只记录第一次达到置信度阈值时的预测类别和 time-step，用于后续与最终结果对比及统计
            # print("time-step",i,"confidence",conf_vals.max().item(),"conf_vals.max().item() >= self.confidence_thr",conf_vals.max().item() >= self.confidence_thr,"pred_cls",pred_cls)
            if early_pred is None and conf_vals.max().item() >= self.confidence_thr:
                early_pred = pred_cls.detach().clone()
                early_step = i + 1  # 将 time-step 记为 1-based 方便理解
                if self.true_stop:
                    for _ in range(self.max_timestep-i-1):
                        accu_list.append(accu)
                    break

            if (i + 1) % 100 == 0:
                print(f"SNN inference: {i}/{self.max_timestep}")

        # 计算最终预测类别
        if not self.true_stop:
            final_probs = F.softmax(accu, dim=1)
            _, final_pred = final_probs.max(dim=1)
            self.sample_count = self.sample_count + 1

            # 如果存在早退时刻的预测，则统计与最终预测的 mismatch 率和平均早退 time-step
            if early_pred is not None:
                mismatches = (early_pred != final_pred).sum().item()
                total = early_pred.numel()
                self.mismatch_num += mismatches
                self.mismatch_den += total

                mismatch_rate = self.mismatch_num / self.mismatch_den if self.mismatch_den > 0 else 0.0

                # 统计平均早退 time-step：对本 batch 中“发生早退”的样本记一次
                self.early_exit_timestep_sum += early_step
                self.early_exit_timestep_cnt += 1
                avg_early_step = (
                    self.early_exit_timestep_sum / self.early_exit_timestep_cnt
                    if self.early_exit_timestep_cnt > 0 else 0.0
                )
                self.total_latency = self.total_latency + self.latency_per_time[early_step - 1]
                
                print_str = f"IntegerSNNWarapperElastic mismatch rate: {mismatch_rate:.6f} " \
                    f"({self.mismatch_num}/{self.mismatch_den}), " \
                    f"avg early-exit time-step: {avg_early_step:.4f} " \
                    f"final prediction: {final_pred.item()} " \
                    f"early prediction: {early_pred.item()} " \
                    f"early time-step: {early_step} "
            else:
                self.total_latency = self.total_latency + self.latency_per_time[-1]
                print_str = f""
            print(print_str + f"avg latency: {self.total_latency/self.sample_count:.4f} ")
        else:
            self.sample_count = self.sample_count + 1
            if early_pred is not None:
                self.early_exit_timestep_sum += early_step
                self.early_exit_timestep_cnt += 1
                avg_early_step = (
                    self.early_exit_timestep_sum / self.early_exit_timestep_cnt
                    if self.early_exit_timestep_cnt > 0 else 0.0
                )
                self.total_latency = self.total_latency + self.latency_per_time[early_step - 1]
                print_str = f"avg early-exit time-step: {avg_early_step:.4f} "
            else:
                self.total_latency = self.total_latency + self.latency_per_time[-1]
                print_str = f""
            print(print_str + f"avg latency: {self.total_latency/self.sample_count:.4f} ")

        if verbose:
            return accu, accu_list
        else:
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


class IntegerSNNWarapperTest1(nn.Module):
    def __init__(self, ANNModel, modelName, bit ,max_timestep=32):
        super(IntegerSNNWarapperTest1,self).__init__()
        self.model = ANNModel
        self.bit = bit
        self.max_timestep = max_timestep
        self.modelName = modelName

        spiking_inference_model_fusebn(self.model,max_timestep)
    
    def reset(self):
        reset_function(self.model)
        
    def forward(self,x,verbose=False):
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
        accu_list = []
        for i in range(self.max_timestep):
            if i == 0:
                input = x
            else:
                input = torch.zeros(x[0].shape).to(x.device)
            
            output = self.model(input)
            # self.inputAccu = self.inputAccu + self.model.module.input
            # self.midFeatureAccu = self.midFeatureAccu + self.model.module.midFeature
            # self.midFeatureAccu1 = self.midFeatureAccu1 + self.model.module.midFeature1
            if i == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if verbose:
                accu_list.append(accu)
            if (i+1) % 100 == 0:
                print(f"SNN inference: {i}/{self.max_timestep}")
        if verbose:
            return accu, accu_list
        else:
            return accu

class IntegerSNNWarapperTest2(nn.Module):
    def __init__(self, ANNModel, modelName, bit ,max_timestep=32):
        super(IntegerSNNWarapperTest2,self).__init__()
        self.model = ANNModel
        self.bit = bit
        self.max_timestep = max_timestep
        self.modelName = modelName

        # spiking_inference_model_fusebn(self.model,max_timestep)
    
    def reset(self):
        reset_function(self.model)
        
    def forward(self,x1,x2,verbose=False):
        accu = None
        # self.midFeatureAccu = 0.0
        # self.midFeatureAccu1 = 0.0
        # self.inputAccu = 0.0
        if self.modelName == 'resnet18' or self.modelName == 'resnet50' or self.modelName == 'resnet34' or self.modelName == 'resnet12' or self.modelName == 'resnet20':
            x1 = self.model.module.conv1.m.m.quan_a_fn(x1)/self.model.module.conv1.m.m.quan_a_fn.s
            x2 = self.model.module.conv1.m.m.quan_a_fn(x2)/self.model.module.conv1.m.m.quan_a_fn.s
        elif self.modelName == 'mobilenetv2':
            x1 = self.model.module.features[0][0].m.m.quan_a_fn(x1)/self.model.module.features[0][0].m.m.quan_a_fn.s
            x2 = self.model.module.features[0][0].m.m.quan_a_fn(x2)/self.model.module.features[0][0].m.m.quan_a_fn.s
        elif self.modelName == 'vgg16':
            x1 = self.model.module.features[0].m.m.quan_a_fn(x1)/self.model.module.features[0].m.m.quan_a_fn.s
            x2 = self.model.module.features[0].m.m.quan_a_fn(x2)/self.model.module.features[0].m.m.quan_a_fn.s
        
        # x = get_subtensors(x,sample_grain=2**(self.bit-1))
        accu_list = []
        for i in range(self.max_timestep):
            if i == 0:
                input = x1
            elif i == 3:
                input = x2 - x1
            else:
                input = torch.zeros(x1[0].shape).to(x1.device)
            
            output = self.model(input)
            # self.inputAccu = self.inputAccu + self.model.module.input
            # self.midFeatureAccu = self.midFeatureAccu + self.model.module.midFeature
            # self.midFeatureAccu1 = self.midFeatureAccu1 + self.model.module.midFeature1
            if i == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if verbose:
                accu_list.append(accu)
            if (i+1) % 100 == 0:
                print(f"SNN inference: {i}/{self.max_timestep}")
        if verbose:
            return accu, accu_list
        else:
            return accu