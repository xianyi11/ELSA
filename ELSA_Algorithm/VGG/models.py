import torch
import torch as t
from resnet import BasicBlock, Bottleneck
from resnet20 import BasicBlockCifar
from resnet12 import BasicBlockResNet12
from mobilenetV2 import InvertedResidual
import glo
import torch.nn.functional as F
import numpy as np
import scipy

# using glorot initialization
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)

def conv_block(in_channels,out_channels):
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2,stride=2)
    )
    return conv

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad



class LsqQuan(t.nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__()
        self.bit = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(torch.tensor(1.0),requires_grad=True)

    def __repr__(self):
        return f"LsqQuan(thd_pos={self.thd_pos}, thd_neg={self.thd_neg}, s={self.s.data}, per_channel={self.per_channel})"

    
    def init_from(self, x, weight=True, *args, **kwargs):
        # threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=300, eps=1e-10)
        # self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        if self.bit >= 16:
            return 
        
        if weight == True:
            if self.per_channel:
                self.s = t.nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            else:
                self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        else:
            pass
            # if self.bit > 4:
            # self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            # print(self.s)
            threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.thd_pos), n_trial=1000, eps=1e-10)
            self.s.data = torch.tensor(threshold / (self.thd_pos),dtype=torch.float32).cuda()
        #     # print(torch.quantile(x.detach().mean(dim=0),0.95), torch.quantile(x.detach().mean(dim=0),0.05))
        #     self.s = t.nn.Parameter( (torch.quantile(x.detach().mean(dim=0),0.99)-torch.quantile(x.detach().mean(dim=0),0.01)) / ((self.thd_pos - self.thd_neg)))
        #     print(self.s)


    def forward(self, x, clip=True):
        if self.bit >= 16:
            self.s.data = torch.tensor(1.0).to(x.device)
            return x
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        s_scale = grad_scale(self.s, s_grad_scale)
        # print(s_scale,s_scale.grad)
        # print("self.thd_neg",self.thd_neg, "self.thd_pos", self.thd_pos)
        x = x / s_scale
        if clip:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

# def cal_l1_loss_full(A):
#     l1_loss = 0
#     granurality = 256
#     print(A.shape)

#     if A.shape[-1] % granurality != 0:
#         pad_len = int((int(A.shape[-1]/granurality)+1)*granurality - A.shape[-1])
#         A = F.pad(A,(0,0,0,pad_len),mode='constant',value=0)
    
#     # print("A.shape:",A.shape)
#     A_gran = A.reshape(int(A.shape[0]/granurality),granurality,A.shape[1])

#     loss_groups = torch.norm(A_gran.float(), p=2, dim=(1), keepdim=False, out=None, dtype=None)
#     l1_loss += loss_groups.sum()
#     return l1_loss

def LsqQuanFunc(x,thd_pos,thd_neg,s):
    
    s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)        
    s_scale = grad_scale(s, s_grad_scale)
    # print(s_scale,s_scale.grad)
    # print("self.thd_neg",self.thd_neg, "self.thd_pos", self.thd_pos)
    x = x / s_scale
    x = t.clamp(x, thd_neg, thd_pos)
    x = round_pass(x)
    x = x * s_scale
    return x

def reshape_to_activation(inputs):
    return inputs.reshape(1, -1, 1, 1)

def reshape_to_weight(inputs):
    return inputs.reshape(-1, 1, 1, 1)

def reshape_to_bias(inputs):
    return inputs.reshape(-1)


class QuanAvgPool(t.nn.Module):
    def __init__(self,m, quan_out_fn):
        super(QuanAvgPool,self).__init__()
        assert isinstance(m, t.nn.AvgPool2d) or isinstance(m, t.nn.AdaptiveAvgPool2d), "average pooling!!!"
        self.m = m
        self.quan_out_fn = quan_out_fn
        self.is_init = False
    def forward(self,x):
        if self.is_init == False:
            x = self.m(x)
            self.quan_out_fn.init_from(x,weight=False)
            self.is_init = True
            return x

        x = self.m(x)
        x = self.quan_out_fn(x)
        # print("train AvgPool output:",(x/self.quan_out_fn.s).abs()[0,0,0,:])
        return x
        
class QuanInferAvgPool(t.nn.Module):
    def __init__(self,m:QuanAvgPool, last_quan_out_fn, name):
        super(QuanInferAvgPool,self).__init__()
        self.m = m
        self.last_quan_out_fn = last_quan_out_fn
        self.kernel_size = self.m.m.kernel_size
        self.s = last_quan_out_fn.s/(m.quan_out_fn.s*self.kernel_size*self.kernel_size)
        self.thd_pos = m.quan_out_fn.thd_pos
        self.thd_neg = m.quan_out_fn.thd_neg
        self.name = name
        self.first = True
        N,M = get_M_N(self.s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)

        # *self.m.kernel_size*self.m.kernel_size
    def forward(self,x):
        if self.first:
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"in")        

        # print("Infer: pooling",self.m.m(x).mean())
        # print("Infer: quantized pooling",((float(self.M)/float(2**self.N))*self.m.m(x))[0:4,0,0,:])
        x = torch.round((float(self.M)/float(2**self.N))*(self.m.m(x)*self.kernel_size*self.kernel_size))
        x = torch.clip(x,max=self.thd_pos,min=self.thd_neg)
        # print("Infer: quantized output",x[0:4,0,0,:])
        # print("Infer: quantized output",x.mean())

        if self.first:
            self.first = False
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"out")
        # print("Infer AvgPool output:", x.abs()[0,0,0,:])
        
        return x


def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    '''
    This function collect the activation data and find the optimized clipping 
    threshold using KL_div as metric. Since this method is originated from 
    post-training quantization which adopted in Tensor-RT, we keep the number of 
    bits here.
    Args:
        data(numpy array): activation data
        n_bit(int): 
        n_trial(int): the searching steps.
        eps(float): add eps at the average bin step for numberical stability.
    
    '''

    n_lvl = quantization_level+1  # quantization levels
    n_half_lvls = (quantization_level+1)//2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(data.flatten(),
                                  bins=np.linspace(-data_max,
                                                   data_max,
                                                   num=n_bin_edge))

    mid_idx = int((len(hist)) / 2)
    start_idx = 100
    # log the threshold and corresponding KL-divergence
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(hist[mid_idx - i * n_half_lvls:mid_idx +
                                i * n_half_lvls])
        # merge the outlier
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()
        # perform quantization: bins merge and expansion
        reshape_dim = int(len(ref_dist) / n_lvl)
        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        # merge bins for quantization
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = (ref_dist_reshape != 0
                        )  # obtain the mask of non-zero bins
        # in each merged large bin, get the average bin_count
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        # expand the merged bins back
        expand_bin_count = np.expand_dims(average_bin_count,
                                          axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(candidate_dist / candidate_dist.sum(),
                                     ref_dist / ref_dist.sum())
        #log threshold and KL-divergence
        current_th = np.abs(
            bin_edge[mid_idx - i * n_half_lvls])  # obtain current threshold
        kl_result[i -
                  start_idx, 0], kl_result[i -
                                           start_idx, 1] = current_th, kl_div

    # based on the logged kl-div result, find the threshold correspond to the smallest kl-div
    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    print("Threshold calibration of current layer finished!")

    return th_sel



class QuanConv2dFuseBN(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, quan_out_fn=None,momentum=0.9, eps=1e-5,is_first=False):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.m = m
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.quan_a_fn = quan_a_fn
        
        self.momentum = momentum
        self.eps = eps
        self.register_parameter('beta', t.nn.Parameter(torch.zeros(m.out_channels)))
        self.register_parameter('gamma', t.nn.Parameter(torch.ones(m.out_channels)))
        self.register_buffer('running_mean', torch.zeros(m.out_channels))
        self.register_buffer('running_var', torch.ones(m.out_channels))

        self.weight = t.nn.Parameter(m.weight.detach())
        # print(self.weight.mean())
        self.quan_w_fn.init_from(m.weight)
        # self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None
        self.is_init = False
        self.is_first = is_first
        # if self.is_first:
        #     self.quan_a_fn.init_from(m.weight)
        # self.l1_loss = 0
        self.l2_loss = 0
        self.absoluteValue = 0

    def forward(self, x):
        
        running_std = torch.sqrt(self.running_var + self.eps)
        weight = self.weight * reshape_to_weight(self.gamma / running_std)
        bias = reshape_to_bias(self.beta - self.gamma * self.running_mean / running_std)

        # print("===========================Training Conv2d==================================")
        # if self.is_first:
        #     print("Training input int", (x).abs().mean())
        #     # print((input*self.m.quan_a_fn.s)[0,0,0,0:16])
        # else:
        #     print("Training input int", (x).abs().mean())


        quantized_weight = self.quan_w_fn(weight)

        # print("self.is_init",self.is_init)
        if self.is_init == False:
            if self.is_first:
                self.quan_a_fn.init_from(x,weight=False)
            out = self._conv_forward(x, weight,bias = None) + bias.reshape(1,-1,1,1)
            self.quan_out_fn.init_from(out,weight=False)
            self.is_init = True
            return out
        
        l2_loss1 = 0               
        if self.is_first:
            x = self.quan_a_fn(x)
            l2_loss1 = (x/self.quan_a_fn.s - self.quan_a_fn.thd_pos).sum()

        out = self._conv_forward(x, quantized_weight,bias = None)        
        quantized_bias = self.quan_out_fn(bias)
        # quantized_bias = bias
        quantized_out = torch.clip(self.quan_out_fn(out,clip=False) + quantized_bias.reshape(1,-1,1,1),min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)
        # self.l2_loss = l2_loss1 + (torch.nn.functional.relu(quantized_out)/self.quan_out_fn.s - self.quan_out_fn.thd_pos).sum()
        # # self.l2_loss = torch.log(1+torch.abs(quantized_out)).sum()
        # self.absoluteValue = torch.abs(quantized_out.detach()/self.quan_out_fn.s.detach()).sum().item()
        # x_abs = torch.abs(quantized_out)/self.quan_out_fn.s
        # self.l2_loss = l2_loss1 + (x_abs - (1/147)*x_abs*x_abs*x_abs).sum()
        # self.absoluteValue = (torch.abs(quantized_out)/self.quan_out_fn.s).sum()
        return quantized_out


# def l1_regularization(model):
#     l1_loss = 0.0
#     def l1_regularization_inner(model):
#         nonlocal l1_loss
#         children = list(model.named_children())
#         for name, child in children:
#             if isinstance(child, QuanConv2dFuseBN):
#                 l1_loss += model._modules[name].l1_loss
#             elif isinstance(child, QuanLinear):
#                 l1_loss += model._modules[name].l1_loss
#             else:
#                 l1_regularization_inner(child)
#     l1_regularization_inner(model)
#     return l1_loss

def l2_regularization(model):
    l2_loss = 0.0
    def l2_regularization_inner(model):
        nonlocal l2_loss
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, QuanConv2dFuseBN) or isinstance(child, QuanLinear):
                l2_loss += model._modules[name].l2_loss
            else:
                l2_regularization_inner(child)
    l2_regularization_inner(model)
    return l2_loss

def get_absoluteValue(model):
    absoluteValue = 0.0
    def get_absoluteValue_inner(model):
        nonlocal absoluteValue
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, QuanConv2dFuseBN) or isinstance(child, QuanLinear):
                absoluteValue += model._modules[name].absoluteValue
            else:
                get_absoluteValue_inner(child)
    get_absoluteValue_inner(model)
    return absoluteValue

class QuanInferConv2dFuseBN(t.nn.Conv2d):
    def __init__(self, m: QuanConv2dFuseBN, last_quan_out_fn, is_first=False, name="act"):
        assert type(m) == QuanConv2dFuseBN
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=True if m.bias is not None else False,
                    padding_mode=m.padding_mode)

        self.is_first = is_first
        self.m = m  
        self.name = name 
        self.last_quan_out_fn = last_quan_out_fn
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos

        running_std = torch.sqrt(self.m.running_var + self.m.eps)
        self.weight.data = m.weight.cuda()
        # print(self.weight.mean())
        weight = self.weight * reshape_to_weight(self.m.gamma / running_std)
        bias = reshape_to_bias(self.m.beta - self.m.gamma * self.m.running_mean / running_std)

        self.weight = t.nn.Parameter(m.quan_w_fn(weight.detach())/m.quan_w_fn.s,requires_grad=False)
        self.bias = t.nn.Parameter(m.quan_out_fn(bias.detach())/m.quan_out_fn.s,requires_grad=False)
            #print(self.bias)

        if is_first:
            s = m.quan_a_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        else:
            s = last_quan_out_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        
        N,M = get_M_N(s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)
        self.s = s
        self.first = True
    
    def forward(self,x):
        if self.is_first:
            x = self.m.quan_a_fn(x)/self.m.quan_a_fn.s
        input = x + 0.0
        if self.first:
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"in")        
        wx = self._conv_forward(x, self.weight,bias=None)
        # if self.is_first:
        #     print("infer conv:",((float(self.M)/float(2**self.N))*wx)[0,0,0,:])
        wx = torch.round((float(self.M)/float(2**self.N))*wx)
        # if self.is_first:
        #     print("infer quantized conv:",wx[0,0,0,:])
        x = wx + self.bias.reshape(1,-1,1,1)
        x = torch.clip(x,max=self.thd_pos,min=self.thd_neg)
        # if self.is_first:
        #     print("infer output:",x[0,0,0,:])
            # print("infer output:",torch.nn.functional.relu(x).mean())
        if self.first:
            self.first = False
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"out")
        # print("infer output:",torch.abs(torch.nn.functional.relu(x)).mean())
        # print("infer output:", x.abs()[0,0,0,:])
        # print("===========================Inference Conv2d==================================")
        # if self.last_quan_out_fn is None:
        #     print("infer input int", (input[0]).abs().mean())
        # else:
        #     print("infer input int", (input[0]).abs().mean())
        # print("infer output without relu:",torch.abs(x[0]).mean())
        # print("infer output relu:",torch.abs(torch.nn.functional.relu(x[0])).mean())
            # print((input*self.last_quan_out_fn.s)[0,0,0,0:16])
        # print("infer bias int:", self.bias.abs().mean())
        # print("infer weight int:", self.weight.abs().mean())
        # print("infer input int:", input.abs().mean())
        # print("infer output int:", x.abs().mean())
        # print("infer output int:", x.abs().int()[0,0,0,:])
        # print((x)[0,0,0,0:16])

        return x

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, quan_out_fn=None, is_first=False):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        # self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None
        self.is_init = False
        self.is_first = is_first
        # if self.is_first:
        #     self.quan_a_fn.init_from(m.weight)
    def forward(self, x):

        # if self.is_init == False:
        #     if self.is_first:
        #         self.quan_a_fn.init_from(x)
        #     out = self._conv_forward(x, self.weight,bias = None)
        #     self.quan_out_fn.init_from(out)
        #     self.is_init = True
        #     return out

        quantized_weight = self.quan_w_fn(self.weight)
        
        l2_loss1 = 0               
        if self.is_first:
            x = self.quan_a_fn(x)
            l2_loss1 = (x/self.quan_a_fn.s - self.quan_a_fn.thd_pos).sum()

        out = self._conv_forward(x, quantized_weight,bias = None)        
        quantized_out = torch.clip(self.quan_out_fn(out,clip=False), min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)
        self.l2_loss = l2_loss1 + (torch.nn.functional.relu(quantized_out)/self.quan_out_fn.s - self.quan_out_fn.thd_pos).sum()
        # self.l2_loss = torch.log(1+torch.abs(quantized_out)).sum()
        self.absoluteValue = torch.abs(quantized_out.detach()/self.quan_out_fn.s.detach()).sum().item()
        return quantized_out

class QuanInferConv2d(t.nn.Conv2d):
    def __init__(self, m: QuanConv2d, last_quan_out_fn, is_first=False, name="act"):
        assert type(m) == QuanConv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=True if m.bias is not None else False,
                    padding_mode=m.padding_mode)

        self.is_first = is_first
        self.m = m  
        self.name = name 
        self.last_quan_out_fn = last_quan_out_fn
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos
        self.weight = t.nn.Parameter(m.quan_w_fn(m.weight.detach())/m.quan_w_fn.s,requires_grad=False)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.quan_out_fn(m.bias.detach())/m.quan_out_fn.s,requires_grad=False)
            #print(self.bias)

        if is_first:
            s = m.quan_a_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        else:
            s = last_quan_out_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        N,M = get_M_N(s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)
        self.s = s
        self.first = True
    
    def forward(self,x):
        if self.is_first:
            x = self.m.quan_a_fn(x)/self.m.quan_a_fn.s                              
        if self.first:
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"in")        
        wx = torch.round((float(self.M)/float(2**self.N))*self._conv_forward(x, self.weight,bias=None))
        x = wx + self.bias.reshape(1,-1,1,1)
        x = torch.clip(x,max=self.thd_pos,min=self.thd_neg)
        if self.first:
            self.first = False
            save_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name+"out")
        return x


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_out_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_out_fn = quan_out_fn
        self.m = m

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        # self.quan_out_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

        self.is_init = False
        # self.l1_loss = 0
        self.l2_loss = 0.0
        self.absoluteValue = 0.0
        
    def forward(self, x):        
        quantized_weight = self.quan_w_fn(self.weight)

        if self.is_init == False:
            out = t.nn.functional.linear(x, self.weight, bias=None) + self.bias.reshape(1,-1)
            self.quan_out_fn.init_from(out,weight=False)
            self.is_init = True
            return out

        out = t.nn.functional.linear(x, quantized_weight, bias=None)
        if self.bias is not None:
            quantized_bias = self.quan_out_fn(self.bias)
        else:
            quantized_bias = None
        quantized_out = self.quan_out_fn(out,clip=False) + quantized_bias.reshape(1,-1)
        quantized_out = torch.clip(quantized_out,min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)

        # self.l1_loss = cal_l1_loss_full(quantized_out.flatten(1))
        # self.l2_loss = 0
        # self.absoluteValue = torch.abs(quantized_out.detach()/self.quan_out_fn.s.detach()).sum().item()
        return quantized_out

class QuanInferLinear(t.nn.Linear):
    def __init__(self, m: QuanLinear, last_quan_out_fn, is_first=False, name="act"):
        assert type(m) == QuanLinear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.m = m
        self.is_first = is_first
        self.first = True
        self.name = name
        self.last_quan_out_fn = last_quan_out_fn
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos
        self.weight = t.nn.Parameter(m.quan_w_fn(m.weight.detach())/m.quan_w_fn.s,requires_grad=False)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.quan_out_fn(m.bias.detach())/m.quan_out_fn.s,requires_grad=False)
        s = last_quan_out_fn.s*m.quan_w_fn.s/m.quan_out_fn.s
        N,M = get_M_N(s)

        if torch.is_tensor(M):
            self.M = t.nn.Parameter(torch.tensor(M.item(),dtype=int),requires_grad=False)
        else:
            self.M = t.nn.Parameter(torch.tensor(M,dtype=int),requires_grad=False)
        if torch.is_tensor(N):
            self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        else:
            self.N = t.nn.Parameter(torch.tensor(N,dtype=int),requires_grad=False)

        self.s_b = m.quan_out_fn.s
    
    def forward(self,x):
        if self.first:
            save_fc_input_for_bin(x[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name)

        # print("Linear infer output:",torch.abs(x).mean())

        out = torch.round((float(self.M)/float(2**self.N))*t.nn.functional.linear(x, self.weight)) + self.bias
        out = torch.clip(out,max=self.thd_pos,min=self.thd_neg)
        if self.first:
            self.first = False
            save_fc_input_for_bin(out[0].unsqueeze(0), glo.get_value("output_bin_qann_dir"),self.name)

        # print("===========================Inference Linear==================================")
        # print("infer input int:", x.abs().mean())
        # print("infer output int:", out.abs().mean())        
        # print("Linear infer output:",torch.abs(out).mean())
        return out

class AdditionQuan(t.nn.Module):
    def __init__(
        self,
        quan_a_fn:LsqQuan,
    ):
        super().__init__()
        self.quan_a_fn = quan_a_fn
        self.is_init = False
        self.thd_pos = quan_a_fn.thd_pos
        self.thd_neg = quan_a_fn.thd_neg
    
    def forward(self,x1,x2):
        if self.is_init == False:
            x = x1 + x2
            self.quan_a_fn.init_from(x,weight=False)
            self.is_init = True
            return x
        else:
            # print("=======================training addition=======================")
            # print("input:",x1.mean(),x2.mean())
            out = self.quan_a_fn(x1+x2)
            # print("output:",out.mean())
        return out


def get_M1_N1_M2_N2(s1,s2):
    T1 = s1 + 0.0
    N = torch.tensor(0)
    T2 = s2 + 0.0
    while(torch.abs(T1)<128 or torch.abs(T2)<128):
        N += 1
        T1 *= 2
        T2 *= 2
    return N,torch.round(T1),torch.round(T2)


class AdditionInfer(t.nn.Module):
    def __init__(
        self,
        m:AdditionQuan,
        quan_input1_fn:LsqQuan,
        quan_input2_fn:LsqQuan,
    ):
        super().__init__()

        s1 = quan_input1_fn.s/m.quan_a_fn.s
        s2 = quan_input2_fn.s/m.quan_a_fn.s
        self.s1 = quan_input1_fn.s
        self.s2 = quan_input2_fn.s
        N,M1,M2 = get_M1_N1_M2_N2(s1,s2)
        self.m = m
        self.N = t.nn.Parameter(torch.tensor(N.item(),dtype=int),requires_grad=False)
        self.M1 = t.nn.Parameter(torch.tensor(M1.item(),dtype=int),requires_grad=False)
        self.M2 = t.nn.Parameter(torch.tensor(M2.item(),dtype=int),requires_grad=False)
        self.thd_neg = m.thd_neg
        self.thd_pos = m.thd_pos

    def forward(self,x1,x2):
        # print("=======================Inference Addition=======================")
        # print("inference input1:",x1[0].abs().mean())
        # print("inference input2:",x2[0].abs().mean())
        x = torch.round((x1*self.M1+x2*self.M2)/2**self.N)
        x = torch.clip(x,self.thd_neg,self.thd_pos)
        # print("inference output:",x[0].abs().mean())
        return x

#def get_M_N(s):
#    N = 10
#    M = torch.round(s*(2**N))
#    return N,M

def get_M_N(s):
    T = s + 0.0
    N = 0
    while(torch.abs(T)<128):
        N += 1
        T *= 2
    return N,torch.round(T)

def set_init_false(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, QuanLinear):
            model._modules[name].is_init = True
        elif isinstance(child, QuanConv2d) or isinstance(child, QuanConv2dFuseBN) or isinstance(child, QuanAvgPool) or isinstance(child,AdditionQuan):
            model._modules[name].is_init = True
        if not is_need:
            set_init_false(child)

def set_init_true(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, QuanLinear):
            model._modules[name].is_init = False
        elif isinstance(child, QuanConv2d) or isinstance(child, QuanConv2dFuseBN) or isinstance(child, QuanAvgPool) or isinstance(child,AdditionQuan):
            model._modules[name].is_init = False
        if not is_need:
            set_init_true(child)

index = 0
last_quan_out_fn = None
def quantized_inference_model(model,bit):
    global index, last_quan_out_fn
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, BasicBlock) or isinstance(child, BasicBlockCifar) or isinstance(child, BasicBlockResNet12):
            model._modules[name].conv1 = QuanInferConv2d(m=child.conv1, is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv2 = QuanInferConv2d(m=child.conv2, is_first=(index==0), last_quan_out_fn=child.conv1.m.quan_out_fn, name=f"act{index}")
            index = index + 1
            if child.downsample is not None:
                child.downsample[0] = QuanInferConv2d(m=child.downsample[0], is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
                index = index + 1
            last_quan_out_fn= child.conv2.m.quan_out_fn
            is_need = True
        elif isinstance(child, Bottleneck):
            model._modules[name].conv1 = QuanInferConv2d(m=child.conv1, is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv2 = QuanInferConv2d(m=child.conv2, is_first=(index==0), last_quan_out_fn=child.conv1.m.quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv3 = QuanInferConv2d(m=child.conv3, is_first=(index==0), last_quan_out_fn=child.conv2.m.quan_out_fn, name=f"act{index}")
            index = index + 1
            if child.downsample is not None:
                child.downsample[0] = QuanInferConv2d(m=child.downsample[0], is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
                index = index + 1
            last_quan_out_fn= child.conv3.m.quan_out_fn
            is_need = True
        elif isinstance(child, QuanLinear):
            model._modules[name] = QuanInferLinear(m=child, is_first=(index==0),last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            last_quan_out_fn= child.quan_out_fn
            index = index + 1
            is_need = True
        elif isinstance(child, QuanConv2d):
            model._modules[name] = QuanInferConv2d(m=child, is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            last_quan_out_fn= child.quan_out_fn
            index = index + 1
            is_need = True
        if not is_need:
            quantized_inference_model(child,bit)



index = 0
last_quan_out_fn = None
def quantized_inference_model_fusebn(model):
    global index, last_quan_out_fn
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, BasicBlock) or isinstance(child, BasicBlockCifar) or isinstance(child, BasicBlockResNet12):
            model._modules[name].conv1 = QuanInferConv2dFuseBN(m=child.conv1, is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv2 = QuanInferConv2dFuseBN(m=child.conv2, is_first=(index==0), last_quan_out_fn=child.conv1.m.quan_out_fn, name=f"act{index}")
            index = index + 1
            if isinstance(child.downsample,t.nn.Sequential):
                child.downsample[0] = QuanInferConv2dFuseBN(m=child.downsample[0], is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
                child.ResidualAdd = AdditionInfer(m=child.ResidualAdd,quan_input1_fn=child.conv2.m.quan_out_fn,quan_input2_fn=child.downsample[0].m.quan_out_fn)
                index = index + 1
            else:
                child.ResidualAdd = AdditionInfer(m=child.ResidualAdd,quan_input1_fn=child.conv2.m.quan_out_fn,quan_input2_fn=last_quan_out_fn)
            last_quan_out_fn= child.ResidualAdd.m.quan_a_fn
            is_need = True
        elif isinstance(child, InvertedResidual):
            model._modules[name].conv[0] = QuanInferConv2dFuseBN(m=child.conv[0], is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv[3] = QuanInferConv2dFuseBN(m=child.conv[3], is_first=(index==0), last_quan_out_fn=model._modules[name].conv[0].m.quan_out_fn, name=f"act{index}")
            index = index + 1
            last_quan_out_fn = child.conv[3].m.quan_out_fn
            if len(child.conv) == 8:
                model._modules[name].conv[6] = QuanInferConv2dFuseBN(m=child.conv[6], is_first=(index==0), last_quan_out_fn=model._modules[name].conv[3].m.quan_out_fn, name=f"act{index}")
                index = index + 1
                last_quan_out_fn = model._modules[name].conv[6].m.quan_out_fn
            is_need = True
        elif isinstance(child, Bottleneck):
            model._modules[name].conv1 = QuanInferConv2dFuseBN(m=child.conv1, is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv2 = QuanInferConv2dFuseBN(m=child.conv2, is_first=(index==0), last_quan_out_fn=child.conv1.m.quan_out_fn, name=f"act{index}")
            index = index + 1
            model._modules[name].conv3 = QuanInferConv2dFuseBN(m=child.conv3, is_first=(index==0), last_quan_out_fn=child.conv2.m.quan_out_fn, name=f"act{index}")
            index = index + 1
            if isinstance(child.downsample,t.nn.Sequential):
                child.downsample[0] = QuanInferConv2dFuseBN(m=child.downsample[0], is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
                child.ResidualAdd = AdditionInfer(m=child.ResidualAdd,quan_input1_fn=child.conv3.m.quan_out_fn,quan_input2_fn=child.downsample[0].m.quan_out_fn)
                index = index + 1
            else:
                child.ResidualAdd = AdditionInfer(m=child.ResidualAdd,quan_input1_fn=child.conv3.m.quan_out_fn,quan_input2_fn=last_quan_out_fn)
            last_quan_out_fn = child.ResidualAdd.m.quan_a_fn
            is_need = True
        elif isinstance(child, QuanLinear):
            model._modules[name] = QuanInferLinear(m=child, is_first=(index==0),last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            last_quan_out_fn= child.quan_out_fn
            index = index + 1
            is_need = True
        elif isinstance(child, QuanConv2dFuseBN):
            model._modules[name] = QuanInferConv2dFuseBN(m=child, is_first=(index==0), last_quan_out_fn=last_quan_out_fn, name=f"act{index}")
            last_quan_out_fn= child.quan_out_fn
            index = index + 1
            is_need = True
        elif isinstance(child, QuanAvgPool):
            model._modules[name] = QuanInferAvgPool(child, last_quan_out_fn=last_quan_out_fn, name=f"AvgPooling{index}")
            index = index + 1
            is_need = True
            last_quan_out_fn = child.quan_out_fn
        if not is_need:
            quantized_inference_model_fusebn(child)


index1 = 0
def quantized_train_model(model,weightBit, actBit):
    global index1
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, BasicBlock) or isinstance(child, BasicBlockCifar) or isinstance(child, BasicBlockResNet12):
            model._modules[name].conv1 = QuanConv2dFuseBN(m=child.conv1, is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            model._modules[name].conv2 = QuanConv2dFuseBN(m=child.conv2, is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            if child.downsample is not None:
                child.downsample[0] = QuanConv2dFuseBN(m=child.downsample[0], is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=model._modules[name].conv2.quan_out_fn)
                child.downsample[0].is_init = True
                index1 = index1 + 1
            # else:
            #     child.downsample = model._modules[name].conv2.quan_out_fn
            child.ResidualAdd = AdditionQuan(quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))            
            is_need = True
        
        elif isinstance(child,InvertedResidual):
            model._modules[name].conv[0] = QuanConv2dFuseBN(m=child.conv[0], is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            model._modules[name].conv[3] = QuanConv2dFuseBN(m=child.conv[3], is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuan(bit=actBit,all_positive=False if len(child.conv) == 8 else False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            model._modules[name].downsample = model._modules[name].conv[3].quan_out_fn
            if len(child.conv) == 8:
                model._modules[name].conv[6] = QuanConv2dFuseBN(m=child.conv[6], is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
                index1 = index1 + 1
                model._modules[name].downsample = model._modules[name].conv[6].quan_out_fn
            is_need = True
        
        elif isinstance(child, Bottleneck):
            model._modules[name].conv1 = QuanConv2dFuseBN(m=child.conv1, is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            model._modules[name].conv2 = QuanConv2dFuseBN(m=child.conv2, is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            model._modules[name].conv3 = QuanConv2dFuseBN(m=child.conv3, is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            if child.downsample is not None:
                child.downsample[0] = QuanConv2dFuseBN(m=child.downsample[0], is_first=(index1==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=model._modules[name].conv3.quan_out_fn)
                index1 = index1 + 1
            # else:
            #     child.downsample = model._modules[name].conv3.quan_out_fn
            child.ResidualAdd = AdditionQuan(quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))            
            is_need = True
        elif isinstance(child, t.nn.Linear):
            model._modules[name] = QuanLinear(m=child, quan_w_fn=LsqQuan(bit=8,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuan(bit=8,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            is_need = True
        elif isinstance(child, t.nn.Conv2d):
            if index1 == 0:
                model._modules[name] = QuanConv2dFuseBN(m=child, is_first=True, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=(actBit),all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            else:
                model._modules[name] = QuanConv2dFuseBN(m=child, is_first=False, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            is_need = True
        elif isinstance(child, t.nn.AvgPool2d) or isinstance(child, t.nn.AdaptiveAvgPool2d):
            model._modules[name] = QuanAvgPool(child, quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index1 = index1 + 1
            is_need = True
        if not is_need:
            quantized_train_model(child,weightBit, actBit)

# index2 = 0
# def quantized_train_model_fusebn(model,weightBit, actBit):
#     global index2
#     children = list(model.named_children())
#     for name, child in children:
#         is_need = False
#         if isinstance(child, t.nn.Linear):
#             model._modules[name] = QuanLinear(m=child, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
#                                             quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
#             index2 = index2 + 1
#             is_need = True
#         elif isinstance(child, t.nn.Conv2d):
#             if index2 == 0:
#                 model._modules[name] = QuanConv2dFuseBN(m=child, is_first=True, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
#                                                 quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
#                                                 quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
#             else:
#                 model._modules[name] = QuanConv2dFuseBN(m=child, is_first=False, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
#                                                 quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
#             index2 = index2 + 1
#             is_need = True
#         if not is_need:
#             quantized_train_model_fusebn(child,weightBit, actBit)


index2 = 0
def quantized_train_model_fusebn(model,weightBit, actBit):
    global index2
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, BasicBlock) or isinstance(child, BasicBlockCifar) or isinstance(child, BasicBlockResNet12):
            model._modules[name].conv1 = QuanConv2dFuseBN(m=child.conv1, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            model._modules[name].conv2 = QuanConv2dFuseBN(m=child.conv2, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            if child.downsample is not None:
                child.downsample[0] = QuanConv2dFuseBN(m=child.downsample[0], is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=model._modules[name].conv2.quan_out_fn)
                child.downsample[0].is_init = True
                index2 = index2 + 1
            # else:
            #     child.downsample = model._modules[name].conv2.quan_out_fn
            child.ResidualAdd = AdditionQuan(quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))            
            is_need = True
        
        elif isinstance(child,InvertedResidual):
            model._modules[name].conv[0] = QuanConv2dFuseBN(m=child.conv[0], is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            model._modules[name].conv[3] = QuanConv2dFuseBN(m=child.conv[3], is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuan(bit=actBit,all_positive=False if len(child.conv) == 8 else False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            model._modules[name].downsample = model._modules[name].conv[3].quan_out_fn
            if len(child.conv) == 8:
                model._modules[name].conv[6] = QuanConv2dFuseBN(m=child.conv[6], is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
                index2 = index2 + 1
                model._modules[name].downsample = model._modules[name].conv[6].quan_out_fn
            is_need = True
        
        elif isinstance(child, Bottleneck):
            model._modules[name].conv1 = QuanConv2dFuseBN(m=child.conv1, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            model._modules[name].conv2 = QuanConv2dFuseBN(m=child.conv2, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            model._modules[name].conv3 = QuanConv2dFuseBN(m=child.conv3, is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            if child.downsample is not None:
                child.downsample[0] = QuanConv2dFuseBN(m=child.downsample[0], is_first=(index2==0), quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=model._modules[name].conv3.quan_out_fn)
                index2 = index2 + 1
            # else:
            #     child.downsample = model._modules[name].conv3.quan_out_fn
            child.ResidualAdd = AdditionQuan(quan_a_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))            
            is_need = True
        elif isinstance(child, t.nn.Linear):
            model._modules[name] = QuanLinear(m=child, quan_w_fn=LsqQuan(bit=4,all_positive=False,symmetric=False,per_channel=False),
                                            quan_out_fn=LsqQuan(bit=8,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            is_need = True
        elif isinstance(child, t.nn.Conv2d):
            if index2 == 0:
                model._modules[name] = QuanConv2dFuseBN(m=child, is_first=True, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_a_fn=LsqQuan(bit=(actBit),all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            else:
                model._modules[name] = QuanConv2dFuseBN(m=child, is_first=False, quan_w_fn=LsqQuan(bit=weightBit,all_positive=False,symmetric=False,per_channel=False),
                                                quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            is_need = True
        elif isinstance(child, t.nn.AvgPool2d) or isinstance(child, t.nn.AdaptiveAvgPool2d):
            model._modules[name] = QuanAvgPool(child, quan_out_fn=LsqQuan(bit=actBit,all_positive=False,symmetric=False,per_channel=False))
            index2 = index2 + 1
            is_need = True
        if not is_need:
            quantized_train_model_fusebn(child,weightBit, actBit)



def save_input_for_bin(input,dir,name):
    B,C,H,W = input.shape
    local_rank = torch.distributed.get_rank()
    
    if local_rank == 0:
        input_list = input.tolist()
        input_binfile = open(f'{dir}/input_{name}_B={B}_C={C}_H={H}_W={W}.bin','wb')
        for i in range(B):
            for j in range(C):
                for n in range(H):
                    for m in range(W):
                        input_binfile.write(int(round(float(input_list[i][j][n][m]))).to_bytes(length=1,byteorder='big',signed =True))
        input_binfile.close()
    
def save_fc_input_for_bin(input,dir,name):
    B,N = input.shape
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        input_list = input.tolist()
        input_binfile = open(f'{dir}/input_{name}_B={B}_N={N}.bin','wb')
        for i in range(B):
            for j in range(N):
                input_binfile.write(int(round(float(input_list[i][j]))).to_bytes(length=1,byteorder='big',signed =True))
        input_binfile.close()

def save_for_bin(model,dir):
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

        if isinstance(child, QuanInferLinear):
            # print("Wrtie bin: QuanInferLinear")
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
        
        if isinstance(child, QuanInferConv2d) or isinstance(child, QuanInferConv2dFuseBN):
            # print("Wrtie bin: QuanInferConv2d")
            weight_list = child.weight.data.tolist()
            weight_binfile = open(f'{dir}/{name}_weight_C1={child.weight.shape[0]}_C2={child.weight.shape[1]}_KH={child.weight.shape[2]}_KW={child.weight.shape[3]}.bin','wb')
            C_out,C_in,KH,KW = child.weight.shape
            for i in range(C_out):
                for j in range(C_in):
                    for n in range(KH):
                        for m in range(KW):
                            weight_binfile.write(int(round(float(weight_list[i][j][n][m]))).to_bytes(length=1,byteorder='big',signed =True))
            weight_binfile.close()

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


# def replace_relu_to_stbif(model,level):
#     children = list(model.named_children())
#     for name, child in children:
#         is_need = False
#         if isinstance(child, nn.ReLU):
#             # model._modules[name] = cext_neuron.MultiStepIFNode(detach_reset=True)
#             model._modules[name] = IFNeuron(q_threshold=1.0,level=level,sym=False)
#             is_need = True
#         # elif isinstance(child, nn.Conv2d) or isinstance(child, nn.BatchNorm2d) or isinstance(child,nn.AvgPool2d) or isinstance(child,nn.MaxPool2d) or isinstance(child,nn.AdaptiveAvgPool2d):
#         #     print(child)
#         #     model._modules[name] = layer.SeqToANNContainer(child)
#         #     is_need = True
#         if not is_need:
#             replace_relu_to_stbif(child,level)


class SimpleCNN(torch.nn.Module):
    def __init__(self) -> None:
        super(SimpleCNN,self).__init__()
        self.conv_block1 = conv_block(1,64) # [B,64,32,512]
        self.conv_block2 = conv_block(64,64) # [B,128,16,256]
        self.conv_block3 = conv_block(64,64) # [B,256,8,128]
        self.conv_block4 = conv_block(64,64) # [B,128,4,64]
        self.conv_block5 = conv_block(64,64) # [B,64,2,32]
        self.conv_block6 = conv_block(64,64) # [B,64,1,16]
        self.flatten=  torch.nn.Flatten(start_dim=1) # [B,256]
        self.fc = torch.nn.Linear(1024,10)
    
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
