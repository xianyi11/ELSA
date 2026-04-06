import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
import re

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import elsa_support  # noqa: F401 — legacy ``partition`` shim for torch.load
from PETile.Tile import Tile
from processElement.STBIFFunction import STBIFNeuron
from PETile.FlitGenerator import FlitGenerator, FlitCombiner
from NetworkOnChip.Router import NOC
from tqdm import tqdm
from router.Switch import SwitchCrossbar
from router.RouteComputer import RouteComputer
import math
import yaml
# repeat:
    # find the input, output, weight, bias, M and N for one layer
    # read the input shape and generate the im2col TLB
    # distribute a tile for the layer 

class OneLayerCal():
    def __init__(self):
        self.weight = None
        self.weight2 = None
        self.bias = None
        self.input = None
        self.input2 = None
        self.output = None
        self.M = None
        self.N = None
        self.LastVthr1 = None
        self.LastVthr2 = None
        self.Vthr = None
        self.midVthr = None
        self.name = None # name in network
        self.type = None # conv, linear, average pooling or residual addition
        self.ops = 0
    
    def printmyself(self):
        print(f"=========================={self.name}============================")
        if self.type is not None:
            print("self.type",self.type)
        if self.weight is not None:
            print("self.weight.shape",self.weight.shape)
        if self.weight2 is not None:
            print("self.weight2.shape",self.weight2.shape)
        if self.bias is not None:
            print("self.bias.shape",self.bias.shape)
        if self.input is not None:
            print("self.input.shape",self.input.shape)
        if self.input2 is not None:
            print("self.input2.shape",self.input2.shape)
        if self.output is not None:
            print("self.output.shape",self.output.shape)
        if self.M is not None:
            print("self.M, self.N",self.M, self.N)
        if self.LastVthr1 is not None:
            print("self.LastVthr1",self.LastVthr1)
        if self.LastVthr2 is not None:
            print("self.LastVthr2",self.LastVthr2)
        if self.midVthr is not None:
            print("self.midVthr",self.midVthr)
        if self.Vthr is not None:
            print("self.Vthr",self.Vthr)
        print("self.ops",self.ops)
            
class VSACompiler(nn.Module):
    def __init__(
        self,
        networkDir,
        connection_path,
        mapping_path,
        tranOccupy_path,
        config_path=None,
    ):
        super(VSACompiler,self).__init__()
        from elsa_support.paths import CONFIG_YAML

        self.dir = networkDir
        self.connection_path = connection_path
        self.mapping_path = mapping_path
        self.tranOccupy_path = tranOccupy_path
        self.layers = []

        if config_path is None:
            config_path = str(CONFIG_YAML)
        self.cfg = None
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)        

        self.NoC = NOC(connection_path, mapping_path, tranOccupy_path)            
        # 对NoC partition 进行数据搜集
        self.InputSearch = {}
        self.InputSearch["InputFlitNum"] = {}
        self.InputSearch["OutputFlitNum"] = {}
        self.InputSearch["avgFlitNumEachToken"] = {}
        self.InputSearch["SRAMNumPerLayer"] = {}
        self.InputSearch["totalSRAMNum"] = self.cfg["TILE"]["SRAMTotalNum"]
        self.InputSearch["VSAWidth"] = self.cfg["processElement"]["adderTree"]["adderNum"]
        self.InputSearch["WidthAfterTiling"] = {}
        self.ElementNumPerSRAM = self.cfg["processElement"]["weightBuffer"]["height"]*self.cfg["processElement"]["weightBuffer"]["inoutWidth"]
        # 每个 token 的 bubble 占流水比例（不包含冷启动），由各层 TILE(flits) 返回后收集
        self.perTokenBubbleRatios = []
        self._bubbleRatioFirstTokenSkipped = False

        self.collectLayers()
        for layer in self.layers:
            layer.printmyself()
    
    
    def allocatePETile(self):
        
        pass
    
    def collectLayers(self):
        connection = torch.load(self.connection_path)
        for i, name in enumerate(list(connection.keys())):
            print(i,name)
            if name.count("attn_qkv") > 0:
                if name.count("attn_qkvk") > 0 or name.count("attn_qkvv") > 0:
                    continue
                curlayerParam = OneLayerCal()   
                curlayerParam.name = f"{name[:-1]}"
                curlayerParam.input = torch.load(glob(self.dir+f"/*{name[:-1]}.in*")[0],map_location="cpu")
                curlayerParam.LastVthr1 = curlayerParam.input.max()
                q = torch.load(glob(self.dir+f"/*{name[:-1]}_q.out*")[0],map_location="cpu")
                k = torch.load(glob(self.dir+f"/*{name[:-1]}_k.out*")[0],map_location="cpu")
                v = torch.load(glob(self.dir+f"/*{name[:-1]}_v.out*")[0],map_location="cpu")
                qkv = torch.stack([q,k,v],dim=0)  # 3,T,B,num_heads,N,dim
                qkv = qkv.permute(1,2,4,0,3,5) #T,B,N,3,num_heads,dim
                qkv = qkv.reshape(qkv.shape[0],qkv.shape[1],qkv.shape[2],-1) # T,B,N,C
                curlayerParam.output = qkv
                curlayerParam.Vthr = torch.unique(curlayerParam.output.float())[4:]
                curlayerParam.weight = torch.load(glob(self.dir+f"/*{name[:-1]}_weight*")[0],map_location="cpu")
                curlayerParam.bias = torch.load(glob(self.dir+f"/*{name[:-1]}_bias*")[0],map_location="cpu")
                curlayerParam.type = "linear"
                T,B,N,C = qkv.shape
                curlayerParam.ops = N*C/3*C/3
                self.layers.append(curlayerParam)
                # SRAM Num = WeightBuffer SRAM Num + MembraneBuffer SRAM Num
                WeightSRAM = math.ceil(curlayerParam.weight.numel()/(self.ElementNumPerSRAM*3)) # 由于dataflow，weight在每个PE里面是重复存储的
                MembraneSRAM = math.ceil((N*C/3)/self.ElementNumPerSRAM) * 3
                self.InputSearch["SRAMNumPerLayer"][curlayerParam.name+"q"] = WeightSRAM + MembraneSRAM
                self.InputSearch["SRAMNumPerLayer"][curlayerParam.name+"k"] = WeightSRAM + MembraneSRAM
                self.InputSearch["SRAMNumPerLayer"][curlayerParam.name+"v"] = WeightSRAM + MembraneSRAM
                
                # 记录每次计算需要多少adderTree
                self.InputSearch["WidthAfterTiling"][curlayerParam.name+"q"] = C/3//self.cfg["TILE"]["PENum"]
                self.InputSearch["WidthAfterTiling"][curlayerParam.name+"k"] = C/3//self.cfg["TILE"]["PENum"]
                self.InputSearch["WidthAfterTiling"][curlayerParam.name+"v"] = C/3//self.cfg["TILE"]["PENum"]
                # print("WeightSRAM",WeightSRAM,"MembraneSRAM",MembraneSRAM,"WidthAfterTiling",C/3//self.cfg["TILE"]["PENum"])
            elif name.count("attn_qkMulti") > 0:
                # q,k multiplication (contain softmax)
                curlayerParam = OneLayerCal()
                curlayerParam.name = f"{name}"
                curlayerParam.input = torch.load(glob(self.dir+f"/*{name}_q.in*")[0],map_location="cpu")
                curlayerParam.LastVthr1 = curlayerParam.input.max()
                curlayerParam.input2 = torch.load(glob(self.dir+f"/*{name}_k.in*")[0],map_location="cpu")
                curlayerParam.LastVthr2 = curlayerParam.input2.max()
                # curlayerParam.weight = torch.load(glob(self.dir+f"/*{name[:-1]}_qkMulti_q_accu.in*")[0],map_location="cpu")
                # curlayerParam.weight2 = torch.load(glob(self.dir+f"/*{name[:-1]}_qkMulti_k_accu.in*")[0],map_location="cpu")
                curlayerParam.weight = torch.load(glob(self.dir+f"/*{name}.out*")[0],map_location="cpu")
                curlayerParam.output = torch.load(glob(self.dir+f"/*{name}_softmax.out*")[0],map_location="cpu")
                curlayerParam.midVthr = curlayerParam.weight.max()
                curlayerParam.Vthr = curlayerParam.output.max()
                curlayerParam.bias = None
                T,B,num_heads,N,dim = curlayerParam.input.shape
                curlayerParam.ops = (num_heads*N)*dim*(N)
                curlayerParam.type = "multiplication_softmax"
                self.layers.append(curlayerParam)
                # SRAM Num = WeightBuffer SRAM Num + MembraneBuffer SRAM Num
                WeightSRAM = math.ceil(2*num_heads*N*dim/self.ElementNumPerSRAM) # 由于dataflow，weight在每个PE里面是重复存储的
                MembraneSRAM = math.ceil(num_heads*N*N/self.ElementNumPerSRAM) * 3
                # print("WeightSRAM",WeightSRAM,"MembraneSRAM",MembraneSRAM,"num_heads",num_heads,"N",N,"dim",dim)
                self.InputSearch["SRAMNumPerLayer"][curlayerParam.name] = WeightSRAM + MembraneSRAM
                # 记录每次计算需要多少adderTree
                self.InputSearch["WidthAfterTiling"][curlayerParam.name] = dim//self.cfg["TILE"]["PENum"]                
            elif name.count("attn_attn") > 0:
                # attn, v multiplication
                curlayerParam = OneLayerCal()   
                curlayerParam.name = f"{name}"
                curlayerParam.input = torch.load(glob(self.dir+f"/*{name}_qk.in*")[0],map_location="cpu")
                curlayerParam.LastVthr1 = curlayerParam.input.max()
                curlayerParam.weight = torch.load(glob(self.dir+f"/*{name}_v.in*")[0],map_location="cpu").sum(dim=0)
                # curlayerParam.weight = torch.load(glob(self.dir+f"/*{name[:-1]}_attn_qk_acc.in*")[0],map_location="cpu")
                # curlayerParam.weight2 = torch.load(glob(self.dir+f"/*{name[:-1]}_attn_v_acc.in*")[0],map_location="cpu")
                curlayerParam.output = torch.load(glob(self.dir+f"/*{name}.out*")[0],map_location="cpu")
                curlayerParam.Vthr = curlayerParam.output.max()
                curlayerParam.bias = None
                T,B,num_heads,N,dim = curlayerParam.output.shape
                curlayerParam.ops = (num_heads*N)*(N)*dim
                curlayerParam.type = "multiplication"
                self.layers.append(curlayerParam)
                # SRAM Num = WeightBuffer SRAM Num + MembraneBuffer SRAM Num
                WeightSRAM = math.ceil((num_heads*N*dim + num_heads*N*N)/self.ElementNumPerSRAM) # 由于dataflow，weight在每个PE里面是重复存储的
                MembraneSRAM = math.ceil(num_heads*N*dim/self.ElementNumPerSRAM) * 3
                self.InputSearch["SRAMNumPerLayer"][curlayerParam.name] = WeightSRAM + MembraneSRAM
                # 记录每次计算需要多少adderTree
                self.InputSearch["WidthAfterTiling"][curlayerParam.name] = dim//self.cfg["TILE"]["PENum"]
            elif name.count("attn_proj") > 0:
                # attn projection
                curlayerParam = OneLayerCal()   
                curlayerParam.name = f"{name}"
                curlayerParam.input = torch.load(glob(self.dir+f"/*{name}.in*")[0],map_location="cpu")
                curlayerParam.LastVthr1 = curlayerParam.input.max()
                curlayerParam.weight = torch.load(glob(self.dir+f"/*{name}_weight*")[0],map_location="cpu")
                curlayerParam.bias = torch.load(glob(self.dir+f"/*{name}_bias*")[0],map_location="cpu")
                curlayerParam.output = torch.load(glob(self.dir+f"/*{name}.out*")[0],map_location="cpu")
                curlayerParam.Vthr = curlayerParam.output.max()
                curlayerParam.type = "linear"
                T,B,N,C = curlayerParam.input.shape
                C1,C = curlayerParam.weight.shape
                curlayerParam.ops = N*C*C1
                self.layers.append(curlayerParam)
                # SRAM Num = WeightBuffer SRAM Num + MembraneBuffer SRAM Num
                WeightSRAM = math.ceil(curlayerParam.weight.numel()/self.ElementNumPerSRAM)# 由于dataflow，weight在每个PE里面是重复存储的
                MembraneSRAM = math.ceil(N*C/self.ElementNumPerSRAM) * 3
                self.InputSearch["SRAMNumPerLayer"][curlayerParam.name] = WeightSRAM + MembraneSRAM
                # 记录每次计算需要多少adderTree
                self.InputSearch["WidthAfterTiling"][curlayerParam.name] = C//self.cfg["TILE"]["PENum"]
            else:
                # read the input, output, weight, bias, M and N for one layer
                curlayerParam = OneLayerCal()
                curlayerParam.name = name
                curlayerParam.input = torch.load(glob(self.dir+f"/*{name}.in*")[0],map_location="cpu")
                curlayerParam.LastVthr1 = curlayerParam.input.max()
                if name.count("addition") > 0:
                    curlayerParam.input2 = torch.load(glob(self.dir+f"/*{name}input2.in*")[0],map_location="cpu")
                    curlayerParam.LastVthr2 = curlayerParam.input2.max()
                else:
                    curlayerParam.weight = torch.load(glob(self.dir+f"/*{name}_weight*")[0],map_location="cpu")
                    curlayerParam.bias = torch.load(glob(self.dir+f"/*{name}_bias*")[0],map_location="cpu")
                curlayerParam.output = torch.load(glob(self.dir+f"/*{name}.out*")[0],map_location="cpu")                
                curlayerParam.Vthr = curlayerParam.output.max()
                # SRAM Num = WeightBuffer SRAM Num + MembraneBuffer SRAM Num
                if name.count("addition") > 0:
                    WeightSRAM = 0
                    MembraneSRAM = math.ceil(curlayerParam.output.shape[-2]*curlayerParam.output.shape[-1]/self.ElementNumPerSRAM) * 3 
                    self.InputSearch["SRAMNumPerLayer"][curlayerParam.name] = WeightSRAM + MembraneSRAM                
                else:
                    WeightSRAM = math.ceil(curlayerParam.weight.numel()/self.ElementNumPerSRAM) # 由于dataflow，weight在每个PE里面是重复存储的
                    MembraneSRAM = math.ceil(curlayerParam.output.shape[-2]*curlayerParam.output.shape[-1]/self.ElementNumPerSRAM) * 3
                    self.InputSearch["SRAMNumPerLayer"][curlayerParam.name] = WeightSRAM + MembraneSRAM
                    
                # 记录每次计算需要多少adderTree
                self.InputSearch["WidthAfterTiling"][curlayerParam.name] = curlayerParam.input.shape[-1]//self.cfg["TILE"]["PENum"]
                # print("self.ElementNumPerSRAM",self.ElementNumPerSRAM,"WeightSRAM",WeightSRAM,"MembraneSRAM",MembraneSRAM,"WidthAfterTiling",curlayerParam.input.shape[-1]//self.cfg["TILE"]["PENum"])
                
                if name.count("addition") > 0:
                    curlayerParam.type = "addition"
                    T,B,N,C = curlayerParam.input.shape
                    curlayerParam.ops = N*C/2
                elif name.count("norm") > 0:
                    curlayerParam.type = "normolization"
                    T,B,N,C = curlayerParam.input.shape
                    curlayerParam.ops = N*C
                elif name.count("head") > 0:
                    curlayerParam.type = "linear"
                    T,B,C = curlayerParam.input.shape
                    C1,C = curlayerParam.weight.shape
                    curlayerParam.ops = C*C1
                elif name.count("fc") > 0:
                    curlayerParam.type = "linear"
                    T,B,N,C = curlayerParam.input.shape
                    C1,C = curlayerParam.weight.shape
                    curlayerParam.ops = N*C*C1                    
                elif name.count("patch_embed") > 0:
                    curlayerParam.type = "conv"
                    M = curlayerParam.weight.shape[0]
                    K = curlayerParam.weight.shape[1]*curlayerParam.weight.shape[2]*curlayerParam.weight.shape[3]
                    N = curlayerParam.output.shape[3]*curlayerParam.output.shape[4]
                    curlayerParam.ops = N*K*M
                self.layers.append(curlayerParam)
                
# def test_VSACompiler():
#     compiler = VSACompiler(r"D:\tools\HPCA2025\simulator\output_bin_snn")
#     compiler.collectLayers()
#     print("compiler.layers[10]",compiler.layers[10].output.sum())
#     compiler.layers[10].printmyself()
#     torch.save(compiler.layers[10],"onelayerParamForTest.pth")
def getRouterBreakdown(calculateInfo, FlitGenerator ,TILE, CrossbarSwitch, RouteEngine, totalCycle, name, type, isConv=False):  
    latency = totalCycle/200000000
    if isConv:
        AysnImg2ColArea = TILE.Im2ColUnit.getArea()
        AysnImg2ColEnergy = TILE.Im2ColUnit.calEnergy(latency)
        VSAOrderControllerArea = TILE.VSAOrderCtrl.getArea()
        VSAOrderControllerEnergy = TILE.VSAOrderCtrl.calEnergy(latency)
        VSAUpdateArbiterArea = TILE.VSAOrderArbiter.getArea()
        VSAUpdateArbiterEnergy = TILE.VSAOrderArbiter.calEnergy(latency)
    FlitGeneratorArea = FlitGenerator.getArea()
    FlitGeneratorEnergy = FlitGenerator.calEnergy(latency)
    FlitDecoderArea = TILE.flitCombiner.getArea()
    FlitDecoderEnergy = TILE.flitCombiner.calEnergy(latency)
    RoutingEngineArea = RouteEngine.getArea()
    RoutingEngineEnergy = RouteEngine.calEnergy(latency)
    CrossbarSwitchArea = CrossbarSwitch.getArea()
    CrossbarSwitchEnergy = CrossbarSwitch.calEnergy(latency)

    if type == "residual_norm":
        LayernormFunctionArea = TILE.layernormFunction.getArea()
        LayernormFunctionEnergy = TILE.layernormFunction.calEnergy(latency)
        AdditionFunctionArea = TILE.AdditionFunction.getArea()
        AdditionFunctionEnergy = TILE.AdditionFunction.calEnergy(latency)
    
    if type == "multiplication_softmax":
        softmaxFunctionArea = TILE.softmaxFunction.getArea()
        softmaxFunctionEnergy = TILE.softmaxFunction.calEnergy(latency)
    
    if isConv:
        calculateInfo[name]["AysnImg2ColArea"] = AysnImg2ColArea
        calculateInfo[name]["AysnImg2ColEnergy"] = AysnImg2ColEnergy
        calculateInfo[name]["VSAOrderControllerArea"] = VSAOrderControllerArea
        calculateInfo[name]["VSAOrderControllerEnergy"] = VSAOrderControllerEnergy
        calculateInfo[name]["VSAUpdateArbiterArea"] = VSAUpdateArbiterArea
        calculateInfo[name]["VSAUpdateArbiterEnergy"] = VSAUpdateArbiterEnergy
    if type == "residual_norm":
        calculateInfo[name]["LayernormFunctionArea"] = LayernormFunctionArea
        calculateInfo[name]["LayernormFunctionEnergy"] = LayernormFunctionEnergy
        calculateInfo[name]["AdditionFunctionArea"] = AdditionFunctionArea
        calculateInfo[name]["AdditionFunctionEnergy"] = AdditionFunctionEnergy
    if type == "multiplication_softmax":
        calculateInfo[name]["softmaxFunctionArea"] = softmaxFunctionArea
        calculateInfo[name]["softmaxFunctionEnergy"] = softmaxFunctionEnergy
        
    calculateInfo[name]["FlitGeneratorArea"] = FlitGeneratorArea
    calculateInfo[name]["FlitGeneratorEnergy"] = FlitGeneratorEnergy
    calculateInfo[name]["FlitDecoderArea"] = FlitDecoderArea
    calculateInfo[name]["FlitDecoderEnergy"] = FlitDecoderEnergy
    calculateInfo[name]["RoutingEngineArea"] = RoutingEngineArea
    calculateInfo[name]["RoutingEngineEnergy"] = RoutingEngineEnergy    
    calculateInfo[name]["CrossbarSwitchArea"] = CrossbarSwitchArea
    calculateInfo[name]["CrossbarSwitchEnergy"] = CrossbarSwitchEnergy    
    
    return calculateInfo


def getPEBreakdownMulti(calculateInfo, PEs, totalCycle, name, head, firstLayer):
    membraneArea = 0.0
    mambraneEnergy = 0.0
    spikeTracerArea = 0.0
    spikeTracerEnergy = 0.0
    weightBufferArea = 0.0
    weightBufferEnergy = 0.0
    inputBufferArea = 0.0
    inputBufferEnergy = 0.0
    adderVectorArea = 0.0
    adderVectorEnergy = 0.0
    fireComponentArea = 0.0
    fireComponentEnergy = 0.0

    for PE in PEs:
        for h in range(head):
            membraneArea = membraneArea + PE.membrane[h].getArea()
            mambraneEnergy = mambraneEnergy + PE.membrane[h].calEnergy(totalCycle/200000000)
            
            spikeTracerArea = spikeTracerArea + PE.spikeTracer[h].getArea()
            spikeTracerEnergy = spikeTracerEnergy + PE.spikeTracer[h].calEnergy(totalCycle/200000000)
            
            weightBufferArea = weightBufferArea + PE.weightBufferQ[h].getArea()  + PE.weightBufferK[h].getArea()
            weightBufferEnergy = weightBufferEnergy + PE.weightBufferQ[h].calEnergy(totalCycle/200000000) + PE.weightBufferK[h].calEnergy(totalCycle/200000000)
            
        inputBufferArea = inputBufferArea + PE.inputBuffer.getArea()
        inputBufferEnergy = inputBufferEnergy + PE.inputBuffer.calEnergy(totalCycle/200000000)    
            
        adderVectorArea = adderVectorArea + (PE.adder.getArea() if firstLayer else PE.adderTree.getArea())
        adderVectorEnergy = adderVectorEnergy + (PE.adder.calEnergy(totalCycle/200000000) if firstLayer else PE.adderTree.calEnergy(totalCycle/200000000))

        fireComponentArea = fireComponentArea + PE.fireComponent.getArea()
        fireComponentEnergy = fireComponentEnergy + PE.fireComponent.calEnergy(totalCycle/200000000)    

    # print("PE.adder.fcount",PE.adder.fCount)
    # print("PE.fireComponent.fcount",PE.fireComponent.fCount)
    
        readCount = 0
        writeCount = 0
        for h in range(head):
            for sram in PE.membrane[h].srams:
                readCount = readCount + sram.readCount
                writeCount = writeCount + sram.writeCount
        # print("PE.membrane.readCount",readCount)
        # print("PE.membrane.writeCount",writeCount)

        readCount = 0
        writeCount = 0
        for h in range(head):
            for sram in PE.spikeTracer[h].srams:
                readCount = readCount + sram.readCount
                writeCount = writeCount + sram.writeCount
        # print("PE.spikeTracer.readCount",readCount)
        # print("PE.spikeTracer.writeCount",writeCount)

        readCount = 0
        writeCount = 0
        for h in range(head):
            readCount = readCount + PE.weightBufferQ[h].cram.readCount + PE.weightBufferK[h].cram.readCount
            writeCount = writeCount + PE.weightBufferQ[h].cram.writeCount + PE.weightBufferK[h].cram.writeCount
        # print("PE.weightBuffer.readCount",readCount)
        # print("PE.weightBuffer.writeCount",writeCount)


        readCount = 0
        writeCount = 0
        for sram in PE.inputBuffer.srams:
            readCount = readCount + sram.readCount
            writeCount = writeCount + sram.writeCount
        # print("PE.inputBuffer.readCount",readCount)
        # print("PE.inputBuffer.writeCount",writeCount)

    calculateInfo[name]["membraneArea"] = membraneArea
    calculateInfo[name]["mambraneEnergy"] = mambraneEnergy
    calculateInfo[name]["spikeTracerArea"] = spikeTracerArea
    calculateInfo[name]["spikeTracerEnergy"] = spikeTracerEnergy
    calculateInfo[name]["weightBufferArea"] = weightBufferArea
    calculateInfo[name]["weightBufferEnergy"] = weightBufferEnergy
    calculateInfo[name]["inputBufferArea"] = inputBufferArea
    calculateInfo[name]["inputBufferEnergy"] = inputBufferEnergy
    calculateInfo[name]["adderVectorArea"] = adderVectorArea
    calculateInfo[name]["adderVectorEnergy"] = adderVectorEnergy
    calculateInfo[name]["fireComponentArea"] = fireComponentArea
    calculateInfo[name]["fireComponentEnergy"] = fireComponentEnergy

    return calculateInfo

def getPEBreakdown(calculateInfo, PEs, totalCycle, name, head, firstLayer):
    membraneArea = 0.0
    mambraneEnergy = 0.0
    spikeTracerArea = 0.0
    spikeTracerEnergy = 0.0
    weightBufferArea = 0.0
    weightBufferEnergy = 0.0
    inputBufferArea = 0.0
    inputBufferEnergy = 0.0
    adderVectorArea = 0.0
    adderVectorEnergy = 0.0
    fireComponentArea = 0.0
    fireComponentEnergy = 0.0

    for PE in PEs:
        for h in range(head):
            membraneArea = membraneArea + PE.membrane[h].getArea()
            mambraneEnergy = mambraneEnergy + PE.membrane[h].calEnergy(totalCycle/200000000)
            
            spikeTracerArea = spikeTracerArea + PE.spikeTracer[h].getArea()
            spikeTracerEnergy = spikeTracerEnergy + PE.spikeTracer[h].calEnergy(totalCycle/200000000)
            
            weightBufferArea = weightBufferArea + PE.weightBuffer[h].getArea()
            weightBufferEnergy = weightBufferEnergy + PE.weightBuffer[h].calEnergy(totalCycle/200000000)
            
        inputBufferArea = inputBufferArea + PE.inputBuffer.getArea()
        inputBufferEnergy = inputBufferEnergy + PE.inputBuffer.calEnergy(totalCycle/200000000)    
            
        adderVectorArea = adderVectorArea + (PE.adder.getArea() if firstLayer else PE.adderTree.getArea())
        adderVectorEnergy = adderVectorEnergy + (PE.adder.calEnergy(totalCycle/200000000) if firstLayer else PE.adderTree.calEnergy(totalCycle/200000000))

        fireComponentArea = fireComponentArea + PE.fireComponent.getArea()
        fireComponentEnergy = fireComponentEnergy + PE.fireComponent.calEnergy(totalCycle/200000000)    

    # print("PE.adder.fcount",PE.adder.fCount)
    # print("PE.fireComponent.fcount",PE.fireComponent.fCount)
    
        readCount = 0
        writeCount = 0
        for h in range(head):
            for sram in PE.membrane[h].srams:
                readCount = readCount + sram.readCount
                writeCount = writeCount + sram.writeCount
        # print("PE.membrane.readCount",readCount)
        # print("PE.membrane.writeCount",writeCount)

        readCount = 0
        writeCount = 0
        for h in range(head):
            for sram in PE.spikeTracer[h].srams:
                readCount = readCount + sram.readCount
                writeCount = writeCount + sram.writeCount
        # print("PE.spikeTracer.readCount",readCount)
        # print("PE.spikeTracer.writeCount",writeCount)

        readCount = 0
        writeCount = 0
        for h in range(head):
            for sram in PE.weightBuffer[h].srams:
                readCount = readCount + sram.readCount
                writeCount = writeCount + sram.writeCount
        # print("PE.weightBuffer.readCount",readCount)
        # print("PE.weightBuffer.writeCount",writeCount)


        readCount = 0
        writeCount = 0
        for sram in PE.inputBuffer.srams:
            readCount = readCount + sram.readCount
            writeCount = writeCount + sram.writeCount
        # print("PE.inputBuffer.readCount",readCount)
        # print("PE.inputBuffer.writeCount",writeCount)

    calculateInfo[name]["membraneArea"] = membraneArea
    calculateInfo[name]["mambraneEnergy"] = mambraneEnergy
    calculateInfo[name]["spikeTracerArea"] = spikeTracerArea
    calculateInfo[name]["spikeTracerEnergy"] = spikeTracerEnergy
    calculateInfo[name]["weightBufferArea"] = weightBufferArea
    calculateInfo[name]["weightBufferEnergy"] = weightBufferEnergy
    calculateInfo[name]["inputBufferArea"] = inputBufferArea
    calculateInfo[name]["inputBufferEnergy"] = inputBufferEnergy
    calculateInfo[name]["adderVectorArea"] = adderVectorArea
    calculateInfo[name]["adderVectorEnergy"] = adderVectorEnergy
    calculateInfo[name]["fireComponentArea"] = fireComponentArea
    calculateInfo[name]["fireComponentEnergy"] = fireComponentEnergy

    return calculateInfo



def _print_layer_bubble(compiler, layer_name):
    """每层结束时打印该层 bubble 占流水比例统计。"""
    if not getattr(compiler, "_currentLayerBubbleRatios", None):
        return
    arr = np.array(compiler._currentLayerBubbleRatios)
    if len(arr) == 0:
        print("[%s] bubble ratio: no tokens" % layer_name)
        return
    print("[%s] bubble ratio (excl. cold start): count=%d, mean=%.4f, min=%.4f, max=%.4f" % (
        layer_name, len(arr), float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))))


def run_vit_small_vary_t(
    datapath,
    connectionpath,
    mappingpath,
    occupyPath,
    config_path=None,
    time_step=32,
    output_dir=None,
):
    from elsa_support.unified_elisa_pipeline import run_unified_transformer_pipeline

    return run_unified_transformer_pipeline(
        datapath,
        connectionpath,
        mappingpath,
        occupyPath,
        config_path=config_path,
        time_step=time_step,
        output_dir=output_dir,
        calculate_info_pth="calculateInfoConv_vitsmall_no_CRAM_baseline.pth",
        column_finish_pth="columnFinishCyclePerlayer_vitsmall_no_CRAM_baseline.pth",
        input_search_pth="InputSearch_baseline.pth",
    )
