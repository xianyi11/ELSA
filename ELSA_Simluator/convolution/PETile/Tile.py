import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
from processElement.processElement import ProcessElement
from .Im2ColTLB import Img2ColTLB, VSAOrderController, VSAUpdateArbiter
# from VSACompiler import OneLayerCal
from router.Flit import Flit, payLoad
from typing import List, Optional
from .FlitGenerator import FlitGenerator,FlitCombiner
from processElement.STBIFFunction import STBIFNeuron
from tqdm import tqdm
import math
import time

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class Tile(VSAModule):
    def __init__(self, layerParam, routeInfo:tuple, first=True):
        super(Tile,self).__init__()
        self.PENum = cfg["TILE"]["PENum"]
        self.layerParam = layerParam
        self.routeInfo = routeInfo
        self.flitCombiner = FlitCombiner()
        self.outcycle = 0
        self.duringcycle = 0
        self.cycleForUpdate = 0
        self.lastFlitInputTime = 0
        self.layertype = layerParam.type
        self.first = first

        if self.layerParam.type == "conv":
            # stride, KW, KH, IW, IH
            stride = self.layerParam.input.shape[-1]//self.layerParam.output.shape[-1]
            IW = self.layerParam.input.shape[-1]
            IH = self.layerParam.input.shape[-2]
            KW = self.layerParam.weight.shape[-1]
            KH = self.layerParam.weight.shape[-2]
            self.Im2ColUnit = Img2ColTLB(stride,KW,KH,IW,IH,padding=KW//2)
            self.VSAOrderCtrl = VSAOrderController()
            self.VSAOrderArbiter = VSAUpdateArbiter(KH=KH,KW=KW,stride=stride,padding=KW//2,IH=IH,IW=IW)

            self.N = self.layerParam.weight.shape[0]
            self.K = self.layerParam.weight.shape[1]*self.layerParam.weight.shape[2]*self.layerParam.weight.shape[3]
            self.M = self.layerParam.output.shape[3]*self.layerParam.output.shape[4]
            self.tileN = self.N//self.PENum
            print("M,K,N",self.M,self.K,self.N)
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileN) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.M, self.layerParam.N),matrixShape=(self.M,self.K,self.N),first=self.first, mapLayerNum = layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
            self.loadWeight()
        elif self.layerParam.type == "linear":
            self.M = self.layerParam.input.shape[1]
            self.K = self.layerParam.weight.shape[1]
            self.N = self.layerParam.weight.shape[0]
            self.tileN = int(math.ceil(self.N/self.PENum))
            if self.tileN*self.PENum != self.N:
                pad = (0, 0, self.tileN*self.PENum - self.N, 0)
                self.layerParam.weight = torch.nn.functional.pad(self.layerParam.weight, pad, "constant", 0)
                self.N = self.layerParam.weight.shape[0]
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileN) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.M, self.layerParam.N),matrixShape=(self.M,self.K,self.N),first=self.first, mapLayerNum = layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
            self.loadWeight()
        elif self.layerParam.type == "pool": # pooling is considered as a special convolution
            stride = self.layerParam.input.shape[-1]/self.layerParam.output.shape[-1]
            KW = self.layerParam.input.shape[-1]/self.layerParam.output.shape[-1]
            KH = self.layerParam.input.shape[-1]/self.layerParam.output.shape[-1]
            IW = self.layerParam.input.shape[-1]
            IH = self.layerParam.input.shape[-2]
            C = self.layerParam.input.shape[-3]
            self.layerParam.weight = torch.ones(C,C,KW,KH)
            self.N = self.layerParam.weight.shape[0]
            self.K = self.layerParam.weight.shape[1]*self.layerParam.weight.shape[2]*self.layerParam.weight.shape[3]
            self.M = self.layerParam.output.shape[3]*self.layerParam.output.shape[4]
            self.tileN = self.N//self.PENum
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileN) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.M, self.layerParam.N),matrixShape=(self.M,self.K,self.N),first=self.first, mapLayerNum = layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
            self.loadWeight()
        elif self.layerParam.type == "residual" or self.layerParam.type == "quantize": # identity, no matrix multiplication
            self.N = self.layerParam.input.shape[2]
            self.K = self.M = self.layerParam.input.shape[-1]*self.layerParam.input.shape[-2]
            self.tileN = self.N//self.PENum
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileN) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.M, self.layerParam.N),matrixShape=(self.M,self.K,self.N),first=self.first, mapLayerNum = layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = 0
                PE.fireComponent.M = 1            
        
        self.spikeNum = 0
        self.spikeNumAfterImg2Col = 0
        # spine-level pipeline: bubble = idle cycles waiting for next spine's data
        self.totalBubbleCycle = 0
        self.totalComputeCycle = 0
        self.firstSpineArrivalTime = None  # first spine's arrival (exclude cold start from ratio)
    def reset(self):
        self.outcycle = 0
        self.duringcycle = 0        
        self.spikeNum = 0
        self.totalBubbleCycle = 0
        self.totalComputeCycle = 0
        self.firstSpineArrivalTime = None
    
    def calEnergy(self, latency):
        totalEnergy = 0.0
        for PE in self.PEs:
            totalEnergy = totalEnergy + PE.calEnergy(latency)
            print("PEs",PE.calEnergy(latency))
        for flitGenerator in self.flitGenerators:
            totalEnergy = totalEnergy + flitGenerator.calEnergy(latency)
            print("flitGenerators",flitGenerator.calEnergy(latency))
        totalEnergy = totalEnergy + self.flitCombiner.calEnergy(latency)
        print("flitCombiner",self.flitCombiner.calEnergy(latency))
        if self.layerParam.type == "conv" or self.layerParam.type == "pool":
            totalEnergy = totalEnergy + self.Im2ColUnit.calEnergy(latency) + self.VSAOrderCtrl.calEnergy(latency) + self.VSAOrderArbiter.calEnergy(latency)
            print("self.Im2ColUnit",self.Im2ColUnit.calEnergy(latency),"self.VSAOrderCtrl",self.VSAOrderCtrl.calEnergy(latency),"self.VSAOrderArbiter", self.VSAOrderArbiter.calEnergy(latency))
        print("total Energy",totalEnergy)
        return totalEnergy

    def getArea(self):
        totalArea = 0.0
        for PE in self.PEs:
            totalArea = totalArea + PE.getArea()
        for flitGenerator in self.flitGenerators:
            totalArea = totalArea + flitGenerator.getArea()
        totalArea = totalArea + self.flitCombiner.getArea()
        if self.layerParam.type == "conv" or self.layerParam.type == "pool":
            totalArea = totalArea + self.Im2ColUnit.getArea() + self.VSAOrderCtrl.getArea() + self.VSAOrderArbiter.getArea()
        return totalArea

    def loadWeight(self):
        # initialize weight data
        weight = self.layerParam.weight
        weight2D = weight.reshape(weight.shape[0],-1)
        assert self.tileN*self.PENum == self.N, "The tileN*PENum must equal the N"
        for i,PE in enumerate(self.PEs):
            #load weight
            print("self.tileN",self.tileN)
            for k in range(self.K):
                block_id = k//PE.weightBuffer.tileHN
                row_id = k%PE.weightBuffer.tileHN
                PE.weightBuffer.input_data(weight2D[i*self.tileN:(i+1)*self.tileN,k], block_id, row_id)

            #load bias
            for m in range(self.M):
                block_id = m//PE.membrane.tileWN
                column_id = m%PE.membrane.tileWN
                if self.layerParam.bias is None:
                    PE.membrane.input_data(torch.zeros(self.tileN)+2**(self.layerParam.N-1), block_id, column_id)
                else:
                    PE.membrane.input_data(torch.zeros(self.tileN)+2**(self.layerParam.N-1) + self.layerParam.bias[i*self.tileN:(i+1)*self.tileN]*(2**(self.layerParam.N)), block_id, column_id)
            # initial spike tracer
            for m in range(self.M):
                block_id = m//PE.membrane.tileWN
                column_id = m%PE.membrane.tileWN
                if self.layerParam.bias is None:
                    PE.spikeTracer.input_data(torch.zeros(self.tileN), block_id, column_id)
                else:
                    PE.spikeTracer.input_data(torch.zeros(self.tileN), block_id, column_id)
                
    def forward(self, flitList:List[Flit]):
        
        output = self.flitCombiner(flitList)
        if output is None:
            return None
        spikes, columnID, maxcycle = output
        # Spine-level pipeline bubble: time tile was idle waiting for this spine's data
        spine_arrival_time = maxcycle
        if self.firstSpineArrivalTime is None:
            self.firstSpineArrivalTime = spine_arrival_time
        tile_ready_time = self.lastFlitInputTime + self.cycleForUpdate
        bubble_this_spine = max(0, spine_arrival_time - tile_ready_time)
        self.totalBubbleCycle += bubble_this_spine
        # If the addition between the time when last input flit enters and the calculate time when last input flit consume is larger than the time when current filt enter, waiting begin.
        # print("maxcycle",maxcycle,"self.lastFlitInputTime",self.lastFlitInputTime,"self.cycleForUpdate",self.cycleForUpdate)
        maxcycle = max(maxcycle, self.lastFlitInputTime + self.cycleForUpdate)
        self.spikeNum = self.spikeNum + len(spikes)
            
        if self.layertype == "conv":
            spikesCol = []
            for spike in spikes:
                onespikeCol = self.Im2ColUnit(spike)
                # print(len(onespikeCol),onespikeCol)
                for s in onespikeCol:
                    spikesCol.append(s)
            spikes = spikesCol
            
        self.spikeNumAfterImg2Col = self.spikeNumAfterImg2Col + len(spikes)
                
        outFlitsList = [[] for i in range(self.PENum)]
        curPEId = 0
        for PE, flitGenerator in zip(self.PEs, self.flitGenerators):
            if self.layertype == "linear":
                update=True
                outspike, column_id = PE(spikes,update=True,tarColumnId=columnID)
                outFlitsList[curPEId] = flitGenerator(outspike,columnId=column_id)

            elif self.layertype == "conv":
                # calculate the input rowID and columnID
                i = columnID//self.layerParam.input.shape[-1]
                j = columnID%self.layerParam.input.shape[-1]
                
                # calculate the output rowID and TarcolumnID
                update, outputIList, outputJLsit = self.VSAOrderArbiter(i,j)
                if update:
                    index = 0
                    for outputI,outputJ in zip(outputIList, outputJLsit):
                        # print("outputIList, outputJLsit",outputIList, outputJLsit)
                        # print("update",update,"i,j",i,j,"outputI, outputJ",outputI, outputJ,"tarcolumnID",outputI*self.layerParam.output.shape[-1]+outputJ)
                        # for i,queue in enumerate(PE.inputBuffer.spikeQueues):
                        #     print(f"queueId:{i},queueLen:{queue.qsize()}")
                        #     print(list(queue.queue))
                        # print(PE.inputBuffer.CIdtoQdTLB)
                        # update: the update column is equal to the TarcolumnID
                        tarcolumnID = outputI*self.layerParam.output.shape[-1]+outputJ
                        if index == 0:
                            outspike, column_id = PE(spikes,update=True,tarColumnId=tarcolumnID)
                        else:
                            outspike, column_id = PE([],update=True,tarColumnId=tarcolumnID)
                        index = index + 1

                        flitList = flitGenerator(outspike,columnId=column_id)
                        outFlitsList[curPEId].append(flitList)

                else:
                    tarcolumnID = 0
                    outspike, column_id = PE(spikes,update=False,tarColumnId=tarcolumnID)
                
                # if i == 8 and j == 6:
                #     print("update",update,"i,j",i,j,"outputI, outputJ",outputI, outputJ,"tarcolumnID",outputI*self.layerParam.output.shape[-1]+outputJ, "outspike",outspike)
            curPEId = curPEId + 1                    
        
        duringcycle = max((PE.computationCycle), self.duringcycle)
        self.cycleForUpdate = duringcycle - self.duringcycle
        self.totalComputeCycle += self.cycleForUpdate
        # print("PE.computationCycle",PE.computationCycle,"duringcycle",duringcycle,"maxcycle",maxcycle,"self.duringcycle",self.duringcycle,"self.cycleForUpdate",self.cycleForUpdate)
        self.duringcycle = duringcycle + 0
        self.lastFlitInputTime = maxcycle
        self.outcycle = maxcycle + self.cycleForUpdate
        # if self.layerParam.name.count("model.module.layer1.0.conv2\n") > 0 or self.layerParam.name.count("model.module.layer1.0.conv1\n") > 0:
        #     print("flit.time",flitList[0].time,"maxcycle",maxcycle,"self.cycleForUpdate",self.cycleForUpdate,"self.outcycle",self.outcycle)            
            
        if update:
            # for calculate latency
            for outFlits in outFlitsList:
                for flit in outFlits:
                    if isinstance(flit,List):
                        for lflit in flit:                        
                            lflit.time = lflit.time + maxcycle + self.cycleForUpdate
                    else:
                        flit.time = flit.time + maxcycle + self.cycleForUpdate
            return outFlitsList, self.outcycle
        else:
            None
    

# def test_Tile_convolution_bias():
#     from copy import deepcopy

#     torch.set_printoptions(profile="full")
    
#     layerParam = torch.load(r"D:\tools\HPCA2025\simulator_CARBON\onelayerParamForTest.pth")
#     print(layerParam.printmyself())

#     M1 = layerParam.M
#     N1 = layerParam.N
#     bias = layerParam.bias
#     weight = layerParam.weight
#     input = layerParam.input
#     groudtruth = layerParam.output
#     neuron = STBIFNeuron(M=M1,N=N1,pos_max=7,neg_min=0,bias=bias.unsqueeze(1))    
#     T = layerParam.input.shape[0]
#     outputList = []
#     stride = layerParam.input.shape[-1]//layerParam.output.shape[-1]
#     KW = layerParam.weight.shape[-1]
#     K = layerParam.weight.shape[1]*layerParam.weight.shape[2]*layerParam.weight.shape[3]
#     firstConv = None
    
#     for t in range(T):    
#         wx = torch.nn.functional.conv2d(input[t], weight, stride=stride, padding=KW//2)
#         if t == 0:
#             firstConv = (wx*M1).reshape(wx.shape[1],-1) + 2**(N1-1) + bias.unsqueeze(1)*(2**(N1))
#         outputList.append(neuron(wx)+0)
    
    
#     print(firstConv[:,0])
#     print(outputList[0].reshape(outputList[0].shape[1],-1)[:,0])
#     output1 = torch.stack(outputList,dim=0)
    
#     # print(output1[0,0,0,0])
#     # print(groudtruth[0,0,0,0])
#     assert (output1 == groudtruth).all(), "output1 != groudtruth"
    
#     TILE = Tile(layerParam=layerParam,routeInfo=(1,0)) #move to the east router

#     output1 = output1.reshape(output1.shape[0],output1.shape[1],output1.shape[2],-1)


#     flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
#     flitCombiner = FlitCombiner()
#     InputFlitNum = 0
#     OutputFlitNum = 0
    
#     perTimeStepCycle = []
    
    
#     outSpikesList = []
#     for t in tqdm(range(T)):
#         # if t > 1:
#         #     break
#         H,W = input.shape[-2],input.shape[-1]
#         input2D = input[t].reshape(input[t].shape[1],-1)
#         r = c = 0
#         outSpikes1 = torch.zeros(output1.shape[2],output1.shape[3])
#         InputRows = input2D.shape[0]
#         InputColumns = input2D.shape[1]
#         print("input2D",input2D.shape)
#         spikeNum = 0
#         maxOutcycle = 0
        
#         for m in tqdm(range(InputColumns)):
#             # if m > 100:
#             #     break
#             # print("r,c",r,c)
#             columnId = r*W+c
#             # if (input2D[:,columnId] == 0).all():
#             #     r,c = TILE.VSAOrderCtrl(r,c)
#             #     continue
#             flits = flitGenerator(spikes=input2D[:,columnId], columnId=columnId)
#             InputFlitNum = InputFlitNum + len(flits)
#             for flit in flits:
#                 spikeNum = spikeNum + len(flit.Payload.rowId)
                
            
#             TILE.flitCombiner.tailNum = TILE.PENum - 1
#             output = TILE(flits) # list[Flit]
            
#             if output is None:
#                 r,c = TILE.VSAOrderCtrl(r,c)
#                 continue
            
#             outFlitsList, outcycle = output
            
#             maxLen = 0
#             for peid in range(TILE.PENum):
#                 maxLen = max(maxLen,len(outFlitsList[peid]))
            
#             for i in range(maxLen):
#                 for peid in range(TILE.PENum):
#                     OutputFlitNum = OutputFlitNum + len(outFlitsList[peid][i])
#                     output = flitCombiner(outFlitsList[peid][i])
            
#                 if output is None:
#                     r,c = TILE.VSAOrderCtrl(r,c)
#                     continue

#                 outSpikes, columnID, maxcycle = output
                
#                 maxOutcycle = max(maxOutcycle,maxcycle)
                        
#                 for spike in outSpikes:
#                     row_id, column_id, sign = spike
#                     outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1
            
#             # print("column_id",column_id)
            
#             r,c = TILE.VSAOrderCtrl(r,c)
#         outSpikesList.append(outSpikes1)    
#         perTimeStepCycle.append(maxOutcycle)
        
#         print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum(), "inputNum:",TILE.spikeNum)
        
#     outSpikesList = torch.stack(outSpikesList,dim=0)
#     print(outSpikesList.shape)
#     print(output1.shape)
#     print(firstConv[:,109])
#     print(outSpikesList[0][:,109])
#     print(output1[0,0,:,109])
#     print("InputFlitNum",InputFlitNum,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
#     print(torch.sum(outSpikesList == output1[:,0,:,:])/output1.numel())
#     torch.save(outSpikesList,"outSpikesList.pth")
#     torch.save(output1,"output1.pth")
    
#     # assert (outSpikesList == output1).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"

# def test_Tile_linear_withoutbias():
#     from copy import deepcopy

#     torch.set_printoptions(profile="full")
#     # calculate the correct result
#     M = cfg["processElement"]["input"]["M"]
#     K = cfg["processElement"]["input"]["K"]
#     N = cfg["processElement"]["input"]["N"]
#     PENum = cfg["TILE"]["PENum"]
    
#     weight = (torch.rand(N,K)*16).int()-8
#     spikes_rands = torch.rand(K,M)
#     spikes = torch.zeros(K,M).int()
#     spikes[spikes_rands < 0.05] = -1
#     spikes[spikes_rands > 0.95] = 1
    
#     M1 = cfg["processElement"]["fireComponent"]["M"]
#     N1 = cfg["processElement"]["fireComponent"]["N"]

#     neuron = STBIFNeuron(M=M1,N=N1,pos_max=8,neg_min=-7,bias=None)
    
#     output = weight@spikes 

#     refoutSpikes1 = neuron(output)
    
#     refSpikes1 = deepcopy(refoutSpikes1)
    
#     refoutSpikes2 = neuron(output)

#     refSpikes2 = deepcopy(refoutSpikes2)
    
#     layerParam = OneLayerCal()
#     layerParam.weight = weight
#     layerParam.input = spikes
#     layerParam.output = refSpikes1
#     layerParam.M = M1
#     layerParam.N = N1
#     layerParam.name = "linear"
#     layerParam.type = "linear"
    
#     TILE = Tile(layerParam=layerParam,routeInfo=(1,0)) #move to the east router
#     tileK = K//PENum
    
#     # begin calculation
#     # initialize the input spikes
#     outSpikes1 = torch.zeros(N,M)
#     outSpikes2 = torch.zeros(N,M)
#     flitGens = [FlitGenerator(RouteInfo=(1,0),PEId=i, TileValue=tileK) for i in range(PENum)]
#     flitCombiner = FlitCombiner()
    
#     for m in tqdm(range(M)):
#         flitsList = []
#         for i, flitGenerator in enumerate(flitGens):
#             flits = flitGenerator(spikes=spikes[i*tileK:(i+1)*tileK,m], columnId=m)
#             flitsList.append(flits) # list[list[Flit]]
        
#         for i in range(PENum):
#             output = TILE(flitsList[i]) # list[Flit]
        
#         outFlitsList, outcycle = output
        
        
#         for outFlits in outFlitsList:
#             output = flitCombiner(outFlits)
        
#         outSpikes, columnID, maxcycle = output
                
#         for spike in outSpikes:
#             row_id, column_id, sign = spike
#             outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1
#             # print(column_id)

#         # for spike in outSpikes:
#         #     row_id, column_id, sign = spike
#         #     outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + 1 if sign == 0 else -1

#     for m in tqdm(range(M)):
#         flitsList = []
#         for i, flitGenerator in enumerate(flitGens):
#             flits = flitGenerator(spikes=spikes[i*tileK:(i+1)*tileK,m], columnId=m)
#             flitsList.append(flits) # list[list[Flit]]
        
#         for i in range(PENum):
#             output = TILE(flitsList[i]) # list[Flit]
        
#         outFlitsList, outcycle = output
        
        
#         for outFlits in outFlitsList:
#             output = flitCombiner(outFlits)
        
#         outSpikes, columnID, maxcycle = output
                
#         for spike in outSpikes:
#             row_id, column_id, sign = spike
#             outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + 1 if sign == 0 else -1

#     # spikesList = torch.split(spikes, 1)
#     # print(outSpikes1)
#     # print(refSpikes1.int())
#     # print(outSpikes2)
#     # print(refSpikes2.int())
#     assert (outSpikes1 == refSpikes1).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
#     assert (outSpikes2 == refSpikes2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
#     # print(PE.accumulator.fCount, 2*torch.sum(torch.abs(spikes)))

# def test_Tile_linear_bias():
#     from copy import deepcopy

#     torch.set_printoptions(profile="full")
#     # calculate the correct result
#     M = cfg["processElement"]["input"]["M"]
#     K = cfg["processElement"]["input"]["K"]
#     N = cfg["processElement"]["input"]["N"]
#     PENum = cfg["TILE"]["PENum"]
    
#     weight = (torch.rand(K,N)*16).int()-8
#     spikes_rands = torch.rand(K,M)
#     spikes = torch.zeros(K,M).int()
#     spikes[spikes_rands < 0.05] = -1
#     spikes[spikes_rands > 0.95] = 1
#     bias = (torch.rand(N)*16).int()-8
    
#     M1 = 123
#     N1 = 11

#     neuron = STBIFNeuron(M=M1,N=N1,pos_max=7,neg_min=0,bias=bias.unsqueeze(1))
    
#     output = (weight.T@spikes)

#     refoutSpikes1 = neuron(output)
    
#     refSpikes1 = deepcopy(refoutSpikes1)
    
#     refoutSpikes2 = neuron(output)

#     refSpikes2 = deepcopy(refoutSpikes2)
    
#     layerParam = OneLayerCal()
#     layerParam.weight = weight.T
#     layerParam.input = spikes
#     layerParam.output = refSpikes1
#     layerParam.bias = bias
#     layerParam.M = M1
#     layerParam.N = N1
#     layerParam.name = "linear"
#     layerParam.type = "linear"
    
#     TILE = Tile(layerParam=layerParam,routeInfo=(1,0)) #move to the east router
#     tileK = K//PENum
    
#     # begin calculation
#     # initialize the input spikes
#     outSpikes1 = torch.zeros(N,M)
#     outSpikes2 = torch.zeros(N,M)
#     flitGens = [FlitGenerator(RouteInfo=(1,0),PEId=i, TileValue=tileK) for i in range(PENum)]
#     flitCombiner = FlitCombiner()
    
#     maxOutcycle = 0
    
#     for m in tqdm(range(M)):
#         flitsList = []
#         for i, flitGenerator in enumerate(flitGens):
#             flits = flitGenerator(spikes=spikes[i*tileK:(i+1)*tileK,m], columnId=m)
#             flitsList.append(flits) # list[list[Flit]]
        
#         for i in range(PENum):
#             output = TILE(flitsList[i]) # list[Flit]
        
#         outFlitsList, outcycle = output
        
#         for outFlits in outFlitsList:
#             output = flitCombiner(outFlits)
        
#         outSpikes, columnID, maxcycle = output
        
#         maxOutcycle = max(maxOutcycle,maxcycle)
                
#         for spike in outSpikes:
#             row_id, column_id, sign = spike
#             outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1
#             # print(column_id)

#         # for spike in outSpikes:
#         #     row_id, column_id, sign = spike
#         #     outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + 1 if sign == 0 else -1

#     maxOutcycle1stInfer = maxOutcycle + 0
#     print("maxOutcycle1stInfer",maxOutcycle1stInfer,"TILE.duringcycle",TILE.duringcycle)
#     maxOutcycle = 0
#     TILE.reset()
    
#     for m in tqdm(range(M)):
#         flitsList = []
#         for i, flitGenerator in enumerate(flitGens):
#             flits = flitGenerator(spikes=spikes[i*tileK:(i+1)*tileK,m], columnId=m)
#             for flit in flits:
#                 flit.time = maxOutcycle1stInfer
#             flitsList.append(flits) # list[list[Flit]]
        
        
                
#         for i in range(PENum):
#             output = TILE(flitsList[i]) # list[Flit]
        
#         outFlitsList, outcycle = output
        
        
#         for outFlits in outFlitsList:
#             output = flitCombiner(outFlits)
        
#         outSpikes, columnID, maxcycle = output

#         maxOutcycle = max(maxOutcycle,maxcycle)
        
#         for spike in outSpikes:
#             row_id, column_id, sign = spike
#             outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + 1 if sign == 0 else -1

#     maxOutcycleInfer2 = maxOutcycle + 0
#     print("maxOutcycleInfer2",maxOutcycleInfer2,"TILE.duringcycle",TILE.duringcycle)
#     # spikesList = torch.split(spikes, 1)
#     # print(outSpikes1)
#     # print(refSpikes1.int())
#     # print(outSpikes2)
#     # print(refSpikes2.int())
#     assert (outSpikes1 == refSpikes1).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
#     assert (outSpikes2 == refSpikes2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
#     # print(PE.accumulator.fCount, 2*torch.sum(torch.abs(spikes)))


