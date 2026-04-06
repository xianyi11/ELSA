import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
from processElement.functionalModule import FunctionalNeurons
from processElement.processElement import ProcessElement, ProcessElementAttention
from processElement.membrane import spikeTracer
from .Im2ColTLB import Img2ColTLB, VSAOrderController, VSAUpdateArbiter
# from VSACompiler import OneLayerCal
from router.Flit import Flit, payLoad
from typing import List, Optional
from .FlitGenerator import FlitGenerator,FlitCombiner
from processElement.STBIFFunction import STBIFNeuron
from tqdm import tqdm
import math

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
        self.head = 1
        self.softmax_call_time = 0

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

            self.M = self.layerParam.weight.shape[0]
            self.K = self.layerParam.weight.shape[1]*self.layerParam.weight.shape[2]*self.layerParam.weight.shape[3]
            self.N = self.layerParam.output.shape[3]*self.layerParam.output.shape[4]
            self.tileN = self.N//self.PENum
            print("M,K,N",self.M,self.K,self.N)
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileN, N=self.N) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.M, self.layerParam.N),matrixShape=(self.M,self.K,self.N),first=self.first,mapLayerNum=self.layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
            self.loadWeight()
        elif self.layerParam.type == "linear":
            self.N = self.layerParam.input.shape[2]
            self.K = self.layerParam.weight.shape[1]
            self.M = self.layerParam.weight.shape[0]
            print("TILE in linear: self.N, self.K, self.M",self.N, self.K, self.M)
            sym = True
            if self.layerParam.name.count("fc1") > 0:
                sym = False
            self.tileM = self.M//self.PENum
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileM, N=self.N) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.LastVthr1, self.layerParam.Vthr),matrixShape=(self.tileM,self.K,self.N),first=self.first,tileM=self.tileM,sym=sym,mapLayerNum=self.layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
                if self.layerParam.name.count("qkvv") > 0:
                    PE.transpose = True
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
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileN, N=self.N) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.M, self.layerParam.N),matrixShape=(self.M,self.K,self.N),first=self.first,mapLayerNum=self.layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
            self.loadWeight()
        elif self.layerParam.type == "residual_norm": # identity, no matrix multiplication
            self.N = self.layerParam.input.shape[2]
            self.K = self.M = self.layerParam.input.shape[3]
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=0,TileValue=self.M, N=self.N)]
            print("self.layerParam.midVthr, self.layerParam.Vthr",self.layerParam.midVthr, self.layerParam.Vthr)
            self.layernormFunction = FunctionalNeurons(quantizeParam=(self.layerParam.midVthr, self.layerParam.Vthr),function='layernorm',head=1,squenceLen=self.K,squenceNum=self.N)
            self.AdditionFunction = FunctionalNeurons(quantizeParam=(self.layerParam.LastVthr1, self.layerParam.LastVthr2, self.layerParam.midVthr),function='residual_addition',head=1,squenceLen=self.K,squenceNum=self.N)
            self.input_register1 = torch.zeros(self.M)
            self.input_register2 = torch.zeros(self.M)
            self.input_count = 0
            self.PENum = 1
            self.loadWeight()
        elif self.layerParam.type == "multiplication":
            self.M = self.layerParam.weight.shape[3]
            self.N = self.layerParam.input.shape[3]
            self.K = self.layerParam.weight.shape[2]
            self.H = self.layerParam.weight.shape[1]
            self.head = self.H
            self.tileM = self.M//self.PENum
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=i,TileValue=self.tileM, N=self.N) for i in range(self.PENum)]
            self.PEs = [ProcessElement(quantizeParam=(self.layerParam.LastVthr1, self.layerParam.Vthr),head=self.H,matrixShape=(self.tileM,self.K,self.N),first=self.first,tileM=self.tileM,mapLayerNum=self.layerParam.mapLayerNum) for i in range(self.PENum)]
            for PE in self.PEs:
                PE.fireComponent.N = self.layerParam.N
                PE.fireComponent.M = self.layerParam.M
            self.loadWeight()
        elif self.layerParam.type == "multiplication_softmax":
            T,B,H,L,N = self.layerParam.input.shape
            self.N = L
            self.K = N
            self.M = L
            self.head = H
            tileNum1 = self.N//int(math.sqrt(self.PENum))
            tileNum2 = self.N - self.N//int(math.sqrt(self.PENum))
            self.tileN = [tileNum1,tileNum2]
            self.tileM = [tileNum1,tileNum2]
            self.flitGenerators = [FlitGenerator(RouteInfo=routeInfo,PEId=0,TileValue=self.tileN[0],N=self.N)]
            self.softmaxFunction = FunctionalNeurons(quantizeParam=(self.layerParam.midVthr, self.layerParam.Vthr),function='softmax',head=H,squenceLen=self.M,squenceNum=self.N)
            # for simplify, we use fake-quantization
            self.PEs = [ProcessElementAttention(quantizeParam=(self.layerParam.LastVthr1 ,self.layerParam.LastVthr2, self.layerParam.midVthr),matrixShape=(tileNum1,self.K,tileNum1),head=self.head,first=False,mapLayerNum=self.layerParam.mapLayerNum),
                        ProcessElementAttention(quantizeParam=(self.layerParam.LastVthr1 ,self.layerParam.LastVthr2, self.layerParam.midVthr),matrixShape=(tileNum1,self.K,tileNum2),head=self.head,first=False,mapLayerNum=self.layerParam.mapLayerNum),
                        ProcessElementAttention(quantizeParam=(self.layerParam.LastVthr1 ,self.layerParam.LastVthr2, self.layerParam.midVthr),matrixShape=(tileNum2,self.K,tileNum1),head=self.head,first=False,mapLayerNum=self.layerParam.mapLayerNum),
                        ProcessElementAttention(quantizeParam=(self.layerParam.LastVthr1 ,self.layerParam.LastVthr2, self.layerParam.midVthr),matrixShape=(tileNum2,self.K,tileNum2),head=self.head,first=False,mapLayerNum=self.layerParam.mapLayerNum),
                        ]
            for PE in self.PEs:
                PE.fireComponent.N = 0
                PE.fireComponent.M = 1
                PE.head = self.head
            self.loadWeight()
            
        
        self.spikeNum = 0
        self.spikeNumAfterImg2Col = 0
        self.debug = True
        self.maxcycleRecord = 0
        # 每个 token 的 bubble 统计（不包含冷启动）：由 forward 写入，供 VSACompiler 读取
        self.lastTokenTotalCycles = 0
        self.lastTokenComputationCycles = 0
        self.lastTokenBubbleRatio = 0.0
    def _update_bubble_ratio(self, Outupdate=False):
        """根据本 token 的 input 起始 cycle 与 outcycle、cycleForUpdate 更新 bubble 占流水比例。仅当本次 forward 为正常输入（非 Outupdate）时更新。"""
        if Outupdate or not hasattr(self, "_lastInputStartCycle"):
            return
        self.lastTokenTotalCycles = max(0, self.outcycle - self._lastInputStartCycle)
        self.lastTokenComputationCycles = self.cycleForUpdate
        if self.lastTokenTotalCycles > 0:
            self.lastTokenBubbleRatio = (self.lastTokenTotalCycles - self.lastTokenComputationCycles) / self.lastTokenTotalCycles
        else:
            self.lastTokenBubbleRatio = 0.0

    def reset(self):
        self.outcycle = 0
        self.duringcycle = 0        
        self.spikeNum = 0
    
    def calEnergy(self, latency):
        totalEnergy = 0.0
        if self.layerParam.type != "residual_norm":
            for PE in self.PEs:
                totalEnergy = totalEnergy + PE.calEnergy(latency)
        else:
            totalEnergy = totalEnergy + self.layernormFunction.calEnergy(latency) + self.AdditionFunction.calEnergy(latency)
        if self.layerParam.type.count("softmax") > 0:
            totalEnergy = totalEnergy + self.softmaxFunction.calEnergy(latency)
        for flitGenerator in self.flitGenerators:
            totalEnergy = totalEnergy + flitGenerator.calEnergy(latency)
        totalEnergy = totalEnergy + self.flitCombiner.calEnergy(latency)
        if self.layerParam.type == "conv" or self.layerParam.type == "pool":
            totalEnergy = totalEnergy + self.Im2ColUnit.calEnergy(latency) + self.VSAOrderCtrl.calEnergy(latency) + self.VSAOrderArbiter.calEnergy(latency)
        return totalEnergy

    def getArea(self):
        totalArea = 0.0
        if self.layerParam.type != "residual_norm":
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
        if self.layerParam.type == "conv" or self.layerParam.type == "linear" or self.layerParam.type == "multiplication":
            weight = self.layerParam.weight
            if self.layerParam.type == "multiplication":
                weight2D = weight.reshape(-1,weight.shape[-1]) # H*K,M
            else:
                weight2D = weight.reshape(weight.shape[0],-1).T # K,M
            
            assert self.tileM*self.PENum == self.M, "The tileM*PENum must equal the M"
            
            for h in range(self.head):        
                for i,PE in enumerate(self.PEs):
                    #load weight
                    for k in range(self.K):
                        block_id = k//PE.weightBuffer[h].tileHN
                        row_id = k%PE.weightBuffer[h].tileHN
                        PE.weightBuffer[h].input_data(weight2D[k+h*self.N, (i*self.tileM):((i+1)*self.tileM)], block_id, row_id)

                    #load bias
                    for n in range(self.N):
                        block_id = n//PE.membrane[h].tileHN
                        row_id = n%PE.membrane[h].tileHN
                        if self.layerParam.bias is None:
                            PE.membrane[h].input_data(torch.zeros(self.tileM)+0.5*self.layerParam.Vthr, block_id, row_id)
                        else:
                            PE.membrane[h].input_data(torch.zeros(self.tileM)+0.5*self.layerParam.Vthr + self.layerParam.bias[(i*self.tileM):((i+1)*self.tileM)], block_id, row_id)

                    # initial spike tracer
                    for n in range(self.N):
                        block_id = n//PE.membrane[h].tileHN
                        row_id = n%PE.membrane[h].tileHN
                        PE.spikeTracer[h].input_data(torch.zeros(self.tileM), block_id, row_id)

        if self.layerParam.type == "multiplication_softmax":
            # print("self.M",self.M,"self.tileM",self.tileM)
            assert self.tileM[0]+self.tileM[1] == self.M, "The tileN*PENum must equal the N"
            
            for h in range(self.head):
                for i,PE in enumerate(self.PEs):
                    #load bias
                    for n in range(PE.N):
                        PE.weightBufferQ[h].input_data(torch.zeros(self.K),i, direction=0)

                    for n in range(PE.M):
                        PE.weightBufferK[h].input_data(torch.zeros(self.K),i, direction=1)

                    for n in range(PE.N):
                        block_id = n//PE.weightBlockWidth
                        row_id = n%PE.weightBlockWidth
                        PE.membrane[h].input_data(torch.zeros(PE.M)+0.5*self.layerParam.midVthr, n, direction=0)
                        PE.spikeTracer[h].input_data(torch.zeros(PE.M), block_id, row_id)
                for n in range(self.N):
                    block_id = n//PE.weightBlockWidth
                    row_id = n%PE.weightBlockWidth
                    self.softmaxFunction.membrane[h].input_data(torch.zeros(self.M)+0.5*self.layerParam.Vthr, block_id, row_id)
                    self.softmaxFunction.spikeTracer[h].input_data(torch.zeros(self.M), block_id, row_id)
            
        if self.layerParam.type == "residual_norm":
            for h in range(self.head):
                for n in range(self.N):
                    block_id = n//self.layernormFunction.membrane[h].tileHN
                    row_id = n%self.layernormFunction.membrane[h].tileHN
                    # PE..membrane[h].input_data(torch.zeros(self.tileN)+0.5*self.layerParam.VThr, i, direction=0)
                    self.layernormFunction.membrane[h].input_data(torch.zeros(self.M)+0.5*self.layerParam.Vthr, block_id, row_id)
                    self.layernormFunction.spikeTracer[h].input_data(torch.zeros(self.M), block_id, row_id)
                    self.AdditionFunction.membrane[h].input_data(torch.zeros(self.M)+0.5*self.layerParam.midVthr, block_id, row_id)
                    self.AdditionFunction.spikeTracer[h].input_data(torch.zeros(self.M), block_id, row_id)
                    self.layernormFunction.module.layerNorm.weight.data = self.layerParam.weight
                    self.layernormFunction.module.layerNorm.bias.data = self.layerParam.bias
            
    def forward(self, flitList:Optional[List[Flit]], Outupdate=False, OutRowId=0, timeStep=0):
        functionCycle = 0
        if Outupdate == False:
            if self.layertype == "residual_norm":
                self.input_count = self.input_count + 1
            # print("=================================TILE======================================")
            # for flit in flitList:
            #     flit.printmyself()
            output = self.flitCombiner(flitList)
            if output is None:
                return None
            # 本 token 首个 flit 到达时间，用于计算 bubble 占流水比例（不包含冷启动由 compiler 侧排除）
            self._lastInputStartCycle = min(f.time for f in flitList)
            spikes, rowColID, maxcycle, colFirst = output
            # print("spikes",spikes)
            # print("colFirst",colFirst)
            # If the addition between the time when last input flit enters and the calculate time when last input flit consume is larger than the time when current filt enter, waiting begin.
            # print("maxcycle",maxcycle,"self.lastFlitInputTime",self.lastFlitInputTime,"self.cycleForUpdate",self.cycleForUpdate)
            maxcycle = max(maxcycle, self.lastFlitInputTime + self.cycleForUpdate)
            self.maxcycleRecord = maxcycle
            self.spikeNum = self.spikeNum + len(spikes)
            
            if self.layertype == "conv":
                spikesCol = []
                for spike in spikes:
                    onespikeCol = self.Im2ColUnit(spike)
                    for s in onespikeCol:
                        spikesCol.append(s)
                spikes = spikesCol
            
            self.spikeNumAfterImg2Col = self.spikeNumAfterImg2Col + len(spikes)
        else:
            maxcycle = self.maxcycleRecord
            colFirst = False

        outFlitsList = [[] for i in range(self.PENum)]
        curPEId = 0
        
        # define the headId
        if Outupdate == False:
            headId= rowColID//self.N
            rowID = rowColID%self.N
        else:
            headId= OutRowId//self.N
            rowID = OutRowId%self.N
        
        if self.layertype == "linear":
            curPEId = 0
            for PE, flitGenerator in zip(self.PEs, self.flitGenerators):
                update=True
                # print("receive spike number: len(spikes)",len(spikes))                    
                outspike, _ = PE(spikes,update=True,tarRowId=rowID,headId=headId)
                outFlitsList[curPEId] = flitGenerator(outspike,columnId=rowID,headId=headId, ColFirst=False)
                curPEId = curPEId + 1
        elif self.layertype == "multiplication":
            curPEId = 0
            for PE, flitGenerator in zip(self.PEs, self.flitGenerators):
                update=True
                for spikeId in range(len(spikes)):
                    spikes[spikeId] = (spikes[spikeId][0] % self.N, spikes[spikeId][1], spikes[spikeId][2])
                outspike, _ = PE(spikes,update=True,tarRowId=rowID,headId=headId)
                # print("headId",headId,"outspike",outspike)
                outFlitsList[curPEId] = flitGenerator(outspike,columnId=rowID,headId=headId, ColFirst=False)
                curPEId = curPEId + 1
        elif self.layertype == "multiplication_softmax":
            PERowId = 0 if rowID < self.tileN[0] else 1 
            PEColId = 0 if rowID < self.tileN[0] else 1
            rowIDForsoftmax = rowID
            rowID = rowID if rowID < self.tileN[0] else rowID-self.tileN[0]
            PEIdlist = []
            if colFirst:
                for i in range(2):
                    PEIdlist.append(i*2 + PEColId)
            else:
                for i in range(2):
                    PEIdlist.append(PERowId*2 + i)

            outspikeAllPEs = []
            spikeTracerAllPEs = []
            for PEId in PEIdlist:
                # print("headId",headId,"PERowId",PERowId,"PEColId",PEColId,"rowID",rowID,"rowIDForsoftmax",rowIDForsoftmax, "colFirst",colFirst, "PEIdlist",PEIdlist)
                if not Outupdate:
                    update = False
                    for spikeId in range(len(spikes)):
                        if not colFirst:
                            spikes[spikeId] = (rowID, spikes[spikeId][1], spikes[spikeId][2])
                        else:
                            spikes[spikeId] = (spikes[spikeId][0], rowID, spikes[spikeId][2]) 
                    self.PEs[PEId](spikes,update=False,right=(not colFirst),tarRowId=rowID,headId=headId)
                else:
                    row_block_id = rowID//self.PEs[PEId].spikeTracer[headId].tileHN
                    row_inner_id = rowID%self.PEs[PEId].spikeTracer[headId].tileHN                
                    spikeTracerAllPEs.append(self.PEs[PEId].spikeTracer[headId].output_data(row_block_id,row_inner_id))
                    outspike, _ = self.PEs[PEId]([],update=True,right=True,tarRowId=rowID,headId=headId)
                    outspikeAllPEs.append(outspike)
            if Outupdate:
                update=True
                outspike = torch.cat(outspikeAllPEs)
                # if self.debug and rowIDForsoftmax == 2 and headId==5:
                #     print("outspike*self.layerParam.midVthr",outspike*self.layerParam.midVthr)
                    # self.debug = False
                spiketracer = torch.cat(spikeTracerAllPEs)
                outspike, functionCycle = self.softmaxFunction(outspike, spiketracer, rowIDForsoftmax, headId = headId, timeStep=timeStep)
                # if self.debug and rowIDForsoftmax == 2 and headId==5:
                #     print("outspike*self.layerParam.Vthr",outspike*self.layerParam.Vthr)
                    # self.debug = False
                outFlitsList[0] = self.flitGenerators[0](outspike,columnId=rowIDForsoftmax,headId=headId, ColFirst=False)
        elif self.layertype == "conv":
            curPEId = 0
            for PE, flitGenerator in zip(self.PEs, self.flitGenerators):
                # calculate the input rowID and columnID
                i = rowID//self.layerParam.input.shape[-1]
                j = rowID%self.layerParam.input.shape[-1]

                # calculate the output rowID and TarcolumnID
                update, outputIList, outputJLsit = self.VSAOrderArbiter(i,j)
                if update:
                    index = 0
                    for outputI,outputJ in zip(outputIList, outputJLsit):
                        # print("update",update,"i,j",i,j,"outputI, outputJ",outputI, outputJ,"tarcolumnID",outputI*self.layerParam.output.shape[-1]+outputJ)
                        # update: the update column is equal to the TarcolumnID
                        tarcolumnID = outputI*self.layerParam.output.shape[-1] + outputJ
                        if index == 0:
                            outspike, column_id = PE(spikes,update=True,tarColumnId=tarcolumnID)
                        else:
                            outspike, column_id = PE([],update=True,tarColumnId=tarcolumnID)
                        index = index + 1

                        outFlitsList[curPEId].append(flitGenerator(outspike,columnId=column_id))

                else:
                    tarcolumnID = 0
                    outspike, column_id = PE(spikes,update=False,tarColumnId=tarcolumnID)
                curPEId = curPEId + 1      
        elif self.layertype == "residual_norm":
            if self.input_count == 1:
                for spike in spikes:
                    row_id,column_id,sign = spike
                    self.input_register1[column_id] = 1 if sign==0 else -1
                return None

            if self.input_count == 2:
                for spike in spikes:
                    row_id,column_id,sign = spike
                    self.input_register2[column_id] = 1 if sign==0 else -1
                self.input_count = 0

            if self.input_count == 0:
                update=True

                # if (self.input_register1 == 0).all() and (self.input_register2 == 0).all(): # if no input, pass this calculation
                #     outFlitsList[0] = self.flitGenerators[0](torch.zeros(self.M),columnId=rowID,headId=0, ColFirst=False)
                #     self.cycleForUpdate = 1
                #     self.lastFlitInputTime = maxcycle
                #     self.outcycle = maxcycle + self.cycleForUpdate
                #     for outFlits in outFlitsList:
                #         for flit in outFlits:
                #             if isinstance(flit,List):
                #                 for lflit in flit:                        
                #                     lflit.time = lflit.time + maxcycle + self.cycleForUpdate
                #             else:
                #                 flit.time = flit.time + maxcycle + self.cycleForUpdate
                #     return outFlitsList, self.outcycle

                row_block_id = rowID//self.AdditionFunction.spikeTracer[0].tileHN
                row_inner_id = rowID%self.AdditionFunction.spikeTracer[0].tileHN
                spiketracer = self.AdditionFunction.spikeTracer[0].output_data(row_block_id,row_inner_id)

                # print("self.input_register1",self.input_register1)
                # print("self.input_register2",self.input_register2)
                outspike, functionCycle1 = self.AdditionFunction(self.input_register1, self.input_register2, rowID, headId = 0)
                # if self.debug:
                #     print("addition output",outspike)
                #     self.debug = False
                self.input_register1[:] = 0
                self.input_register2[:] = 0
                
                # print("outspike",outspike)
                # print("spiketracer",spiketracer)
                outspike, functionCycle2 = self.layernormFunction(outspike, spiketracer, rowID, headId = 0,timeStep=timeStep)
                outFlitsList[0] = self.flitGenerators[0](outspike,columnId=rowID,headId=0, ColFirst=False)
                
                # print("functionCycle1, functionCycle2",functionCycle1, functionCycle2)
                self.cycleForUpdate = functionCycle1 + functionCycle2
                self.lastFlitInputTime = maxcycle
                self.outcycle = maxcycle + self.cycleForUpdate
                self._update_bubble_ratio(Outupdate=False)
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
                    return None
        
        for PE in self.PEs:
            duringcycle = max((PE.computationCycle + functionCycle), self.duringcycle)
        self.cycleForUpdate = duringcycle - self.duringcycle
        # print("PE.adder.fCount",PE.adder.fCount,"self.duringcycle",self.duringcycle,"self.cycleForUpdate",self.cycleForUpdate,"maxcycle",maxcycle,"self.maxcycleRecord",self.maxcycleRecord)
        self.duringcycle = duringcycle + 0
        self.lastFlitInputTime = maxcycle
        self.outcycle = maxcycle + self.cycleForUpdate
        self._update_bubble_ratio(Outupdate=Outupdate)
        if update:
            # print("PE.computationCycle",PE.computationCycle)
            # for calculate latency
            for outFlits in outFlitsList:
                for flit in outFlits:
                    if isinstance(flit,List):
                        for lflit in flit:                        
                            lflit.time = lflit.time + maxcycle + self.cycleForUpdate
                    else:
                        flit.time = flit.time + maxcycle + self.cycleForUpdate
            # for outFlits in outFlitsList:
            #     for flit in outFlits:
            #         flit.printmyself()
            return outFlitsList, self.outcycle
        else:
            return None

# class OneLayerCal():
#     def __init__(self):
#         self.weight = None
#         self.weight2 = None
#         self.bias = None
#         self.input = None
#         self.input2 = None
#         self.output = None
#         self.M = None
#         self.N = None
#         self.LastVthr1 = None
#         self.LastVthr2 = None
#         self.Vthr = None
#         self.midVthr = None
#         self.name = None # name in network
#         self.type = None # conv, linear, average pooling or residual addition
#         self.ops = 0
    
#     def printmyself(self):
#         print(f"=========================={self.name}============================")
#         if self.type is not None:
#             print("self.type",self.type)
#         if self.weight is not None:
#             print("self.weight.shape",self.weight.shape)
#         if self.weight2 is not None:
#             print("self.weight.shape",self.weight2.shape)
#         if self.bias is not None:
#             print("self.bias.shape",self.bias.shape)
#         if self.input is not None:
#             print("self.input.shape",self.input.shape)
#         if self.input2 is not None:
#             print("self.input2.shape",self.input2.shape)
#         if self.output is not None:
#             print("self.output.shape",self.output.shape)
#         if self.M is not None:
#             print("self.M, self.N",self.M, self.N)
#         if self.LastVthr1 is not None:
#             print("self.LastVthr1",self.LastVthr1)
#         if self.LastVthr2 is not None:
#             print("self.LastVthr2",self.LastVthr2)
#         if self.midVthr is not None:
#             print("self.midVthr",self.midVthr)
#         if self.Vthr is not None:
#             print("self.Vthr",self.Vthr)
#         # print("self.ops",self.ops)

def test_Tile_linear_bias():
    from copy import deepcopy
    from VSACompiler import layerParam
    
    torch.set_printoptions(profile="full")
    
    # layerParam = torch.load("/home/kang_you/simulator_new_transformer/test_layers/blocks.0.attn_qkv")
    print(layerParam.printmyself())
    
    T,B,L,N = layerParam.output.shape
    qkv = layerParam.output.reshape(T,B,L,3,N//3)
    q,k,v = qkv.unbind(dim=3)

    M,K = layerParam.weight.shape
    qkv_weight = layerParam.weight.reshape(3,M//3,K)
    q_weight,k_weight,v_weight = qkv_weight.unbind(dim=0)
    
    # layerParam.weight = q_weight
    # layerParam.output = q
    
    bias = layerParam.bias
    weight = layerParam.weight
    input = layerParam.input
    groudtruth = layerParam.output    
    neuron = STBIFNeuron(threshold=torch.unique(groudtruth)[-1],pos_max=7,neg_min=-8,bias=None)
    T = layerParam.input.shape[0]
    
    outputList = []
    wxList = []
    for t in range(T):    
        wx = torch.nn.functional.linear(input[t], weight)
        wx = wx.reshape(B,L,3,N//3)
        q,k,v = wx.unbind(dim=2)
        print("q.shape",q.shape)
        # print(neuron(wx))
        wxList.append(q)
        outputList.append(neuron(q)+0.0)
    
    wx1 = torch.stack(wxList,dim=0)
    output1 = torch.stack(outputList,dim=0)
    print(wx1[0,0,0,:])
    print(output1[0,0,0,:])
    # print(groudtruth[0,0,0,:])
    

    assert (output1 == groudtruth).all(), "output1 != groudtruth"

    # M1 = layerParam.M
    # N1 = layerParam.N
    # bias = layerParam.bias
    # weight = layerParam.weight
    # input = layerParam.input
    # groudtruth = layerParam.output
    # neuron = STBIFNeuron(M=M1,N=N1,pos_max=7,neg_min=0,bias=bias.unsqueeze(1))    
    # T = layerParam.input.shape[0]
    # outputList = []
    # stride = layerParam.input.shape[-1]//layerParam.output.shape[-1]
    # KW = layerParam.weight.shape[-1]
    # K = layerParam.weight.shape[1]*layerParam.weight.shape[2]*layerParam.weight.shape[3]
    # firstConv = None
    
    # for t in range(T):    
    #     wx = torch.nn.functional.conv2d(input[t], weight, stride=stride, padding=KW//2)
    #     if t == 0:
    #         firstConv = (wx*M1).reshape(wx.shape[1],-1) + 2**(N1-1) + bias.unsqueeze(1)*(2**(N1))
    #     outputList.append(neuron(wx)+0)
    
    
    # print(firstConv[:,0])
    # print(outputList[0].reshape(outputList[0].shape[1],-1)[:,0])
    # output1 = torch.stack(outputList,dim=0)
    
    # # print(output1[0,0,0,0])
    # # print(groudtruth[0,0,0,0])
    # assert (output1 == groudtruth).all(), "output1 != groudtruth"
    
    # TILE = Tile(layerParam=layerParam,routeInfo=(1,0)) #move to the east router

    # output1 = output1.reshape(output1.shape[0],output1.shape[1],output1.shape[2],-1)


    # flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
    # flitCombiner = FlitCombiner()
    # InputFlitNum = 0
    # OutputFlitNum = 0
    
    # perTimeStepCycle = []
    
    
    # outSpikesList = []
    # for t in tqdm(range(T)):
    #     # if t > 1:
    #     #     break
    #     H,W = input.shape[-2],input.shape[-1]
    #     input2D = input[t].reshape(input[t].shape[1],-1)
    #     r = c = 0
    #     outSpikes1 = torch.zeros(output1.shape[2],output1.shape[3])
    #     InputRows = input2D.shape[0]
    #     InputColumns = input2D.shape[1]
    #     print("input2D",input2D.shape)
    #     spikeNum = 0
    #     maxOutcycle = 0
        
    #     for m in tqdm(range(InputColumns)):
    #         # if m > 100:
    #         #     break
    #         # print("r,c",r,c)
    #         columnId = r*W+c
    #         # if (input2D[:,columnId] == 0).all():
    #         #     r,c = TILE.VSAOrderCtrl(r,c)
    #         #     continue
    #         flits = flitGenerator(spikes=input2D[:,columnId], columnId=columnId)
    #         InputFlitNum = InputFlitNum + len(flits)
    #         for flit in flits:
    #             spikeNum = spikeNum + len(flit.Payload.rowId)
                
            
    #         TILE.flitCombiner.tailNum = TILE.PENum - 1
    #         output = TILE(flits) # list[Flit]
            
    #         if output is None:
    #             r,c = TILE.VSAOrderCtrl(r,c)
    #             continue
            
    #         outFlitsList, outcycle = output
            
    #         maxLen = 0
    #         for peid in range(TILE.PENum):
    #             maxLen = max(maxLen,len(outFlitsList[peid]))
            
    #         for i in range(maxLen):
    #             for peid in range(TILE.PENum):
    #                 OutputFlitNum = OutputFlitNum + len(outFlitsList[peid][i])
    #                 output = flitCombiner(outFlitsList[peid][i])
            
    #             if output is None:
    #                 r,c = TILE.VSAOrderCtrl(r,c)
    #                 continue

    #             outSpikes, columnID, maxcycle = output
                
    #             maxOutcycle = max(maxOutcycle,maxcycle)
                        
    #             for spike in outSpikes:
    #                 row_id, column_id, sign = spike
    #                 outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1
            
    #         # print("column_id",column_id)
            
    #         r,c = TILE.VSAOrderCtrl(r,c)
    #     outSpikesList.append(outSpikes1)    
    #     perTimeStepCycle.append(maxOutcycle)
        
    #     print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum(), "inputNum:",TILE.spikeNum)
        
    # outSpikesList = torch.stack(outSpikesList,dim=0)
    # print(outSpikesList.shape)
    # print(output1.shape)
    # print(firstConv[:,109])
    # print(outSpikesList[0][:,109])
    # print(output1[0,0,:,109])
    # print("InputFlitNum",InputFlitNum,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
    # print(torch.sum(outSpikesList == output1[:,0,:,:])/output1.numel())
    # torch.save(outSpikesList,"outSpikesList.pth")
    # torch.save(output1,"output1.pth")
    
    # assert (outSpikesList == output1).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"

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


