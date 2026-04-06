import torch
from basicModule import VSAModule
from .accumulators import Adder, AdderTree
from .fireComponent import fireComponent
from .inputBuffer import inputBuffer
from .membrane import membrane, spikeTracer,membraneCRAM
from .weightBuffer import weightBuffer,weightBufferCRAM
from .spikeEncoder import spikeEncoder, spikeEncoderRow
from .STBIFFunction import STBIFNeuron
from tqdm import tqdm
import math

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class ProcessElement(VSAModule):
    def __init__(self, quantizeParam, matrixShape, head=1, identity=False, depthWise=False, kernelSize=1, first=True, tileM = None, sym=True, mapLayerNum=1):
        super(ProcessElement,self).__init__()
        self.parallelism = cfg["processElement"]["parallelism"]
        self.identity = identity
        self.kernelSize = kernelSize
        self.depthWise = depthWise
        self.mapLayerNum = mapLayerNum
        self.first = first
        self.head = head
        self.transpose = False # 针对Value矩阵，需要做转置
        # self.M = cfg["processElement"]["input"]["M"]
        # self.K = cfg["processElement"]["input"]["K"]
        # self.N = cfg["processElement"]["input"]["N"]
        self.M, self.K, self.N = matrixShape
        if tileM is None:
            self.tileM = self.M//cfg["TILE"]["PENum"]
        else:
            self.tileM = tileM

        # determine the number of adders and calculate the number of heads that are calculated simultaneously
        self.maxAdderNumber = cfg["processElement"]["adderTree"]["adderNum"]
        # self.headParallelism = min(math.floor(self.maxAdderNumber/self.tileM),self.head)

        self.weightBlockWidth = cfg["processElement"]["weightBuffer"]["height"]
        self.weightBufferWordNum = math.ceil(self.weightBlockWidth/(math.ceil(self.tileM/cfg["processElement"]["weightBuffer"]["inoutWidth"])))
        self.membraneBlockWidth = cfg["processElement"]["membrane"]["height"]
        self.membraneWordNum = math.ceil(self.membraneBlockWidth/(math.ceil(self.tileM/cfg["processElement"]["membrane"]["inoutWidth"])))
        self.LastvThr, self.vThr = quantizeParam
        print("self.N, self.K, self.M, self.LastvThr, self.vThr, self.head", self.N, self.K,self.M,self.LastvThr, self.vThr, self.head)
        
        # initilize storage
        self.inputBuffer = inputBuffer(queueDepth=cfg["processElement"]["inputBuffer"]["queueDepth"], parallelism=self.parallelism,first=self.first)
        self.weightBuffer = [weightBuffer(K=self.K,M=self.tileM) for h in range(head)]
        self.spikeTracer = [spikeTracer(N=self.N,M=self.tileM) for h in range(head)]
        self.membrane = [membrane(N=self.N,M=self.tileM) for h in range(head)]

        # print("self.inputBuffer.sram number:",len(self.inputBuffer.srams))
        # print("self.weightBuffer.sram number:",len(self.weightBuffer.srams),self.weightBuffer.SRAMNumber)
        # print("self.spikeTracer.sram number:",len(self.spikeTracer.srams),self.spikeTracer.SRAMNumber)
        # print("self.membrane.sram number:",len(self.membrane.srams),self.membrane.SRAMNumber)
        
        # initilize computation component
        # self.accumulator = accumulator(treeWidth=self.tileN, parallelism=self.parallelism)
        self.adder = Adder()
        self.adderTree = AdderTree()
        self.fireComponent = fireComponent(width=self.tileM,vthr=self.vThr,sym=sym)
        self.fireComponent.set_vThr(self.vThr)
        
        # output interface
        self.spikeEncoder = spikeEncoderRow()
        
        # for latency calculation
        self.computationCycle = 0
    
    def getArea(self):
        if self.first:
            print("self.adder.getArea()",self.adder.getArea())
            storageMemory = 0
            for head in range(self.head):
                storageMemory = storageMemory + self.weightBuffer[head].getArea() + self.spikeTracer[head].getArea() + self.membrane[head].getArea()
            return self.inputBuffer.getArea() + storageMemory + self.adder.getArea() + self.fireComponent.getArea()
        else:
            print("self.adderTree.getArea()",self.adderTree.getArea())
            storageMemory = 0
            for head in range(self.head):
                storageMemory = storageMemory + self.weightBuffer[head].getArea() + self.spikeTracer[head].getArea() + self.membrane[head].getArea()
            return self.inputBuffer.getArea() + storageMemory + self.adderTree.getArea() + self.fireComponent.getArea()
    
    def calEnergy(self,latency):
        storageMemory = 0
        for head in range(self.head):
            storageMemory = storageMemory + self.weightBuffer[head].calEnergy(latency) + self.spikeTracer[head].calEnergy(latency) + self.membrane[head].calEnergy(latency)
        if self.first:
            return self.inputBuffer.calEnergy(latency) + storageMemory + self.adder.calEnergy(latency) + self.fireComponent.calEnergy(latency)
        else:
            return self.inputBuffer.calEnergy(latency) + storageMemory + self.adderTree.calEnergy(latency) + self.fireComponent.calEnergy(latency)
        
    def forward(self,dataStream, update=False, tarRowId=0, headId=0):
        # dataStream = virtualChannel.Clientread()
        
        dataCount = 0
        lastdataCount = 0
        memPotential = None
        spikeTracer = None
        adderCount = 0
        PESpikeNumber = 0
        # column_id = dataStream[0][1]
        # column_block_id = column_id//self.membrane.tileWN
        # column_inner_id = column_id%self.membrane.tileWN
        # memPotential = self.membrane.output_data(column_block_id,column_inner_id)
        # spikeTracer = self.spikeTracer.output_data(column_block_id,column_inner_id)
        inputBufferLoadCycle = 0
        if len(dataStream) > 0:
            while(1):
                while(1):
                    success = self.inputBuffer.get_data(dataStream[dataCount],weightBlockWidth=self.weightBlockWidth,membraneWordNum=self.membraneWordNum)
                    if success:
                        dataCount = dataCount + 1
                        if dataCount == len(dataStream):
                            break
                    else:
                        break
                    # elif dataCount != len(dataStream):
                    #     dataCount = dataCount - 1
                    #     break
                    # elif dataCount == len(dataStream):
                    #     break
                while(1):
                    spikes, empty, isColEnd = self.inputBuffer.output_data()
                    inputBufferLoadCycle = math.ceil((dataCount-lastdataCount)/self.inputBuffer.queueNum)
                    lastdataCount = dataCount + 0.0
                    # print(spikes)
                    if empty:
                        self.inputBuffer.columnRecorder = []
                        self.inputBuffer.curOutQueueId = 0
                        adderCount = 0
                        break
                    if self.first:
                        PESpikeNumber = PESpikeNumber + len(spikes)
                        for spike in spikes:
                            row_id,column_id,sign = spike
                            block_id = row_id//self.weightBlockWidth
                            row_id = row_id%self.weightBlockWidth
                            column_block_id = column_id//self.membrane[headId].tileWN
                            column_inner_id = column_id%self.membrane[headId].tileWN
                            # if self.identity:
                            #     memPotential = self.membrane.output_data(column_block_id,column_inner_id)
                            #     memPotential = self.adder(sign, [weightData])
                            #     self.membrane.input_data(memPotential,column_block_id,column_inner_id)
                            # else:
                            if self.transpose == False: #正常情况下只要1次读取就可以
                                memPotential = self.membrane[headId].output_data(block_id,row_id)
                            else: # 转置情况下，每个SRAM做16次读取
                                for i in range(self.weightBuffer.inoutWidth):
                                    memPotential = self.membrane[headId].output_data(block_id,row_id)
                            weightData = self.weightBuffer[headId].output_data(column_block_id,column_inner_id,sign)*self.LastvThr
                            # print("memPotential.shape",memPotential.shape,"weightData.shape",weightData.shape, "block_id,row_id",block_id,row_id)
                            memPotential = self.adder(memPotential, [weightData])
                            if self.transpose == False: #正常情况下只要1次读取就可以
                                self.membrane[headId].input_data(memPotential,block_id,row_id)
                            else: # 转置情况下，每个SRAM做16次读取
                                for i in range(self.weightBuffer.inoutWidth):
                                    self.membrane[headId].input_data(memPotential,block_id,row_id)

                        weightLoadCycle = math.ceil((len(spikes)*math.ceil(weightData.shape[0]/self.weightBuffer[0].inoutWidth))/(math.ceil(self.weightBuffer[0].SRAMTotalNum/self.mapLayerNum)))
                        membraneLoadCycle = 1 # 只载入一行membrane
                        membraneSaveCycle = 1 # 只保存一行membrane
                        AddCycle = math.ceil(weightData.shape[0]/(math.ceil(self.adderTree.adderNum/self.mapLayerNum)))
                        self.computationCycle = self.computationCycle + max(inputBufferLoadCycle, max(max(membraneLoadCycle,weightLoadCycle), max(AddCycle, membraneSaveCycle))) # 流水执行input load，weight/membrane load，addertree操作以及membrane save，取最大的那个块。
                        # print("inputBufferLoadCycle",inputBufferLoadCycle,"weightLoadCycle",weightLoadCycle,"AddCycle",AddCycle)
                        # print("self.mapLayerNum",self.mapLayerNum,"len(spikes)",len(spikes),'self.computationCycle',self.computationCycle,"math.ceil(weightData[0].shape[0]/self.adderTree.adderNum)",math.ceil(weightData[0].shape[0]/self.adderTree.adderNum))
                    else:
                        # print(f"================================================spikesNum={len(spikes)}===========================================")
                        # print("spikes",spikes)
                        PESpikeNumber = PESpikeNumber + len(spikes)
                        if len(spikes) > 0:
                            row_id,column_id,sign = spikes[-1]
                            row_block_id = row_id//self.membrane[headId].tileHN
                            row_inner_id = row_id%self.membrane[headId].tileHN
                            if memPotential is None:
                                if self.transpose == False:
                                    memPotential = self.membrane[headId].output_data(row_block_id,row_inner_id)
                                else: # 转置情况下，每个SRAM做16次读取
                                    for i in range(self.weightBuffer[0].inoutWidth):
                                        memPotential = self.membrane[headId].output_data(row_block_id,row_inner_id)
                                self.computationCycle = self.computationCycle + 1 # for read out latency
                            weightData = []
                            for spike in spikes:
                                row_id,column_id,sign = spike                                
                                column_block_id = column_id//self.weightBuffer[headId].tileHN
                                column_inner_id = column_id%self.weightBuffer[headId].tileHN
                                # print("headId, row_id,column_id,sign",headId,row_id,column_id,sign)
                                # print("weight:",self.weightBuffer[headId].output_data(column_block_id,column_inner_id,sign)*self.LastvThr)
                                weightData.append(self.weightBuffer[headId].output_data(column_block_id,column_inner_id,sign)*self.LastvThr)

                            # print(memPotential.shape,weightData[0].shape)
                            memPotential = self.adderTree(memPotential, weightData)
                            adderCount = adderCount + 1
                            if isColEnd == True:
                                if self.transpose == False:
                                    self.membrane[headId].input_data(memPotential,row_block_id,row_inner_id)
                                else: # 转置情况下，每个SRAM做16次读取
                                    for i in range(self.weightBuffer[0].inoutWidth):
                                        self.membrane[headId].input_data(memPotential,row_block_id,row_inner_id)
                                memPotential = None

                            # print("len(spikes)",len(spikes),"weightData[0].shape[0]",weightData[0].shape[0],"self.weightBuffer[0].inoutWidth",self.weightBuffer[0].inoutWidth,"self.weightBuffer[0].SRAMTotalNum",self.weightBuffer[0].SRAMTotalNum,"self.mapLayerNum",self.mapLayerNum)
                            weightLoadCycle = math.ceil((len(spikes)*math.ceil(weightData[0].shape[0]/self.weightBuffer[0].inoutWidth))/(math.ceil(self.weightBuffer[0].SRAMTotalNum)))
                            membraneLoadCycle = 1 # 只载入一行membrane
                            membraneSaveCycle = 1 # 只保存一行membrane
                            AddCycle = math.ceil(weightData[0].shape[0]/(math.ceil(self.adderTree.adderNum/self.mapLayerNum)))
                            self.computationCycle = self.computationCycle + max(inputBufferLoadCycle, max(max(membraneLoadCycle,weightLoadCycle), max(AddCycle, membraneSaveCycle))) # 流水执行input load，weight/membrane load，addertree操作以及membrane save，取最大的那个块。
                            # print("inputBufferLoadCycle",inputBufferLoadCycle,"weightLoadCycle",weightLoadCycle,"AddCycle",AddCycle)
                            # print("self.mapLayerNum",self.mapLayerNum,"len(spikes)",len(spikes),'self.computationCycle',self.computationCycle,"math.ceil(weightData[0].shape[0]/self.adderTree.adderNum)",math.ceil(weightData[0].shape[0]/self.adderTree.adderNum))
                    
                if dataCount == len(dataStream):
                    break

        outspike = None
        if update:
        # fire
            row_block_id = tarRowId//self.membrane[headId].tileHN
            row_inner_id = tarRowId%self.membrane[headId].tileHN
            memPotential_mid = self.membrane[headId].output_data(row_block_id,row_inner_id)
            spikeTracer = self.spikeTracer[headId].output_data(row_block_id,row_inner_id)
            outspike,memPotential = self.fireComponent(spikeTracer,memPotential_mid)
        # update
            spikeTracer = spikeTracer + outspike
            self.membrane[headId].input_data(memPotential,row_block_id,row_inner_id)
            self.spikeTracer[headId].input_data(spikeTracer,row_block_id,row_inner_id)
            self.computationCycle = self.computationCycle + 1 # fire spike
            # print("update!!! headId, outspike",headId,outspike)
            for i in range(self.K - PESpikeNumber): # 算上漏读的的weight
                weightData = [self.weightBuffer[headId].output_data(0,0,0)]
            self.computationCycle = self.computationCycle + math.ceil(((self.K - PESpikeNumber)*math.ceil(weightData[0].shape[0]/self.weightBuffer[0].inoutWidth))/(math.ceil(self.weightBuffer[0].SRAMTotalNum)))
            # print(math.ceil(((self.K - PESpikeNumber)*math.ceil(weightData.shape[0]/self.weightBuffer[0].inoutWidth))/(math.ceil(self.weightBuffer[0].SRAMTotalNum))))
                        
        return outspike*self.vThr, tarRowId

        # output encoding
        # outspikes = []
        # outspikes = self.spikeEncoder(outspike,tarRowId,outspikes)
        # return outspikes


class ProcessElementAttention(VSAModule):
    def __init__(self, quantizeParam, matrixShape, head=1, identity=False, depthWise=False, kernelSize=1, first=True, tileN = None, softmax=False, mapLayerNum=1):
        super(ProcessElementAttention,self).__init__()
        self.parallelism = cfg["processElement"]["parallelism"]
        self.identity = identity
        self.kernelSize = kernelSize
        self.depthWise = depthWise
        self.first = first
        self.head = head
        self.softmax = softmax
        self.mapLayerNum = mapLayerNum

        # self.M = cfg["processElement"]["input"]["M"]
        # self.K = cfg["processElement"]["input"]["K"]
        # self.N = cfg["processElement"]["input"]["N"]
        self.N, self.K, self.M = matrixShape

        # determine the number of adders and calculate the number of heads that are calculated simultaneously
        self.maxAdderNumber = cfg["processElement"]["adderTree"]["adderNum"]
        self.headParallelism = min(math.floor(self.maxAdderNumber/self.N),self.head)
        print("self.K, self.N, self.M, self.headParallelism",self.K, self.N, self.M, self.headParallelism)

        self.weightBlockWidth = cfg["processElement"]["weightBuffer"]["height"]
        self.weightBufferWordNum = math.ceil(self.weightBlockWidth/(math.ceil(self.N/cfg["processElement"]["weightBuffer"]["inoutWidth"])))
        self.membraneBlockWidth = cfg["processElement"]["membrane"]["height"]
        self.membraneWordNum = math.ceil(self.membraneBlockWidth/(math.ceil(self.N/cfg["processElement"]["membrane"]["inoutWidth"])))
        self.lastvthr1,self.lastvthr2,self.vthr = quantizeParam
        
        # initilize storage
        self.inputBuffer = inputBuffer(queueDepth=cfg["processElement"]["inputBuffer"]["queueDepth"], parallelism=self.parallelism,first=self.first)
        self.weightBufferQ = [weightBufferCRAM(N=self.N,K=self.K) for h in range(head)]
        self.weightBufferK = [weightBufferCRAM(K=self.M,N=self.K) for h in range(head)]
        self.spikeTracer = [spikeTracer(N=self.N,M=self.M) for h in range(head)] 
        self.membrane = [membraneCRAM(N=self.N,M=self.M) for h in range(head)] 
        
        # initilize computation component
        # self.accumulator = accumulator(treeWidth=self.tileN, parallelism=self.parallelism)
        self.adder = Adder()
        self.adderTree = AdderTree()
        self.fireComponent = fireComponent(width=self.M,vthr=self.vthr)
        self.fireComponent.set_vThr(self.vthr)
        self.rowUpdator = torch.zeros(self.K)
        
        # output interface
        self.spikeEncoder = spikeEncoderRow()
        
        # for latency calculation
        self.computationCycle = 0
        self.debug = True
    
    def getArea(self):
        if self.first:
            print("self.adder.getArea()",self.adder.getArea())
            storageMemory = 0
            for head in range(self.head):
                storageMemory = storageMemory + self.weightBufferQ[head].getArea() + self.weightBufferK[head].getArea() + self.spikeTracer[head].getArea() + self.membrane[head].getArea()
            return self.inputBuffer.getArea() + storageMemory + self.adder.getArea() + self.fireComponent.getArea()
        else:
            print("self.adderTree.getArea()",self.adderTree.getArea())
            storageMemory = 0
            for head in range(self.head):
                storageMemory = storageMemory + self.weightBufferQ[head].getArea() + self.weightBufferK[head].getArea() + self.spikeTracer[head].getArea() + self.membrane[head].getArea()
            return self.inputBuffer.getArea() + storageMemory + self.adderTree.getArea() + self.fireComponent.getArea()
    
    def calEnergy(self,latency):
        if self.first:
            storageMemory = 0
            for head in range(self.head):
                self.weightBufferQ[head].calEnergy(latency) + self.weightBufferK[head].calEnergy(latency) + self.spikeTracer[head].calEnergy(latency) + self.membrane[head].calEnergy(latency)
            return self.inputBuffer.calEnergy(latency) + storageMemory + self.adder.calEnergy(latency) + self.fireComponent.calEnergy(latency)
        else:
            storageMemory = 0
            for head in range(self.head):
                self.weightBufferQ[head].calEnergy(latency) + self.weightBufferK[head].calEnergy(latency) + self.spikeTracer[head].calEnergy(latency) + self.membrane[head].calEnergy(latency)
            return self.inputBuffer.calEnergy(latency) + storageMemory + self.adderTree.calEnergy(latency) + self.fireComponent.calEnergy(latency)
        
    def forward(self,dataStream, right=True, update=False, tarRowId=0, headId = 0):
        # dataStream = virtualChannel.Clientread()
        
        dataCount = 0
        lastdataCount = 0
        memPotential = None
        spikeTracer = None
        isColEnd = False
        PESpikeNumber = 0
        # column_id = dataStream[0][1]
        # column_block_id = column_id//self.membrane.tileWN
        # column_inner_id = column_id%self.membrane.tileWN
        # memPotential = self.membrane.output_data(column_block_id,column_inner_id)
        # spikeTracer = self.spikeTracer.output_data(column_block_id,column_inner_id)
        if len(dataStream) > 0:
            while(1):
                while(1):
                    # print("dataCount",dataCount)
                    success = self.inputBuffer.get_data(dataStream[dataCount],right=right,weightBlockWidth=self.weightBlockWidth,membraneWordNum=self.membraneWordNum)
                    if success:
                        dataCount = dataCount + 1
                        if dataCount == len(dataStream):
                            isColEnd = True
                            break
                    else:
                        # dataCount = dataCount - 1
                        break
                    # elif dataCount == len(dataStream):
                    #     break
                while(1):
                    spikes, empty, isQueueEnd = self.inputBuffer.output_data()
                    inputBufferLoadCycle = math.ceil((dataCount-lastdataCount)/self.inputBuffer.queueNum)
                    lastdataCount = dataCount + 0.0
                    # print("len(spikes)",len(spikes))
                    # print("isQueueEnd",isQueueEnd,"isColEnd",isColEnd,"empty",empty)
                    if empty:
                        self.inputBuffer.columnRecorder = []
                        self.inputBuffer.curOutQueueId = 0
                        break
                    # print(spikes)
                    PESpikeNumber = PESpikeNumber + len(spikes)
                    if len(spikes) > 0:
                        if right:
                            # print("calculating: isQueueEnd",isQueueEnd,"isColEnd",isColEnd)
                            row_id,column_id,sign = spikes[-1]
                            if memPotential is None:
                                memPotential = self.membrane[headId].output_data(row_id,direction=0)
                                self.computationCycle = self.computationCycle + 1 # for read out latency
                            weightData = []
                            for spike in spikes:
                                row_id,column_id,sign = spike
                                self.rowUpdator[column_id] = 1 if sign == 0 else -1
                                # print("self.weightBufferK[headId].output_data(column_id,sign,direction=0)",self.weightBufferK[headId].output_data(column_id,sign,direction=0)*self.lastvthr1)
                                weightData.append(self.weightBufferK[headId].output_data(column_id,sign,direction=0)*self.lastvthr1)
                            memPotential = self.adderTree(memPotential, weightData)
                            if isQueueEnd == True and isColEnd == True:
                                self.membrane[headId].input_data(memPotential,row_id,direction=0)
                                spikeTracerQ = self.weightBufferQ[headId].output_data(rowColId=row_id,sign=0,direction=0)
                                # print("self.weightBufferQ[headId].input_data(self.rowUpdator+spikeTracerQ,rowColId=row_id,direction=0)",self.rowUpdator*self.lastvthr+spikeTracerQ)
                                self.weightBufferQ[headId].input_data(self.rowUpdator*self.lastvthr1+spikeTracerQ,rowColId=row_id,direction=0)
                                self.rowUpdator[:] = 0
                                memPotential = None
                        else:
                            # print("calculating: isQueueEnd",isQueueEnd,"isColEnd",isColEnd)
                            row_id,column_id,sign = spikes[-1]
                            if memPotential is None:
                                memPotential = self.membrane[headId].output_data(column_id,direction=1)
                                self.computationCycle = self.computationCycle + 1 # for read out latency
                            weightData = []
                            for spike in spikes:
                                row_id,column_id,sign = spike
                                self.rowUpdator[row_id] = 1 if sign == 0 else -1
                                # print("self.weightBufferQ[headId].output_data(row_id,sign,direction=1)",self.weightBufferQ[headId].output_data(row_id,sign,direction=1)*self.lastvthr2)
                                weightData.append(self.weightBufferQ[headId].output_data(row_id,sign,direction=1)*self.lastvthr2)
                            memPotential = self.adderTree(memPotential, weightData)
                            if isQueueEnd == True and isColEnd == True:
                                self.membrane[headId].input_data(memPotential,column_id,direction=1)
                                spikeTracerK = self.weightBufferK[headId].output_data(rowColId=column_id,sign=0,direction=1)
                                # print("self.weightBufferK.input_data(self.rowUpdator+spikeTracerK,rowColId=column_id,direction=1)",self.rowUpdator*self.lastvthr+spikeTracerK)
                                self.weightBufferK[headId].input_data(self.rowUpdator*self.lastvthr2+spikeTracerK,rowColId=column_id,direction=1)
                                self.rowUpdator[:] = 0
                                memPotential = None

                        # print("weightData[0].shape[0]",weightData[0].shape[0],"self.weightBuffer[0].inoutWidth",self.weightBufferQ[0].inoutWidth,"self.weightBufferQ[0].SRAMTotalNum",self.weightBufferQ[0].CRAMTotalNum)
                        weightLoadCycle = math.ceil((len(spikes)*math.ceil(weightData[0].shape[0]/self.weightBufferQ[0].inoutWidth))/(math.ceil(self.weightBufferQ[0].CRAMTotalNum)))
                        membraneLoadCycle = 1 # 只载入一行membrane
                        membraneSaveCycle = 1 # 只保存一行membrane
                        AddCycle = math.ceil(weightData[0].shape[0]/(math.ceil(self.adderTree.adderNum)))
                        self.computationCycle = self.computationCycle + max(inputBufferLoadCycle, max(max(membraneLoadCycle,weightLoadCycle), max(AddCycle, membraneSaveCycle))) # 流水执行input load，weight/membrane load，addertree操作以及membrane save，取最大的那个块。
                        # print("inputBufferLoadCycle",inputBufferLoadCycle,"weightLoadCycle",weightLoadCycle,"AddCycle",AddCycle)
                        # print("self.mapLayerNum",self.mapLayerNum,"len(spikes)",len(spikes),'self.computationCycle',self.computationCycle,"math.ceil(weightData[0].shape[0]/self.adderTree.adderNum)",math.ceil(weightData[0].shape[0]/self.adderTree.adderNum))

                    
                if dataCount == len(dataStream):
                    break

        outspike = None
        if update:
        # fire
            row_block_id = tarRowId//self.spikeTracer[0].tileHN
            row_inner_id = tarRowId%self.spikeTracer[0].tileHN
            memPotential_mid = self.membrane[headId].output_data(rowColId=tarRowId,direction=0)
            spikeTracer = self.spikeTracer[headId].output_data(row_block_id,row_inner_id)
            # if self.debug and tarRowId == 2:
            #     print("memPotential_mid",memPotential_mid)
            #     print("spikeTracer",spikeTracer)
                # outspike,memPotential = self.fireComponent(spikeTracer,memPotential_mid,verbose=True)
            # else:
            outspike,memPotential = self.fireComponent(spikeTracer,memPotential_mid)
            # if self.debug and tarRowId == 2:
            #     print("outspike",outspike)
        # update
            spikeTracer = spikeTracer + outspike
            self.membrane[headId].input_data(memPotential,tarRowId,direction=0)
            self.spikeTracer[headId].input_data(spikeTracer,row_block_id,row_inner_id)
            if torch.abs(outspike).sum() != 0:
                self.computationCycle = self.computationCycle + 1

            if right:
                for i in range(self.K - PESpikeNumber): # 算上漏读的的weight
                    weightData = [self.weightBufferQ[headId].output_data(0,0,0)]
                self.computationCycle = self.computationCycle + math.ceil(((self.K - PESpikeNumber)*math.ceil(weightData[0].shape[0]/self.weightBufferQ[0].inoutWidth))/(math.ceil(self.weightBufferQ[0].CRAMTotalNum)))
            else:
                for i in range(self.K - PESpikeNumber): # 算上漏读的的weight
                    weightData = [self.weightBufferK[headId].output_data(0,0,0)]
                    self.computationCycle = self.computationCycle + math.ceil(((self.K - PESpikeNumber)*math.ceil(weightData[0].shape[0]/self.weightBufferK[0].inoutWidth))/(math.ceil(self.weightBufferK[0].CRAMTotalNum)))
            # print(math.ceil(((self.K - PESpikeNumber)*math.ceil(weightData.shape[0]/self.weightBufferQ[0].inoutWidth))/(math.ceil(self.weightBufferQ[0].CRAMTotalNum))))
            # outspikes = []
            # outspikes = self.spikeEncoder(outspike,tarRowId,outspikes)
            # return outspikes
        return outspike, tarRowId


def test_processElement_withbias():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # calculate the correct result
    M = cfg["processElement"]["input"]["M"]
    K = cfg["processElement"]["input"]["K"]
    N = cfg["processElement"]["input"]["N"]
    tileM = M//cfg["TILE"]["PENum"]
    print("N,K,M,tileM",N,K,M,tileM)
    
    vThrLast = 0.41    
    vThr = 0.51
    weight = (torch.rand(K,tileM)*2)-1
    spikes_rands = torch.rand(N,K)
    spikes = torch.zeros(N,K)
    spikes[spikes_rands < 0.05] = -vThrLast
    spikes[spikes_rands > 0.95] = vThrLast
    bias = (torch.rand(tileM)*2)-1
    
    neuron = STBIFNeuron(threshold=vThr,pos_max=7,neg_min=0,bias=bias.unsqueeze(0))
    
    output = spikes@weight 

    refoutSpikes1 = neuron(output)
    
    refSpikes1 = deepcopy(refoutSpikes1)

    refoutSpikes2 = neuron(output)

    refSpikes2 = deepcopy(refoutSpikes2)

    PE = ProcessElement(quantizeParam=(vThrLast, vThr),matrixShape=(M,K,N),first=False)
    
    # initialize weight/bias data
    for i in range(K):
        block_id = i//PE.weightBuffer.tileHN
        row_id = i%PE.weightBuffer.tileHN
        PE.weightBuffer.input_data(weight[i,:], block_id, row_id)
        
    for i in range(N):
        block_id = i//PE.membrane.tileHN
        column_id = i%PE.membrane.tileHN
        PE.membrane.input_data(torch.zeros(tileM)+0.5*vThr+bias*vThr, block_id, column_id)
        PE.spikeTracer.input_data(torch.zeros(tileM), block_id, column_id)

    # begin calculation
    # initialize the input spikes
    
    outSpikes1 = torch.zeros(N,tileM)
    outSpikes2 = torch.zeros(N,tileM)
    for n in tqdm(range(N)):
        dataStream = []
        for k in range(K):
            if spikes[n,k] > 0:
                dataStream.append((n,k,0))
            elif spikes[n,k] < 0:
                dataStream.append((n,k,1))
        outSpikes = PE(dataStream, update=True, tarRowId=n)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + vThr if sign == 0 else -vThr
        outSpikes = PE(dataStream, update=True, tarRowId=n)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + vThr if sign == 0 else -vThr

    print("TimeStep1: point accuracy:",(torch.abs(outSpikes1 - refSpikes1)<1e-4).sum()/refSpikes1.numel())
    print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-4).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpikes1 - refSpikes1)<1e-4).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    assert (torch.abs(outSpikes2 - refSpikes2)<1e-4).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    # print(PE.adderTree.fCount, 2*torch.sum(torch.abs(spikes)))

def test_processElementAttention():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    # calculate the correct result
    N = cfg["processElement"]["input"]["N"]
    K = cfg["processElement"]["input"]["K"]
    tileN = N//int(math.sqrt(cfg["TILE"]["PENum"]))
    tileK = K//int(math.sqrt(cfg["TILE"]["PENum"]))
    print("N,K,tileN,tileK",N,K,tileN,tileK)
    
    vThrLast = 0.41    
    vThr = 0.51
    
    spikes_rands = torch.rand(N,K)
    query = torch.zeros(N,K)
    query[spikes_rands < 0.5] = -vThrLast
    query[spikes_rands > 0.5] = vThrLast

    spikes_rands = torch.rand(K,N)
    key = torch.zeros(K,N)
    key[spikes_rands < 0.5] = -vThrLast
    key[spikes_rands > 0.5] = vThrLast
    
    queryQANN = query + query
    keyQANN = key + key
        
    # print("query",query)
    # print("key",key)
    outputRef = queryQANN@keyQANN
    # print("outputRef",outputRef)
    outputRef = torch.clip(torch.round(outputRef/vThr)*vThr,min=0,max=7*vThr)
    
    PE = ProcessElementAttention(quantizeParam=(vThrLast, vThr),matrixShape=(K,N),first=False)
    
    # initialize the spike tracer of query and key in weight buffer
    for i in range(N):
        PE.weightBufferQ[0].input_data(torch.zeros(K),i, direction=0)
        PE.weightBufferK[0].input_data(torch.zeros(K),i, direction=1)
    
    # initialize the output membrane and spiketracer
    for i in range(N):
        block_id = i//PE.weightBlockWidth
        row_id = i%PE.weightBlockWidth
        PE.membrane[0].input_data(torch.zeros(N)+0.5*vThr, i, direction=0)
        PE.spikeTracer[0].input_data(torch.zeros(N), block_id, row_id)

    # begin calculation
    # initialize the input spikes
    T = 15
    outSpikes1 = torch.zeros(T,N,N)
    for n in tqdm(range(N)):
        dataStream = []
        # query input
        # print("query input:",query[n,:])
        for k in range(K):
            if query[n,k] == vThrLast:
                dataStream.append((n,k,0))
            elif query[n,k] == -vThrLast:
                dataStream.append((n,k,1))
        # print("spiking query input:",dataStream)
        PE(dataStream, right=True, update=False, tarRowId=n)
        # key input
        dataStream = []
        # print("key input:",key[:,n])
        for k in range(K):
            if key[k,n] == vThrLast:
                dataStream.append((k,n,0))
            elif key[k,n] == -vThrLast:
                dataStream.append((k,n,1))
        # print("spiking key input:",dataStream)
        PE(dataStream, right=False, update=False, tarRowId=n)

    for n in tqdm(range(N)):
        outSpikes = PE([], right=True, update=True, tarRowId=n)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes1[0][row_id][column_id] = outSpikes1[0][row_id][column_id] + vThr if sign == 0 else -vThr

    for n in tqdm(range(N)):
        dataStream = []
        # query input
        for k in range(K):
            if query[n,k] > 0:
                dataStream.append((n,k,0))
            elif query[n,k] < 0:
                dataStream.append((n,k,1))
        PE(dataStream, right=True, update=False, tarRowId=n)
        # key input
        dataStream = []
        for k in range(K):
            if key[k,n] > 0:
                dataStream.append((k,n,0))
            elif key[k,n] < 0:
                dataStream.append((k,n,1))
        PE(dataStream, right=False, update=False, tarRowId=n)

    for t in range(T-1):
        for n in tqdm(range(N)):
            outSpikes = PE([], right=True, update=True, tarRowId=n)
            for spike in outSpikes:
                row_id, column_id, sign = spike
                outSpikes1[t+1][row_id][column_id] = outSpikes1[t+1][row_id][column_id] + vThr if sign == 0 else -vThr

    outSpiking = torch.sum(outSpikes1,dim=0)
    # print("outSpiking",outSpiking)
    # print("membrane potantial:",PE.membrane.cram.memory)
    # print("outputRef",outputRef)
    print(outSpiking.shape, outputRef.shape)
    print("TimeStep1: point accuracy:",(torch.abs(outSpiking - outputRef)<1e-4).sum()/outputRef.numel())
    # print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-2).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpiking - outputRef)<1e-4).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    print(PE.adderTree.fCount)

def test_processElement_withbias():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # calculate the correct result
    M = cfg["processElement"]["input"]["M"]
    K = cfg["processElement"]["input"]["K"]
    N = cfg["processElement"]["input"]["N"]
    tileM = M//cfg["TILE"]["PENum"]
    print("N,K,M,tileM",N,K,M,tileM)
    
    vThrLast = 0.41    
    vThr = 0.51
    weight = (torch.rand(K,tileM)*2)-1
    spikes_rands = torch.rand(N,K)
    spikes = torch.zeros(N,K)
    spikes[spikes_rands < 0.05] = -vThrLast
    spikes[spikes_rands > 0.95] = vThrLast
    bias = (torch.rand(tileM)*2)-1
    
    neuron = STBIFNeuron(threshold=vThr,pos_max=7,neg_min=0,bias=bias.unsqueeze(0))
    
    output = spikes@weight 

    refoutSpikes1 = neuron(output)
    
    refSpikes1 = deepcopy(refoutSpikes1)

    refoutSpikes2 = neuron(output)

    refSpikes2 = deepcopy(refoutSpikes2)

    PE = ProcessElement(quantizeParam=(vThrLast, vThr),matrixShape=(M,K,N),first=False)
    
    # initialize weight/bias data
    for i in range(K):
        block_id = i//PE.weightBuffer[0].tileHN
        row_id = i%PE.weightBuffer[0].tileHN
        PE.weightBuffer[0].input_data(weight[i,:], block_id, row_id)
        
    for i in range(N):
        block_id = i//PE.membrane[0].tileHN
        column_id = i%PE.membrane[0].tileHN
        PE.membrane[0].input_data(torch.zeros(tileM)+0.5*vThr+bias*vThr, block_id, column_id)
        PE.spikeTracer[0].input_data(torch.zeros(tileM), block_id, column_id)

    # begin calculation
    # initialize the input spikes
    
    outSpikes1 = torch.zeros(N,tileM)
    outSpikes2 = torch.zeros(N,tileM)
    for n in tqdm(range(N)):
        dataStream = []
        for k in range(K):
            if spikes[n,k] > 0:
                dataStream.append((n,k,0))
            elif spikes[n,k] < 0:
                dataStream.append((n,k,1))
        outSpikes = PE(dataStream, update=True, tarRowId=n)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + vThr if sign == 0 else -vThr
        outSpikes = PE(dataStream, update=True, tarRowId=n)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + vThr if sign == 0 else -vThr

    print("TimeStep1: point accuracy:",(torch.abs(outSpikes1 - refSpikes1)<1e-2).sum()/refSpikes1.numel())
    print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-2).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpikes1 - refSpikes1)<1e-2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    assert (torch.abs(outSpikes2 - refSpikes2)<1e-2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    # print(PE.adderTree.fCount, 2*torch.sum(torch.abs(spikes)))

def test_processElement_multiplicationMultiHead():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # calculate the correct result
    N = cfg["processElement"]["input"]["N"]
    K = cfg["processElement"]["input"]["K"]
    H = cfg["processElement"]["head"]
    print("N,K,head",N,K,H)
    
    vThrLast = 0.41    
    vThr = 0.51
    weight = (torch.rand(H,N,K)*2)-1
    spikes_rands = torch.rand(H,N,N)
    spikes = torch.zeros(H,N,N)
    spikes[spikes_rands < 0.5] = -vThrLast
    spikes[spikes_rands > 0.5] = vThrLast
    
    neuron = STBIFNeuron(threshold=vThr,pos_max=7,neg_min=0,bias=None)
    
    output = spikes@weight 

    refoutSpikes1 = neuron(output)
    
    refSpikes1 = deepcopy(refoutSpikes1)

    refoutSpikes2 = neuron(output)

    refSpikes2 = deepcopy(refoutSpikes2)

    PE = ProcessElement(quantizeParam=(vThrLast, vThr),matrixShape=(K,N,N),head=H,first=False,tileM=K)
    
    # initialize weight/bias data
    for h in range(H):
        for i in range(N):
            block_id = i//PE.weightBuffer[h].tileHN
            row_id = i%PE.weightBuffer[h].tileHN
            PE.weightBuffer[h].input_data(weight[h,i,:], block_id, row_id)
        
        for i in range(N):
            block_id = i//PE.membrane[h].tileHN
            column_id = i%PE.membrane[h].tileHN
            PE.membrane[h].input_data(torch.zeros(K)+0.5*vThr, block_id, column_id)
            PE.spikeTracer[h].input_data(torch.zeros(K), block_id, column_id)

    # begin calculation
    # initialize the input spikes
    
    outSpikes1 = torch.zeros(H,N,K)
    outSpikes2 = torch.zeros(H,N,K)
    for h in range(H):
        for n in tqdm(range(N)):
            dataStream = []
            for k in range(N):
                if spikes[h,n,k] > 0:
                    dataStream.append((n,k,0))
                elif spikes[h,n,k] < 0:
                    dataStream.append((n,k,1))
            outSpikes = PE(dataStream, update=True, tarRowId=n, headId=h)
            for spike in outSpikes:
                row_id, column_id, sign = spike
                outSpikes1[h][row_id][column_id] = outSpikes1[h][row_id][column_id] + vThr if sign == 0 else -vThr
            outSpikes = PE(dataStream, update=True, tarRowId=n, headId=h)
            for spike in outSpikes:
                row_id, column_id, sign = spike
                outSpikes2[h][row_id][column_id] = outSpikes2[h][row_id][column_id] + vThr if sign == 0 else -vThr

    print("TimeStep1: point accuracy:",(torch.abs(outSpikes1 - refSpikes1)<1e-2).sum()/refSpikes1.numel())
    print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-2).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpikes1 - refSpikes1)<1e-2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    assert (torch.abs(outSpikes2 - refSpikes2)<1e-2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    # print(PE.adderTree.fCount, 2*torch.sum(torch.abs(spikes)))

def test_processElementAttentionTiling():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    # calculate the correct result
    N = cfg["processElement"]["input"]["N"]
    K = cfg["processElement"]["input"]["K"]
    tileN = N//int(math.sqrt(cfg["TILE"]["PENum"]))
    tileK = K//int(math.sqrt(cfg["TILE"]["PENum"]))
    print("N,K,tileN,tileK",N,K,tileN,tileK)
    
    vThrLast = 0.41    
    vThr = 0.51
    
    spikes_rands = torch.rand(N,K)
    query = torch.zeros(N,K)
    query[spikes_rands < 0.5] = -vThrLast
    query[spikes_rands > 0.5] = vThrLast

    spikes_rands = torch.rand(K,N)
    key = torch.zeros(K,N)
    key[spikes_rands < 0.5] = -vThrLast
    key[spikes_rands > 0.5] = vThrLast
    
    queryQANN = query + query
    keyQANN = key + key
        
    # print("query",query)
    # print("key",key)
    outputRef = queryQANN@keyQANN
    # print("outputRef",outputRef)
    outputRef = torch.clip(torch.round(outputRef/vThr)*vThr,min=0,max=7*vThr)
    
    PEs = [ProcessElementAttention(quantizeParam=(vThrLast, vThr),matrixShape=(K,tileN),first=False) for i in range(cfg["TILE"]["PENum"])]
    
    # begin tiling
    T = 15
    outSpikes1 = torch.zeros(T,N,N)
    for i,PE in enumerate(PEs):   
        rowId = i//int(math.sqrt(cfg["TILE"]["PENum"]))
        colId = i%int(math.sqrt(cfg["TILE"]["PENum"]))
        # initialize the spike tracer of query and key in weight buffer
        for i in range(tileN):
            PE.weightBufferQ[0].input_data(torch.zeros(K),i, direction=0)
            PE.weightBufferK[0].input_data(torch.zeros(K),i, direction=1)
        
        # initialize the output membrane and spiketracer
        for i in range(tileN):
            block_id = i//PE.weightBlockWidth
            row_id = i%PE.weightBlockWidth
            PE.membrane[0].input_data(torch.zeros(tileN)+0.5*vThr, i, direction=0)
            PE.spikeTracer[0].input_data(torch.zeros(tileN), block_id, row_id)

        # begin calculation
        # initialize the input spikes
        for n in tqdm(range(tileN)):
            dataStream = []
            # query input
            # print("query input:",query[n,:])
            for k in range(K):
                if query[n+rowId*tileN,k] == vThrLast:
                    dataStream.append((n,k,0))
                elif query[n+rowId*tileN,k] == -vThrLast:
                    dataStream.append((n,k,1))
            # print("spiking query input:",dataStream)
            PE(dataStream, right=True, update=False, tarRowId=n)
            # key input
            dataStream = []
            # print("key input:",key[:,n])
            for k in range(K):
                if key[k,n+colId*tileN] == vThrLast:
                    dataStream.append((k,n,0))
                elif key[k,n+colId*tileN] == -vThrLast:
                    dataStream.append((k,n,1))
            # print("spiking key input:",dataStream)
            PE(dataStream, right=False, update=False, tarRowId=n)

        for n in tqdm(range(tileN)):
            outSpikes = PE([], right=True, update=True, tarRowId=n)
            for spike in outSpikes:
                row_id, column_id, sign = spike
                outSpikes1[0][row_id+rowId*tileN][column_id+colId*tileN] = outSpikes1[0][row_id+rowId*tileN][column_id+colId*tileN] + vThr if sign == 0 else -vThr

        for n in tqdm(range(tileN)):
            dataStream = []
            # query input
            for k in range(K):
                if query[n+rowId*tileN,k] > 0:
                    dataStream.append((n,k,0))
                elif query[n+rowId*tileN,k] < 0:
                    dataStream.append((n,k,1))
            PE(dataStream, right=True, update=False, tarRowId=n)
            # key input
            dataStream = []
            for k in range(K):
                if key[k,n+colId*tileN] > 0:
                    dataStream.append((k,n,0))
                elif key[k,n+colId*tileN] < 0:
                    dataStream.append((k,n,1))
            PE(dataStream, right=False, update=False, tarRowId=n)

        for t in range(T-1):
            for n in tqdm(range(tileN)):
                outSpikes = PE([], right=True, update=True, tarRowId=n)
                for spike in outSpikes:
                    row_id, column_id, sign = spike
                    outSpikes1[t+1][row_id+rowId*tileN][column_id+colId*tileN] = outSpikes1[t+1][row_id+rowId*tileN][column_id+colId*tileN] + vThr if sign == 0 else -vThr

    outSpiking = torch.sum(outSpikes1,dim=0)
    # print("outSpiking",outSpiking)
    # print("membrane potantial:",PE.membrane.cram.memory)
    # print("outputRef",outputRef)
    print(outSpiking.shape, outputRef.shape)
    print("TimeStep1: point accuracy:",(torch.abs(outSpiking - outputRef)<1e-4).sum()/outputRef.numel())
    # print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-2).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpiking - outputRef)<1e-4).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    print(PE.adderTree.fCount)

def test_processElementAttentionMultiHead():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    # calculate the correct result
    N = cfg["processElement"]["input"]["N"]
    K = cfg["processElement"]["input"]["K"]
    H = cfg["processElement"]["head"]
    print("N,K,head",N,K,H)
    
    vThrLast = 0.41    
    vThr = 0.51
    
    spikes_rands = torch.rand(H,N,K)
    query = torch.zeros(H,N,K)
    query[spikes_rands < 0.5] = -vThrLast
    query[spikes_rands > 0.5] = vThrLast

    spikes_rands = torch.rand(H,K,N)
    key = torch.zeros(H,K,N)
    key[spikes_rands < 0.5] = -vThrLast
    key[spikes_rands > 0.5] = vThrLast
    
    queryQANN = query + query
    keyQANN = key + key
        
    # print("query",query)
    # print("key",key)
    outputRef = queryQANN@keyQANN
    # print("outputRef",outputRef)
    outputRef = torch.clip(torch.round(outputRef/vThr)*vThr,min=0,max=7*vThr)
    
    PE = ProcessElementAttention(quantizeParam=(vThrLast, vThr),matrixShape=(K,N),head=H,first=False)
    
    # query = query.reshape(H*N,K)
    # key = key.transpose(0,1).reshape(K,H*N)
    # N = H*N
    # initialize the spike tracer of query and key in weight buffer
    for h in range(H):
        for i in range(N):
            PE.weightBufferQ[h].input_data(torch.zeros(K),i, direction=0)
            PE.weightBufferK[h].input_data(torch.zeros(K),i, direction=1)
    
    # initialize the output membrane and spiketracer
    for h in range(H):
        for i in range(N):
            block_id = i//PE.weightBlockWidth
            row_id = i%PE.weightBlockWidth
            PE.membrane[h].input_data(torch.zeros(N)+0.5*vThr, i, direction=0)
            PE.spikeTracer[h].input_data(torch.zeros(N), block_id, row_id)

    # begin calculation
    # initialize the input spikes
    T = 15
    outSpikes1 = torch.zeros(T,H,N,N)
    for h in range(H):
        for n in tqdm(range(N)):
            dataStream = []
            # query input
            # print("query input:",query[n,:])
            for k in range(K):
                if query[h,n,k] == vThrLast:
                    dataStream.append((n,k,0))
                elif query[h,n,k] == -vThrLast:
                    dataStream.append((n,k,1))
            # print("spiking query input:",dataStream)
            PE(dataStream, right=True, update=False, tarRowId=n, headId=h)
            # key input
            dataStream = []
            # print("key input:",key[:,n])
            for k in range(K):
                if key[h,k,n] == vThrLast:
                    dataStream.append((k,n,0))
                elif key[h,k,n] == -vThrLast:
                    dataStream.append((k,n,1))
            # print("spiking key input:",dataStream)
            PE(dataStream, right=False, update=False, tarRowId=n, headId=h)

        for n in tqdm(range(N)):
            outSpikes = PE([], right=True, update=True, tarRowId=n, headId=h)
            for spike in outSpikes:
                row_id, column_id, sign = spike
                outSpikes1[0][h][row_id][column_id] = outSpikes1[0][h][row_id][column_id] + vThr if sign == 0 else -vThr

        for n in tqdm(range(N)):
            dataStream = []
            # query input
            for k in range(K):
                if query[h,n,k] > 0:
                    dataStream.append((n,k,0))
                elif query[h,n,k] < 0:
                    dataStream.append((n,k,1))
            PE(dataStream, right=True, update=False, tarRowId=n, headId=h)
            # key input
            dataStream = []
            for k in range(K):
                if key[h,k,n] > 0:
                    dataStream.append((k,n,0))
                elif key[h,k,n] < 0:
                    dataStream.append((k,n,1))
            PE(dataStream, right=False, update=False, tarRowId=n, headId=h)

        for t in range(T-1):
            for n in tqdm(range(N)):
                outSpikes = PE([], right=True, update=True, tarRowId=n, headId=h)
                for spike in outSpikes:
                    row_id, column_id, sign = spike
                    outSpikes1[t+1][h][row_id][column_id] = outSpikes1[t+1][h][row_id][column_id] + vThr if sign == 0 else -vThr

    outSpiking = torch.sum(outSpikes1,dim=0)
    # print("outSpiking",outSpiking)
    # print("membrane potantial:",PE.membrane.cram.memory)
    # print("outputRef",outputRef)
    print(outSpiking.shape, outputRef.shape)
    print("TimeStep1: point accuracy:",(torch.abs(outSpiking - outputRef)<1e-4).sum()/outputRef.numel())
    # print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-2).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpiking - outputRef)<1e-4).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    print(PE.adderTree.fCount)

def test_processElementAttentionMultiHeadWithSoftmax():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    # calculate the correct result
    N = cfg["processElement"]["input"]["N"]
    K = cfg["processElement"]["input"]["K"]
    H = cfg["processElement"]["head"]
    print("N,K,head",N,K,H)
    
    vThrLast = 0.41    
    vThr = 0.51
    
    spikes_rands = torch.rand(H,N,K)
    query = torch.zeros(H,N,K)
    query[spikes_rands < 0.5] = -vThrLast
    query[spikes_rands > 0.5] = vThrLast

    spikes_rands = torch.rand(H,K,N)
    key = torch.zeros(H,K,N)
    key[spikes_rands < 0.5] = -vThrLast
    key[spikes_rands > 0.5] = vThrLast
    
    queryQANN = query + query
    keyQANN = key + key
        
    # print("query",query)
    # print("key",key)
    outputRef = queryQANN@keyQANN
    # print("outputRef",outputRef)
    outputRef = torch.clip(torch.round(outputRef/vThr)*vThr,min=0,max=7*vThr)
    
    PE = ProcessElementAttention(quantizeParam=(vThrLast, vThr),matrixShape=(K,N),head=H,first=False)
    
    # query = query.reshape(H*N,K)
    # key = key.transpose(0,1).reshape(K,H*N)
    # N = H*N
    # initialize the spike tracer of query and key in weight buffer
    for h in range(H):
        for i in range(N):
            PE.weightBufferQ[h].input_data(torch.zeros(K),i, direction=0)
            PE.weightBufferK[h].input_data(torch.zeros(K),i, direction=1)
    
    # initialize the output membrane and spiketracer
    for h in range(H):
        for i in range(N):
            block_id = i//PE.weightBlockWidth
            row_id = i%PE.weightBlockWidth
            PE.membrane[h].input_data(torch.zeros(N)+0.5*vThr, i, direction=0)
            PE.spikeTracer[h].input_data(torch.zeros(N), block_id, row_id)

    # begin calculation
    # initialize the input spikes
    T = 15
    outSpikes1 = torch.zeros(T,H,N,N)
    for h in range(H):
        for n in tqdm(range(N)):
            dataStream = []
            # query input
            # print("query input:",query[n,:])
            for k in range(K):
                if query[h,n,k] == vThrLast:
                    dataStream.append((n,k,0))
                elif query[h,n,k] == -vThrLast:
                    dataStream.append((n,k,1))
            # print("spiking query input:",dataStream)
            PE(dataStream, right=True, update=False, tarRowId=n, headId=h)
            # key input
            dataStream = []
            # print("key input:",key[:,n])
            for k in range(K):
                if key[h,k,n] == vThrLast:
                    dataStream.append((k,n,0))
                elif key[h,k,n] == -vThrLast:
                    dataStream.append((k,n,1))
            # print("spiking key input:",dataStream)
            PE(dataStream, right=False, update=False, tarRowId=n, headId=h)

        for n in tqdm(range(N)):
            outSpikes = PE([], right=True, update=True, tarRowId=n, headId=h)
            for spike in outSpikes:
                row_id, column_id, sign = spike
                outSpikes1[0][h][row_id][column_id] = outSpikes1[0][h][row_id][column_id] + vThr if sign == 0 else -vThr

        for n in tqdm(range(N)):
            dataStream = []
            # query input
            for k in range(K):
                if query[h,n,k] > 0:
                    dataStream.append((n,k,0))
                elif query[h,n,k] < 0:
                    dataStream.append((n,k,1))
            PE(dataStream, right=True, update=False, tarRowId=n, headId=h)
            # key input
            dataStream = []
            for k in range(K):
                if key[h,k,n] > 0:
                    dataStream.append((k,n,0))
                elif key[h,k,n] < 0:
                    dataStream.append((k,n,1))
            PE(dataStream, right=False, update=False, tarRowId=n, headId=h)

        for t in range(T-1):
            for n in tqdm(range(N)):
                outSpikes = PE([], right=True, update=True, tarRowId=n, headId=h)
                for spike in outSpikes:
                    row_id, column_id, sign = spike
                    outSpikes1[t+1][h][row_id][column_id] = outSpikes1[t+1][h][row_id][column_id] + vThr if sign == 0 else -vThr

    outSpiking = torch.sum(outSpikes1,dim=0)
    # print("outSpiking",outSpiking)
    # print("membrane potantial:",PE.membrane.cram.memory)
    # print("outputRef",outputRef)
    print(outSpiking.shape, outputRef.shape)
    print("TimeStep1: point accuracy:",(torch.abs(outSpiking - outputRef)<1e-4).sum()/outputRef.numel())
    # print("TimeStep2: point accuracy:",(torch.abs(outSpikes2 - refSpikes2)<1e-2).sum()/refSpikes2.numel())
    
    assert (torch.abs(outSpiking - outputRef)<1e-4).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    print(PE.adderTree.fCount)


