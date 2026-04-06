import torch
from basicModule import VSAModule
from .accumulators import Adder, AdderTree
from .fireComponent import fireComponent
from .inputBuffer import inputBuffer
from .membrane import membrane, spikeTracer
from .weightBuffer import weightBuffer
from .spikeEncoder import spikeEncoder
from .STBIFFunction import STBIFNeuron
from tqdm import tqdm
import math

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class ProcessElement(VSAModule):
    def __init__(self, quantizeParam, matrixShape, identity=False, depthWise=False, kernelSize=1, first=True, mapLayerNum=1):
        super(ProcessElement,self).__init__()
        self.parallelism = cfg["processElement"]["parallelism"]
        self.identity = identity
        self.kernelSize = kernelSize
        self.depthWise = depthWise
        self.first = first
        # self.M = cfg["processElement"]["input"]["M"]
        # self.K = cfg["processElement"]["input"]["K"]
        # self.N = cfg["processElement"]["input"]["N"]
        self.M, self.K, self.N = matrixShape
        print("self.M, self.K, self.N",self.M, self.K, self.N)
        self.tileN = self.N//cfg["TILE"]["PENum"]
        self.weightBlockWidth = cfg["processElement"]["weightBuffer"]["height"]
        self.weightBufferWordNum = math.ceil(self.weightBlockWidth/(math.ceil(self.tileN/cfg["processElement"]["weightBuffer"]["inoutWidth"])))
        self.membraneBlockWidth = cfg["processElement"]["membrane"]["height"]
        self.membraneWordNum = math.ceil(self.membraneBlockWidth/(math.ceil(self.tileN/cfg["processElement"]["membrane"]["inoutWidth"])))
        self.M1, self.N1 = quantizeParam
        
        # initilize storage
        self.inputBuffer = inputBuffer(queueDepth=cfg["processElement"]["inputBuffer"]["queueDepth"], parallelism=self.parallelism,first=self.first)
        self.weightBuffer = weightBuffer(N=self.tileN,K=self.K)
        self.spikeTracer = spikeTracer(N=self.tileN,M=self.M)
        self.membrane = membrane(N=self.tileN,M=self.M)
        
        # initilize computation component
        # self.accumulator = accumulator(treeWidth=self.tileN, parallelism=self.parallelism)
        self.adder = Adder()
        self.adderTree = AdderTree()
        self.fireComponent = fireComponent(width=self.tileN)
        self.fireComponent.set_MN(self.M1,self.N1)
        self.mapLayerNum = mapLayerNum
        
        # output interface
        self.spikeEncoder = spikeEncoder()
        
        # for latency calculation
        self.computationCycle = 0
    
    def getArea(self):
        if self.first:
            # print("self.adder.getArea()",self.adder.getArea())
            return self.inputBuffer.getArea() + self.weightBuffer.getArea() + self.spikeTracer.getArea() + self.membrane.getArea() + self.adder.getArea() + self.fireComponent.getArea()
        else:
            # print("self.adderTree.getArea()",self.adderTree.getArea())
            return self.inputBuffer.getArea() + self.weightBuffer.getArea() + self.spikeTracer.getArea() + self.membrane.getArea() + self.adderTree.getArea() + self.fireComponent.getArea()
    
    def calEnergy(self,latency):
        # print("IN PE: self.inputBuffer",self.inputBuffer.calEnergy(latency))
        # print("IN PE: self.weightBuffer",self.weightBuffer.calEnergy(latency))
        # print("IN PE: self.spikeTracer",self.spikeTracer.calEnergy(latency))
        # print("IN PE: self.membrane",self.membrane.calEnergy(latency))
        # print("IN PE: self.adder",self.adder.calEnergy(latency))
        # print("IN PE: self.adderTree",self.adderTree.calEnergy(latency))
        # print("IN PE: self.fireComponent",self.fireComponent.calEnergy(latency))
        if self.first:
            return self.inputBuffer.calEnergy(latency) + self.weightBuffer.calEnergy(latency) + self.spikeTracer.calEnergy(latency) + self.membrane.calEnergy(latency) + self.adder.calEnergy(latency) + self.fireComponent.calEnergy(latency)
        else:
            return self.inputBuffer.calEnergy(latency) + self.weightBuffer.calEnergy(latency) + self.spikeTracer.calEnergy(latency) + self.membrane.calEnergy(latency) + self.adderTree.calEnergy(latency) + self.fireComponent.calEnergy(latency)
        
    def forward(self,dataStream, update=False, tarColumnId=0):
        # dataStream = virtualChannel.Clientread()

        lastdataCount = 0
        dataCount = 0
        memPotential = None
        spikeTracer = None
        # column_id = dataStream[0][1]
        # column_block_id = column_id//self.membrane.tileWN
        # column_inner_id = column_id%self.membrane.tileWN
        # memPotential = self.membrane.output_data(column_block_id,column_inner_id)
        # spikeTracer = self.spikeTracer.output_data(column_block_id,column_inner_id)
        # self.computationCycle = self.computationCycle + math.ceil(len(dataStream)/self.inputBuffer.queueNum)
        inputBufferLoadCycle = 0
        PEspikeNum = 0
        if len(dataStream) > 0:
            while(1):
                while(1):
                    success,flashQueueId = self.inputBuffer.get_data(dataStream[dataCount],weightBlockWidth=self.weightBlockWidth,membraneWordNum=self.membraneWordNum)
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
                if not success or update: # 如果not success，两种情况，第一种是queue全部被分配需要清空，另一种是queue满了需要被清空；如果update说明这一列对应的所有脉冲都要输出，因为neuron要发放了
                    # 考虑inputBuffer load的情况，并行度是inputBuffer的数量，因此load时间是本次load的data数量除以并行度
                    inputBufferLoadCycle = math.ceil((dataCount-lastdataCount)/self.inputBuffer.queueNum)
                    lastdataCount = dataCount + 0.0
                    if success and update:
                        if tarColumnId not in self.inputBuffer.CIdtoQdTLB.keys(): # 说明已经加完了，直接break就可以
                            break
                        else:
                            flashQueueId = self.inputBuffer.CIdtoQdTLB[tarColumnId] # 取出要加的行对应的QueueId
                    spikes, empty, isColEnd = self.inputBuffer.output_data(flashQueueId)
                    if len(spikes) > 0:
                        PEspikeNum = PEspikeNum + len(spikes)
                        column_id = spikes[-1][2]
                        column_block_id = column_id//self.membrane.tileWN
                        column_inner_id = column_id%self.membrane.tileWN
                        if memPotential is None:
                            memPotential = self.membrane.output_data(column_block_id,column_inner_id)
                        weightData = []

                        for spike in spikes:
                            block_id,row_id,column_id,sign = spike
                            weightData.append(self.weightBuffer.output_data(block_id,row_id,sign)*self.M1)

                        # 40块小SRAM，每块SRAM读取64bit也就是16个weight，并行度有440，一次可以读出440*16个weights，这边考虑随机分配一次可以读出220*16个weight（50%利用率），然后再考虑一个tile分配多个layer的情况
                        # print("self.weightBuffer.SRAMTotalNum",self.weightBuffer.SRAMTotalNum)
                        weightLoadCycle = math.ceil((len(spikes)*math.ceil(weightData[0].shape[0]/self.weightBuffer.inoutWidth))/(math.ceil(self.weightBuffer.SRAMTotalNum/self.mapLayerNum)))

                        membraneLoadCycle = 1 # 只载入一行membrane
                        memPotential = self.adderTree(memPotential, weightData)
                        if isColEnd == True:
                            self.membrane.input_data(memPotential,column_block_id,column_inner_id)
                            memPotential = None
                        AddCycle = math.ceil(weightData[0].shape[0]/(math.ceil(self.adderTree.adderNum/self.mapLayerNum))) # computation cycle, ，然后再考虑一个tile分配多个layer的情况
                        membraneSaveCycle = 1 # 只保存一行membrane
                        self.computationCycle = self.computationCycle + max(inputBufferLoadCycle, max(max(membraneLoadCycle,weightLoadCycle), max(AddCycle, membraneSaveCycle))) # 流水执行input load，weight/membrane load，addertree操作以及membrane save，取最大的那个块。
                        # print("inputBufferLoadCycle",inputBufferLoadCycle,"weightLoadCycle",weightLoadCycle,"AddCycle",AddCycle)
                        # print("self.mapLayerNum",self.mapLayerNum,"len(spikes)",len(spikes),'self.computationCycle',self.computationCycle,"math.ceil(weightData[0].shape[0]/self.adderTree.adderNum)",math.ceil(weightData[0].shape[0]/self.adderTree.adderNum))
                                        
                if dataCount == len(dataStream):
                    break
        
        outspike = None
        if update:
        # fire
            if tarColumnId not in self.inputBuffer.CIdtoQdTLB.keys(): # 说明已经加完了，直接break就可以
                pass
            else:
                # update如果没有数据输入也需要将buffer里面的spike处理掉，不然输出不对。
                flashQueueId = self.inputBuffer.CIdtoQdTLB[tarColumnId] # 取出要加的行对应的QueueId，update时没有spike输入还是要把spike清空的，这个问题会导致输出不一致。
                spikes, empty, isColEnd = self.inputBuffer.output_data(flashQueueId) 
                if len(spikes) > 0:
                    column_id = spikes[-1][2]
                    column_block_id = column_id//self.membrane.tileWN
                    column_inner_id = column_id%self.membrane.tileWN
                    if memPotential is None:
                        memPotential = self.membrane.output_data(column_block_id,column_inner_id)
                    weightData = []
                    for spike in spikes:
                        block_id,row_id,column_id,sign = spike
                        weightData.append(self.weightBuffer.output_data(block_id,row_id,sign)*self.M1)
                    # self.computationCycle = self.computationCycle + int(math.ceil(len(spikes)/self.SRAMNumHeight)) - 1
                    membraneLoadCycle = 1 # 只载入一行membrane

                    # 40块小SRAM，每块SRAM读取64bit也就是16个weight，并行度有40，一次可以读出40*16个weights，这边考虑随机分配一次可以读出20*16个weight（50%利用率）
                    weightLoadCycle = math.ceil((len(spikes)*math.ceil(weightData[0].shape[0]/self.weightBuffer.inoutWidth))/(self.weightBuffer.SRAMTotalNum//2)) 

                    memPotential = self.adderTree(memPotential, weightData)
                    if isColEnd == True:
                        self.membrane.input_data(memPotential,column_block_id,column_inner_id)
                        memPotential = None
                    AddCycle = math.ceil(weightData[0].shape[0]/self.adderTree.adderNum) # computation cycle
                    membraneSaveCycle = 1 # 只保存一行membrane
                    self.computationCycle = self.computationCycle + max(AddCycle,max(max(weightLoadCycle,membraneLoadCycle), membraneSaveCycle)) # update不考虑input buffer的load           

            # if self.K - PEspikeNum > 0 and PEspikeNum > 0:
            #     for i in range(self.K - PEspikeNum):
            #         self.weightBuffer.output_data(0,0,0)*self.M1
            #     weightLoadCycle = math.ceil(((self.K - PEspikeNum)*math.ceil(weightData[0].shape[0]/self.weightBuffer.inoutWidth))/(self.weightBuffer.SRAMTotalNum//2)) 
            #     self.computationCycle = self.computationCycle + weightLoadCycle

            column_block_id = tarColumnId//self.membrane.tileWN
            column_inner_id = tarColumnId%self.membrane.tileWN
            memPotential_mid = self.membrane.output_data(column_block_id,column_inner_id)
            spikeTracer = self.spikeTracer.output_data(column_block_id,column_inner_id)
            outspike,memPotential = self.fireComponent(spikeTracer,memPotential_mid)

            # print("tarColumnId",tarColumnId)
            # if tarColumnId == 111:
            #     print(memPotential_mid)
            #     print(outspike)
        # update
            spikeTracer = spikeTracer + outspike
            self.membrane.input_data(memPotential,column_block_id,column_inner_id)
            self.spikeTracer.input_data(spikeTracer,column_block_id,column_inner_id)

        return outspike, tarColumnId

        # output encoding
        # outspikes = []
        # outspikes = self.spikeEncoder(outspike,column_id,outspikes)
        # return outspikes


def test_processElement_withbias():
    from copy import deepcopy
    torch.set_printoptions(profile="full")
    # calculate the correct result
    M = cfg["processElement"]["input"]["M"]
    K = cfg["processElement"]["input"]["K"]
    N = cfg["processElement"]["input"]["N"]
    tileN = N//cfg["TILE"]["PENum"]
    
    weight = (torch.rand(tileN,K)*16).int()-8
    spikes_rands = torch.rand(K,M)
    spikes = torch.zeros(K,M).int()
    spikes[spikes_rands < 0.05] = -1
    spikes[spikes_rands > 0.95] = 1
    bias = (torch.rand(tileN)*16).int()-8
    
    M1 = 122
    N1 = 10
    neuron = STBIFNeuron(M=M1,N=N1,pos_max=7,neg_min=0,bias=bias.unsqueeze(1))
    
    output = weight@spikes 

    refoutSpikes1 = neuron(output)
    
    refSpikes1 = deepcopy(refoutSpikes1)

    refoutSpikes2 = neuron(output)

    refSpikes2 = deepcopy(refoutSpikes2)

    PE = ProcessElement(quantizeParam=(M1,N1),matrixShape=(M,K,N))
    
    # initialize weight/bias data
    for i in range(K):
        block_id = i//PE.weightBuffer.tileHN
        row_id = i%PE.weightBuffer.tileHN
        PE.weightBuffer.input_data(weight[:,i], block_id, row_id)
        
    for i in range(M):
        block_id = i//PE.membrane.tileWN
        column_id = i%PE.membrane.tileWN
        PE.membrane.input_data(torch.zeros(tileN)+2**(N1-1)+bias*(2**N1), block_id, column_id)

    # begin calculation
    # initialize the input spikes
    
    outSpikes1 = torch.zeros(tileN,M)
    outSpikes2 = torch.zeros(tileN,M)
    for m in tqdm(range(M)):
        dataStream = []
        for k in range(K):
            if spikes[k,m] == 1:
                dataStream.append((k,m,0))
            elif spikes[k,m] == -1:
                dataStream.append((k,m,1))
        outSpikes = PE(dataStream, update=True, tarColumnId=m)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1
        outSpikes = PE(dataStream, update=True, tarColumnId=m)
        for spike in outSpikes:
            row_id, column_id, sign = spike
            outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + 1 if sign == 0 else -1

    assert (outSpikes1 == refSpikes1).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    assert (outSpikes2 == refSpikes2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
    print(PE.adder.fCount, 2*torch.sum(torch.abs(spikes)))



# def test_processElement():
#     from copy import deepcopy
#     torch.set_printoptions(profile="full")
#     # calculate the correct result
#     M = cfg["processElement"]["input"]["M"]
#     K = cfg["processElement"]["input"]["K"]
#     N = cfg["processElement"]["tile"]["tileN"]
    
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

#     PE = ProcessElement()
    
#     # initialize weight/bias data
#     for i in range(K):
#         block_id = i//PE.weightBuffer.tileHN
#         row_id = i%PE.weightBuffer.tileHN
#         PE.weightBuffer.input_data(weight[:,i], block_id, row_id)
        
#     for i in range(M):
#         block_id = i//PE.membrane.tileWN
#         column_id = i%PE.membrane.tileWN
#         PE.membrane.input_data(torch.zeros(N)+2**(N1-1), block_id, column_id)

#     # begin calculation
#     # initialize the input spikes
    
#     outSpikes1 = torch.zeros(N,M)
#     outSpikes2 = torch.zeros(N,M)
#     for m in tqdm(range(M)):
#         dataStream = []
#         for k in range(K):
#             if spikes[k,m] == 1:
#                 dataStream.append((k,m,0))
#             elif spikes[k,m] == -1:
#                 dataStream.append((k,m,1))
#         outSpikes = PE(dataStream)
#         for spike in outSpikes:
#             row_id, column_id, sign = spike
#             outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1
#         outSpikes = PE(dataStream)
#         for spike in outSpikes:
#             row_id, column_id, sign = spike
#             outSpikes2[row_id][column_id] = outSpikes2[row_id][column_id] + 1 if sign == 0 else -1

#     assert (outSpikes1 == refSpikes1).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
#     assert (outSpikes2 == refSpikes2).all(), "test failed!!! outspikes != refSpikes in 1-st time-step"
#     print(PE.accumulator.fCount, 2*torch.sum(torch.abs(spikes)))

#     ClockTime = 1/2e8
#     totalLatency = ClockTime*(PE.accumulator.fCount)/cfg["processElement"]["parallelism"]
#     totalEnergy = 0.0
#     totalArea = 0.0
    
    
#     print("input buffer statics:")
#     readCount = 0
#     writeCount = 0
#     totalsramEnergy = 0.0
#     totalsramArea = 0.0
#     print("====>readCount:",PE.inputBuffer.readCount)
#     print("====>writeCount:",PE.inputBuffer.writeCount)
#     # print("====>calEnergy:",PE.inputBuffer.calEnergy(totalLatency))
#     # print("====>Area:",PE.inputBuffer.getArea())
#     for i,sram in enumerate (PE.inputBuffer.srams):
#         print(f"==>sram{i}:")
#         print("====>readCount:",sram.readCount)
#         print("====>writeCount:",sram.writeCount)
#         print("====>calEnergy:",sram.calEnergy(totalLatency))
#         print("====>Area:",sram.getArea())
#         readCount += sram.readCount
#         writeCount += sram.writeCount
#         totalsramEnergy += sram.calEnergy(totalLatency)
#         totalsramArea += sram.getArea()
#     print(f"==>Input Buffer Total: readCount={readCount}, writeCount={writeCount}, Energy={totalsramEnergy}, Area={totalsramArea}")
#     totalEnergy += totalsramEnergy
#     totalArea += totalsramArea
    
#     print("weight buffer statics:")
#     readCount = 0
#     writeCount = 0
#     totalsramEnergy = 0.0
#     totalsramArea = 0.0
#     for i,sram in enumerate (PE.weightBuffer.srams):
#         print(f"==>sram{i}:")
#         print("====>readCount:",sram.readCount)
#         print("====>writeCount:",sram.writeCount)
#         print("====>calEnergy:",sram.calEnergy(totalLatency))
#         print("====>Area:",sram.getArea())
#         readCount += sram.readCount
#         writeCount += sram.writeCount
#         totalsramEnergy += sram.calEnergy(totalLatency)
#         totalsramArea += sram.getArea()
#     print(f"==>Weight Buffer Total: readCount={readCount}, writeCount={writeCount}, Energy={totalsramEnergy}, Area={totalsramArea}")
#     totalEnergy += totalsramEnergy
#     totalArea += totalsramArea


#     print("membrane buffer statics:")
#     readCount = 0
#     writeCount = 0
#     totalsramEnergy = 0.0
#     totalsramArea = 0.0
#     for i,sram in enumerate (PE.membrane.srams):
#         print(f"==>sram{i}:")
#         print("====>readCount:",sram.readCount)
#         print("====>writeCount:",sram.writeCount)
#         print("====>calEnergy:",sram.calEnergy(totalLatency))
#         print("====>Area:",sram.getArea())
#         readCount += sram.readCount
#         writeCount += sram.writeCount
#         totalsramEnergy += sram.calEnergy(totalLatency)
#         totalsramArea += sram.getArea()
#     print(f"==>Membrane Buffer Total: readCount={readCount}, writeCount={writeCount}, Energy={totalsramEnergy}, Area={totalsramArea}")
#     totalEnergy += totalsramEnergy
#     totalArea += totalsramArea

#     # readCount = 0
#     # writeCount = 0
#     # totalEnergy = 0.0
#     print("spikeTracer buffer statics:")
#     readCount = 0
#     writeCount = 0
#     totalsramEnergy = 0.0
#     for i,sram in enumerate (PE.spikeTracer.srams):
#         print(f"==>sram{i}:")
#         print("====>readCount:",sram.readCount)
#         print("====>writeCount:",sram.writeCount)
#         print("====>calEnergy:",sram.calEnergy(totalLatency))
#         print("====>Area:",sram.getArea())
#         readCount += sram.readCount
#         writeCount += sram.writeCount
#         totalsramEnergy += sram.calEnergy(totalLatency)
#         totalsramArea += sram.getArea()
#     print(f"==>SpikeTracer Buffer Total: readCount={readCount}, writeCount={writeCount}, Energy={totalsramEnergy}, Area={totalsramArea}")        
#     totalEnergy += totalsramEnergy
#     totalArea += totalsramArea

#     print("accumulator statics:")
#     print("==>fCount:",PE.accumulator.fCount)
#     print("==>calEnergy:",PE.accumulator.calEnergy(totalLatency))
#     print("==>Area:",PE.accumulator.getArea()*cfg["processElement"]["parallelism"])
#     totalEnergy += PE.accumulator.calEnergy(totalLatency)
#     totalArea += PE.accumulator.getArea()*cfg["processElement"]["parallelism"]

    
#     print("firecomponent statics:")
#     print("==>fCount:",PE.fireComponent.fCount)
#     print("==>calEnergy:",PE.fireComponent.calEnergy(totalLatency))
#     print("==>Area:",PE.fireComponent.getArea())
#     totalEnergy += PE.fireComponent.calEnergy(totalLatency)
#     totalArea += PE.fireComponent.getArea()
#     print(f"Total: totalEnergy={totalEnergy}, totalArea={totalArea}, totalLatency={totalLatency}")
