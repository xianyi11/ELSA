import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
import math
from .membrane import spikeTracer, membrane
from .fireComponent import fireComponent
from .accumulators import Adder
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


class softmaxFunction(VSAModule):
    def __init__(self):
        super(softmaxFunction,self).__init__()
        self.fCount = 0
        self.fEnergy = cfg["router"]["softmaxFunction"]["fEnergy"]
        self.area = cfg["router"]["softmaxFunction"]["area"]
        self.staticPower = cfg["router"]["softmaxFunction"]["leakage"]
        self.parallelism = cfg["router"]["softmaxFunction"]["parallelism"]
    
    def forward(self, input, spiketracer,timeStep=0):
        sequenceLength = input.shape[0]
        if timeStep == 0:
            needCycle = math.ceil(sequenceLength/self.parallelism) * 7
            self.fCount = self.fCount + math.ceil(sequenceLength/self.parallelism)
            return torch.nn.functional.softmax(input), needCycle
        else:
            needCycle = math.ceil(sequenceLength/self.parallelism) * 14
            self.fCount = self.fCount + math.ceil(sequenceLength/self.parallelism)*2
            return torch.nn.functional.softmax(input+spiketracer) - torch.nn.functional.softmax(spiketracer), needCycle

class layerNormFunction(VSAModule):
    def __init__(self,length):
        super(layerNormFunction,self).__init__()
        self.fCount = 0
        self.fEnergy = cfg["router"]["layerNormFunction"]["fEnergy"]
        self.area = cfg["router"]["layerNormFunction"]["area"]
        self.staticPower = cfg["router"]["layerNormFunction"]["leakage"]
        self.parallelism = cfg["router"]["layerNormFunction"]["parallelism"]
        self.layerNorm = torch.nn.LayerNorm(length)
    
    def forward(self, input, spiketracer,timeStep=0):
        sequenceLength = input.shape[0]
        # print("self.partialSum",self.partialSum)
        if timeStep == 0:           
            needCycle = math.ceil(sequenceLength/self.parallelism) * 10
            self.fCount = self.fCount + math.ceil(sequenceLength/self.parallelism)
            return self.layerNorm(input+spiketracer), needCycle
        else:
            needCycle = math.ceil(sequenceLength/self.parallelism) * 20
            self.fCount = self.fCount + math.ceil(sequenceLength/self.parallelism)*2
            return self.layerNorm(input+spiketracer) - self.layerNorm(spiketracer), needCycle

class AdditionFunction(VSAModule):
    def __init__(self):
        super(AdditionFunction,self).__init__()
        self.fCount = 0
        self.fEnergy = cfg["router"]["AdditionFunction"]["fEnergy"]
        self.area = cfg["router"]["AdditionFunction"]["area"]
        self.staticPower = cfg["router"]["AdditionFunction"]["leakage"]
        self.parallelism = cfg["router"]["AdditionFunction"]["parallelism"]
        # self.layerNorm = torch.nn.LayerNorm(length)
    
    def forward(self, input, spiketracer,timeStep=0):
        sequenceLength = input.shape[0]
        needCycle = math.ceil(sequenceLength/self.parallelism)
        self.fCount = self.fCount + needCycle
        return input + spiketracer, needCycle

class FunctionalNeurons(VSAModule):
    def __init__(self, quantizeParam, function, head, squenceNum, squenceLen):
        super(FunctionalNeurons,self).__init__()
        self.function = function
        self.squenceLen = squenceLen
        self.squenceNum = squenceNum
        self.head = head
        if self.function == "softmax":
            self.module = softmaxFunction()
        if self.function == "layernorm":
            self.module = layerNormFunction(squenceLen)
        if self.function == "residual_addition":
            self.module = AdditionFunction()
        self.spikeTracer = [spikeTracer(N=self.squenceNum,M=self.squenceLen) for h in range(head)]
        self.membrane = [membrane(N=self.squenceNum,M=self.squenceLen) for h in range(head)]
        if self.function == "residual_addition":
            self.LastvThr1,self.LastvThr2, self.vThr = quantizeParam
        else:
            self.LastvThr1, self.vThr = quantizeParam
            self.LastvThr2 = self.LastvThr1
        if self.function == "softmax":
            self.fireComponent = fireComponent(width=self.squenceLen,vthr=self.vThr,sym=False)
        else:
            self.fireComponent = fireComponent(width=self.squenceLen,vthr=self.vThr,sym=True)
        self.adder = Adder()
        self.adderNum = cfg["processElement"]["accumulator"]["adderNum"]
        self.debug = True
        self.outZero = False
        self.inZero = False
        
    def calEnergy(self, latency, record=False):
        storageMemory = 0.0
        for h in range(self.head):
            storageMemory = storageMemory + self.spikeTracer[h].calEnergy(latency) + self.membrane[h].calEnergy(latency)

        return self.fireComponent.calEnergy(latency) + storageMemory + self.module.calEnergy(latency) + self.adder.calEnergy(latency)

    def getArea(self):
        storageMemory = 0.0
        for h in range(self.head):
            storageMemory = storageMemory + self.spikeTracer[h].getArea() + self.membrane[h].getArea()

        return self.fireComponent.getArea() + storageMemory + self.module.getArea() + self.adder.getArea()

    def forward(self, x, spiketracer, rowId, headId = 0, timeStep=0):
        
        row_block_id = rowId//self.spikeTracer[headId].tileHN
        row_inner_id = rowId%self.spikeTracer[headId].tileHN
        
        self.inZero = (x.abs().sum() == 0)
        
        # if self.function == "layernorm" and rowId == 1 and headId == 0:
        #     if self.debug:
        #         print("timeStep",timeStep,"after layerNorm input",x*self.LastvThr1)
        #         print("timeStep",timeStep,"spiketracer",spiketracer*self.LastvThr2)
        if self.inZero and not self.outZero:
            moduleCycle = 0
            spiketracer = self.spikeTracer[headId].output_data(row_block_id,row_inner_id)
            membrane = self.membrane[headId].output_data(row_block_id,row_inner_id)
            outspike,memPotential = self.fireComponent(spiketracer,membrane)            

            membraneLoadCycle = 1 # 只载入一行membrane
            membraneSaveCycle = 1 # 只保存一行membrane
            neuronCycle = max(membraneLoadCycle, membraneSaveCycle)
                    
            spiketracer = spiketracer + outspike
            self.membrane[headId].input_data(memPotential,row_block_id,row_inner_id)
            self.spikeTracer[headId].input_data(spiketracer,row_block_id,row_inner_id)                        
            self.outZero = (outspike.abs().sum() == 0)

        elif self.inZero and self.outZero: #跳过这次计算
            moduleCycle = 0
            neuronCycle = 0
            outspike = torch.zeros(x.shape)
        else:
            output, moduleCycle = self.module(x*self.LastvThr1, spiketracer*self.LastvThr2, timeStep)
            # if self.function == "layernorm" and rowId == 1 and headId == 0:
            #     if self.debug:
            #         print("after layerNorm output",output)
            #         print("spiketracer",spiketracer)
            #         print("output[0:64]",output[0:64])
            
            spiketracer = self.spikeTracer[headId].output_data(row_block_id,row_inner_id)
            membrane = self.membrane[headId].output_data(row_block_id,row_inner_id)
            
            memPotential_mid = self.adder(membrane, [output])
            # if self.function == "layernorm" and rowId == 1 and headId == 0:
            #     if self.debug:
            #         print("layerNorm neurons membrane",membrane)
            #         print("layerNorm neurons memPotential_mid",memPotential_mid)

            # if self.function == "softmax" and rowId == 2 and headId == 5:
            #     if self.debug:
            #         outspike,memPotential = self.fireComponent(spiketracer,memPotential_mid,verbose=True)
            # else:
            outspike,memPotential = self.fireComponent(spiketracer,memPotential_mid)

            # if self.function == "layernorm" and rowId == 1:
            #     if self.debug:
            #         print("outspike",outspike*self.vThr)
            #         self.debug = False
            
            membraneLoadCycle = 1 # 只载入一行membrane
            membraneSaveCycle = 1 # 只保存一行membrane
            integrateCycle = math.ceil(self.squenceLen/self.adderNum)
            neuronCycle = max(membraneLoadCycle, membraneSaveCycle, integrateCycle)
                    
            spiketracer = spiketracer + outspike
            self.membrane[headId].input_data(memPotential,row_block_id,row_inner_id)
            self.spikeTracer[headId].input_data(spiketracer,row_block_id,row_inner_id)
        
            self.outZero = (outspike.abs().sum() == 0)
        
        return outspike, moduleCycle + neuronCycle


