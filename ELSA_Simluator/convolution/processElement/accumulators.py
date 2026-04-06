import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
import math
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


class accumulator(VSAModule):
    def __init__(self, treeWidth, parallelism=1):
        super(accumulator,self).__init__()
        self.treeWidth = treeWidth
        self.parallelism = parallelism
        self.fCount = 0
        self.fEnergy = cfg["processElement"]["accumulator"]["fEnergy"]
        self.area = cfg["processElement"]["accumulator"]["area"]
        self.staticPower = cfg["processElement"]["accumulator"]["leakage"]
        self.partialSum = None
    
    def reset(self):
        self.partialSum = None
    
    def forward(self, weightDatas):
        self.fCount = self.fCount + len(weightDatas)
        # print("self.partialSum",self.partialSum)
        for weightData in weightDatas:
            if self.partialSum is None:
                self.partialSum = weightData
            else:
                self.partialSum = self.partialSum + weightData
        return self.partialSum

class AdderTree(VSAModule):
    def __init__(self):
        super(AdderTree,self).__init__()
        self.fEnergy = cfg["processElement"]["adderTree"]["fEnergy"]
        self.area = cfg["processElement"]["adderTree"]["area"]
        self.staticPower = cfg["processElement"]["adderTree"]["leakage"]    
        self.adderNum = cfg["processElement"]["adderTree"]["adderNum"]    
        self.addCount = 0
        
    def getArea(self):
        return self.area
    
    def forward(self, memPotential, weightData):
        # self.fCount = self.fCount + len(weightData)
        for weight in weightData:
            # print("memPotential.shape",memPotential.shape, "weight.shape",weight.shape)
            memPotential = memPotential + weight
        if weight.shape[0]/self.adderNum < 1:
            self.fCount = self.fCount + weight.shape[0]/self.adderNum
        else:
            self.fCount = self.fCount + math.ceil(weight.shape[0]/self.adderNum)
        
        
        self.addCount = math.ceil(weight.shape[0]/self.adderNum)
        return memPotential

class Adder(VSAModule):
    def __init__(self):
        super(Adder,self).__init__()
        self.fEnergy = cfg["processElement"]["accumulator"]["fEnergy"]
        self.area = cfg["processElement"]["accumulator"]["area"]
        self.staticPower = cfg["processElement"]["accumulator"]["leakage"]    
        self.width = cfg["processElement"]["weightBuffer"]["inoutWidth"]*8
        self.parallelism = cfg["processElement"]["parallelism"]
        self.adderNum = cfg["processElement"]["adderTree"]["adderNum"]    
        self.addCount = 0
    
    def getArea(self):
        return self.area*self.parallelism
    
    def forward(self, memPotential, weightData):
        # self.fCount = self.fCount + len(weightData)
        for weight in weightData:
            # print("memPotential.shape",memPotential.shape, "weight.shape",weight.shape)
            memPotential = memPotential + weight
            if weight.shape[0]/self.width < 1:
                self.fCount = self.fCount + weight.shape[0]/self.width
            else:
                self.fCount = self.fCount + math.ceil(weight.shape[0]/self.width)
            self.addCount = math.ceil(weight.shape[0]/self.width)
        return memPotential

def test_accumulator():
    parallelism = 4
    accumu = accumulator(128, parallelism=parallelism)
    addcounts = 10
    partialSum = 0
    for i in range(addcounts):
        weightDatas = [(torch.rand(128)*32).int()-16 for i in range(parallelism)]
        for j in range(parallelism):
            partialSum = partialSum + weightDatas[j]
        test_partialSum = accumu(weightDatas)

    print(partialSum)
    print(test_partialSum)
    assert (test_partialSum == partialSum).all(), "test_partialSum is not equal to partialSum"
    
            
            