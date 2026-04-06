import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
from .STBIFFunction import STBIFNeuron
import math

cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class fireComponent(VSAModule):
    def __init__(self, width):
        super(fireComponent,self).__init__()
        self.width = cfg["processElement"]["membrane"]["inoutWidth"]*8
        self.SMax = cfg["processElement"]["fireComponent"]["SMax"]
        self.SMin = cfg["processElement"]["fireComponent"]["SMin"]
        self.M = 0
        self.N = 0
        self.vThr = 2**self.N

        self.fCount = 0
        self.fEnergy = cfg["processElement"]["fireComponent"]["fEnergy"]
        self.area = cfg["processElement"]["fireComponent"]["area"]
        self.staticPower = cfg["processElement"]["fireComponent"]["leakage"]
        self.tileN = width
    
    def set_MN(self,M,N):
        self.N = N
        self.M = M
        self.vThr = 2**self.N
    
    def forward(self,spikeTracer, membrane):
        if self.width >= self.tileN:
            self.fCount = self.fCount + 1
        else:
            self.fCount = self.fCount + math.ceil((self.tileN+0.0)/self.width)
        
        positiveSpikes = torch.logical_and(membrane >= self.vThr, spikeTracer < self.SMax).int()
        negativeSpikes = torch.logical_and(membrane < 0, spikeTracer > self.SMin).int()
        outSpikes = (positiveSpikes - negativeSpikes)
        membrane = membrane - outSpikes*self.vThr
        return outSpikes, membrane


def test_firecomponent():
    firecom = fireComponent(128)
    
    
    M1 = cfg["processElement"]["fireComponent"]["M"]
    N1 = cfg["processElement"]["fireComponent"]["N"]
    SMax = cfg["processElement"]["fireComponent"]["SMax"]
    SMin = cfg["processElement"]["fireComponent"]["SMin"]

    spikesList = []
    membraneList = []
    inferTime = 16

    truespikesList = []
    truemembraneList = []
    
    neuron = STBIFNeuron(M=M1,N=N1,pos_max=SMax,neg_min=SMin,bias=None)

    spikeTracer = (torch.zeros(128)).int()
    for t in range(inferTime):
        input = (torch.rand(128)*255-128).int()
        if t == 0:
            spikes, updatemembrane = firecom(spikeTracer,membrane=2**(N1-1)+input*M1)
        else:
            spikes, updatemembrane = firecom(spikeTracer,membrane=updatemembrane+input*M1)
        spikeTracer = spikeTracer + spikes
        spikesList.append(spikes+0.0)
        membraneList.append(updatemembrane+0.0)
        truespikes = neuron(input)
        truespikesList.append(truespikes+0.0)
        truemembraneList.append(neuron.q+0.0)
    
    spikesList = torch.stack(spikesList,dim=0)
    membraneList = torch.stack(membraneList,dim=0)
    truespikesList = torch.stack(truespikesList,dim=0)
    truemembraneList = torch.stack(truemembraneList,dim=0)
    print(spikesList)
    print(truespikesList)
    
    assert (truespikesList == spikesList).all(), "truespikes is not equal to spikes"
    assert (truemembraneList == membraneList).all(), "true membrane potential is not equal to membrane potential"