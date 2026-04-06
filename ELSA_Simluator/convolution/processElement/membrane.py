import numpy
import torch
import torch.nn as nn
from .SRAM import SRAM
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
import math
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


class membrane(VSAModule):
    # careful!!!! The M is tiled by PE number P
    def __init__(self, N, M, **kwargs):
        super(membrane,self).__init__()
        self.N = N
        self.M = M
        self.srams = []
        self.busWidth = cfg["processElement"]["membrane"]["height"]
        self.inoutWidth = cfg["processElement"]["membrane"]["inoutWidth"]
        self.mux = cfg["processElement"]["membrane"]["mux"]
        self.temporature = cfg["processElement"]["membrane"]["temporature"]
        self.voltage = cfg["processElement"]["membrane"]["voltage"]
        self.periphery_vt = cfg["processElement"]["membrane"]["periphery_vt"]

        self.SRAMNumber = M//self.busWidth if M % self.busWidth == 0 else M//self.busWidth + 1
        self.tileWN = self.busWidth
        self.tileHN = 1
        
        for i in range(self.SRAMNumber):
            self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt, belong="membrane"))
        self.SRAMNumber = math.ceil(self.N/self.inoutWidth)*self.SRAMNumber
        self.area = self.SRAMNumber*self.srams[0].area
    
    def calEnergy(self,latency):
        totalEnergy = 0.0
        for sram in self.srams:
            totalEnergy = totalEnergy + sram.calEnergy(latency)
        return totalEnergy

    def input_data(self, data, block_id, row_id):
        self.srams[block_id].write(data,row_id)
        
    def output_data(self, block_id, row_id):
        return self.srams[block_id].read(row_id)[:self.N]

    # def parallel_output_data(self, block_id_list, row_id_list):
    #     data = []
    #     for block_id,row_id in zip(block_id_list, row_id_list):
    #         data.append(self.srams[block_id].read(row_id))
    #     return data


class spikeTracer(VSAModule):
    # careful!!!! The M is tiled by PE number P
    def __init__(self, N, M, **kwargs):
        super(spikeTracer,self).__init__()
        self.N = N
        self.M = M
        self.srams = []
        self.busWidth = cfg["processElement"]["membrane"]["height"]
        self.inoutWidth = cfg["processElement"]["membrane"]["inoutWidth"]
        self.mux = cfg["processElement"]["membrane"]["mux"]
        self.temporature = cfg["processElement"]["membrane"]["temporature"]
        self.voltage = cfg["processElement"]["membrane"]["voltage"]
        self.periphery_vt = cfg["processElement"]["membrane"]["periphery_vt"]

        self.SRAMNumber = M//self.busWidth if M % self.busWidth == 0 else M//self.busWidth + 1
        self.tileWN = self.busWidth
        self.tileHN = 1

        for i in range(self.SRAMNumber):
            self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt, belong="spiketracer"))    
        self.SRAMNumber = math.ceil(self.N/self.inoutWidth)*self.SRAMNumber
        self.area = self.SRAMNumber*self.srams[0].area

    def calEnergy(self,latency):
        totalEnergy = 0.0
        for sram in self.srams:
            totalEnergy = totalEnergy + sram.calEnergy(latency)
        return totalEnergy
        
    def input_data(self, data, block_id, row_id):
        self.srams[block_id].write(data,row_id)
        
    def output_data(self, block_id, row_id):
        return self.srams[block_id].read(row_id)[:self.N]


def test_membrane():
    width = 25088
    inputNum = 128
    mem = membrane(inputNum,width)
    answer = torch.zeros(inputNum,width)
    for i in range(width):
        randomData = (torch.rand(inputNum)*256).int() - 128
        answer[:, i] = randomData
        block_id = i//mem.tileWN
        row_id = i%mem.tileWN
        mem.input_data(randomData, block_id, row_id)

    output = torch.zeros(inputNum,width)
    for i in range(width):
        block_id = i//mem.tileWN
        row_id = i%mem.tileWN
        output[:,i] = mem.output_data(block_id, row_id)
        
    assert (answer == output).all(), "sram output is not equal to input"
    
    
    for sram in mem.srams:
        print(sram.readCount,sram.writeCount)
    
    
def test_spikeTracer():
    width = 25088
    inputNum = 128
    mem = spikeTracer(inputNum,width)
    answer = torch.zeros(inputNum,width)
    for i in range(width):
        randomData = (torch.rand(inputNum)*256).int() - 128
        answer[:, i] = randomData
        block_id = i//mem.tileWN
        row_id = i%mem.tileWN
        mem.input_data(randomData, block_id, row_id)

    output = torch.zeros(inputNum,width)
    for i in range(width):
        block_id = i//mem.tileWN
        row_id = i%mem.tileWN
        output[:,i] = mem.output_data(block_id, row_id)
        
    assert (answer == output).all(), "sram output is not equal to input"
    
    
    for sram in mem.srams:
        print(sram.readCount,sram.writeCount)
    
    