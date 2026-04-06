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

class weightBuffer(VSAModule):
    # careful!!!! The M is tiled by PE number P
    def __init__(self, N, K):
        super(weightBuffer,self).__init__()
        self.K = K
        self.N = N
        self.busWidth = cfg["processElement"]["weightBuffer"]["height"]
        self.inoutWidth = cfg["processElement"]["weightBuffer"]["inoutWidth"]
        self.mux = cfg["processElement"]["weightBuffer"]["mux"]
        self.temporature = cfg["processElement"]["weightBuffer"]["temporature"]
        self.voltage = cfg["processElement"]["weightBuffer"]["voltage"]
        self.periphery_vt = cfg["processElement"]["weightBuffer"]["periphery_vt"]
        self.SRAMTotalNum = max(64,math.ceil(N*K/(cfg["processElement"]["weightBuffer"]["height"]*cfg["processElement"]["weightBuffer"]["inoutWidth"])))
        print("weight Buffer SRAMTotalNum",self.SRAMTotalNum)
        
        # self.SRAMNumber = cfg["processElement"]["parallelism"]
        self.SRAMNumber = int(math.ceil(K/cfg["processElement"]["weightBuffer"]["height"]))
        self.SRAMNumHeight = int(math.ceil(K/cfg["processElement"]["weightBuffer"]["height"]))
        self.tileHN = cfg["processElement"]["weightBuffer"]["height"]

        # print("K,self.SRAMNumber,self.tileHN",K,self.SRAMNumber,self.tileHN)
        self.tileWN = 1
        self.srams = []
        for i in range(self.SRAMNumber):
            self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt, belong="weight"))
        self.SRAMNumber = math.ceil(self.N/self.inoutWidth)*self.SRAMNumber
        self.area = self.SRAMNumber*self.srams[0].area
    
    def calEnergy(self,latency):
        totalEnergy = 0.0
        for sram in self.srams:
            totalEnergy = totalEnergy + sram.calEnergy(latency,True)
        return totalEnergy
    
    def input_data(self, data, block_id, row_id):
        self.srams[block_id].write(data,row_id)
        
    def output_data(self, block_id, row_id, sign):
        return self.srams[block_id].read(row_id) if sign == 0 else -self.srams[block_id].read(row_id)

    def parallel_output_data(self, block_id_list, row_id_list,sign_list):
        data = []
        for block_id,row_id,sign in zip(block_id_list, row_id_list, sign_list):
            data.append(self.srams[block_id].read(row_id) if sign == 0 else -self.srams[block_id].read(row_id))
        return data


def test_weightBuffer():
    width = 4608
    inputNum = 128
    weigthbuffer = weightBuffer(128,4608)
    answer = torch.zeros(inputNum,width)
    for i in range(width):
        randomData = (torch.rand(inputNum)*256).int() - 128
        answer[:, i] = randomData
        block_id = i//weigthbuffer.tileHN
        row_id = i%weigthbuffer.tileHN
        # print(weigthbuffer.tileHN, block_id, row_id)
        weigthbuffer.input_data(randomData, block_id, row_id)

    output = torch.zeros(inputNum,width)
    for i in range(width):
        block_id = i//weigthbuffer.tileHN
        row_id = i%weigthbuffer.tileHN
        output[:,i] = weigthbuffer.output_data(block_id, row_id, 0)
        
    assert (answer == output).all(), "sram output is not equal to input"
    
    
    for sram in weigthbuffer.srams:
        print(sram.readCount,sram.writeCount)
    




