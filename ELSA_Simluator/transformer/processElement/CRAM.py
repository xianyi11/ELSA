import numpy
import torch
import torch.nn as nn
import pandas as pd
from basicModule import VSAModule
import math

class CRAM(VSAModule):
    def __init__(self,busWidth,inoutWidth,matrixWidth, matrixHeight, mux,temporature, voltage, periphery_vt, belong="weight"):
        super(CRAM,self).__init__()
        self.busWidth = busWidth
        self.inoutWidth = inoutWidth
        self.matrixWidth = matrixWidth
        self.matrixHeight = matrixHeight
        self.readCount = 0
        self.writeCount = 0
        self.mux = mux
        self.temporature = temporature
        self.voltage = voltage
        self.periphery_vt = periphery_vt
        self.belong = belong
        self.get_power_area_latency()
        self.memory = torch.zeros(self.matrixHeight, self.matrixWidth)
        # print("self.memory",self.memory.shape)
        
    def write(self,data,rowColId,direction):
        # direction: (0->rowId) (1->colId)
        # assert self.inoutWidth == data.shape[0], "The length of input data must be same with the inout-width of the sram"
        dataBlockLen = self.inoutWidth if direction == 0 else self.busWidth
        self.writeCount += math.ceil(data.shape[0]/dataBlockLen)
        if direction == 0:
            self.memory[rowColId] = data
        else:
            self.memory[:,rowColId] = data

    def fake_write(self,data,rowColId,direction):
        # direction: (0->rowId) (1->colId)
        # assert self.inoutWidth == data.shape[0], "The length of input data must be same with the inout-width of the sram"
        dataBlockLen = self.inoutWidth if direction == 0 else self.busWidth
        if direction == 0:
            self.memory[rowColId] = data
        else:
            self.memory[:,rowColId] = data


    def read(self,rowColId,direction):
        # direction: (0->rowId) (1->colId)
        if direction == 0:
            data = self.memory[rowColId]
        else:
            data = self.memory[:,rowColId]
        dataBlockLen = self.inoutWidth if direction == 0 else self.busWidth
        self.readCount += math.ceil(data.shape[0]/dataBlockLen)
        return data


    def fake_read(self,rowColId,direction):
        # direction: (0->rowId) (1->colId)
        if direction == 0:
            data = self.memory[rowColId]
        else:
            data = self.memory[:,rowColId]
        dataBlockLen = self.inoutWidth if direction == 0 else self.busWidth
        return data

    def get_power_area_latency(self):            
        if self.belong == "membrane":
            # self.staticPower = 52.198e-6 # W
            # self.readEnergy = 0.469e-12 # J
            # self.wrtieEnergy = 0.389e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 4586.798*1e-6 # mm^2

            #SRAM
            self.staticPower = 261.494e-9 # W
            self.readEnergy = 0.730e-12 # J
            self.wrtieEnergy = 0.480e-12 # J
            self.accessLatency = 0 # s
            self.area = 1695.830*1e-6 # mm^2
        elif self.belong == "spiketracer":
            # CRAM
            # self.staticPower = 27.580e-6 # W
            # self.readEnergy = 0.244e-12 # J
            # self.wrtieEnergy = 0.203e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 2344.950*1e-6 # mm^2

            #SRAM
            self.staticPower = 91.121e-9 # W
            self.readEnergy = 0.256e-12 # J
            self.wrtieEnergy = 0.170e-12 # J
            self.accessLatency = 0 # s
            self.area = 608.522*1e-6 # mm^2
        elif self.belong == "weight":
            # CRAM
            # self.staticPower = 27.580e-6 # W
            # self.readEnergy = 0.535e-12 # J
            # self.wrtieEnergy = 0.446e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 2344.950*1e-6 # mm^2

            #SRAM
            self.staticPower = 91.121e-9 # W
            self.readEnergy = 0.256e-12 # J
            self.wrtieEnergy = 0.170e-12 # J
            self.accessLatency = 0 # s
            self.area = 608.522*1e-6 # mm^2


    



