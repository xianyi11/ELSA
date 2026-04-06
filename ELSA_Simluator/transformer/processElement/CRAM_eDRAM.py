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
        # get from the cacti
        # df = pd.read_excel(r"D:\tools\HPCA2025\simulator_CARBON\datasheet\SRAMDatasheet.xls",header=1)
        # data = df.loc[(df['io'] == self.inoutWidth*8) & (df['word'] == self.busWidth) & (df['mux'] == self.mux) &(df['T']==self.temporature) & (df['V']==self.voltage) & (df['periphery Vt']==self.periphery_vt)]
        # if self.belong == "weight":
        #     self.staticPower = 176.408e-9 # W
        #     self.readEnergy = 0.494e-12 # J
        #     self.wrtieEnergy = 0.320e-12 # J
        #     self.accessLatency = 0 # s
        #     self.area = 1173.248*1e-6 # mm^2
        # elif self.belong == "membrane":
        #     self.staticPower = 347.408e-9 # W
        #     self.readEnergy = 1.493e-12 # J
        #     self.wrtieEnergy = 1.157e-12 # J
        #     self.accessLatency = 0 # s
        #     self.area = 4012.0428*1e-6 # mm^2
        # elif self.belong == "spiketracer":
        #     self.staticPower = 93.030e-9 # W
        #     self.readEnergy = 0.397e-12 # J
        #     self.wrtieEnergy = 0.310e-12 # J
        #     self.accessLatency = 0 # s
        #     self.area = 1022.907*1e-6 # mm^2
            
        if self.belong == "weight":
            # 64 * 64
            self.staticPower = 11.558e-9 # W
            self.readEnergy = 0.686e-12 # J
            self.wrtieEnergy = 0.789e-12 # J
            self.accessLatency = 0 # s
            self.area = 210.545*1e-6 # mm^2
            # 8 * 16
            # self.staticPower = 14.492e-9 # W
            # self.readEnergy = 0.049e-12 # J
            # self.wrtieEnergy = 0.135e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 115.757*1e-6 # mm^2
        elif self.belong == "membrane":
            # dual-port sram
            # self.staticPower = 177.978e-9 # W
            # self.readEnergy = 0.764e-12 # J
            # self.wrtieEnergy = 0.592e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 2016.539*1e-6 # mm^2
            # single-port sram
            # 64x128
            self.staticPower = 23.115e-9*1.5 # W
            self.readEnergy = 1.373e-12*1.5 # J
            self.wrtieEnergy = 1.577e-12*1.5 # J
            self.accessLatency = 0 # s
            self.area = 412.090*1e-6*1.5 # mm^2
            # 8x16
            # self.staticPower = 29.979e-9 # W
            # self.readEnergy = 0.268e-12 # J
            # self.wrtieEnergy = 0.1e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 227.3*1e-6 # mm^2

        elif self.belong == "spiketracer":
            # 64 * 64
            self.staticPower = 11.558e-9 # W
            self.readEnergy = 0.686e-12 # J
            self.wrtieEnergy = 0.789e-12 # J
            self.accessLatency = 0 # s
            self.area = 210.545*1e-6 # mm^2
            # 8 * 64
            # self.staticPower = 14.492e-9 # W
            # self.readEnergy = 0.049e-12 # J
            # self.wrtieEnergy = 0.135e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 115.757*1e-6 # mm^2
        elif self.belong == "inputBuffer":
# 64 * 32
            self.staticPower = 9.452e-9 # W
            self.readEnergy = 0.345e-12 # J
            self.wrtieEnergy = 0.339e-12 # J
            self.accessLatency = 0 # s
            self.area = 131.342*1e-6 # mm^2