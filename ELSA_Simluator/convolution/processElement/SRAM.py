import numpy
import torch
import torch.nn as nn
import pandas as pd
from basicModule import VSAModule
import math

class SRAM(VSAModule):
    def __init__(self,busWidth,inoutWidth, mux,temporature, voltage, periphery_vt, belong="weight"):
        super(SRAM,self).__init__()
        self.busWidth = busWidth
        self.inoutWidth = inoutWidth
        self.readCount = 0
        self.writeCount = 0
        self.mux = mux
        self.temporature = temporature
        self.voltage = voltage
        self.periphery_vt = periphery_vt
        self.belong = belong
        self.get_power_area_latency()
        self.memory = []
        for i in range(self.busWidth):
            self.memory.append(torch.zeros(self.inoutWidth))
    
    def write(self,data,row_id):
        # assert self.inoutWidth == data.shape[0], "The length of input data must be same with the inout-width of the sram"
        if self.inoutWidth >= data.shape[0]:
            self.writeCount += 1
            self.memory[row_id] = data
        else:
            self.writeCount += math.ceil(data.shape[0]/self.inoutWidth)
            self.memory[row_id] = data

    def read(self,row_id):
        data = self.memory[row_id]
        if self.inoutWidth >= data.shape[0]:
            self.readCount += 1
        else:
            self.readCount += math.ceil(data.shape[0]/self.inoutWidth)
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
            # 64 * 16
            self.staticPower = 91.121e-9 # W
            self.readEnergy = 0.256e-12 # J
            self.wrtieEnergy = 0.170e-12 # J
            self.accessLatency = 0 # s
            self.area = 608.522*1e-6 # mm^2
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
            # 64x16
            self.staticPower = 261.494e-9 # W
            self.readEnergy = 0.730e-12 # J
            self.wrtieEnergy = 0.480e-12 # J
            self.accessLatency = 0 # s
            self.area = 1695.830*1e-6 # mm^2
            # 8x16
            # self.staticPower = 29.979e-9 # W
            # self.readEnergy = 0.268e-12 # J
            # self.wrtieEnergy = 0.1e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 227.3*1e-6 # mm^2
            
        elif self.belong == "spiketracer":
            # 64*16
            self.staticPower = 91.121e-9 # W
            self.readEnergy = 0.256e-12 # J
            self.wrtieEnergy = 0.170e-12 # J
            self.accessLatency = 0 # s
            self.area = 608.522*1e-6 # mm^2
            # 8 * 64
            # self.staticPower = 14.492e-9 # W
            # self.readEnergy = 0.049e-12 # J
            # self.wrtieEnergy = 0.135e-12 # J
            # self.accessLatency = 0 # s
            # self.area = 115.757*1e-6 # mm^2
        elif self.belong == "inputBuffer":
            self.staticPower = 48.388e-9 # W
            self.readEnergy = 0.136e-12 # J
            self.wrtieEnergy = 0.092e-12 # J
            self.accessLatency = 0 # s
            self.area = 311.456*1e-6 # mm^2




