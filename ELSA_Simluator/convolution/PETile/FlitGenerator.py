import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
from router.Flit import Flit, payLoad
from typing import List, Optional
from processElement.SRAM import SRAM
from copy import deepcopy

cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


class FlitGenerator(VSAModule):
    def __init__(self,RouteInfo, PEId, TileValue):
        super(FlitGenerator,self).__init__()
        self.RouteInfo = RouteInfo
        self.PEId = PEId
        self.fEnergy = cfg["TILE"]["FlitGenerator"]["fEnergy"]
        self.area = cfg["TILE"]["FlitGenerator"]["area"]
        self.staticPower = cfg["TILE"]["FlitGenerator"]["leakage"]    
        self.TileValue = TileValue
        
    def forward(self,spikes:torch.tensor, columnId:int):
        self.fCount = self.fCount + 1
        curSpikePoint = 0
        flitNum = 0
        flitList = []
        IndexWidth = cfg["NOC"]["rowColumnBitWidth"]
        N = spikes.shape[0]
        while(1):    
            if curSpikePoint >= N:
                break
            if spikes[curSpikePoint] == 0:
                curSpikePoint = curSpikePoint + 1
                continue
            if spikes[curSpikePoint] != 0:
                payloadColumn = []
                payloadRow = []
                sign = []
                if flitNum == 0:
                    flit = Flit(head=True)
                    payloadSize = flit.payloadSize
                    spikeCount = (payloadSize - IndexWidth)//(IndexWidth+1)
                    # print("payloadSize",payloadSize,"head",spikeCount, torch.abs(spikes).sum())
                    payloadColumn.append(columnId)
                    flitType = "01"
                    flitRouteInfo = self.RouteInfo
                    flitNum = flitNum + 1
                else:
                    flit = Flit(head=False)
                    payloadSize = flit.payloadSize
                    spikeCount = (payloadSize)//(IndexWidth+1)
                    # print("payloadSize",payloadSize,"body",spikeCount, torch.abs(spikes).sum())
                    flitNum = flitNum + 1
                    flitRouteInfo = None
                    flitType = "00"
                    
                while(1):
                    if curSpikePoint >= N:
                        break
                    if spikes[curSpikePoint] == 0:
                        curSpikePoint = curSpikePoint + 1
                        continue
                    if spikes[curSpikePoint] != 0:
                        spikeCount = spikeCount - 1
                        payloadRow.append(curSpikePoint+self.PEId*self.TileValue)
                        if spikes[curSpikePoint] == -1:
                            sign.append(1)
                        else:
                            sign.append(0)
                        curSpikePoint = curSpikePoint + 1
                        if spikeCount == 0:
                            break
                
                thePayload = payLoad()
                thePayload.setup(columnId=payloadColumn,rowId=payloadRow,sign=sign)
                flit.setup(VC=self.PEId, Type=flitType, RouteInfo=flitRouteInfo,Payload=thePayload, Check=None)            
                # self,VC=None,Type=None,RouteInfo=None,Payload=None,Check=None
                flitList.append(flit)

        if len(flitList) != 0:
            flitList[-1].tail=True
            flitList[-1].Type="10"
        else:
            flit = Flit(head=True,tail=True)
            thePayload = payLoad()
            payloadColumn = [columnId]
            payloadRow = []
            sign = []
            flitType = "01"
            flitRouteInfo = self.RouteInfo
            thePayload.setup(columnId=payloadColumn,rowId=payloadRow,sign=sign)
            flit.setup(VC=self.PEId, Type=flitType, RouteInfo=flitRouteInfo,Payload=thePayload, Check=None)
            flitList.append(flit)
        return flitList

class FlitCombiner(VSAModule):
    def __init__(self):
        super(FlitCombiner,self).__init__()
        self.fEnergy = cfg["TILE"]["FlitCombiner"]["fEnergy"]
        self.area = cfg["TILE"]["FlitCombiner"]["area"]
        self.staticPower = cfg["TILE"]["FlitCombiner"]["leakage"]
        # 负责同步以及合并flits为一列spike元素
        
        self.busWidth = cfg["TILE"]["FlitCombiner"]["height"]
        self.inoutWidth = cfg["TILE"]["FlitCombiner"]["inoutWidth"]
        self.mux = cfg["TILE"]["FlitCombiner"]["mux"]
        self.temporature = cfg["TILE"]["FlitCombiner"]["temporature"]
        self.voltage = cfg["TILE"]["FlitCombiner"]["voltage"]
        self.periphery_vt = cfg["TILE"]["FlitCombiner"]["periphery_vt"]
        self.PENums = cfg["TILE"]["PENum"]
        self.flitGenTime = cfg["NOC"]["flitGenTime"]

        self.srams = []
        self.bufferNum = 1
        self.wordsize = cfg["Word"]
        for i in range(self.bufferNum):
            self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt))

        self.spikes = []
        self.tailNum = 0
        self.maxcycle = 0
        self.columnID = -1
        
    def forward(self,flitList:List[Flit]):
        if len(flitList) == 0:
            self.tailNum = self.tailNum + 1
        else:
            if self.columnID == -1:
                self.columnID = flitList[0].Payload.columnId[0]
            for i,flit in enumerate(flitList):
                if i == 0:
                    columnID = flit.Payload.columnId[0]
                    assert self.columnID == columnID, f"From FlitCombiner!!!! the columnID mush be same. self.columnID:{self.columnID} != columnID:{columnID}"
                for rowId,sign in zip(flit.Payload.rowId, flit.Payload.sign):
                    self.spikes.append((rowId,columnID,sign))
                # self.maxcycle = max(self.maxcycle, flit.time + len(flitList)*self.flitGenTime) # the worst case, we need self.flitGenTime cycle to generate one flit
                self.maxcycle = max(self.maxcycle, flit.time)
                # print("in flitcombiner:",flit.time,"len(flitList)*self.flitGenTime",len(flitList)*self.flitGenTime,"self.maxcycle",self.maxcycle)
                if flit.tail:
                    self.tailNum = self.tailNum + 1
        if self.tailNum == self.PENums:            
            if self.columnID == -1: #no spike through
                self.tailNum = 0
                self.spikes = []
                self.maxcycle = 0
                return None
            self.writeCount = len(self.spikes)
            self.readCount = len(self.spikes)
            spikes = deepcopy(self.spikes)
            self.spikes = []
            self.tailNum = 0
            maxcycle = deepcopy(self.maxcycle)
            columnID = self.columnID + 0
            self.maxcycle = 0
            self.columnID = -1
            return spikes, columnID, maxcycle
        else:
            return None
    

def test_FlitGenerator():
    N = 128
    spikes = torch.zeros(N).int()
    spikes_rands = torch.rand(N)
    spikes[spikes_rands < 0.2] = -1
    spikes[spikes_rands > 0.8] = 1
    print(spikes)
    genSpike = FlitGenerator(RouteInfo=(1,1), PEId=0, TileValue=N)
    flitList = genSpike(spikes,columnId=2)
    for i, flit in enumerate(flitList):
        print(f"==================flit{i}====================")
        flit.printmyself()
    
def test_FlitCombiner():
    N = 128
    spikes = torch.zeros(N).int()
    # spikes_rands = torch.rand(N)
    # spikes[spikes_rands < 0.2] = -1
    # spikes[spikes_rands > 0.8] = 1
    genSpike = FlitGenerator(RouteInfo=(1,1), PEId=0,TileValue=N)
    flitList1 = genSpike(spikes,columnId=2)
    flitList2 = genSpike(spikes,columnId=2)
    flitList3 = genSpike(spikes,columnId=2)
    flitList4 = genSpike(spikes,columnId=2)

    for i, flit in enumerate(flitList1):
        print(f"==================flit{i}====================")
        flit.printmyself()
    
    for i,flit in enumerate(flitList1):
        flit.time = i
    for i,flit in enumerate(flitList2):
        flit.time = i
    for i,flit in enumerate(flitList3):
        flit.time = i
    for i,flit in enumerate(flitList4):
        flit.time = i
        
    fc = FlitCombiner()
    
    fc(flitList1)
    fc(flitList2)
    fc(flitList3)
    spikes, columnID, maxcycle = fc(flitList4)
    print(spikes, columnID, maxcycle)