import torch
from basicModule import VSAModule
from .Flit import Flit, Credit
from .RouteComputer import RouteComputer
from .Switch import SwitchCrossbar
from .SwitchAllocator import SwitchAllocator
from .IngressUnit import IngressUnit
from .InputUnit import InputUnit
from .EgressUnit import EgressUnit
from .OutputUnit import OutputsUnit
from .VCAllocator import VCAllocator
from basicModule import RouterDirection, RouteMap
import queue
from typing import List, Optional
from random import random
from .Flit import payLoad


import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    

class Router(VSAModule):
    def __init__(self):
        super(Router,self).__init__()
        self.RouterComputer = RouteComputer()
        self.Switch = SwitchCrossbar()
        self.SwitchAllocator = SwitchAllocator()
        self.IngressUnit = [IngressUnit() for i in range(cfg["NOC"]["ingressRouter"])]
        self.InputUnit = [InputUnit() for i in range(4)]
        self.EgressUnit = [EgressUnit() for i in range(cfg["NOC"]["egressRouter"])]
        self.OutputUnit = [OutputsUnit() for i in range(4)]
        self.VCAllocator = VCAllocator()
        self.cycleCount = 0
        self.flitCount = 0
        self.route = None

    def forward(self,flitLocal:Optional[Flit], flitLocalID:Optional[int],flitRoute:Optional[Flit], flitRouteID:Optional[int]):
        if flitLocal is not None:
            # RC
            if flitLocal.head:
                flitLocal = self.IngressUnit[flitLocalID](flitLocal)
                self.IngressUnit[flitLocalID].Route = self.RouterComputer(flitLocal)
                flitLocal = self.IngressUnit[flitLocalID](flitLocal)
                flitLocal.time = flitLocal.time + 1
            # VA
            if flitLocal.head:
                if self.IngressUnit[flitLocalID].Route != RouterDirection.Local:
                    self.outputVCId = self.VCAllocator(self.OutputUnit[RouteMap[self.IngressUnit[flitLocalID].Route]].GlobalState)
                    self.OutputUnit[RouteMap[self.IngressUnit[flitLocalID].Route]].GlobalState[self.outputVCId] = "A"
                else:
                    self.outputVCId = self.VCAllocator(self.EgressUnit[RouteMap[self.IngressUnit[flitLocalID].Route]].GlobalState)
                    self.EgressUnit[RouteMap[self.IngressUnit[flitLocalID].Route]].GlobalState[self.outputVCId] = "A"
                # if self.outputVCId == -1: #virtual channel is full
                #     flitLocal.time = flitLocal.time + 1
                #     return -1
                self.IngressUnit[flitLocalID].OutputVC[flitLocal.VC] = self.outputVCId
                flitLocal = self.IngressUnit[flitLocalID](flitLocal)
                flitLocal.time = flitLocal.time + 1
            # SA
            self.SwitchAllocator(flitLocal)
            if flitLocal.head:
                flitLocal.time = flitLocal.time + 1
            # ST
            if self.IngressUnit[flitLocalID].Route != RouterDirection.Local:
                self.OutputUnit[RouteMap[self.IngressUnit[flitLocalID].Route]](self.Switch(flitLocal))
                self.OutputUnit[RouteMap[self.IngressUnit[flitLocalID].Route]].InputVC[self.IngressUnit[flitLocalID].OutputVC[flitLocal.VC]] = flitLocal.VC
                flitLocal.VC = self.IngressUnit[flitLocalID].OutputVC[flitLocal.VC]
            else:
                self.EgressUnit[RouteMap[self.IngressUnit[flitLocalID].Route]](self.Switch(flitLocal))
                self.EgressUnit[RouteMap[self.IngressUnit[flitLocalID].Route]].InputVC[self.IngressUnit[flitLocalID].OutputVC[flitLocal.VC]] = flitLocal.VC
                flitLocal.VC = self.IngressUnit[flitLocalID].OutputVC[flitLocal.VC]
            flitLocal.time = flitLocal.time + 1
            self.flitCount = self.flitCount + 1
            return flitLocal
        elif flitRoute is not None:
            # RC
            if flitRoute.head:
                flitRoute = self.InputUnit[flitRouteID](flitRoute)
                self.InputUnit[flitRouteID].Route = self.RouterComputer(flitRoute)
                flitRoute = self.InputUnit[flitRouteID](flitRoute)
                flitRoute.time = flitRoute.time + 1
            # VA
            if flitRoute.head:
                if self.InputUnit[flitRouteID].Route != RouterDirection.Local:
                    self.outputVCId = self.VCAllocator(self.OutputUnit[RouteMap[self.InputUnit[flitRouteID].Route]].GlobalState)
                    self.OutputUnit[RouteMap[self.InputUnit[flitRouteID].Route]].GlobalState[self.outputVCId] = "A"
                else:
                    self.outputVCId = self.VCAllocator(self.EgressUnit[RouteMap[self.InputUnit[flitRouteID].Route]].GlobalState)
                    self.EgressUnit[RouteMap[self.InputUnit[flitRouteID].Route]].GlobalState[self.outputVCId] = "A"
                # if self.outputVCId == -1: #virtual channel is full
                #     flitLocal.time = flitLocal.time + 1
                #     return -1
                self.InputUnit[flitRouteID].OutputVC[flitRoute.VC] = self.outputVCId
                flitRoute = self.InputUnit[flitRouteID](flitRoute)
                flitRoute.time = flitRoute.time + 1
            # SA
            self.SwitchAllocator(flitRoute)
            if flitRoute.head:
                flitRoute.time = flitRoute.time + 1
            # ST
            if self.InputUnit[flitRouteID].Route != RouterDirection.Local:
                self.OutputUnit[RouteMap[self.InputUnit[flitRouteID].Route]](self.Switch(flitRoute))
                self.OutputUnit[RouteMap[self.InputUnit[flitRouteID].Route]].InputVC[self.InputUnit[flitRouteID].OutputVC[flitRoute.VC]] = flitRoute.VC
                flitRoute.VC = self.InputUnit[flitRouteID].OutputVC[flitRoute.VC]
            else:
                self.EgressUnit[RouteMap[self.InputUnit[flitRouteID].Route]](self.Switch(flitRoute))
                self.EgressUnit[RouteMap[self.InputUnit[flitRouteID].Route]].InputVC[self.InputUnit[flitRouteID].OutputVC[flitRoute.VC]] = flitRoute.VC
                flitRoute.VC = self.InputUnit[flitRouteID].OutputVC[flitRoute.VC]

            flitRoute.time = flitRoute.time + 1
            self.flitCount = self.flitCount + 1
            return flitRoute
        else:
            return None

def GenFlits(spikes,VCId,RouteInfo):
    N = cfg["processElement"]["input"]["N"]
    columnId = 1
    curSpikePoint = 0
    flitNum = 0
    flitList = []
    IndexWidth = cfg["NOC"]["rowColumnBitWidth"]
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
                payloadColumn.append(columnId)
                flitType = "01"
                flitRouteInfo = RouteInfo
                flitNum = flitNum + 1
            else:
                flit = Flit(head=False)
                payloadSize = flit.payloadSize
                spikeCount = (payloadSize)//(IndexWidth+1)
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
                    payloadRow.append(curSpikePoint)
                    if spikes[curSpikePoint] == -1:
                        sign.append(1)
                    else:
                        sign.append(0)
                    curSpikePoint = curSpikePoint + 1
                    if spikeCount == 0:
                        break
            
            thePayload = payLoad()
            thePayload.setup(columnId=payloadColumn,rowId=payloadRow,sign=sign)
            flit.setup(VC=VCId, Type=flitType, RouteInfo=flitRouteInfo,Payload=thePayload, Check=None)            
            # self,VC=None,Type=None,RouteInfo=None,Payload=None,Check=None
            flitList.append(flit)
    return flitList

def GenRandomInputs(N,VCId,RouteInfo):
    spikes = torch.zeros(N).int()
    spikes_rands = torch.rand(N)
    spikes[spikes_rands < 0.2] = -1
    spikes[spikes_rands > 0.8] = 1        
    flitList = GenFlits(spikes,VCId,RouteInfo)
    flitList[-1].tail=True
    flitList[-1].Type="10"
    return spikes, flitList

def test_router_basic():

    N = cfg["processElement"]["input"]["N"]
    
    VSARouter = Router()
    
    # from local 1
    localId = 1
    spikes, flitList = GenRandomInputs(N,localId,(1,1))

    print(spikes)
    # print(flitList)
    for i, flit in enumerate(flitList):
        print(f"=========================input flit{i}=========================")
        print("column:",flit.Payload.columnId)
        print("rowId:",flit.Payload.rowId)
        print("sign:",flit.Payload.sign)
        print("routerInfo:",flit.RouteInfo)

    print(spikes)
    # print(flitList)
    for i, flit in enumerate(flitList):
        print(f"=========================input flit{i}=========================")
        flit.printmyself()


    outFlitList = []
    for i, flit in enumerate(flitList):
        outFlitList.append(VSARouter(flitLocal=flit,flitLocalID=localId, flitRoute=None, flitRouteID=None))
        # outFlitList.append(VSARouter(flitLocal=None,flitLocalID=None, flitRoute=flit, flitRouteID=localId))    

    for i, flit in enumerate(outFlitList):
        print(f"=========================output flit{i}=========================")
        flit.printmyself()
        
    print("cycle:",VSARouter.cycleCount)
    print("flitcount:",VSARouter.flitCount)
    print("routeDirection:",VSARouter.route)
    
def test_router_multi_input():
    N = cfg["processElement"]["input"]["N"]
    VSARouter = Router()
    localId = 0

    VCID1 = 0
    spikes1, flitList1 = GenRandomInputs(N,VCID1,(1,1))
    
    VCID2 = 1
    spikes2, flitList2 = GenRandomInputs(N,VCID2,(1,1))    
    
    VCID3 = 2
    spikes3, flitList3 = GenRandomInputs(N,VCID3,(1,1))    
    
    VCID4 = 3
    spikes4, flitList4 = GenRandomInputs(N,VCID4,(1,1))    
    
    outFlitList = []

    for i, flit in enumerate(flitList1):
        if i != 0:
            flit.time = outFlitList[-1].time
        flit = VSARouter(flitLocal=flitList1[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList.append(flit)

    for i, flit in enumerate(flitList2):
        if i != 0:
            flit.time = outFlitList[-1].time
        flit = VSARouter(flitLocal=flitList2[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList.append(flit)

    for i, flit in enumerate(flitList3):
        if i != 0:
            flit.time = outFlitList[-1].time
        flit = VSARouter(flitLocal=flitList3[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList.append(flit)

    for i, flit in enumerate(flitList4):
        if i != 0:
            flit.time = outFlitList[-1].time
        flit = VSARouter(flitLocal=flitList4[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList.append(flit)

    # for i in range(minvalue):
    #     outFlitList.append(VSARouter(flitLocal=flitList1[i],flitLocalID=localId, flitRoute=None, flitRouteID=None))
    #     outFlitList.append(VSARouter(flitLocal=flitList2[i],flitLocalID=localId, flitRoute=None, flitRouteID=None))
    #     outFlitList.append(VSARouter(flitLocal=flitList3[i],flitLocalID=localId, flitRoute=None, flitRouteID=None))
    #     outFlitList.append(VSARouter(flitLocal=flitList4[i],flitLocalID=localId, flitRoute=None, flitRouteID=None))
    
    for i, flit in enumerate(outFlitList):
        print(f"=========================output flit{i}=========================")
        flit.printmyself()
        
    # print("cycle:",VSARouter.cycleCount)
    print("flitcount:",VSARouter.flitCount)
    print("routeDirection:",VSARouter.route)
    

def test_router_multi_inputchannel():    
    N = cfg["processElement"]["input"]["N"]
    VSARouter = Router()
    localId = 0

    VCID1 = 0
    spikes1, flitList1 = GenRandomInputs(N,VCID1,(1,1))
    
    VCID2 = 1
    spikes2, flitList2 = GenRandomInputs(N,VCID2,(1,1))    
    
    VCID3 = 2
    spikes3, flitList3 = GenRandomInputs(N,VCID3,(1,1))    
    
    VCID4 = 3
    spikes4, flitList4 = GenRandomInputs(N,VCID4,(1,1))    
    

    len1 = len(flitList1)
    len2 = len(flitList2)
    len3 = len(flitList3)
    len4 = len(flitList4)

    minLen = min(len1,len2,len3,len4)
    
    outFlitList1 = []
    outFlitList2 = []
    outFlitList3 = []
    outFlitList4 = []
    
    for i in range(minLen):
        if i != 0:
            flitList1[i].time = outFlitList1[-1].time
        flit = VSARouter(flitLocal=flitList1[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList1.append(flit)

        if i != 0:
            flitList2[i].time = outFlitList2[-1].time
        flit = VSARouter(flitLocal=flitList2[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList2.append(flit)

        if i != 0:
            flitList3[i].time = outFlitList3[-1].time
        flit = VSARouter(flitLocal=flitList3[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList3.append(flit)
    
        if i != 0:
            flitList4[i].time = outFlitList4[-1].time
        flit = VSARouter(flitLocal=flitList4[i],flitLocalID=localId, flitRoute=None, flitRouteID=None)
        outFlitList4.append(flit) 
    
    for i in range(minLen):
        print(f"=========================channel{1}: output flit{i}=========================")
        outFlitList1[i].printmyself()
        
        print(f"=========================channel{2}: output flit{i}=========================")
        outFlitList2[i].printmyself()
        
        print(f"=========================channel{3}: output flit{i}=========================")
        outFlitList3[i].printmyself()

        print(f"=========================channel{4}: output flit{i}=========================")
        outFlitList4[i].printmyself()

def test_router_multi_router():
    N = cfg["processElement"]["input"]["N"]
    VSARouter = Router()
    localId = 0

    localId1 = 0
    spikes1, flitList1 = GenRandomInputs(N,0,(1,1))
    
    localId2 = 1
    spikes2, flitList2 = GenRandomInputs(N,0,(1,1))    
    
    localId3 = 2
    spikes3, flitList3 = GenRandomInputs(N,0,(1,1))    
    
    localId4 = 3
    spikes4, flitList4 = GenRandomInputs(N,0,(1,1))    
    

    len1 = len(flitList1)
    len2 = len(flitList2)
    len3 = len(flitList3)
    len4 = len(flitList4)

    minLen = min(len1,len2,len3,len4)
    
    outFlitList1 = []
    outFlitList2 = []
    outFlitList3 = []
    outFlitList4 = []
    
    for i in range(minLen):
        if i != 0:
            flitList1[i].time = outFlitList1[-1].time
        flit = VSARouter(flitLocal=None,flitLocalID=None, flitRoute=flitList1[i], flitRouteID=localId1)
        outFlitList1.append(flit)

        if i != 0:
            flitList2[i].time = outFlitList2[-1].time
        flit = VSARouter(flitLocal=None,flitLocalID=None, flitRoute=flitList2[i], flitRouteID=localId2)
        outFlitList2.append(flit)

        if i != 0:
            flitList3[i].time = outFlitList3[-1].time
        flit = VSARouter(flitLocal=None,flitLocalID=None, flitRoute=flitList3[i], flitRouteID=localId3)
        outFlitList3.append(flit)
    
        if i != 0:
            flitList4[i].time = outFlitList4[-1].time
        flit = VSARouter(flitLocal=None,flitLocalID=None, flitRoute=flitList4[i], flitRouteID=localId4)
        outFlitList4.append(flit) 
    
    for i in range(minLen):
        print(f"=========================router{1}: output flit{i}=========================")
        outFlitList1[i].printmyself()
        
        print(f"=========================router{2}: output flit{i}=========================")
        outFlitList2[i].printmyself()
        
        print(f"=========================router{3}: output flit{i}=========================")
        outFlitList3[i].printmyself()

        print(f"=========================router{4}: output flit{i}=========================")
        outFlitList4[i].printmyself()
