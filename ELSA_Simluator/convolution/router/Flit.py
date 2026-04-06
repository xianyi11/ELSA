import torch
from basicModule import VSAModule
import math

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    

class Flit(VSAModule):
    def __init__(self, head=False, tail=False):
        super(Flit,self).__init__()
        self.flitSize = cfg["NOC"]["flitSize"]
        self.VCSize = math.ceil(math.log2(cfg["NOC"]["virtualChannelNum"]))
        self.TypeSize = 2
        self.RouteInfoSize = math.ceil(math.log2(cfg["NOC"]["meshHeight"])) + math.ceil(math.log2(cfg["NOC"]["meshWidth"]))
        self.checkSize = cfg["NOC"]["checkSize"]
        self.head = head
        self.tail = tail
        if self.head:
            self.payloadSize = self.flitSize - self.VCSize - self.TypeSize - self.RouteInfoSize - self.checkSize # 64-2-2-6-8=46
        else:
            self.payloadSize = self.flitSize - self.VCSize - self.TypeSize - self.checkSize # 64-2-2-8=52
        self.VC = None # virtual channel ID
        self.Type = None # 00 body, 01 head, 10 tail
        self.RouteInfo = None # (m,n)
        self.Payload = None # data
        self.Check = None # Dont Care
        
        # recode trace
        self.time = 0
        

    def setup(self,VC=None,Type=None,RouteInfo=None,Payload=None,Check=None):
        self.VC = VC
        self.Type = Type
        self.RouteInfo = RouteInfo
        self.Payload = Payload
        self.Check = Check
        self.time = 0
    
    def printmyself(self):
        print("virtual channel id:",self.VC)
        print("Type (00 body, 01 head, 10 tail):",self.Type)
        print("Route Information:",self.RouteInfo)
        print("Payload column:",self.Payload.columnId)
        print("Payload rowId:",self.Payload.rowId)
        print("Payload sign:",self.Payload.sign)
        print("output cycle in router:",self.time)

class payLoad(VSAModule):
    def __init__(self):
        self.columnId = []
        self.rowId = []
        self.sign = []
    
    def setup(self,columnId, rowId, sign):
        self.columnId = columnId
        self.rowId = rowId
        self.sign = sign

class Credit(VSAModule):
    def __init__(self, head=False):
        super(Credit,self).__init__()
        self.VCSize = math.ceil(math.log2(cfg["NOC"]["virtualChannelNum"]))
        self.TypeSize = 2
        self.checkSize = cfg["NOC"]["creditCheckSize"]
        self.CreditSize = self.VCSize + self.TypeSize + self.checkSize
        self.VC = None
        self.Type = None # 00 body, 01 head, 10 tail
        self.Check = None

    def setup(self,VC=None,Type=None,Check=None):
        self.VC = VC
        self.Type = Type
        self.Check = Check
