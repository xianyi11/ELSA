import torch
from basicModule import VSAModule
from .Flit import Flit, Credit
from .RouteComputer import RouteComputer
from basicModule import RouterDirection
import queue

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class EgressUnit(VSAModule):
    def __init__(self):
        super(EgressUnit,self).__init__()
        self.flitSize = cfg["NOC"]["flitSize"]
        self.virtualChannelNum = cfg["NOC"]["virtualChannelNum"]
        self.flitCount = [0 for i in range(self.virtualChannelNum)]
        self.CreditCount = [0 for i in range(self.virtualChannelNum)]
        self.GlobalState = ["I" for i in range(self.virtualChannelNum)] #  Either idle (I), active (A), or waiting for credits (C).
        self.InputVC = [0 for i in range(self.virtualChannelNum)]
        self.flitBuffer = [ queue.Queue(maxsize=cfg["NOC"]["flitBufferNumPerVC"]) for i in range(self.virtualChannelNum) ]

    def forward(self,flit:Flit):
        if self.GlobalState[flit.VC] == "I":
            self.GlobalState[flit.VC] = "A"
        elif self.GlobalState[flit.VC] == "A":
            self.flitCount[flit.VC] = self.flitCount[flit.VC] + 1
            if flit.tail: # IF the flit is the tail flit
                self.GlobalState[flit.VC] = "I"
        return flit
        
        
        