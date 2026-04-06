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

class IngressUnit(VSAModule):
    def __init__(self):
        super(IngressUnit,self).__init__()
        self.flitSize = cfg["NOC"]["flitSize"]
        self.virtualChannelNum = cfg["NOC"]["virtualChannelNum"]
        self.flitCount = [0 for i in range(self.virtualChannelNum)]
        self.CreditCount = [0 for i in range(self.virtualChannelNum)]
        self.GlobalState = ["I" for i in range(self.virtualChannelNum)] #  idle (I), routing (R), waiting for an output VC (V), active (A), or waiting for credits (C)
        self.Route = RouterDirection.Unknown
        self.OutputVC = [0 for i in range(self.virtualChannelNum)]
        self.flitBuffer = [ queue.Queue(maxsize=cfg["NOC"]["flitBufferNumPerVC"]) for i in range(self.virtualChannelNum) ]
    
    def forward(self,flit:Flit):
        # Router Computing
        if self.GlobalState[flit.VC] == "I":
            self.GlobalState[flit.VC] = "R"
        elif self.GlobalState[flit.VC] == "R":
            self.GlobalState[flit.VC] = "V"
        elif self.GlobalState[flit.VC] == "V":
            self.GlobalState[flit.VC] = "A"
            # flit.VC = self.OutputVC[flit.VC]
        elif self.GlobalState[flit.VC] == "A":
            self.flitCount[flit.VC] = self.flitCount[flit.VC] + 1
            if flit.tail: # IF the flit is the tail flit
                self.GlobalState[flit.VC] = "I"
        return flit
    
    # def backCredit(self,credit:Credit):
    #     self.CreditCount[credit.VC] = self.CreditCount[credit.VC] + 1
    #     self.flitBuffer[credit.VC].get(block=False)
    #     if self.GlobalState[credit.VC] == "C" and self.CreditCount[credit.VC] == self.flitCount[credit.VC]:
    #         self.GlobalState[credit.VC] = "I"
    #         self.flitBuffer[credit.VC].queue.clear()


