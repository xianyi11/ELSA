import torch
from basicModule import VSAModule
from .Flit import Flit
from .RouteComputer import RouteComputer
from basicModule import RouterDirection


class VCAllocator(VSAModule):
    def __init__(self):
        super(VCAllocator,self).__init__()

    def forward(self, VCState):
        for i, state in enumerate(VCState):
            if state == "I":
                return i
        return -1