import torch
from basicModule import VSAModule,RouterDirection
from .Flit import Flit

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class RouteComputer(VSAModule):
    def __init__(self):
        super(RouteComputer,self).__init__()
        # self.mapping = None
        # with open('../RouterConnection.yaml', 'r', encoding='utf-8') as f:
        #     self.mapping = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.fEnergy = cfg["router"]["RoutingEngine"]["fEnergy"]
        self.area = cfg["router"]["RoutingEngine"]["area"]
        self.staticPower = cfg["router"]["RoutingEngine"]["leakage"]    
    
    def forward(self,flit:Flit):
        (m,n) = flit.RouteInfo
        if m == 0 and n == 0:
            return RouterDirection.Local
        elif m > 0 and n == 0:
            m = m - 1
            flit.RouteInfo = (m,n)
            return RouterDirection.East
        elif m < 0 and n == 0:
            m = m + 1
            flit.RouteInfo = (m,n)
            return RouterDirection.West
        elif n > 0:
            n = n - 1
            flit.RouteInfo = (m,n)
            return RouterDirection.North
        else:
            n = n + 1
            flit.RouteInfo = (m,n)
            return RouterDirection.South
