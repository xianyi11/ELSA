import torch.nn as nn
from enum import Enum

class VSAModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.staticPower = 0 # W
        self.readEnergy = 0 # J
        self.wrtieEnergy = 0 # J
        self.accessLatency = 0 # s
        self.area = 0 # mm^2
        self.readCount = 0 
        self.writeCount = 0
        self.fCount = 0
        self.fEnergy = 0 # J
    
    def calEnergy(self,latency,record=False):
        # if self.readCount > 0 and record:
        #     print("static Power:",self.staticPower*latency,"self.readCount",self.readCount, "readEnergy",self.readEnergy*self.readCount, "self.writeCount", self.writeCount,"wrtieEnergy", self.wrtieEnergy*self.writeCount)
        return self.staticPower*latency + self.readEnergy*self.readCount + self.wrtieEnergy*self.writeCount + self.fEnergy*self.fCount
    
    def getArea(self):
        return self.area
    
class RouterDirection(Enum):
    Local = 1
    North = 2
    West = 3
    South = 4
    East = 5
    Unknown = 6

RouteMap = {RouterDirection.Local:0, RouterDirection.North:0, RouterDirection.West:1, RouterDirection.South:1, RouterDirection.East:1}