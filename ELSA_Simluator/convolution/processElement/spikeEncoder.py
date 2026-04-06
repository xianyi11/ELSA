import numpy
import torch
import torch.nn as nn
from .SRAM import SRAM
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
# convert spike vector to spike packets

class spikeEncoder(VSAModule):
    def __init__(self):
        super(spikeEncoder,self).__init__()
        self.fEnergy = cfg["processElement"]["spikeEncoder"]["fEnergy"]
        self.fCount = 0 

    def forward(self,spikes,column_id,outspikes):
        
        for i,spike in enumerate(spikes):
            if spike == 0:
                continue
            elif spike == 1:
                outspikes.append((i,column_id,0))
                self.fCount = self.fCount + 1
            elif spike == -1:
                outspikes.append((i,column_id,1))
                self.fCount = self.fCount + 1
        return outspikes
    
def test_spikeEncoder():
    encoder = spikeEncoder()
    width = 128
    spikes = (torch.rand(width)*4-2).int()
    columnId = 12
    outspikes = []
    
    outspikes = encoder(spikes,columnId,outspikes)
    print(spikes)
    print(outspikes)
    print(encoder.fCount)
    


