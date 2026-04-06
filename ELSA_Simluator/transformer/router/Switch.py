from basicModule import VSAModule

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class SwitchCrossbar(VSAModule):
    def __init__(self):
        super(SwitchCrossbar,self).__init__()
        self.fEnergy = cfg["router"]["switchCrossbar"]["fEnergy"]
        self.area = cfg["router"]["switchCrossbar"]["area"]
        self.staticPower = cfg["router"]["switchCrossbar"]["leakage"]    
    
    def forward(self,x):
        return x
