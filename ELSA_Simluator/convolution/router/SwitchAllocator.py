from basicModule import VSAModule

import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

class SwitchAllocator(VSAModule):
    def __init__(self):
        super(SwitchAllocator,self).__init__()
    
    def forward(self,x):
        return x
