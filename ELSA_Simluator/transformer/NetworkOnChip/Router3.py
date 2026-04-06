import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
import math
import mapping.partition as partition
import random
import sys
sys.path.append("./mapping")


cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


# ResNet18_Connection = {}
# ResNet18_mapping = [[],]

class NOC(VSAModule):
    def __init__(self, connection_path="", mapping_path="", tranOccupy_path="", first=False):
        super(NOC,self).__init__()
        self.meshHeight = cfg["NOC"]["meshHeight"]
        self.meshWidth = cfg["NOC"]["meshWidth"]
        self.nodeNum = self.meshHeight * self.meshWidth
        self.mapping = torch.load(mapping_path) # 在2D mesh上的mapping情况，里面是Node类型，以及每个router的占用情况
        self.connection = torch.load(connection_path) # connection是一个邻接表，用字典存储，存储了layer之间的连接情况
        self.tranOccupy = torch.load(tranOccupy_path) # tranOccupy是一个邻接表，用字典存储，存储了NoC link的占用情况
        self.NoCDatawidth = cfg["NOC"]["NoCLinkDataWidth"]
        self.flitSize = cfg["NOC"]["flitSize"]
        self.RouterCycle = cfg["router"]["flitCycle"]
        self.NoCCycle = self.flitSize/self.NoCDatawidth
        self.flitNumber = 0
        self.first = first
        print(self.connection)
        # for y in range(self.meshHeight):
        #     for x in range(self.meshWidth):
        #         print(f"({y},{x})",self.mapping[y][x])        
        
        self.tokenSpineOutputTime = {} # 每一个output spine的输出time
        self.transmitTraffic = [[0 for i in range(self.nodeNum)] for i in range(self.nodeNum)] # NoC Link上的传输量，二维矩阵
    
    def set_mapping(self, mapping): #设置网络每一层之间的mapping
        self.mapping = mapping

    def getMapLayerNum(self, curLayerId):
        if self.first: #第一次的一对一mapping
            return 1
        else:
            for y in range(self.meshHeight):
                for x in range(self.meshWidth):
                    if curLayerId in self.mapping[y][x].layerIdSet:
                        return len(self.mapping[y][x].layerIdSet)
            print(f"The layer {curLayerId} is not mapped to VSA!!!!")
            return 1
        # exit(0)
    
    def updateLastLayersTime(self, curLayerId, spineTokenTime): # 记录上一个layer在chip中的每一个token的扇出时间，这个要在forward之前运行一下，把输入载入
        # print(spineTokenTime)
        self.tokenSpineOutputTime[curLayerId] = spineTokenTime
    
    def forward(self, curLayerId, spineTokenId, timestep, flitNumber): # 通过NoC确定token真正的输入时间，考虑数据依赖以及NoC上的传输时间
        self.flitNumber = self.flitNumber + flitNumber
        SpineInputTime = 0 # 考虑过数据依赖以及NoC的输入时间
        lastLayersId = self.connection[curLayerId] # 先寻找连接到该层的前面网络层
        # if curLayerId.count("layer2.0.downsample.0") > 0: 
        #     print("curLayerId",curLayerId,"lastLayersId",lastLayersId)
        if self.first: #第一次的一对一mapping
            for lastLayerId in lastLayersId:
                curSpineInputTime = self.tokenSpineOutputTime[lastLayerId][timestep][spineTokenId] + (self.RouterCycle + self.NoCCycle) * flitNumber
                SpineInputTime = max(SpineInputTime, curSpineInputTime)
            return SpineInputTime
        for lastLayerId in lastLayersId: # 查询每一层网络层
            # spineInputTime = 0 # 总的spine的输入时间，每所有当前某个Spine输入时间的最大值
            curSpineInputTime = 0 #当前某个Spine的输入时间
            # 找到对应的2D mesh中的node
            x1 = 0; y1 = 0
            x2 = 0; y2 = 0
            SNode = None
            TNode = None
            for y in range(self.meshHeight):
                for x in range(self.meshWidth):
                    if lastLayerId in self.mapping[y][x].layerIdSet: # 如果找到了起始结点
                        x1 = x; y1 = y; SNode = self.mapping[y][x]
                    if curLayerId in self.mapping[y][x].layerIdSet: # 如果找到了终止结点
                        x2 = x; y2 = y; TNode = self.mapping[y][x]
            # 如果两个结点相同，说明是自己传输到自己：
            if SNode.Id == TNode.Id: 
                # 查询输入时间
                curSpineInputTime = self.tokenSpineOutputTime[lastLayerId][timestep][spineTokenId]
                # 只考虑router上的时间，加上router的时间：
                curSpineInputTime = curSpineInputTime + self.RouterCycle * SNode.occupy * flitNumber
                # if curLayerId.count("layer2.0.downsample.0") > 0: 
                #     print("lastLayersId",lastLayersId,"self.RouterCycle",self.RouterCycle,"SNode.occupy",SNode.occupy,"flitNumber",flitNumber,"curSpineInputTime",curSpineInputTime)
                pos = y1 * self.meshWidth + x1
                self.transmitTraffic[pos][pos] = self.transmitTraffic[pos][pos] + flitNumber * self.flitSize
                # print("0", f"(x_1, y_1)=({x1},{y1}), (x_2, y_2)=({x2},{y2})", "pos",pos, "SNode.occupy",SNode.occupy)
            else: # 如果两个结点不相同，那么就按照概率进行传输，每个flit按照特定的概率传输，选择时间最长的那个flit
                maxNocLinkTranTime = 0
                for flitId in range(flitNumber):
                    x_now = x1 # 当前位置
                    y_now = y1 
                    dx = (1 if x2 - x1 >= 0 else -1)# flit的传输方向
                    dy = (1 if y2 - y1 >= 0 else -1) 
                    curNocLinkTranTime = (1 + flitId) * self.RouterCycle * SNode.occupy # 当前flit传输需要的时间，以及包括产生这个flit的时间
                    
                    while(1): # 一直走直到终点
                        if x_now == x2 and y_now == y2: # 到达终点
                            maxNocLinkTranTime = max(maxNocLinkTranTime, curNocLinkTranTime) # 更新flit时间
                            break
                        elif x_now == x2 and (not y_now == y2): # 需要在y轴方向传输
                            curNode = self.mapping[y_now][x_now]
                            pos1 = y_now * self.meshWidth + x_now
                            pos2 = (y_now+dy) * self.meshWidth + x_now
                            # NoC link time + Router time
                            curNocLinkTranTime = curNocLinkTranTime + math.ceil(self.tranOccupy[pos1][pos2] * self.NoCCycle) + self.RouterCycle * curNode.occupy
                            self.transmitTraffic[pos1][pos2] = self.transmitTraffic[pos1][pos2] + self.flitSize
                            # print("1. curNocLinkTranTime",curNocLinkTranTime, f"(x_now, y_now)=({x_now},{y_now}), (x_1, y_1)=({x1},{y1}), (x_2, y_2)=({x2},{y2})", "self.tranOccupy[pos1][pos2]",self.tranOccupy[pos1][pos2],"curNode.occupy",curNode.occupy)
                            y_now = y_now + dy
                        elif (not x_now == x2) and y_now == y2: # 需要在x轴方向传输
                            curNode = self.mapping[y_now][x_now]
                            pos1 = y_now * self.meshWidth + x_now
                            pos2 = y_now * self.meshWidth + x_now + dx
                            # NoC link time + Router time
                            curNocLinkTranTime = curNocLinkTranTime + math.ceil(self.tranOccupy[pos1][pos2] * self.NoCCycle) + self.RouterCycle * curNode.occupy
                            self.transmitTraffic[pos1][pos2] = self.transmitTraffic[pos1][pos2] + self.flitSize
                            # print("2. curNocLinkTranTime",curNocLinkTranTime, f"(x_now, y_now)=({x_now},{y_now}), (x_1, y_1)=({x1},{y1}), (x_2, y_2)=({x2},{y2})", "self.tranOccupy[pos1][pos2]",self.tranOccupy[pos1][pos2],"curNode.occupy",curNode.occupy)
                            x_now = x_now + dx
                        else: # 只往x方向走：
                            curNode = self.mapping[y_now][x_now]
                            pos1 = y_now * self.meshWidth + x_now
                            pos2 = y_now * self.meshWidth + x_now + dx
                            # NoC link time + Router time
                            curNocLinkTranTime = curNocLinkTranTime + math.ceil(self.tranOccupy[pos1][pos2] * self.NoCCycle) + self.RouterCycle * curNode.occupy
                            self.transmitTraffic[pos1][pos2] = self.transmitTraffic[pos1][pos2] + self.flitSize
                            # print("4 curNocLinkTranTime",curNocLinkTranTime, f"(x_now, y_now)=({x_now},{y_now}), (x_1, y_1)=({x1},{y1}), (x_2, y_2)=({x2},{y2})", "self.tranOccupy[pos1][pos2]",self.tranOccupy[pos1][pos2],"curNode.occupy",curNode.occupy)
                            x_now = x_now + dx
                
                    # 得到所有flit中的最长运输时间
                    maxNocLinkTranTime = max(maxNocLinkTranTime, curNocLinkTranTime)        
                    
                    # 得到输入的时间
                    curSpineInputTime = self.tokenSpineOutputTime[lastLayerId][timestep][spineTokenId]
                        
                    print("curSpineInputTime",curSpineInputTime,"maxNocLinkTranTime",maxNocLinkTranTime)
                    # 得到最终的时间
                    curSpineInputTime = maxNocLinkTranTime + curSpineInputTime
                    # if curLayerId.count("layer2.0.downsample.0") > 0: 
                    #     print("self.RouterCycle",self.RouterCycle,"SNode.occupy",SNode.occupy,"flitNumber",flitNumber,"curSpineInputTime",curSpineInputTime,"maxNocLinkTranTime",maxNocLinkTranTime)

            
            # 得到所有层中最长的传输时间：
            # if curLayerId.count("layer2.0.downsample.0") > 0: 
            #     print("line160:",SpineInputTime,curSpineInputTime)
            SpineInputTime = max(SpineInputTime, curSpineInputTime)
                    
        return SpineInputTime



        
        
        
        
        
        
        
        
