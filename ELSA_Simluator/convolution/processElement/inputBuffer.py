import numpy
import torch
import torch.nn as nn
from .SRAM import SRAM
import queue
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


class inputBuffer(VSAModule):
    def __init__(self, queueDepth, parallelism=1, first=True):
        # careful!!!! The K is tiled by PE number P
        super(inputBuffer,self).__init__()
        self.queueDepth = queueDepth
        self.parallelism = parallelism
        self.queueNum = cfg["processElement"]["inputBuffer"]["queueNum"]
        self.readEnergy = cfg["processElement"]["inputBuffer"]["readEnergy"]
        self.writeEnergy = cfg["processElement"]["inputBuffer"]["writeEnergy"]

        self.busWidth = cfg["processElement"]["inputBuffer"]["queueDepth"]
        self.inoutWidth = cfg["processElement"]["inputBuffer"]["inoutWidth"]
        self.mux = cfg["processElement"]["inputBuffer"]["mux"]
        self.temporature = cfg["processElement"]["inputBuffer"]["temporature"]
        self.voltage = cfg["processElement"]["inputBuffer"]["voltage"]
        self.periphery_vt = cfg["processElement"]["inputBuffer"]["periphery_vt"]

        self.spikeQueues = []
        self.columnQueue = queue.Queue(self.queueNum)
        self.CIdtoQdTLB = {}
        self.QIdtoCdTLB = {}
        self.curFreeQueueId = 0
        self.lastBlockId = -1
        self.curQueueId = 0
        self.curOutQueueId = 0
        self.readCount = 0
        self.writeCount = 0
        self.first = first
        for i in range(self.queueNum):
            q = queue.Queue(self.queueDepth)
            self.spikeQueues.append(q)

        self.srams = []
        for i in range(self.queueNum):
            self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt,belong="inputBuffer"))
        self.area = len(self.srams)*self.srams[0].area

    def calEnergy(self,latency):
        totalEnergy = 0.0
        for sram in self.srams:
            totalEnergy = totalEnergy + sram.calEnergy(latency)
        return totalEnergy
    
    def get_data(self, data, weightBlockWidth, membraneWordNum):
        # data format (row_id, column_id, sign)
        assert isinstance(data,tuple), "inputBuffer, get_data: the data format must be tuple!!!!"
        # for i,queue in enumerate(self.spikeQueues):
        #     print(f"queueId:{i},queueLen:{queue.qsize()}")
        self.writeCount = self.writeCount + 1
        blockId = data[0]//weightBlockWidth
        rowId = data[0]%weightBlockWidth
        if data[1] not in self.CIdtoQdTLB.keys() and not self.columnQueue.full(): # 如果TLB没有查到这一列的spike所分配的queue id，并且queue还没有全部被分配
            self.CIdtoQdTLB[data[1]] = self.curFreeQueueId # 分配queue
            self.QIdtoCdTLB[self.curFreeQueueId] = data[1] # 分配queue
            self.columnQueue.put(self.CIdtoQdTLB[data[1]])
            self.curFreeQueueId = self.curFreeQueueId + 1
        elif data[1] not in self.CIdtoQdTLB.keys() and self.columnQueue.full(): # 如果TLB没有查到这一列的spike所分配的queue id，并且queue还全部被分配
            queueIdtoflash = self.columnQueue.get(block=False)
            delColumnId = self.QIdtoCdTLB[queueIdtoflash]
            del self.QIdtoCdTLB[queueIdtoflash]
            del self.CIdtoQdTLB[delColumnId]
            self.curFreeQueueId = queueIdtoflash
            return False, queueIdtoflash # 返回插入失败，强制输出某一个queue里的所有值

        self.curQueueId = self.CIdtoQdTLB[data[1]]
        
        if self.spikeQueues[self.curQueueId].qsize() >= self.parallelism:
            return False, self.curQueueId
        # print("self.CIdtoQdTLB",self.CIdtoQdTLB,"self.columnQueue.qsize()",self.columnQueue.qsize(),"self.columnQueue",list(self.columnQueue.queue))
        # print(f"data[1]:{data[1]}, self.curQueueId:{self.curQueueId}")
        self.spikeQueues[self.curQueueId].put((blockId,rowId,data[1],data[2]), block=False)
        self.srams[self.curQueueId].writeCount = self.srams[self.curQueueId].writeCount + 1
        return True, 0
                        
        # return data format (block_id, row_id, column_id, sign)
    
    def output_data(self,queueId):
        outputData = []
        # print("len(self.spikeQueues[queueId])",self.spikeQueues[queueId].qsize())
        for i in range(self.parallelism):
            if self.spikeQueues[queueId].empty():
                # print("output all!!!!")
                return outputData,False,True
            if len(outputData) > 0 and outputData[-1][2] != self.spikeQueues[queueId].queue[0][2]:
                return outputData,False,True
            outputData.append(self.spikeQueues[queueId].get(block=False))
            self.srams[queueId].readCount = self.srams[queueId].readCount + 1
            self.readCount = self.readCount + 1
        return outputData,False,True
    
    def output_data_all(self):
        outputData = []
        while(1):
            if self.spikeQueues[self.curOutQueueId].empty():
                self.curOutQueueId = self.curOutQueueId + 1
                if self.curOutQueueId == len(self.columnRecorder):
                    return outputData,True,True
            else:
                break
        for i in range(self.parallelism):
            if self.spikeQueues[self.curOutQueueId].empty():
                return outputData,False,True
            if len(outputData) > 0 and outputData[-1][2] != self.spikeQueues[self.curOutQueueId].queue[0][2]:
                return outputData,False,True
            outputData.append(self.spikeQueues[self.curOutQueueId].get(block=False))
            self.srams[self.curOutQueueId].readCount = self.srams[self.curOutQueueId].readCount + 1
            self.readCount = self.readCount + 1
        return outputData,False,False
                



def generateRandomSpikes(maxRow, colId):
    rowId = int(torch.rand(1)*maxRow)
    sign = int(torch.rand(1)*2)
    return(rowId, colId, sign)
    

def test_inputBuffer():
    queueDepth = 128
    parallelism = 4
    inputbuffer = inputBuffer(queueDepth=queueDepth, parallelism=parallelism)
    weightBlockWidth = 512
    spikeCount = 16
    correct = []
    while(1):
        spike = generateRandomSpikes(25088,10)
        isSuccess = inputbuffer.get_data(spike, weightBlockWidth)
        if isSuccess:
            spikeCount = spikeCount - 1
        else:
            break
        if spikeCount == 0:
            break
        print((spike[0]//weightBlockWidth, spike[0]%weightBlockWidth,spike[1],spike[2]))
    
    while(1):
        outputSpike,is_empty = inputbuffer.output_data()
        if is_empty:
            break
        print(outputSpike)


