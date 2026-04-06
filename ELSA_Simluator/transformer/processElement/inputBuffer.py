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
        self.columnRecorder = []
        self.lastBlockId = -1
        self.curQueueId = 0
        self.curOutQueueId = 0
        self.readCount = 0
        self.writeCount = 0
        self.first = first
        if self.first:
            for i in range(self.parallelism):
                q = queue.Queue(self.queueDepth)
                self.spikeQueues.append(q)
        else:
            for i in range(self.queueNum):
                q = queue.Queue(self.queueDepth)
                self.spikeQueues.append(q)

        self.srams = []
        if self.first:
            for i in range(self.parallelism):
                self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt,belong="inputBuffer"))
        else:
            for i in range(self.queueNum):
                self.srams.append(SRAM(busWidth=self.busWidth,inoutWidth=self.inoutWidth, mux=self.mux,temporature=self.temporature, voltage=self.voltage, periphery_vt=self.periphery_vt,belong="inputBuffer"))

        self.area = len(self.srams)*self.srams[0].area

    def calEnergy(self,latency):
        totalEnergy = 0.0
        for sram in self.srams:
            totalEnergy = totalEnergy + sram.calEnergy(latency)
        return totalEnergy
    
    def get_data(self, data, weightBlockWidth, right=True, membraneWordNum=0):
        # data format (row_id, column_id, sign)
        assert isinstance(data,tuple), "inputBuffer, get_data: the data format must be tuple!!!!"
        if self.first:
            self.writeCount = self.writeCount + 1
            blockId = data[0]//weightBlockWidth
            rowId = data[0]%weightBlockWidth

            # columnBlockId = data[1]//membraneWordNum
            # colId = data[1]%membraneWordNum
            if self.spikeQueues[self.curQueueId].full():
                return False
            self.spikeQueues[self.curQueueId].put((blockId,rowId,data[1],data[2]), block=False)
            self.srams[self.curQueueId].writeCount = self.srams[self.curQueueId].writeCount + 1
            self.curQueueId = (self.curQueueId+1)%self.parallelism
            return True
        else:
            self.writeCount = self.writeCount + 1
            if right:
                if data[0] not in self.columnRecorder:
                    self.columnRecorder.append(data[0])
                self.curQueueId = self.columnRecorder.index(data[0])
            else:
                if data[1] not in self.columnRecorder:
                    self.columnRecorder.append(data[1])
                self.curQueueId = self.columnRecorder.index(data[1])
            
            # print("input buffer, data:",data,"self.columnRecorder",self.columnRecorder,"columnBlockId","self.curQueueId",self.curQueueId,"right",right)
            
            if self.spikeQueues[self.curQueueId].full():
                return False
            self.spikeQueues[self.curQueueId].put((data[0],data[1],data[2]), block=False)
            self.srams[self.curQueueId].writeCount = self.srams[self.curQueueId].writeCount + 1
            return True
                        
        # return data format (block_id, row_id, column_id, sign)
    
    def is_empty(self):
        empty = True
        for i,q in enumerate(self.spikeQueues):
            empty = empty and q.empty()
        return empty
            
    
    def output_data(self):
        outputData = []
        if self.first:
            self.readCount = self.readCount + self.parallelism
            is_empty = True 
            for i, q in enumerate(self.spikeQueues):
                is_empty = is_empty and q.empty()
                if q.empty():
                    continue
                outputData.append(q.get(block=False))
                self.srams[i].readCount = self.srams[i].readCount + 1
            return outputData,is_empty,False
        else:
            # for i,q in enumerate(self.spikeQueues):
            #     print("id=",i,",size=",q.qsize())
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
                # if len(outputData) > 0 and outputData[-1][2] != self.spikeQueues[self.curOutQueueId].queue[0][2]:
                #     return outputData,False,True
                outputData.append(self.spikeQueues[self.curOutQueueId].get(block=False))
                self.srams[self.curOutQueueId].readCount = self.srams[self.curOutQueueId].readCount + 1
                self.readCount = self.readCount + 1

            if self.spikeQueues[self.curOutQueueId].empty():
                return outputData,False,True
            else:
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


