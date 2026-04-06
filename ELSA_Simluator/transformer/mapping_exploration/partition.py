import torch
import heapq
import math

# 先构建node和Edge

class Node():
    def __init__(self, layerIdSet, SRAMNumber, AdderNumber):
        self.layerIdSet = layerIdSet
        self.Id = str(layerIdSet)
        self.requireSRAM = SRAMNumber
        self.requireAdder = AdderNumber
        self.occupy = 0 # 该node router的占用情况
        self.YAllocate = 0.5 #该Node往四个port的分配情况
        self.XAllocate = 0.5 #该Node往四个port的分配情况
        self.oneAdderCycle = 0

    def __str__(self):
        return f"Node(layerId={self.Id}, requireSRAM={self.requireSRAM}, requireAdder={self.requireAdder}, occupy={self.occupy})"


class Edge():
    def __init__(self, SNode, TNode, FlitNumber, avgFlitNumSpine):
        self.SNode = SNode
        self.TNode = TNode
        self.FlitNumber = FlitNumber
        self.avgFlitNumSpine = avgFlitNumSpine
    
    def __lt__(self, other):
        return self.FlitNumber > other.FlitNumber    

    def __str__(self):
        return f"Edge(SNode={self.SNode}, TNode={self.TNode}, FlitNumber={self.FlitNumber}, avgFlitNumSpine={self.avgFlitNumSpine})"

if __name__ == "__main__":

    searchInput = torch.load("datasheets/InputSearch.pth")
    ConnectionNext = torch.load("datasheets/ViT_Connection_Next.pth")
    T = 32

    # 构建每一层的输出Flit数量，需要的SRAM数量以及需要的加法器数量

    inputFlitPerLayer = {}
    inputFlitPerSpine = {}

    for layerId in ConnectionNext.keys():
        if layerId == "transmitTraffic":
            continue
        # if layerId == "model.module.fc\n":
        #     pass
        # else:
        inputFlitPerLayer[layerId] = searchInput["InputFlitNum"][layerId] + 0.0
        inputFlitPerSpine[layerId] = searchInput["avgFlitNumEachToken"][layerId]/T + 0.0
        # print("layerId",layerId, "avgFlitNumEachLayer",searchInput["avgFlitNumEachToken"][layerId]/T,"outputFlitNum",searchInput["OutputFlitNum"][layerId])
    
    d = {}
    layerNames = []
    for key in ConnectionNext.keys():
        # if key == "model.module.fc\n":
        #     pass
        # else:
        if key.count("addition") > 0:
            d[key] = 0
        else:
            d[key] = searchInput["SRAMNumPerLayer"][key]
        layerNames.append(key)
    D = searchInput["totalSRAMNum"]
    a = {}
    for key in ConnectionNext.keys():
        # if key == "model.module.fc\n":
        #     pass
        # else:
        if key.count("addition") > 0:
            a[key] = 0
        else:
            a[key] = searchInput["WidthAfterTiling"][key]
    A = 240

    # print(len(list(a.keys())))
    # print(len(list(d.keys())))
    # print(len(list(outputFlitPerLayer.keys())))
    # print("totalSRAMNum",searchInput["totalSRAMNum"])

    # 构建连接对，连接对的值是传输的flit

    EdgeList = []
    NodeList = {}

    # 构建点和边集合
    for layerId in layerNames:
        if layerId.count("qkvq") > 0:
            node = Node(set([layerId,layerId[:-4]+"qkvk"]), d[layerId] + d[layerId[:-4]+"qkvk"], a[layerId] + a[layerId[:-4]+"qkvk"])        
            node.oneAdderCycle = math.ceil((a[layerId] + a[layerId[:-4]+"qkvk"])/A)
            NodeList[node.Id] = node
            continue
        if layerId.count("qkvk") > 0:
            continue
        node = Node(set([layerId]), d[layerId], a[layerId])
        node.oneAdderCycle = math.ceil(a[layerId]/A)
        NodeList[node.Id] = node

    for SlayerId in layerNames:
        for TlayerId in ConnectionNext[SlayerId]:
            # if TlayerId == "model.module.avgpool\n" or SlayerId == "model.module.avgpool\n":
            #     continue
            FlitNumber = inputFlitPerLayer[TlayerId]
            AvgFlitNumSpine = inputFlitPerSpine[TlayerId]
            if SlayerId.count("qkvq") > 0 or SlayerId.count("qkvk") > 0:
                EdgeList.append(Edge(NodeList[str(set([SlayerId[:-4]+"qkvq", SlayerId[:-4]+"qkvk"]))], NodeList[str(set([TlayerId]))], FlitNumber, AvgFlitNumSpine))
            elif TlayerId.count("qkvq") > 0 or TlayerId.count("qkvk") > 0:
                EdgeList.append(Edge(NodeList[str(set([SlayerId]))], NodeList[str(set([TlayerId[:-4]+"qkvq", TlayerId[:-4]+"qkvk"]))], FlitNumber, AvgFlitNumSpine))
            else:
                if TlayerId.count("addition") > 0:
                    AvgFlitNumSpine = AvgFlitNumSpine / 2
                EdgeList.append(Edge(NodeList[str(set([SlayerId]))], NodeList[str(set([TlayerId]))], FlitNumber, AvgFlitNumSpine))
            print(EdgeList[-1])

    # 将边集合按照FlitNumber构造优先队列：
    # heapq.heapify(EdgeList)
    # for edge in EdgeList:
    #     print(edge)
    
    # 人为加一个条件，想要qkvq和qkvk以及fc1和fc2合并：
    
    for maxFlitEdge in EdgeList:
        SNode = maxFlitEdge.SNode
        TNode = maxFlitEdge.TNode
        FlitNumber = maxFlitEdge.FlitNumber
        SRequiredSRAM = SNode.requireSRAM
        SRequiredAdder = SNode.requireAdder
        TRequiredSRAM = TNode.requireSRAM
        TRequiredAdder = TNode.requireAdder
        if SNode.Id.count("fc1") > 0 and TNode.Id.count("fc2") > 0:
            # 先构造新的Node
            newNode = Node(SNode.layerIdSet | TNode.layerIdSet, SRequiredSRAM + TRequiredSRAM, SRequiredAdder + TRequiredAdder)
            newNode.oneAdderCycle = math.ceil((SRequiredAdder + TRequiredAdder)/A)
            # print(newNode)
            # print("==========================================")
            # 更新边集
            for edge in EdgeList:
                if edge.SNode == SNode or edge.SNode == TNode:
                    edge.SNode = newNode
                if edge.TNode == SNode or edge.TNode == TNode:
                    edge.TNode = newNode
            # 更新点集，先删除原来的点，再添加新的点
            del NodeList[SNode.Id]
            del NodeList[TNode.Id]
            NodeList[newNode.Id] = newNode

    # 选择最大的边进行合并：
    for maxFlitEdge in EdgeList:
        SNode = maxFlitEdge.SNode
        TNode = maxFlitEdge.TNode
        FlitNumber = maxFlitEdge.FlitNumber
        SRequiredSRAM = SNode.requireSRAM
        SRequiredAdder = SNode.requireAdder
        TRequiredSRAM = TNode.requireSRAM
        TRequiredAdder = TNode.requireAdder
        # print(maxFlitEdge,"SRequiredSRAM",SRequiredSRAM,"TRequiredSRAM",TRequiredSRAM,"D",D,"SRequiredAdder",SRequiredAdder,"TRequiredAdder",TRequiredAdder,"A",A)
        if SRequiredSRAM + TRequiredSRAM <= D and SRequiredAdder + TRequiredAdder <= A: # 如果满足约束条件，那么合并
            # 先构造新的Node
            if "addition1" in SNode.Id and "addition2" in TNode.Id or "addition2" in SNode.Id and "addition1" in TNode.Id:
                continue
            
            newNode = Node(SNode.layerIdSet | TNode.layerIdSet, SRequiredSRAM + TRequiredSRAM, SRequiredAdder + TRequiredAdder)
            newNode.oneAdderCycle = math.ceil((SRequiredAdder + TRequiredAdder)/A)
            # print(newNode)
            # print("==========================================")
            # 更新边集
            for edge in EdgeList:
                if edge.SNode == SNode or edge.SNode == TNode:
                    edge.SNode = newNode
                if edge.TNode == SNode or edge.TNode == TNode:
                    edge.TNode = newNode
            # 更新点集，先删除原来的点，再添加新的点
            del NodeList[SNode.Id]
            del NodeList[TNode.Id]
            NodeList[newNode.Id] = newNode

    SNode = NodeList[list(NodeList.keys())[0]]
    TNode = NodeList[list(NodeList.keys())[-1]]
    newNode = Node(SNode.layerIdSet | TNode.layerIdSet, SRequiredSRAM + TRequiredSRAM, SRequiredAdder + TRequiredAdder)
    newNode.oneAdderCycle = math.ceil((SRequiredAdder + TRequiredAdder)/A)
    for edge in EdgeList:
        if edge.SNode == SNode or edge.SNode == TNode:
            edge.SNode = newNode
        if edge.TNode == SNode or edge.TNode == TNode:
            edge.TNode = newNode
    # 更新点集，先删除原来的点，再添加新的点
    del NodeList[SNode.Id]
    del NodeList[TNode.Id]
    NodeList[newNode.Id] = newNode    
    # 合并Q,K,V
    
    # 构建partition
    partition = [] 
    for key in NodeList.keys():
        partition.append(NodeList[key])

    # 构建partition之后的EdgeList
    EdgeList_partition = []
    # 建立visit矩阵避免重复线
    Vis = [[0 for i in range(len(partition))] for j in range(len(partition))]
    for SlayerId in layerNames:
        for TlayerId in ConnectionNext[SlayerId]:
            # if TlayerId == "model.module.avgpool\n" or SlayerId == "model.module.avgpool\n":
            #     continue
            # 寻找SNode和TNode
            SNode = None
            TNode = None
            for node in partition:
                if SlayerId in node.layerIdSet:
                    SNode = node
                if TlayerId in node.layerIdSet:
                    TNode = node
            if Vis[partition.index(SNode)][partition.index(TNode)] == 1: #如果已经被访问过，那么跳过
                for edge in EdgeList_partition: # 更新FlitNumber
                    if edge.SNode == SNode and edge.TNode == TNode:
                        # print(edge)
                        edge.FlitNumber = edge.FlitNumber + inputFlitPerLayer[TlayerId]
                        edge.avgFlitNumSpine = edge.avgFlitNumSpine + inputFlitPerSpine[TlayerId]
                continue
            Vis[partition.index(SNode)][partition.index(TNode)] = 1
            FlitNumber = inputFlitPerLayer[TlayerId]
            AvgFlitNumSpine = inputFlitPerSpine[TlayerId]
            EdgeList_partition.append(Edge(SNode, TNode, FlitNumber, AvgFlitNumSpine))

    # print(EdgeList_partition)
    torch.save(partition, "partition.pth")
    torch.save(EdgeList_partition, "EdgeList.pth")
    # print(f"==============================Node After Partition NodeNum={len(partition)}==============================")
    for node in partition:
        print(node)
    
    # for edge in EdgeList_partition:
    #     print(f"{edge.SNode.Id} ----> {edge.TNode.Id}, {edge.avgFlitNumSpine}")
