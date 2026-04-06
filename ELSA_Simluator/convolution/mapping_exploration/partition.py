import torch
import heapq

# 先构建node和Edge

class Node():
    def __init__(self, layerIdSet, SRAMNumber, AdderNumber):
        self.layerIdSet = layerIdSet
        self.Id = str(layerIdSet)
        self.requireSRAM = SRAMNumber
        self.requireAdder = AdderNumber
        self.occupy = 0 # 该node router的占用情况
        self.Allocates = [0.25,0.25,0.25,0.25] #该Node往四个port的分配情况

    def __str__(self):
        return f"Node(layerId={self.Id}, requireSRAM={self.requireSRAM}, requireAdder={self.requireAdder}, occupy={self.occupy}, self.Allocates={self.Allocates})"


class Edge():
    def __init__(self, SNode, TNode, FlitNumber, avgFlitNumSpine):
        self.SNode = SNode
        self.TNode = TNode
        self.FlitNumber = FlitNumber
        self.avgFlitNumSpine = avgFlitNumSpine
        self.transmitTime = 0
    
    def __lt__(self, other):
        return self.FlitNumber > other.FlitNumber    

    def __str__(self):
        return f"Edge(SNode={self.SNode}, TNode={self.TNode}, FlitNumber={self.FlitNumber}, avgFlitNumSpine={self.avgFlitNumSpine}, self.transmitTime={self.transmitTime})"

if __name__ == "__main__":

    searchInput = torch.load("datasheets/SearchInput_ResNet34_a4w4_with_Noc.pth")
    info = torch.load("../outputs/calculateInfoConv_ResNet34_a4w4_with_Noc_VSA_AER.pth")
    ConnectionNext = torch.load("../mapping/ResNet34_Connection_Next.pth")
    T = len(info[list(info.keys())[0]]["perTimeStepCycle"])

    # 构建每一层的输出Flit数量，需要的SRAM数量以及需要的加法器数量

    inputFlitPerLayer = {}
    inputFlitPerSpine = {}

    # print(list(info.keys()))
    for layerId in info.keys():
        if layerId == "transmitTraffic":
            continue
        # if layerId == "model.module.fc\n":
        #     pass
        # else:
        if layerId == "model.module.conv1\n":
            inputFlitPerLayer["model.module.maxpool\n"] = info[layerId]["InputFlitNum"]
            inputFlitPerSpine["model.module.maxpool\n"] = searchInput["avgFlitNumEachLayer"]["model.module.maxpool\n"]/T
        if layerId == "model.module.layer4.1.conv2\n":
            inputFlitPerLayer["model.module.avgpool\n"] = info[layerId]["InputFlitNum"]
            inputFlitPerSpine["model.module.avgpool\n"] = info[layerId]["InputFlitNum"]/T
        inputFlitPerLayer[layerId] = info[layerId]["InputFlitNum"]
        if layerId == "model.module.fc\n":
            inputFlitPerSpine[layerId] = info[layerId]["InputFlitNum"]/T
        else:
            inputFlitPerSpine[layerId] = searchInput["avgFlitNumEachLayer"][layerId]/T
        # print("layerId",layerId, "avgFlitNumEachLayer",searchInput["avgFlitNumEachLayer"][layerId]/T,"outputFlitNum",info[layerId]["outputFlitNum"])

    d = {}
    layerNames = []
    for key in ConnectionNext.keys():
        # if key == "model.module.fc\n":
        #     continue
        # else:
        if key == "model.module.maxpool\n" or key == "model.module.avgpool\n":
            d[key] = 0
        else:
            d[key] = searchInput["SRAMNumPerLayer"][key]
        layerNames.append(key)
    D = searchInput["totalSRAMNum"]
    a = {}
    for key in ConnectionNext.keys():
        # if key == "model.module.fc\n":
        #     continue
        # else:
        if key == "model.module.maxpool\n" or key == "model.module.avgpool\n":
            a[key] = 0
        else:
            a[key] = searchInput["WidthAfterTiling"][key]
    A = 128

    # print(len(list(a.keys())))
    # print(len(list(d.keys())))
    # print(len(list(inputFlitPerLayer.keys())))

    # 构建连接对，连接对的值是传输的flit

    EdgeList = []
    NodeList = {}

    # 构建点和边集合
    for layerId in layerNames:
        node = Node(set([layerId]), d[layerId], a[layerId])
        NodeList[node.Id] = node
        # print(node)

    for SlayerId in layerNames:
        for TlayerId in ConnectionNext[SlayerId]:
            if TlayerId == "model.module.fc\n" or SlayerId == "model.module.fc\n":
                continue
            FlitNumber = inputFlitPerLayer[TlayerId]
            AvgFlitNumSpine = inputFlitPerSpine[TlayerId]
            EdgeList.append(Edge(NodeList[str(set([SlayerId]))], NodeList[str(set([TlayerId]))], FlitNumber, AvgFlitNumSpine))
    
    # for edge in EdgeList:
    #     print("=================================================")
    #     print(edge)
    # 将边集合按照FlitNumber构造优先队列：
    heapq.heapify(EdgeList)

    # 选择最大的边进行合并：
    while(len(EdgeList) > 0):
        maxFlitEdge = heapq.heappop(EdgeList)
        SNode = maxFlitEdge.SNode
        TNode = maxFlitEdge.TNode
        FlitNumber = maxFlitEdge.FlitNumber
        SRequiredSRAM = SNode.requireSRAM
        SRequiredAdder = SNode.requireAdder
        TRequiredSRAM = TNode.requireSRAM
        TRequiredAdder = TNode.requireAdder
        if SRequiredSRAM + TRequiredSRAM <= D and SRequiredAdder + TRequiredAdder <= A: # 如果满足约束条件，那么合并
            # 先构造新的Node
            newNode = Node(SNode.layerIdSet | TNode.layerIdSet, SRequiredSRAM + TRequiredSRAM, SRequiredAdder + TRequiredAdder)
            # print(maxFlitEdge)
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
            if TlayerId == "model.module.fc\n" or SlayerId == "model.module.fc\n":
                continue
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
                        edge.FlitNumber = edge.FlitNumber + inputFlitPerLayer[TlayerId]
                        edge.avgFlitNumSpine = edge.avgFlitNumSpine + inputFlitPerSpine[TlayerId]
                        # if edge.SNode.Id.count("maxpool") > 0 and edge.TNode.Id.count("layer1.0.conv1") > 0:
                        #     print(edge, str(SlayerId), str(TlayerId))
                continue
            Vis[partition.index(SNode)][partition.index(TNode)] = 1
            FlitNumber = inputFlitPerLayer[TlayerId]
            AvgFlitNumSpine = inputFlitPerSpine[TlayerId]
            EdgeList_partition.append(Edge(SNode, TNode, FlitNumber, AvgFlitNumSpine))
            # if SNode.Id.count("maxpool") > 0 and TNode.Id.count("layer1.0.conv1") > 0:
            #     print(EdgeList_partition[-1], str(SlayerId), str(TlayerId))

    # print(EdgeList_partition)
    torch.save(partition, "partition_ResNet34.pth")
    torch.save(EdgeList_partition, "EdgeList_ResNet34.pth")
    print("Layer Num",len(list(info.keys())),"Neural Core Num",len(partition))
    layerNum = 0
    for i, core in enumerate(partition):
        layerNum = layerNum + len(core.layerIdSet)
        print(i,layerNum,core)
    # for edge in EdgeList_partition:
    #     print("============================================")
    #     print(edge)
