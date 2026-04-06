import torch
from partition import Node, Edge
import json
import heapq
# 在partition之后，我们先根据计算顺序确定每个结点的顺序，然后根据结点顺序按照hilbert曲线进行2-D mash的映射，算法基于浙大工作改编
# Modifed Hilbert Curve for Rectangles and Cuboids and Its Application in Entropy Coding for Image and Video 使用这篇工作的方法生成任意维度的hilbert曲线

class Tension():
    def __init__(self, SNode, TNode, tension):        
        self.SNode = SNode
        self.TNode = TNode
        self.tension = tension
    
    def __lt__(self, other):
        return self.tension > other.tension    

    def __str__(self):
        return f"Tension(SNode={self.SNode}, TNode={self.TNode}, Tension={self.tension})"
    
    

# 但是由于能力有限，我们手动写一个6x6的hilbert曲线

# 6x6的hilbert曲线
hilbert_mapping6_6 = [(0,0),(0,1),(1,1),(1,0),(2,0),(3,0),(3,1),(2,1),(2,2),(3,2),(3,3),(2,3),(1,3),(1,2),(0,2), \
                      (0,3),(0,4),(0,5),(1,5),(1,4),(2,4),(2,5),(3,5),(3,4),(4,4),(4,5),(5,5),(5,4),(5,3),(4,3), \
                      (4,2),(5,2),(5,1),(4,1),(4,0),(5,0)]
H = 6
W = 6

# 载入partition
partition = torch.load("partition.pth")

# 按照计算顺序确定每个结点的顺序
NodeListCal = []
EdgeList = torch.load("EdgeList.pth")

# 载入计算顺序
info = torch.load("../calculateInfoConv_vitsmall_no_CRAM.pth")
for key in info.keys(): # 根据计算顺序寻找node
    thisNode = None
    for node in partition: # 遍历所有的node
        if key in node.layerIdSet:
            thisNode = node
            break            
    # 将partition对应的node删除
    if thisNode is not None:
        partition.remove(thisNode)
        # 将node加入NodeListCal
        NodeListCal.append(thisNode)

def NodeListCal_index(node):
    i = 0
    for checknode in NodeListCal:
        if checknode.Id == node.Id:
            return i
        i = i + 1
    return -1

# 构建TensorList并且进行排序
def create_Tension_List(TensionList,hilbert_mapping6_6,NodeListCal):
    # 使用基于势场的mapping算法进一步优化
    # 对于每个2D-Mesh中相邻的结点，我们计算这两个结点的tension，并且排序
    # bug1: 如果两个node互换，node之间的连线应该是不计入tension的
    dx = [1,0]
    dy = [0,1]
    for y in range(H):
        for x in range(W):
            if hilbert_mapping6_6.index((x,y)) >= len(NodeListCal): #该结点没有被分配layers
                continue
            node1 = NodeListCal[hilbert_mapping6_6.index((x,y))]
            # 只考虑左边和下面的节点
            for i in range(len(dx)):
                if x+dx[i] < W and y+dy[i] < H:
                    tension = 0 # tension的定义是potention1 - potention2
                    potention1 = 0 # potention1是当前的势能
                    potention2 = 0 # potention2是移动之后的势能
                    if hilbert_mapping6_6.index((x+dx[i],y+dy[i])) >= len(NodeListCal): #该结点没有被分配layers
                        continue
                    node2 = NodeListCal[hilbert_mapping6_6.index((x+dx[i],y+dy[i]))]
                    # 计算目前的势能
                    #   1. 计算这条边在翻转前影响的总势能
                    for edge in EdgeList:
                        if edge.SNode.Id == node1.Id or edge.TNode.Id == node1.Id or edge.SNode.Id == node2.Id or edge.TNode.Id == node2.Id:
                        # 计算这条边的曼哈顿距离
                            if edge.SNode.Id == node1.Id and edge.TNode.Id == node2.Id or edge.SNode.Id == node2.Id and edge.TNode.Id == node1.Id: # 如果是node1和node2
                                continue
                            x1,y1 = hilbert_mapping6_6[NodeListCal_index(edge.SNode)]
                            x2,y2 = hilbert_mapping6_6[NodeListCal_index(edge.TNode)]
                            distance = abs(x1-x2) + abs(y1-y2)
                            # if node1.Id.count("model.module.layer4.2.conv1") > 0:
                            #     print(f"({x1},{y1}),({x2},{y2})")
                            #     print("node1",node1,"node2",node2,"potention1",potention1,"distance*edge.FlitNumber",distance*edge.FlitNumber)
                            # if node2.Id.count("model.module.layer4.2.conv1") > 0:
                            #     print(f"({x1},{y1}),({x2},{y2})")
                            #     print("node2",node2,"potention1",potention1,"distance*edge.FlitNumber",distance*edge.FlitNumber)
                            potention1 = distance*edge.FlitNumber + potention1
                    #   2. 计算这两个点在翻转后影响的总势能
                    for edge in EdgeList:
                        if edge.SNode.Id == node1.Id or edge.TNode.Id == node1.Id or edge.SNode.Id == node2.Id or edge.TNode.Id == node2.Id:
                            if edge.SNode.Id == node1.Id and edge.TNode.Id == node2.Id or edge.SNode.Id == node2.Id and edge.TNode.Id == node1.Id: # 如果是node1和node2
                                continue
                            if edge.SNode.Id == node1.Id: # 如果是node1且node1是SNode(x1,y1)
                                x1,y1 = hilbert_mapping6_6[NodeListCal_index(edge.SNode)]
                                x1 = x1 + dx[i] # node1
                                y1 = y1 + dy[i]
                                x2,y2 = hilbert_mapping6_6[NodeListCal_index(edge.TNode)]
                            elif edge.TNode.Id == node1.Id: # 如果是node1且node1是TNode(x2,y2)
                                x1,y1 = hilbert_mapping6_6[NodeListCal_index(edge.SNode)]
                                x2,y2 = hilbert_mapping6_6[NodeListCal_index(edge.TNode)]
                                x2 = x2 + dx[i] # node1
                                y2 = y2 + dy[i]
                            elif edge.SNode.Id == node2.Id: # 如果是node2且node2是SNode(x1,y1)
                                x1,y1 = hilbert_mapping6_6[NodeListCal_index(edge.SNode)]
                                x1 = x1 - dx[i] # node2
                                y1 = y1 - dy[i]
                                x2,y2 = hilbert_mapping6_6[NodeListCal_index(edge.TNode)]
                            elif edge.TNode.Id == node2.Id: # 如果是node2且node2是SNode(x1,y1)
                                x1,y1 = hilbert_mapping6_6[NodeListCal_index(edge.SNode)]
                                x2,y2 = hilbert_mapping6_6[NodeListCal_index(edge.TNode)]
                                x2 = x2 - dx[i] # node2
                                y2 = y2 - dy[i]
                            distance = abs(x1-x2) + abs(y1-y2)
                            # if node1.Id.count("model.module.layer4.2.conv1") > 0:
                            #     print(f"({x1},{y1}),({x2},{y2})")
                            #     print("node1",node1,"potention2",potention2,"distance*edge.FlitNumber",distance*edge.FlitNumber)
                            # if node2.Id.count("model.module.layer4.2.conv1") > 0:
                            #     print(f"({x1},{y1}),({x2},{y2})")
                            #     print("node1",node1,"node2",node2,"potention2",potention2,"distance*edge.FlitNumber",distance*edge.FlitNumber)
                            potention2 = distance*edge.FlitNumber + potention2
                    tension = potention1 - potention2 # 如果tensor为正，则有优化空间
                    # print(node1,node2,potention1,potention2)
                    TensionList.append(Tension(node1,node2,tension))
                else:
                    pass
    # 接下来对队列进行排序，构建优先队列
    heapq.heapify(TensionList)
    return TensionList

if __name__ == "__main__":
    # 首先构建TensionList
    TensionList = []
    TensionList = create_Tension_List(TensionList,hilbert_mapping6_6,NodeListCal)
    # for tension in TensionList:
    #     print(tension)
    #     print("==========================================")
    # 选择前N大的tension进行点对位置转换
    N = 1
    maxIter = 10
    iter = 0
    maxTension = 1.0
    while(maxTension > 0 and maxIter >= iter):
        iter = iter + 1
        for i in range(N):
            curtension = heapq.heappop(TensionList)
            print("=================================================")
            print(curtension, iter)
            maxTension = curtension.tension
            if curtension.tension <= 0:
                break
            SNode = curtension.SNode
            TNode = curtension.TNode
            # 交换位置
            # print(SNode,"------>",TNode)
            pos1 = NodeListCal_index(SNode)
            pos2 = NodeListCal_index(TNode)
            NodeListCal[pos1], NodeListCal[pos2] = NodeListCal[pos2], NodeListCal[pos1]
        # 重新构建TensionList
        TensionList = []
        TensionList = create_Tension_List(TensionList,hilbert_mapping6_6,NodeListCal)
                    
    nodeID = []
    for node in NodeListCal:
        nodeID.append(node.Id)

    MeshMapping = [[0 for x in range(W)] for y in range(H)]

    def format_number(num):
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return f"{num:.1f}"
    # 生成echarts的option
    option = {"title": {
        "text": "Hilbert Mapping of ResNet34 in 6x6 2D mesh",
    },
    "tooltip": {},
    "animationDurationUpdate": 1500,
    "animationEasingUpdate": "quinticInOut",
    "series":[
            {
                "type": 'graph',
                "layout": 'none',
                "symbolSize": 50,
                "roam": "true",
                "label": {
                    "show": "true"
                },
                "edgeSymbol": ['circle', 'arrow'],
                "edgeSymbolSize": [4, 10],
                "edgeLabel": {
                    "fontSize": 20
                },
                "data": [],
                "links": [],
                "lineStyle": {
                    "opacity": 0.9,
                    "width": 2,
                    "curveness": 0
                }
        }
    ]
    }

    for node in NodeListCal:
        MeshMapping[hilbert_mapping6_6[NodeListCal.index(node)][0]][hilbert_mapping6_6[NodeListCal.index(node)][1]] = node
        option["series"][0]["data"].append({
            "name": nodeID.index(node.Id),
            "x": hilbert_mapping6_6[NodeListCal.index(node)][1]*100,
            "y": hilbert_mapping6_6[NodeListCal.index(node)][0]*100,
            "symbolSize": 50,
            "itemStyle": {
                "color": "blue"
            },
            "tooltip": {
                "show": "true",
                "formatter": node.Id
            }    
        })

    for i in range(len(NodeListCal),36):
        node = Node(layerIdSet=f"Null{i}", SRAMNumber=0, AdderNumber=0)
        MeshMapping[hilbert_mapping6_6[i][0]][hilbert_mapping6_6[i][1]] = node
        option["series"][0]["data"].append({
            "name": i,
            "x": hilbert_mapping6_6[i][1]*100,
            "y": hilbert_mapping6_6[i][0]*100,
            "symbolSize": 50,
            "itemStyle": {
                "color": "grey"
            }
        })

    for edge in EdgeList:
        # print(edge)
        option["series"][0]["links"].append({
            "source": nodeID.index(edge.SNode.Id),
            "target": nodeID.index(edge.TNode.Id),
            "label": {
                "show": "true",
                "formatter": format_number(edge.FlitNumber)
            },
            "lineStyle": {
                "normal": {
                    "curveness": 0.1
                }
            }
        })


    with open('data.json', 'w') as f:
        json.dump(option, f)


    # 生成echarts的option
    option = {"title": {
        "text": "Hilbert Mapping of ResNet34 in 6x6 2D mesh",
    },
    "tooltip": {},
    "animationDurationUpdate": 1500,
    "animationEasingUpdate": "quinticInOut",
    "series":[
            {
                "type": 'graph',
                "layout": 'none',
                "symbolSize": 50,
                "roam": "true",
                "label": {
                    "show": "true"
                },
                "edgeSymbol": ['circle', 'arrow'],
                "edgeSymbolSize": [4, 10],
                "edgeLabel": {
                    "fontSize": 20
                },
                "data": [],
                "links": [],
                "lineStyle": {
                    "opacity": 0.9,
                    "width": 2,
                    "curveness": 0
                }
        }
    ]
    }

    for node in NodeListCal:
        MeshMapping[hilbert_mapping6_6[NodeListCal.index(node)][0]][hilbert_mapping6_6[NodeListCal.index(node)][1]] = node
        option["series"][0]["data"].append({
            "name": nodeID.index(node.Id),
            "x": hilbert_mapping6_6[NodeListCal.index(node)][1]*100,
            "y": hilbert_mapping6_6[NodeListCal.index(node)][0]*100,
            "symbolSize": 50,
            "itemStyle": {
                "color": "blue"
            },
            "tooltip": {
                "show": "true",
                "formatter": node.Id
            }    
        })

    for i in range(len(NodeListCal),36):
        node = Node(layerIdSet=f"Null{i}", SRAMNumber=0, AdderNumber=0)
        MeshMapping[hilbert_mapping6_6[i][0]][hilbert_mapping6_6[i][1]] = node
        option["series"][0]["data"].append({
            "name": i,
            "x": hilbert_mapping6_6[i][1]*100,
            "y": hilbert_mapping6_6[i][0]*100,
            "symbolSize": 50,
            "itemStyle": {
                "color": "grey"
            }
        })

    for edge in EdgeList:
        # print(edge)
        option["series"][0]["links"].append({
            "source": nodeID.index(edge.SNode.Id),
            "target": nodeID.index(edge.TNode.Id),
            "label": {
                "show": "true",
                "formatter": format_number(edge.avgFlitNumSpine)
            },
            "lineStyle": {
                "normal": {
                    "curveness": 0.1
                }
            }
        })


    with open('SpinflitNum.json', 'w') as f:
        json.dump(option, f)
        

        
    torch.save(MeshMapping, "MeshMapping.pth")
