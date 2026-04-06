# 在进行hilbert曲线以及基于势场的mapping之后，我们根据2D-Mesh的路由使用无关路由算法确定每一条NoC link上的传输量以及最长的传输时间。
# 使用 最长传输时间*输出的spine数量 + critical path上的时间 作为latency，这是因为VSA使用了token/spine-level pipeline
# critical path time = \sum T_router * N + T_link * N
# 最长传输时间 = T_router_max * N_max + T_link_max * N_max
# 这里的N是指某一条NoC link上的某一个token/spine中的平均flit数量*该层的视野域，*_max是指最长路径中对应的变量
# T_router 某一个router的处理时间 = 16 cycle，T_link 某一个NoC link上的传输时间，这边设置为一个cycle两个flit
# 因此这里需要引入新的统计量，也就是每个layer中一个token/spine的平均flit数量

import torch
from partition import Edge, Node
import queue
import json

H = 6
W = 6

class NodeWithManhattan():
    def __init__(self, node, distance, x, y):
        self.node = node
        self.x = x
        self.y = y
        self.distance = distance
        self.traffic = 0
        self.SpineflitNum = 0
        self.occupy = 0
        self.transmitTime = 0
    
    def __lt__(self, other):
        return self.distance < other.distance

    def __str__(self):
        return f"NodeWithManhattan(SNode={self.node}, distance={self.distance})"


def update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,allocate,transmitTime,TranOccupy, T_link, T_router):
    transmitTime[pos2] = max(transmitTime[pos2], transmitTime[pos1] + (TranOccupy[pos1][pos2])*T_link*enNode.SpineflitNum + curNode.occupy*T_router*enNode.SpineflitNum)
    # print(f"pos1={pos1},pos2={pos2},transmitTime[pos2]={transmitTime[pos2]},transmitTime[pos1]={transmitTime[pos1]},TranOccupy[pos1][pos2]={TranOccupy[pos1][pos2]},SpineflitNum={enNode.SpineflitNum},curNode.occupy={curNode.occupy}")
    # 找到对应的下一个节点，更新其traffic
    for enNodeForupdate in encloseNodes:
        if enNodeForupdate.node.Id == nextNode.Id:
            enNodeForupdate.SpineflitNum = enNodeForupdate.SpineflitNum + enNode.SpineflitNum * allocate
            break


# 输出Latency
def cal_Latency(NodeAllocation, TranOccupy, EdgeList, NodePosition, MeshMapping):
    # 计算平均Latency
    T_router = 8
    T_link = 0.5
    
    # 使用类似于defineTraffic里面的方法计算每一条边的maxTransmitTime
    # 1. 确定起始，终止节点位置：
    for edge in EdgeList:
        # print(edge)
        SNode = edge.SNode
        TNode = edge.TNode
        x1,y1 = NodePosition[SNode.Id]
        x2,y2 = NodePosition[TNode.Id]
        MeshMapping[y1][x1].occupy = MeshMapping[y1][x1].occupy + 1
        dx = (1 if x2 - x1 >= 0 else -1)
        dy = (1 if y2 - y1 >= 0 else -1)
        # 如果两个结点相同，说明是自己传输到自己：
        if SNode.Id == TNode.Id:
            curNode = MeshMapping[y1][x1]
            edge.transmitTime = edge.avgFlitNumSpine * T_router
            # print(f"0 : (x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),avgFlitNumSpine={edge.avgFlitNumSpine},occupy={curNode.occupy},transmitTime={edge.transmitTime})")
            continue
        
        # 这里做一个优化，如果x1==x2，或者y1==y2，那么说明这两个点在同一条水平或者垂直线上，分成三条线路进行传输，这样能够balance负载
        if x1 == x2 or y1 == y2:
            if x1 == x2:
                # 2. 确定所有起始，终止结点间的所有点：
                encloseNodes = []
                transmitTime = [0 for i in range(H*W)]
                for x in range(max(min(x1,x2)-1, 0), min(max(x1,x2)+1+1, W)):
                    for y in range(min(y1,y2), max(y1,y2)+1):
                        # print(f"encloseNodes, (x,y)=({x},{y}), (x1,y1)=({x1},{y1}), (x2,y2)=({x2},{y2})")
                        distance = abs(x-x1) + abs(y-y1)
                        encloseNodes.append(NodeWithManhattan(MeshMapping[y][x], distance, x, y))
                        if distance == 0:
                            encloseNodes[-1].SpineflitNum = edge.avgFlitNumSpine
                # 3. 按照曼哈顿距离进行排序
                encloseNodes.sort()        
                # 4. 确定每个节点的传输量
                # print("len(encloseNodes)",len(encloseNodes))
                for enNode in encloseNodes:
                    x_now = enNode.x
                    y_now = enNode.y
                    # print(f"(x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2})")
                    if x_now == x2 and y_now == y2: # 如果已经到了终点
                        pos1 = y_now*W + x_now
                        edge.transmitTime = transmitTime[pos1]
                        # print("pos1",pos1,"edge.transmitTime",edge.transmitTime)
                        break
                    elif x_now == x1 and y_now == y1: # 如果是起点
                        pos1 = y_now*W + x_now
                        y_allocate = NodeAllocation[pos1][0 if dy >= 0 else 1]
                        if x_now + 1 < W:
                            x_allocate_dx = NodeAllocation[pos1][2] # dx方向
                        else:
                            x_allocate_dx = 0
                        if x_now - 1 >= 0: 
                            x_allocate_re_dx = NodeAllocation[pos1][3] # dx方向反向
                        else:
                            x_allocate_re_dx = 0
                        # 归一化
                        y_allocate_tmp = y_allocate/(y_allocate + x_allocate_dx + x_allocate_re_dx + 1e-5)
                        x_allocate_dx_tmp = x_allocate_dx/(y_allocate + x_allocate_dx + x_allocate_re_dx + 1e-5)
                        x_allocate_re_dx_tmp = x_allocate_re_dx/(y_allocate + x_allocate_dx + x_allocate_re_dx + 1e-5)
                        y_allocate = y_allocate_tmp + 0.0
                        x_allocate_dx = x_allocate_dx_tmp + 0.0
                        x_allocate_re_dx = x_allocate_re_dx_tmp + 0.0
                    
                        # 更新路径
                        # 首先进行y轴的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now+dy][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos2 = (y_now+dy)*W + x_now
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,y_allocate,transmitTime,TranOccupy, T_link, T_router)
                        # 然后进行x轴的传输
                        if x_now + 1 < W: # 要满足边界条件
                            curNode = MeshMapping[y_now][x_now]
                            nextNode = MeshMapping[y_now][x_now+1]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = y_now*W + x_now + 1
                            update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,x_allocate_dx,transmitTime,TranOccupy, T_link, T_router)

                        # 然后进行x轴反向的的传输
                        if x_now - 1 >= 0: # 要满足边界条件
                            curNode = MeshMapping[y_now][x_now]
                            nextNode = MeshMapping[y_now][x_now-1]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = y_now*W + x_now - 1
                            update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,x_allocate_re_dx,transmitTime,TranOccupy, T_link, T_router)
                    elif x_now == x2 and (not y_now == y2): # 说明在主干道上
                        # 进行y轴的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now+dy][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now+dy)*W + x_now                    
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    elif (x_now + 1 == x2) and y_now == y2: # 说明不在主干道上但是在相同水平线上
                        # 进行x轴的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now][x_now+1]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now + 1                    
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    elif (x_now - 1 == x2) and y_now == y2: # 说明不在主干道上但是在相同水平线上
                        # 进行x轴反向的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now][x_now-1]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now - 1                    
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    else: # 都不在，往y轴传输
                        nextNode = MeshMapping[y_now+dy][x_now]
                        curNode = MeshMapping[y_now][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now+dy)*W + x_now                    
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
            else: # y1 == y2
                encloseNodes = []
                transmitTime = [0 for i in range(H*W)]
                for x in range(min(x1,x2), max(x1,x2)+1):
                    for y in range(max(min(y1,y2)-1,0), min(max(y1,y2)+1+1,H)):
                        # print(f"encloseNodes, (x,y)=({x},{y}), (x1,y1)=({x1},{y1}), (x2,y2)=({x2},{y2})")
                        distance = abs(x-x1) + abs(y-y1)
                        encloseNodes.append(NodeWithManhattan(MeshMapping[y][x], distance, x, y))
                        if distance == 0:
                            encloseNodes[-1].SpineflitNum = edge.avgFlitNumSpine
            
                # 3. 按照曼哈顿距离进行排序
                encloseNodes.sort()
                # 4. 确定每个节点的传输量
                # print("len(encloseNodes)",len(encloseNodes))
                for enNode in encloseNodes:
                    x_now = enNode.x
                    y_now = enNode.y
                    # print(f"(x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2})")
                    if x_now == x2 and y_now == y2: # 如果已经到了终点
                        pos1 = y_now*W + x_now
                        edge.transmitTime = transmitTime[pos1]
                        # print("edge.transmitTime",edge.transmitTime)
                        break
                    elif x_now == x1 and y_now == y1: # 如果是起点
                        pos1 = y_now*W + x_now
                        if y_now + 1 < H:
                            y_allocate_dy = NodeAllocation[pos1][0]
                        else:
                            y_allocate_dy = 0
                        if y_now - 1 >= 0:
                            y_allocate_re_dy = NodeAllocation[pos1][1]
                        else:
                            y_allocate_re_dy = 0
                        x_allocate = NodeAllocation[pos1][2 if dx >= 0 else 3] # dx方向
                        # 归一化
                        y_allocate_dy_tmp = y_allocate_dy/(y_allocate_dy + y_allocate_re_dy + x_allocate + 1e-5)
                        y_allocate_re_dy_tmp = y_allocate_re_dy/(y_allocate_dy + y_allocate_re_dy + x_allocate + 1e-5)
                        x_allocate_tmp = x_allocate/(y_allocate_dy + y_allocate_re_dy + x_allocate + 1e-5)
                        y_allocate_dy = y_allocate_dy_tmp + 0.0
                        y_allocate_re_dy = y_allocate_re_dy_tmp + 0.0
                        x_allocate = x_allocate_tmp + 0.0
                    
                        # 更新路径
                        # 首先进行x轴的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now][x_now+dx]
                        # 更新traffic邻接矩阵里面的值
                        pos2 = y_now*W + x_now + dx
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,x_allocate,transmitTime,TranOccupy, T_link, T_router)
                        # 然后进行y轴的传输
                        if y_now + 1 < H: # 要满足边界条件
                            curNode = MeshMapping[y_now][x_now]
                            nextNode = MeshMapping[y_now+1][x_now]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = (y_now+1)*W + x_now
                            update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,y_allocate_dy,transmitTime,TranOccupy, T_link, T_router)

                        if y_now - 1 >= 0: # 要满足边界条件
                            # 然后进行y轴反向的的传输
                            curNode = MeshMapping[y_now][x_now]
                            nextNode = MeshMapping[y_now-1][x_now]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = (y_now-1)*W + x_now
                            update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,y_allocate_re_dy,transmitTime,TranOccupy, T_link, T_router)
                    elif y_now == y2 and (not x_now == x2): # 说明在主干道上
                        # 进行x轴的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now][x_now+dx]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now + dx
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    elif x_now == x2 and y_now + 1 == y2: # 说明不在主干道上但是在相同垂直线上
                        # 进行y轴的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now+1][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now+1)*W + x_now
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    elif x_now == x2 and y_now - 1 == y2: # 说明不在主干道上但是在相同垂直线上
                        # 进行y轴反向的传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now-1][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now-1)*W + x_now                  
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    else: # 都不在，往x轴传输
                        curNode = MeshMapping[y_now][x_now]
                        nextNode = MeshMapping[y_now][x_now+dx]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now + dx
                        update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
                    # print("pos1,pos2",pos1,pos2)
            continue    
            
        # 2. 确定所有起始，终止结点间的所有点：
        encloseNodes = []
        transmitTime = [0 for i in range(H*W)]
        for x in range(min(x1,x2), max(x1,x2)+1):
            for y in range(min(y1,y2), max(y1,y2)+1):
                distance = abs(x-x1) + abs(y-y1)
                encloseNodes.append(NodeWithManhattan(MeshMapping[y][x], distance, x, y))
                if distance == 0:
                    encloseNodes[-1].SpineflitNum = edge.avgFlitNumSpine
        # 3. 按照曼哈顿距离进行排序
        encloseNodes.sort()        
        # 4. 确定每个节点的传输量
        for enNode in encloseNodes:
            x_now = enNode.x
            y_now = enNode.y
            # 确定要传输的结点，一个是dx一个是dy
            if x_now == x2 and y_now == y2:
                pos1 = y_now*W + x_now
                edge.transmitTime = transmitTime[pos1]
                # print("edge.transmitTime",edge.transmitTime)
                break
            elif x_now == x2 and y_now != y2:
                # 如果x已经到达位置，那么只进行y轴的传输
                curNode = MeshMapping[y_now][x_now]
                nextNode = MeshMapping[y_now+dy][x_now]
                # 更新traffic邻接矩阵里面的值
                pos1 = y_now*W + x_now
                pos2 = (y_now+dy)*W + x_now
                update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
            elif x_now != x2 and y_now == y2:
                # 如果y已经到达位置，那么只进行x轴的传输
                curNode = MeshMapping[y_now][x_now]
                nextNode = MeshMapping[y_now][x_now+dx]
                # 更新traffic邻接矩阵里面的值
                pos1 = y_now*W + x_now
                pos2 = y_now*W + x_now + dx
                # print(f"2 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,1.0,transmitTime,TranOccupy, T_link, T_router)
            else: # 如果x和y都没有到达位置，那么x和y都要传输，需要更新两个结点以及两个NoC Links
                # 首先进行y轴的传输
                curNode = MeshMapping[y_now][x_now]
                nextNode = MeshMapping[y_now+dy][x_now]
                # 更新traffic邻接矩阵里面的值
                pos1 = y_now*W + x_now
                pos2 = (y_now+dy)*W + x_now
                # print(f"3 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                # 如果是在起始位置：
                if x_now == x1 and y_now == y1:
                    y_allocate = NodeAllocation[pos1][0 if dy >= 0 else 1]
                    x_allocate = NodeAllocation[pos1][2 if dx >= 0 else 3]
                    # 归一化
                    if y_allocate + x_allocate > 0.0:
                        y_allocate_tmp = y_allocate/(y_allocate + x_allocate)
                        x_allocate_tmp = x_allocate/(y_allocate + x_allocate)
                        y_allocate = y_allocate_tmp + 0.0
                        x_allocate = x_allocate_tmp + 0.0
                    else:
                        y_allocate = 0.5
                        x_allocate = 0.5
                else:
                    y_allocate = 0.5
                    x_allocate = 0.5

                update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,y_allocate,transmitTime,TranOccupy, T_link, T_router)

                curNode = MeshMapping[y_now][x_now]
                nextNode = MeshMapping[y_now][x_now+dx]
                # 然后进行x轴的传输
                # 更新traffic邻接矩阵里面的值
                pos2 = y_now*W + x_now + dx
                # print(f"4 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                update_latency(encloseNodes,pos1,pos2,enNode,curNode,nextNode,x_allocate,transmitTime,TranOccupy, T_link, T_router)

    # 找到最长的距离
    MaxDis = 0
    # print("=================================================")
    for edge in EdgeList:
        # print("edge.transmitTime",edge.transmitTime)
        if edge.transmitTime > MaxDis:
            MaxDis = edge.transmitTime
                
    # 在所有edge之中找最长的一条链
    CriticalPath = 0 # 这个是到终点的最长距离
    NodesLen = {} # 到每个节点的距离
    Visited = [0 for i in range(len(EdgeList))] # 这个是用来记录某条边是否访问过    
    
    EdgeQueue = queue.Queue()
    EdgeQueue.put(EdgeList[0])
    NodesLen[EdgeList[0].SNode.Id] = 0
    Visited[0] = 1
    while(not EdgeQueue.empty()): # 访问push出边，寻找结点，然后加入边到queue中
        curEdge = EdgeQueue.get()
        TNode = curEdge.TNode
        SNode = curEdge.SNode
        if TNode.Id in NodesLen.keys(): # 更新NodesLen，如果有两条线到同一个结点取更大的那条线
            NodesLen[TNode.Id] = max(NodesLen[TNode.Id],NodesLen[SNode.Id] + edge.transmitTime)
        else:
            NodesLen[TNode.Id] = NodesLen[SNode.Id] + edge.transmitTime
        for i,findEdge in enumerate(EdgeList):
            # print("findEdge.SNode.Id",findEdge.SNode.Id,"curEdge.TNode.Id",curEdge.TNode.Id)
            if findEdge.SNode.Id == curEdge.TNode.Id and Visited[i] == 0: # 找到了边
                Visited[i] = 1
                EdgeQueue.put(findEdge)        

        if TNode.Id.count("fc") > 0: # 如果到了最终结点
            CriticalPath = max(CriticalPath, NodesLen[TNode.Id])
                
    # 定义最后输出的spine大小：49
    finalSpineNum = 197
    # print(MaxAdjustLatency,MaxDis)
    # print("CriticalPath,MaxDis",CriticalPath,MaxDis)
    # print("NodesLen",NodesLen)
    Latency = CriticalPath + MaxDis*finalSpineNum
    # print(f"Latency={Latency}, CriticalPath={CriticalPath}, MaxDis={MaxDis}")
    return Latency




def update_info(encloseNodes,pos1,pos2,enNode,nextNode,allocate,TranTraffic,TranSpineFlitNum,TranOccupy):
    TranTraffic[pos1][pos2] = TranTraffic[pos1][pos2] + enNode.traffic * allocate
    TranSpineFlitNum[pos1][pos2] = TranSpineFlitNum[pos1][pos2] + enNode.SpineflitNum * allocate           
    TranOccupy[pos1][pos2] = TranOccupy[pos1][pos2] + enNode.occupy * allocate    
    # 找到对应的下一个节点，更新其traffic
    for enNodeForupdate in encloseNodes:
        if enNodeForupdate.node.Id == nextNode.Id:
            enNodeForupdate.traffic = enNodeForupdate.traffic + enNode.traffic * allocate
            enNodeForupdate.SpineflitNum = enNodeForupdate.SpineflitNum + enNode.SpineflitNum * allocate
            enNodeForupdate.occupy = enNodeForupdate.occupy + enNode.occupy * allocate
            break

# 确定每条NoC link上的传输量以及最长的传输时间
# NodeAllocation: 每个结点为起点时y轴和x轴的分配情况
def defineTraffic(MeshMapping, EdgeList, NodeAllocation):
    # 构建邻接矩阵
    TranTraffic  = [[0 for i in range(H*W)] for i in range(H*W)] # 存储每条NoC Link上面的总传输量（FlitNumber）
    TranSpineFlitNum  = [[0 for i in range(H*W)] for i in range(H*W)] # VSA一次性输出多个Flits, 存储每条NoC Link上每次输出的平均Flit数量
    TranOccupy = [[0 for i in range(H*W)] for i in range(H*W)] # 存储每条NoC Link的占用情况，这用来计算大的simulator的NoC latency
    NodePosition = {}
    # 先构建每一个节点的位置信息
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):
            NodePosition[MeshMapping[y][x].Id] = (x,y)
    # 1. 确定起始，终止节点位置：
    for edge in EdgeList:
        # print(edge)
        SNode = edge.SNode
        TNode = edge.TNode
        x1,y1 = NodePosition[SNode.Id]
        x2,y2 = NodePosition[TNode.Id]
        MeshMapping[y1][x1].occupy = MeshMapping[y1][x1].occupy + 1
        dx = (1 if x2 - x1 >= 0 else -1)
        dy = (1 if y2 - y1 >= 0 else -1)
        # 如果两个结点相同，说明是自己传输到自己：
        if SNode.Id == TNode.Id:
            pos = y1*W + x1
            # print(f"0 : (x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos,pos)=({pos},{pos})")
            TranTraffic[pos][pos] = TranTraffic[pos][pos] + edge.FlitNumber * 256
            TranSpineFlitNum[pos][pos] = TranSpineFlitNum[pos][pos] + edge.avgFlitNumSpine
            TranOccupy[pos][pos] = TranOccupy[pos][pos] + 0
            continue
        
        # 这里做一个优化，如果x1==x2，或者y1==y2，那么说明这两个点在同一条水平或者垂直线上，分成三条线路进行传输，这样能够balance负载
        if x1 == x2 or y1 == y2:
            if x1 == x2:
                # 2. 确定所有起始，终止结点间的所有点：49244
                encloseNodes = []
                for x in range(max(min(x1,x2)-1, 0), min(max(x1,x2)+1+1, W)):
                    for y in range(min(y1,y2), max(y1,y2)+1):
                        # print(f"encloseNodes, (x,y)=({x},{y}), (x1,y1)=({x1},{y1}), (x2,y2)=({x2},{y2})")
                        distance = abs(x-x1) + abs(y-y1)
                        encloseNodes.append(NodeWithManhattan(MeshMapping[y][x], distance, x, y))
                        if distance == 0: #说明是始发node
                            encloseNodes[-1].traffic = edge.FlitNumber * 256
                            encloseNodes[-1].SpineflitNum = edge.avgFlitNumSpine
                            encloseNodes[-1].occupy = 1.0
                # 3. 按照曼哈顿距离进行排序
                encloseNodes.sort()        
                # 4. 确定每个节点的传输量
                # print("len(encloseNodes)",len(encloseNodes))
                for enNode in encloseNodes:
                    x_now = enNode.x
                    y_now = enNode.y
                    # print(f"(x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2})")
                    if x_now == x2 and y_now == y2: # 如果已经到了终点
                        continue
                    elif x_now == x1 and y_now == y1: # 如果是起点
                        pos1 = y_now*W + x_now
                        y_allocate = NodeAllocation[pos1][0 if dy >= 0 else 1]
                        if x_now + 1 < W:
                            x_allocate_dx = NodeAllocation[pos1][2] # dx方向
                        else:
                            x_allocate_dx = 0
                        if x_now - 1 >= 0: 
                            x_allocate_re_dx = NodeAllocation[pos1][3] # dx方向反向
                        else:
                            x_allocate_re_dx = 0
                        # 归一化
                        y_allocate_tmp = y_allocate/(y_allocate + x_allocate_dx + x_allocate_re_dx + 1e-5)
                        x_allocate_dx_tmp = x_allocate_dx/(y_allocate + x_allocate_dx + x_allocate_re_dx + 1e-5)
                        x_allocate_re_dx_tmp = x_allocate_re_dx/(y_allocate + x_allocate_dx + x_allocate_re_dx + 1e-5)
                        y_allocate = y_allocate_tmp + 0.0
                        x_allocate_dx = x_allocate_dx_tmp + 0.0
                        x_allocate_re_dx = x_allocate_re_dx_tmp + 0.0
                    
                        # 更新路径
                        # 首先进行y轴的传输
                        nextNode = MeshMapping[y_now+dy][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos2 = (y_now+dy)*W + x_now
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,y_allocate,TranTraffic,TranSpineFlitNum,TranOccupy)
                        # 然后进行x轴的传输
                        if x_now + 1 < W: # 要满足边界条件
                            nextNode = MeshMapping[y_now][x_now+1]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = y_now*W + x_now + 1
                            update_info(encloseNodes,pos1,pos2,enNode,nextNode,x_allocate_dx,TranTraffic,TranSpineFlitNum,TranOccupy)

                        # 然后进行x轴反向的的传输
                        if x_now - 1 >= 0: # 要满足边界条件
                            nextNode = MeshMapping[y_now][x_now-1]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = y_now*W + x_now - 1
                            update_info(encloseNodes,pos1,pos2,enNode,nextNode,x_allocate_re_dx,TranTraffic,TranSpineFlitNum,TranOccupy)
                    elif x_now == x2 and (not y_now == y2): # 说明在主干道上
                        # 进行y轴的传输
                        nextNode = MeshMapping[y_now+dy][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now+dy)*W + x_now                    
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
                    elif (x_now + 1 == x2) and y_now == y2: # 说明不在主干道上但是在相同水平线上
                        # 进行x轴的传输
                        nextNode = MeshMapping[y_now][x_now+1]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now + 1                    
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
                    elif (x_now - 1 == x2) and y_now == y2: # 说明不在主干道上但是在相同水平线上
                        # 进行x轴反向的传输
                        nextNode = MeshMapping[y_now][x_now-1]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now - 1                    
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
                    else: # 都不在，往y轴传输
                        nextNode = MeshMapping[y_now+dy][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now+dy)*W + x_now                    
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
            else: # y1 == y2
                encloseNodes = []
                for x in range(min(x1,x2), max(x1,x2)+1):
                    for y in range(max(min(y1,y2)-1,0), min(max(y1,y2)+1+1,H)):
                        # print(f"encloseNodes, (x,y)=({x},{y}), (x1,y1)=({x1},{y1}), (x2,y2)=({x2},{y2})")
                        distance = abs(x-x1) + abs(y-y1)
                        encloseNodes.append(NodeWithManhattan(MeshMapping[y][x], distance, x, y))
                        if distance == 0: #说明是始发node
                            encloseNodes[-1].traffic = edge.FlitNumber * 256
                            encloseNodes[-1].SpineflitNum = edge.avgFlitNumSpine
                            encloseNodes[-1].occupy = 1.0
            
                # 3. 按照曼哈顿距离进行排序
                encloseNodes.sort()
                # 4. 确定每个节点的传输量
                # print("len(encloseNodes)",len(encloseNodes))
                for enNode in encloseNodes:
                    x_now = enNode.x
                    y_now = enNode.y
                    # print(f"(x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2})")
                    if x_now == x2 and y_now == y2: # 如果已经到了终点
                        continue
                    elif x_now == x1 and y_now == y1: # 如果是起点
                        pos1 = y_now*W + x_now
                        if y_now + 1 < H:
                            y_allocate_dy = NodeAllocation[pos1][0]
                        else:
                            y_allocate_dy = 0
                        if y_now - 1 >= 0:
                            y_allocate_re_dy = NodeAllocation[pos1][1]
                        else:
                            y_allocate_re_dy = 0
                        x_allocate = NodeAllocation[pos1][2 if dx >= 0 else 3] # dx方向
                        # 归一化
                        y_allocate_dy_tmp = y_allocate_dy/(y_allocate_dy + y_allocate_re_dy + x_allocate + 1e-5)
                        y_allocate_re_dy_tmp = y_allocate_re_dy/(y_allocate_dy + y_allocate_re_dy + x_allocate + 1e-5)
                        x_allocate_tmp = x_allocate/(y_allocate_dy + y_allocate_re_dy + x_allocate + 1e-5)
                        y_allocate_dy = y_allocate_dy_tmp + 0.0
                        y_allocate_re_dy = y_allocate_re_dy_tmp + 0.0
                        x_allocate = x_allocate_tmp + 0.0
                    
                        # 更新路径
                        # 首先进行x轴的传输
                        nextNode = MeshMapping[y_now][x_now+dx]
                        # 更新traffic邻接矩阵里面的值
                        pos2 = y_now*W + x_now + dx
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,x_allocate,TranTraffic,TranSpineFlitNum,TranOccupy)
                        # 然后进行y轴的传输
                        if y_now + 1 < H: # 要满足边界条件
                            nextNode = MeshMapping[y_now+1][x_now]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = (y_now+1)*W + x_now
                            update_info(encloseNodes,pos1,pos2,enNode,nextNode,y_allocate_dy,TranTraffic,TranSpineFlitNum,TranOccupy)

                        if y_now - 1 >= 0: # 要满足边界条件
                            # 然后进行y轴反向的的传输
                            nextNode = MeshMapping[y_now-1][x_now]
                            # 更新traffic邻接矩阵里面的值
                            pos2 = (y_now-1)*W + x_now
                            update_info(encloseNodes,pos1,pos2,enNode,nextNode,y_allocate_re_dy,TranTraffic,TranSpineFlitNum,TranOccupy)
                    elif y_now == y2 and (not x_now == x2): # 说明在主干道上
                        # 进行x轴的传输
                        nextNode = MeshMapping[y_now][x_now+dx]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now + dx
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
                    elif x_now == x2 and y_now + 1 == y2: # 说明不在主干道上但是在相同垂直线上
                        # 进行y轴的传输
                        nextNode = MeshMapping[y_now+1][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now+1)*W + x_now
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
                    elif x_now == x2 and y_now - 1 == y2: # 说明不在主干道上但是在相同垂直线上
                        # 进行y轴反向的传输
                        nextNode = MeshMapping[y_now-1][x_now]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = (y_now-1)*W + x_now                  
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)
                    else: # 都不在，往x轴传输
                        nextNode = MeshMapping[y_now][x_now+dx]
                        # 更新traffic邻接矩阵里面的值
                        pos1 = y_now*W + x_now
                        pos2 = y_now*W + x_now + dx
                        update_info(encloseNodes,pos1,pos2,enNode,nextNode,1.0,TranTraffic,TranSpineFlitNum,TranOccupy)                        
                    # print("pos1,pos2",pos1,pos2)
            continue    
            
        # 2. 确定所有起始，终止结点间的所有点：
        encloseNodes = []
        for x in range(min(x1,x2), max(x1,x2)+1):
            for y in range(min(y1,y2), max(y1,y2)+1):
                distance = abs(x-x1) + abs(y-y1)
                encloseNodes.append(NodeWithManhattan(MeshMapping[y][x], distance, x, y))
                if distance == 0: #说明是始发node
                    encloseNodes[-1].traffic = edge.FlitNumber * 256
                    encloseNodes[-1].SpineflitNum = edge.avgFlitNumSpine
                    encloseNodes[-1].occupy = 1.0
        # 3. 按照曼哈顿距离进行排序
        encloseNodes.sort()        
        # 4. 确定每个节点的传输量
        for enNode in encloseNodes:
            x_now = enNode.x
            y_now = enNode.y
            # 确定要传输的结点，一个是dx一个是dy
            if x_now == x2 and y_now == y2:
                break
            elif x_now == x2 and y_now != y2:
                # 如果x已经到达位置，那么只进行y轴的传输
                nextNode = MeshMapping[y_now+dy][x_now]
                # 更新traffic邻接矩阵里面的值
                pos1 = y_now*W + x_now
                pos2 = (y_now+dy)*W + x_now
                TranTraffic[pos1][pos2] = TranTraffic[pos1][pos2] + enNode.traffic
                TranSpineFlitNum[pos1][pos2] = TranSpineFlitNum[pos1][pos2] + enNode.SpineflitNum
                TranOccupy[pos1][pos2] = TranOccupy[pos1][pos2] + enNode.occupy
                # print(f"1 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                # 找到对应的下一个节点，更新其traffic
                for enNodeForupdate in encloseNodes:
                    if enNodeForupdate.node.Id == nextNode.Id:
                        enNodeForupdate.traffic = enNodeForupdate.traffic + enNode.traffic
                        enNodeForupdate.SpineflitNum = enNodeForupdate.SpineflitNum + enNode.SpineflitNum
                        enNodeForupdate.occupy = enNodeForupdate.occupy + enNode.occupy                        
                        break
            elif x_now != x2 and y_now == y2:
                # 如果y已经到达位置，那么只进行x轴的传输
                nextNode = MeshMapping[y_now][x_now+dx]
                # 更新traffic邻接矩阵里面的值
                pos1 = y_now*W + x_now
                pos2 = y_now*W + x_now + dx
                # print(f"2 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                TranTraffic[pos1][pos2] = TranTraffic[pos1][pos2] + enNode.traffic
                TranSpineFlitNum[pos1][pos2] = TranSpineFlitNum[pos1][pos2] + enNode.SpineflitNum
                TranOccupy[pos1][pos2] = TranOccupy[pos1][pos2] + enNode.occupy
                # 找到对应的下一个节点，更新其traffic
                for enNodeForupdate in encloseNodes:
                    if enNodeForupdate.node.Id == nextNode.Id:
                        enNodeForupdate.traffic = enNodeForupdate.traffic + enNode.traffic
                        enNodeForupdate.SpineflitNum = enNodeForupdate.SpineflitNum + enNode.SpineflitNum
                        enNodeForupdate.occupy = enNodeForupdate.occupy + enNode.occupy                        
                        break
            else: # 如果x和y都没有到达位置，那么x和y都要传输，需要更新两个结点以及两个NoC Links
                # 首先进行y轴的传输
                nextNode = MeshMapping[y_now+dy][x_now]
                # 更新traffic邻接矩阵里面的值
                pos1 = y_now*W + x_now
                pos2 = (y_now+dy)*W + x_now
                # print(f"3 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                # 如果是在起始位置：
                if x_now == x1 and y_now == y1:
                    y_allocate = NodeAllocation[pos1][0 if dy >= 0 else 1]
                    x_allocate = NodeAllocation[pos1][2 if dx >= 0 else 3]
                    # 归一化
                    if y_allocate + x_allocate > 0.0:
                        y_allocate_tmp = y_allocate/(y_allocate + x_allocate)
                        x_allocate_tmp = x_allocate/(y_allocate + x_allocate)
                        y_allocate = y_allocate_tmp + 0.0
                        x_allocate = x_allocate_tmp + 0.0
                    else:
                        y_allocate = 0.5
                        x_allocate = 0.5
                else:
                    y_allocate = 0.5
                    x_allocate = 0.5

                TranTraffic[pos1][pos2] = TranTraffic[pos1][pos2] + enNode.traffic * y_allocate
                TranSpineFlitNum[pos1][pos2] = TranSpineFlitNum[pos1][pos2] + enNode.SpineflitNum * y_allocate           
                TranOccupy[pos1][pos2] = TranOccupy[pos1][pos2] + enNode.occupy * y_allocate    
                # 找到对应的下一个节点，更新其traffic
                for enNodeForupdate in encloseNodes:
                    if enNodeForupdate.node.Id == nextNode.Id:
                        enNodeForupdate.traffic = enNodeForupdate.traffic + enNode.traffic * y_allocate
                        enNodeForupdate.SpineflitNum = enNodeForupdate.SpineflitNum + enNode.SpineflitNum * y_allocate
                        enNodeForupdate.occupy = enNodeForupdate.occupy + enNode.occupy * y_allocate
                        break
                nextNode = MeshMapping[y_now][x_now+dx]
                # 然后进行x轴的传输
                # 更新traffic邻接矩阵里面的值
                pos2 = y_now*W + x_now + dx
                # print(f"4 : (x_now,y_now)=({x_now},{y_now}),(x1,y1)=({x1},{y1}),(x2,y2)=({x2},{y2}),(dx,dy)=({dx},{dy}),(pos1,pos2)=({pos1},{pos2})")
                TranTraffic[pos1][pos2] = TranTraffic[pos1][pos2] + enNode.traffic * x_allocate
                TranSpineFlitNum[pos1][pos2] = TranSpineFlitNum[pos1][pos2] + enNode.SpineflitNum * x_allocate
                TranOccupy[pos1][pos2] = TranOccupy[pos1][pos2] + enNode.occupy * x_allocate    
                # 找到对应的下一个节点，更新其traffic
                for enNodeForupdate in encloseNodes:
                    if enNodeForupdate.node.Id == nextNode.Id:
                        enNodeForupdate.traffic = enNodeForupdate.traffic + enNode.traffic * x_allocate
                        enNodeForupdate.SpineflitNum = enNodeForupdate.SpineflitNum + enNode.SpineflitNum * x_allocate
                        enNodeForupdate.occupy = enNodeForupdate.occupy + enNode.occupy * x_allocate
                        break
    # 输出Traffic邻接矩阵
    # print("==============================================")
    # for i in range(H*W):
    #     print(TranTraffic[i])
        
    # # 输出SpineFlitNum邻接矩阵
    # print("==============================================")
    # for i in range(H*W):
    #     print(TranSpineFlitNum[i])
        
    latency = cal_Latency(NodeAllocation=NodeAllocation,TranOccupy=TranOccupy,EdgeList=EdgeList,NodePosition=NodePosition, MeshMapping=MeshMapping)
    # 考虑workload balance，使用TranSpineFlitNum里面非零元素的方差进行考虑
    maximum_bandwidth = 0.0
    for i in range(H*W):
        for j in range(i, H*W):
            maximum_bandwidth = max(maximum_bandwidth,TranOccupy[i][j]+TranOccupy[j][i])
    

    NoC_Link_TranTraffic = []        
    for i in range(H*W):
        for j in range(i+1, H*W):
            Traffic_per_NoC = (TranTraffic[i][j] + TranTraffic[j][i])/256
            if Traffic_per_NoC > 0:
                NoC_Link_TranTraffic.append(Traffic_per_NoC)
    mean = torch.tensor(NoC_Link_TranTraffic).mean()
    std = torch.tensor(NoC_Link_TranTraffic).std()
    
    return maximum_bandwidth*2 + (mean+std*0.5) * 1e-5, latency, TranTraffic, TranSpineFlitNum, TranOccupy


def format_number(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.1f}"
#输入邻接矩阵输出echarts代码
def output_echart(adjustMatric, MeshMapping, name):
    
    from hilbert_mapping import hilbert_mapping6_6
    # 定义 option 的头
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
    # 先输出node信息
    H = len(MeshMapping)
    W = len(MeshMapping[0])
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):    
            node = MeshMapping[y][x]
            if node.Id.count("Null") == 0:
                option["series"][0]["data"].append({
                    "name": hilbert_mapping6_6.index((y,x)),
                    "x": x*100,
                    "y": y*100,
                    "symbolSize": 50,
                    "itemStyle": {
                        "color": "blue"
                    },
                    "tooltip": {
                        "show": "true",
                        "formatter": node.Id
                    }    
                })
            else:
                option["series"][0]["data"].append({
                    "name": hilbert_mapping6_6.index((y,x)),
                    "x": x*100,
                    "y": y*100,
                    "symbolSize": 50,
                    "itemStyle": {
                        "color": "grey"
                    }
                })
    # 然后输出edge信息：
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):
            pos1 = y*W + x
            for pos2 in range(H*W):
                if adjustMatric[pos1][pos2] > 0:
                    option["series"][0]["links"].append({
                    "source": pos1,
                    "target": pos2,
                    "label": {
                        "show": "true",
                        "formatter": format_number(adjustMatric[pos1][pos2])
                    },
                    "lineStyle": {
                        "normal": {
                            "curveness": 0.1
                        }
                    }
                    })
    with open(f'{name}_data.json', 'w') as f:
        json.dump(option, f)

if __name__ == "__main__":
    # 先导入mapping结果以及连接信息
    MeshMapping = torch.load("MeshMapping.pth")
    EdgeList = torch.load("EdgeList.pth")
    # 定义邻接矩阵来存储NoC Link上的传输量
    # 先定义NodeAllocation均是0.5
    NodeAllocation = []
    for i in range(H*W):
        NodeAllocation.append([0.5,0.5,0.5,0.5])

    # 确定latency和分配结果
    var_balance ,latency, TranTraffic, TranSpineFlitNum, TranOccupy = defineTraffic(MeshMapping, EdgeList, NodeAllocation)

    # print(TranTraffic)
    print(f"var_balance_opt={var_balance},latency={latency}, traffic={format_number(torch.tensor(TranTraffic).sum().item()/8)}B")

    # 生成echarts的option
    output_echart(TranTraffic, MeshMapping, "TranTraffic")
    output_echart(TranSpineFlitNum, MeshMapping, "TranSpineFlitNum")
    output_echart(TranOccupy, MeshMapping, "TranOccupy")

    # for edge in EdgeList:
    #     print(edge.transmitTime)

    # 保存最后结果给实际NoC使用
    torch.save(MeshMapping,"Final_MeshMapping.pth")
    torch.save(TranOccupy,"TranOccupy.pth")