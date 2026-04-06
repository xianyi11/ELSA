import random
import math
import torch
from oliviours_routing import defineTraffic, output_echart, format_number
from partition import Node, Edge

# ===================
#  1. 定义辅助函数
# ===================
def normalize_individual(individual):
    """
    将长度为144的一维向量，以每4个元素为一组进行归一化，使得每行元素之和为1.
    individual: list[float], 长度为144
    """
    new_ind = individual[:]
    for row_idx in range(36):
        start = row_idx * 4
        end = start + 4
        row_sum = sum(new_ind[start:end])
        if row_sum == 0:
            # 如果初始化或变异之后出现某一行全0，做一个平分处理
            for i in range(start, end):
                new_ind[i] = 1.0/4
        else:
            for i in range(start, end):
                new_ind[i] /= row_sum
    return new_ind

def create_individual():
    """
    随机创建一个长度为144的一维向量，然后做归一化
    """
    ind = [random.random() for _ in range(36*4)]
    ind = normalize_individual(ind)
    return ind

def crossover(ind1, ind2, prob=0.5):
    """
    简单的单点交叉或者逐位交叉(这里演示逐位交叉)，并保持长度为144
    prob: 每个位置交换基因的概率
    """
    child1, child2 = ind1[:], ind2[:]
    for i in range(len(ind1)):
        if random.random() < prob:
            child1[i], child2[i] = child2[i], child1[i]
    # 交叉后需要归一化
    child1 = normalize_individual(child1)
    child2 = normalize_individual(child2)
    return child1, child2

def mutate(ind, mutation_rate=0.01, mutation_strength=0.1):
    """
    以一定概率对每个基因做微小扰动，然后归一化
    mutation_rate: 每个基因被选中变异的概率
    mutation_strength: 扰动的幅度，可根据需要调整
    """
    new_ind = ind[:]
    for i in range(len(new_ind)):
        if random.random() < mutation_rate:
            # 在[-mutation_strength, mutation_strength]区间内随机扰动
            delta = (random.random() * 2 - 1) * mutation_strength
            new_ind[i] += delta
            # 避免出现负数
            if new_ind[i] < 0:
                new_ind[i] = 0
    new_ind = normalize_individual(new_ind)
    return new_ind

# ===================
#  2. 定义核心优化流程
# ===================

def evaluate(individual, MeshMapping, EdgeList):
    """
    计算该个体的目标函数(latency).
    individual: 长度为144的一维向量
    MeshMapping: 外部已定义好的网格拓扑信息
    """
    # reshape 成 (36 x 4) 矩阵
    NodeAllocation = []
    for row_idx in range(36):
        start = row_idx * 4
        end = start + 4
        NodeAllocation.append(individual[start:end])
    
    # 重置occupy
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):
            MeshMapping[y][x].occupy = 0

    for edge in EdgeList:
        edge.transmitTime = 0
                
    var_balance,latency,TranTraffic,_,_ = defineTraffic(MeshMapping,EdgeList,NodeAllocation)
    metric = var_balance
    # + torch.tensor(TranTraffic).sum().item()/(1024*1024*8*250)
    return metric

def genetic_algorithm_optimization(MeshMapping,
                                   EdgeList,
                                   pop_size=50,
                                   n_generations=100,
                                   crossover_prob=0.7,
                                   mutation_rate=0.01):
    """
    用简单的遗传算法对 defineTraffic 的输入 NodeAllocation 进行寻优
    """
    # 1) 初始化种群
    population = [create_individual() for _ in range(pop_size)]

    best_ind = None
    best_fit = float('inf')

    # 2) 迭代进化
    for gen in range(n_generations):
        # 2.1) 计算适应度
        fitnesses = []
        for ind in population:
            f = evaluate(ind, MeshMapping, EdgeList)
            fitnesses.append(f)

        # 记录当代最优
        for i, f in enumerate(fitnesses):
            if f < best_fit:
                best_fit = f
                best_ind = population[i][:]

        print(f"Generation {gen}, best latency so far: {best_fit}")

        # 2.2) 选择 (这里用简单的锦标赛方式演示)
        new_population = []
        for _ in range(pop_size // 2):
            # 随机选两个个体对比，挑适应度好的那个
            cand1 = random.randint(0, pop_size-1)
            cand2 = random.randint(0, pop_size-1)
            if fitnesses[cand1] < fitnesses[cand2]:
                parent1 = population[cand1]
            else:
                parent1 = population[cand2]

            cand3 = random.randint(0, pop_size-1)
            cand4 = random.randint(0, pop_size-1)
            if fitnesses[cand3] < fitnesses[cand4]:
                parent2 = population[cand3]
            else:
                parent2 = population[cand4]

            # 2.3) 交叉
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # 2.4) 变异
            child1 = mutate(child1, mutation_rate=mutation_rate)
            child2 = mutate(child2, mutation_rate=mutation_rate)

            # 放回新种群
            new_population.append(child1)
            new_population.append(child2)

        # 替换旧种群
        population = new_population

    return best_ind, best_fit

# ===================
#  3. 主函数调用示例
# ===================
if __name__ == "__main__":
    # 你需要提前定义好 MeshMapping 或者其他 needed data
    MeshMapping = torch.load("MeshMapping.pth")
    EdgeList = torch.load("EdgeList.pth")

    best_individual, best_latency = genetic_algorithm_optimization(MeshMapping,
                                                                   EdgeList,
                                                                   pop_size=50,
                                                                   n_generations=8000,
                                                                   crossover_prob=0.7,
                                                                   mutation_rate=0.1)
    print("========================================")
    print("最优个体对应的 latency:", best_latency)
    print("最优个体 (长度=144):", best_individual)
    # 如果需要查看其对应的 NodeAllocation 矩阵：
    best_NodeAllocation = []
    for row_idx in range(36):
        start = row_idx * 4
        end = start + 4
        best_NodeAllocation.append(best_individual[start:end])
    print("对应的 36x4 NodeAllocation 矩阵:")
    for row in best_NodeAllocation:
        print(row)

    # 重置occupy
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):
            MeshMapping[y][x].occupy = 0
    # 如果还需要再次调用 defineTraffic 来查看该分配矩阵下的详细结果
    # 重置occupy
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):
            MeshMapping[y][x].occupy = 0

    for edge in EdgeList:
        edge.transmitTime = 0
                    
    metric_opt, latency, TranTraffic_opt, TranSpineFlitNum_opt, TranOccupy_opt = defineTraffic(MeshMapping, EdgeList, best_NodeAllocation)
    print(f"var_balance_opt={metric_opt},latency={latency}, traffic={format_number(torch.tensor(TranTraffic_opt).sum().item()/8)}B")
    # 你可以在此处输出或处理 TranTraffic_opt, TranSpineFlitNum_opt, TranOccupy_opt
    output_echart(TranTraffic_opt, MeshMapping, "opt_output/TranTraffic_opt")
    output_echart(TranSpineFlitNum_opt, MeshMapping, "opt_output/TranSpineFlitNum_opt")
    output_echart(TranOccupy_opt, MeshMapping, "opt_output/TranOccupy_opt")
    
    # 生成最后的mapping以及TranOccupy
    W = 6
    H = 6
    for y in range(len(MeshMapping)):
        for x in range(len(MeshMapping[y])):
            pos = y*W + x
            MeshMapping[y][x].Allocates = best_NodeAllocation[pos]
    
    print("TranOccupy_opt",TranOccupy_opt)
    
    torch.save(MeshMapping,"Final_MeshMapping.pth")
    torch.save(TranOccupy_opt,"TranOccupy.pth")
    
    maximum_bandwidth = 0.0
    for i in range(H*W):
        for j in range(i, H*W):
            maximum_bandwidth = max(maximum_bandwidth,TranOccupy_opt[i][j]+TranOccupy_opt[j][i])
    print("maximum_bandwidth",maximum_bandwidth)
    
    NoC_Link_TranTraffic = []        
    for i in range(H*W):
        for j in range(i+1, H*W):
            Traffic_per_NoC = (TranTraffic_opt[i][j] + TranTraffic_opt[j][i])/256
            if Traffic_per_NoC > 0:
                NoC_Link_TranTraffic.append(Traffic_per_NoC)
    print(NoC_Link_TranTraffic)
    print("var_balance",torch.tensor(NoC_Link_TranTraffic).var())
    print("var_average",torch.tensor(NoC_Link_TranTraffic).mean())
    print(f"var_balance_opt={metric_opt},latency={latency}, traffic={format_number(torch.tensor(TranTraffic_opt).sum().item()/8)}B")

