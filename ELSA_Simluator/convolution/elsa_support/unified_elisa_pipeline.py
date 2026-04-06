"""
Shared ELSA convolution+NoC simulation loop (from ELSACompiler_ResNet50).
Uses VSACompiler and helpers from ELSACompiler_ResNet50 so all models share one implementation.
"""
from __future__ import annotations

import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from tqdm import tqdm

from PETile.Tile import Tile
from processElement.STBIFFunction import STBIFNeuron
from PETile.FlitGenerator import FlitGenerator, FlitCombiner
from router.Switch import SwitchCrossbar
from router.RouteComputer import RouteComputer

from Compilers.ELSACompiler_ResNet50 import (
    VSACompiler,
    get_Receptive_Field,
    getRouterBreakdown,
    getPEBreakdown,
)


def run_unified_elisa_pipeline(
    datapath,
    connectionpath,
    mappingpath,
    occupyPath,
    config_path=None,
    time_step=24,
    output_dir=None,
    *,
    search_input_simple: str,
    calculate_info_pth: str,
    communication_traffic_pth: str,
    column_finish_pth: str,
    search_input_final: str,
    use_input_time_steps: bool = False,
):
    from elsa_support.paths import CONV_ROOT, CONFIG_YAML
    if config_path is None:
        config_path = str(CONFIG_YAML)
    if output_dir is None:
        output_dir = str(CONV_ROOT / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    compiler = VSACompiler(
        datapath, connectionpath, mappingpath, occupyPath, config_path=config_path
    )
    Time_step = time_step
    calculateInfo = {}
    totalCycle = 0
    columnFinishCycle = {}
    columnFinishCycleLast = {}
    columnFinishCyclePerlayer = []
    firstLayer = True
    overall_sprasity = 0.0
    inputNum = 0
    nonzeroNum = 0
    for layerIndex, layerParam in enumerate(compiler.layers):
        if layerParam.type == "conv" or layerParam.type == "linear":
            nonzeroNum = nonzeroNum + layerParam.input.abs().sum()
            inputNum = inputNum + layerParam.input.numel()

    get_Receptive_Field(compiler = compiler)
    SearchInput = {}
    SearchInput["FlitNumberEachLayer"] = compiler.FlitNumberEachLayer
    SearchInput["SpikeNumberEachLayer"] = compiler.SpikeNumberEachLayer
    SearchInput["WidthAfterTiling"] = compiler.WidthAfterTiling
    SearchInput["SRAMNumPerLayer"] = compiler.SRAMNumPerLayer
    SearchInput["VSAWidth"] = compiler.VSAWidth
    SearchInput["totalSRAMNum"] = compiler.totalSRAMNum
    SearchInput["receptiveField"] = compiler.receptiveField
    print(SearchInput["receptiveField"])
    torch.save(SearchInput, os.path.join(output_dir, search_input_simple))

    print("average sparsity:",1 - nonzeroNum/inputNum)
    for layerIndex, layerParam in enumerate(compiler.layers):
        print(layerIndex, layerParam.name)
        # if layerIndex < 1:
        #     firstLayer = False
        #     continue
        if layerParam.type == "conv":
            layerParam.printmyself()
            # print("columnFinishCycle",columnFinishCycle)
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}
            layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)
            M1 = layerParam.M
            N1 = layerParam.N
            bias = layerParam.bias
            weight = layerParam.weight
            input = layerParam.input
            groudtruth = layerParam.output
            neuron = STBIFNeuron(M=M1,N=N1,pos_max=15,neg_min=0,bias=bias.unsqueeze(1))    
            # Legacy VGG16_CIFAR10 uses T = layer timesteps from exported tensors, not CLI --Time_step.
            T = int(layerParam.input.shape[0]) if use_input_time_steps else Time_step
            outputList = []
            stride = layerParam.input.shape[-1]//layerParam.output.shape[-1]
            KW = layerParam.weight.shape[-1]
            K = layerParam.weight.shape[1]*layerParam.weight.shape[2]*layerParam.weight.shape[3]
            firstConv = None
        
            for t in range(T):    
                wx = torch.nn.functional.conv2d(input[t], weight, stride=stride, padding=KW//2)
                if t == 0:
                    firstConv = (wx*M1).reshape(wx.shape[1],-1) + 2**(N1-1) + bias.unsqueeze(1)*(2**(N1))
                outputList.append(neuron(wx)+0)
        
        
            # print(firstConv[:,0])
            # print(outputList[0].reshape(outputList[0].shape[1],-1)[:,0])
            output1 = torch.stack(outputList,dim=0)
        
            # print(output1.shape,groudtruth.shape)
            if use_input_time_steps:
                print(torch.sum(output1 - groudtruth) / groudtruth.numel())
            else:
                print(torch.sum(output1 - groudtruth[:T]) / groudtruth[:T].numel())
            # assert (output1 == groudtruth).all(), "output1 != groudtruth"
        
            TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=firstLayer) #move to the east router

            output1 = output1.reshape(output1.shape[0],output1.shape[1],output1.shape[2],-1)


            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            flitCombiner = FlitCombiner()
            CrossbarSwitch = SwitchCrossbar()
            RouteEngine = RouteComputer()
            InputFlitNum = 0
            OutputFlitNum = 0
            OOutputFlitNum = 0
            TrueOutputFlitNum = 0
        
            perTimeStepCycle = []
            perTimeStepCyclefine = []
            perLayerLatencyCycle = []
            perLayerBeginCylce = []
            spikeNumBeforeImg2ColPerTimeStep = []
            spikeNumAfterImg2ColPerTimeStep = []
        
            outSpikesList = []
            inputTime = 0
            compiler.NoC.flitNumber = 0
            # compiler.NoC.reset_router_busy_until()  # 本层内下一 spine 到达时若 router 仍忙则等待
            for t in tqdm(range(T)):
                # if t > 0:
                #     break
                columnFinishCycle[t] = {}
                H,W = input.shape[-2],input.shape[-1]
                input2D = input[t].reshape(input[t].shape[1],-1)
                r = c = 0
                outSpikes1 = torch.zeros(output1.shape[2],output1.shape[3])
                InputRows = input2D.shape[0]
                InputColumns = input2D.shape[1]
                spikeNum = 0
                maxOutcycle = 0
                curFlitCycle = 0
                mFirst = True
                for m in tqdm(range(InputColumns)):
                    # print("len(columnFinishCycleLast)",len(columnFinishCycleLast),"len(columnFinishCycle)",len(columnFinishCycle))
                    # print(columnFinishCycle)
                    # if m > 100:
                    #     break
                    # print("r,c",r,c)
                    columnId = r*W+c
                    # if (input2D[:,columnId] == 0).all():
                    #     r,c = TILE.VSAOrderCtrl(r,c)
                    #     continue                    
                    # print("input2D[:,columnId]",input2D[:,columnId])
                    flits = flitGenerator(spikes=input2D[:,columnId], columnId=columnId)
                    # print("columnId:",m,len(flits))
                    curFlitNubmer = 0 #这一个spine需要多少个flit
                    for flit in flits:
                        spikeNum = spikeNum + len(flit.Payload.rowId)
                        if len(flit.Payload.rowId) > 0:
                            InputFlitNum = InputFlitNum + 1
                            curFlitNubmer = curFlitNubmer + 1 #更新flit

                    SpineInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=columnId, timestep=t, flitNumber=curFlitNubmer)
                    # 为每一个flit更新flit的输入时间，通过NOC确定
                    for flit in flits:
                        # 输入层，每个cycle输入一个flit
                        if layerIndex == 0:
                            flit.time = inputTime + 0
                            if len(flit.Payload.rowId) != 0:
                                inputTime = inputTime + 1                            
                        else:
                            # 中间层，按照NOC计算出每个flit的输入时间
                            flit.time = SpineInputTime
                    TILE.flitCombiner.tailNum = TILE.PENum - 1
                    output = TILE(flits) # list[Flit]
                    if output is None:
                        r,c = TILE.VSAOrderCtrl(r,c)
                        continue
                
                    if mFirst:
                        if len(perTimeStepCycle) == 0:
                            perLayerBeginCylce.append(flits[0].time)
                        else:
                            perLayerBeginCylce.append(max(flits[0].time, perTimeStepCycle[-1]))
                        mFirst = False

                    outFlitsList, outcycle = output
                
                
                    maxLen = 0
                    for peid in range(TILE.PENum):
                        maxLen = max(maxLen,len(outFlitsList[peid]))
                
                    for i in range(maxLen):
                    
                        for peid in range(TILE.PENum):
                            output = flitCombiner(outFlitsList[peid][i])
                
                        if output is None:
                            r,c = TILE.VSAOrderCtrl(r,c)
                            continue

                        outSpikes, columnID, maxcycle = output

                        if columnID not in columnFinishCycle[t].keys():
                            columnFinishCycle[t][columnID] = maxcycle
                        else:
                            columnFinishCycle[t][columnID] = max(columnFinishCycle[t][columnID],maxcycle)

                        curFlitCycle = maxcycle + 0.0
                    
                        maxOutcycle = max(maxOutcycle,maxcycle)
                    
                        # print("columnID",columnID)
                        for spike in outSpikes:
                            row_id, column_id, sign = spike
                            outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1

                        flits = flitGenerator(spikes=outSpikes1[:,columnID], columnId=columnID)
                        for flit in flits:
                            if len(flit.Payload.rowId) > 0:
                                TrueOutputFlitNum = TrueOutputFlitNum + 1
                    
                    
                    # print("output flit Time",outFlitsList[0][0][0].time, "out columnID", columnID)
                    # print("column_id",column_id)
                
                    r,c = TILE.VSAOrderCtrl(r,c)

                outSpikesList.append(outSpikes1)    
                perTimeStepCycle.append(maxOutcycle)
                perLayerLatencyCycle.append(TILE.PEs[0].computationCycle)
                spikeNumAfterImg2ColPerTimeStep.append(TILE.spikeNumAfterImg2Col)
                spikeNumBeforeImg2ColPerTimeStep.append(TILE.spikeNum)
            
                # print(f"Timestep{t}: columnFinishCycle",columnFinishCycle[t])
                print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum(), "inputNum:",TILE.spikeNum)
                print("In TILE spikeNum",TILE.spikeNum,"spikeNumAfterImg2Col",TILE.spikeNumAfterImg2Col)

            outSpikesList = torch.stack(outSpikesList,dim=0)
            # print(outSpikesList.shape)
            # print(output1.shape)
            # print(firstConv[:,109])
            # print(outSpikesList[0][:,109])
            # print(output1[0,0,:,109])
            print("InputFlitNum",InputFlitNum,"TrueOutputFlitNum",TrueOutputFlitNum,"InputFlitNumberNoC",compiler.NoC.flitNumber, "perLayerLatencyCycle",perLayerLatencyCycle)
            compiler.FlitNumberEachLayer[layerParam.name] = OutputFlitNum
            compiler.SpikeNumberEachLayer[layerParam.name] = TILE.spikeNumAfterImg2Col
        
            # if i == 0:
            # if totalCycle == 0:
                # totalCycle = perTimeStepCycle[-1]
            for keyT in columnFinishCycle.keys():
                perTimeStepCyclefine.append(0)

            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    perTimeStepCyclefine[keyT] = max(perTimeStepCyclefine[keyT], columnFinishCycle[keyT][key])
        
            totalCycle = perTimeStepCyclefine[-1]
            # spine-level pipeline metrics (ResNet-style); legacy VGG16_CIFAR10 omits these.
            if not use_input_time_steps:
                effective_latency = max(0, totalCycle - (TILE.firstSpineArrivalTime or 0))
                spine_bubble_ratio = (TILE.totalBubbleCycle / effective_latency) if effective_latency > 0 else 0.0
                spine_compute_ratio = (TILE.totalComputeCycle / effective_latency) if effective_latency > 0 else 0.0
            else:
                effective_latency = spine_bubble_ratio = spine_compute_ratio = 0.0

            # 更新NoC中下一层的每一个输入时间
            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)
            # 记录NoC中的一个spine的平均输出flitnum
            compiler.avgFlitNumEachLayer[layerParam.name] = InputFlitNum/(layerParam.input.shape[-1]*layerParam.input.shape[-2]) * math.ceil(math.sqrt(compiler.receptiveField[layerParam.name])) * (math.ceil(layerParam.input.shape[-1]/layerParam.output.shape[-1]))
            print("OutputFlitNum",OutputFlitNum,"M ",(layerParam.output.shape[-1]*layerParam.output.shape[-2]),"receptiveField",  math.ceil(math.sqrt(compiler.receptiveField[layerParam.name])) * (math.ceil(layerParam.input.shape[-1]/layerParam.output.shape[-1])),"avgFlitNumEachLayer",compiler.avgFlitNumEachLayer[layerParam.name])
            # 拿该层的输出算pooling的输入结果
            if compiler.layers[layerIndex + 1].name.count("maxpool") > 0:
                PoolingParam = compiler.layers[layerIndex + 1]
                compiler.avgFlitNumEachLayer[PoolingParam.name] = TrueOutputFlitNum/(PoolingParam.output.shape[-1]*PoolingParam.output.shape[-2]) * math.ceil(math.sqrt(compiler.receptiveField[PoolingParam.name])) * (math.ceil(PoolingParam.input.shape[-1]/PoolingParam.output.shape[-1]))
            

            print("point accuracy:",torch.sum(outSpikesList[0] == output1[0,0,:,:])/output1[0,0,:,:].numel())
            calculateInfo[layerParam.name] = {}
            calculateInfo[layerParam.name]["point accuracy"] = torch.sum(outSpikesList[0] == output1[0,0,:,:])/output1[0,0,:,:].numel()
            calculateInfo[layerParam.name]["InputFlitNum"] = InputFlitNum
            calculateInfo[layerParam.name]["outputFlitNum"] = TrueOutputFlitNum
            calculateInfo[layerParam.name]["perTimeStepCycle"] = perTimeStepCycle
            calculateInfo[layerParam.name]["perTimeStepCyclefine"] = perTimeStepCyclefine
            calculateInfo[layerParam.name]["perLayerLatencyCycle"] = perLayerLatencyCycle
            calculateInfo[layerParam.name]["perLayerBeginCylce"] = perLayerBeginCylce
            calculateInfo[layerParam.name]["numbers of operation"] = layerParam.ops
            calculateInfo[layerParam.name]["numbers of spikes"] = TILE.spikeNum
            calculateInfo[layerParam.name]["numbers of spikeNumAfterImg2Col"] = spikeNumAfterImg2ColPerTimeStep
            calculateInfo[layerParam.name]["numbers of spikeNumBeforeImg2Col"] = spikeNumBeforeImg2ColPerTimeStep
            calculateInfo[layerParam.name]["one spike operation"] = TILE.spikeNumAfterImg2Col*TILE.N
            calculateInfo[layerParam.name]["sparsity"] = layerParam.inputSparsity
            calculateInfo[layerParam.name]["TileN"] = TILE.tileN
            if not use_input_time_steps:
                calculateInfo[layerParam.name]["spine_bubble_cycles"] = TILE.totalBubbleCycle
                calculateInfo[layerParam.name]["spine_total_compute_cycles"] = TILE.totalComputeCycle
                calculateInfo[layerParam.name]["spine_bubble_ratio"] = spine_bubble_ratio
                calculateInfo[layerParam.name]["spine_compute_ratio"] = spine_compute_ratio
                calculateInfo[layerParam.name]["spine_effective_latency"] = effective_latency
                calculateInfo[layerParam.name]["spine_first_arrival_cycle"] = TILE.firstSpineArrivalTime

            CrossbarSwitch.fcount = OutputFlitNum
            RouteEngine.fcount = OutputFlitNum

            calculateInfo = getPEBreakdown(calculateInfo,TILE.PEs,totalCycle,layerParam.name,firstLayer)
            calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name)
                        # begin to calculate PE total area/power
            totalArea = 0.0
            totalEnergy = 0.0
        
            tileArea = TILE.getArea()
            tileEnergy = TILE.calEnergy(totalCycle/200000000)            
        
            totalArea = tileArea + CrossbarSwitch.getArea() + RouteEngine.getArea()
            totalEnergy = tileEnergy + CrossbarSwitch.calEnergy(totalCycle/200000000) + RouteEngine.calEnergy(totalCycle/200000000)
            calculateInfo[layerParam.name]["Total Area"] = totalArea
            calculateInfo[layerParam.name]["Total Energy"] = totalEnergy
            calculateInfo[layerParam.name]["Total Latency"] = totalCycle
            if not use_input_time_steps:
                print(f"[spine pipeline] {layerParam.name.strip()}: bubble_ratio={spine_bubble_ratio:.4f}, compute_ratio={spine_compute_ratio:.4f}, bubble_cycles={TILE.totalBubbleCycle}, first_arrival_cycle={TILE.firstSpineArrivalTime}, effective_latency={effective_latency}, total_latency={totalCycle}")
            print(calculateInfo[layerParam.name])
            firstLayer = False
            # break
        elif layerParam.type == "linear":
            layerParam.printmyself()
            perLayerLatencyCycle = []
            perLayerBeginCylce = []
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}
            layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)
            M1 = layerParam.M
            N1 = layerParam.N
            bias = layerParam.bias
            weight = layerParam.weight
            input = layerParam.input
            groudtruth = layerParam.output
            neuron = STBIFNeuron(M=M1,N=N1,pos_max=15,neg_min=0,bias=bias.unsqueeze(0))    
            T = int(layerParam.input.shape[0]) if use_input_time_steps else Time_step
            outputList = []
            firstLinear = None

        
            for t in range(T):    
                wx = torch.nn.functional.linear(input[t], weight)
                if t == 0:
                    firstLinear = (wx*M1).reshape(wx.shape[1],-1) + 2**(N1-1) + bias.unsqueeze(0)*(2**(N1))
                outputList.append(neuron(wx)+0)
        
        
            # print(firstConv[:,0])
            # print(outputList[0].reshape(outputList[0].shape[1],-1)[:,0])
            output1 = torch.stack(outputList,dim=0)
        
            # print(torch.sum(output1-groudtruth)/groudtruth.numel())
            # assert (output1 == groudtruth).all(), "output1 != groudtruth"
        
            TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=firstLayer) #move to the east router

            output1 = output1.reshape(output1.shape[0],output1.shape[1],output1.shape[2],-1)

            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            flitCombiner = FlitCombiner()
            CrossbarSwitch = SwitchCrossbar()
            RouteEngine = RouteComputer()
            InputFlitNum = 0
            OutputFlitNum = 0
            TrueOutputFlitNum = 0
            perTimeStepCycle = []
            spikeNumBeforeImg2ColPerTimeStep = []
            spikeNumAfterImg2ColPerTimeStep = []        
        
            outSpikesList = []
            compiler.NoC.flitNumber = 0
            # compiler.NoC.reset_router_busy_until()  # 本层内下一 spine 到达时若 router 仍忙则等待
            for t in tqdm(range(T)):
                # if t > 0:
                #     break
                H,W = input.shape[-2],input.shape[-1]
                input2D = input[t].reshape(input[t].shape[1],-1)
                r = c = 0
                outSpikes1 = torch.zeros(output1.shape[2],output1.shape[3])
                InputRows = input2D.shape[0]
                InputColumns = input2D.shape[1]
                spikeNum = 0
                maxOutcycle = 0
                curFlitCycle = 0
            
                for m in tqdm(range(InputColumns)):
                    # if m > 100:
                    #     break
                    # print("r,c",r,c)
                    columnId = m
                    # if (input2D[:,columnId] == 0).all():
                    #     r,c = TILE.VSAOrderCtrl(r,c)
                    #     continue                    
                
                    flits = flitGenerator(spikes=input2D[:,columnId], columnId=columnId)
                    curFlitNubmer = 0
                    for flit in flits:
                        if len(flit.Payload.rowId) > 0:
                            InputFlitNum = InputFlitNum + 1
                            curFlitNubmer = curFlitNubmer + 1
                        spikeNum = spikeNum + len(flit.Payload.rowId)

                    SpineInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=columnId, timestep=t, flitNumber=curFlitNubmer)
                    # 为每一个flit更新flit的输入时间，通过NOC确定
                    for flit in flits:
                        # 输入层，每个cycle输入一个flit
                        if layerIndex == 0:
                            flit.time = inputTime + 0
                            if len(flit.Payload.rowId) != 0:
                                inputTime = inputTime + 1                            
                        else:
                            # 中间层，按照NOC计算出每个flit的输入时间
                            flit.time = SpineInputTime
                                    
                    TILE.flitCombiner.tailNum = TILE.PENum - 1
                    output = TILE(flits) # list[Flit]
                                    
                    outFlitsList, outcycle = output
                
                    maxLen = 0
                    # for peid in range(TILE.PENum):
                    #     maxLen = max(maxLen,len(outFlitsList[peid]))
                
                    for peid in range(TILE.PENum):
                        OutputFlitNum = OutputFlitNum + len(outFlitsList[peid])
                        output = flitCombiner(outFlitsList[peid])
            
                    outSpikes, columnID, maxcycle = output
                                    
                    curFlitCycle = maxcycle + 0.0

                    maxOutcycle = max(maxOutcycle,maxcycle)
                        
                    for spike in outSpikes:
                        row_id, column_id, sign = spike
                        outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + 1 if sign == 0 else -1

                    # print("column_id",column_id)
                    flits = flitGenerator(spikes=outSpikes1[:,columnID], columnId=columnID)
                    for flit in flits:
                        if len(flit.Payload.rowId) > 0:
                            TrueOutputFlitNum = TrueOutputFlitNum + 1
                
                outSpikesList.append(outSpikes1)   
            
                lastlayermaxcycles = 0
                for key in columnFinishCycleLast[t].keys():
                    lastlayermaxcycles = max(lastlayermaxcycles, columnFinishCycleLast[t][key])
                # Legacy VGG16_CIFAR10 linear: perTimeStepCycle uses maxOutcycle only (no +lastlayermaxcycles).
                if use_input_time_steps:
                    columnFinishCycle[t] = {}
                    columnFinishCycle[t][0] = maxOutcycle
                    perTimeStepCycle.append(maxOutcycle)
                    perLayerBeginCylce.append(lastlayermaxcycles)
                else:
                    perTimeStepCycle.append(maxOutcycle + lastlayermaxcycles)
                    perLayerLatencyCycle.append(TILE.PEs[0].computationCycle)
                    columnFinishCycle[t] = {}
                    columnFinishCycle[t][0] = maxOutcycle + lastlayermaxcycles
                    perLayerBeginCylce.append(lastlayermaxcycles)

                spikeNumAfterImg2ColPerTimeStep.append(TILE.spikeNumAfterImg2Col)
                spikeNumBeforeImg2ColPerTimeStep.append(TILE.spikeNum)

                print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum(), "inputNum:",TILE.spikeNum)
                                
            outSpikesList = torch.stack(outSpikesList,dim=0)
            # print(outSpikesList.shape)
            # print(output1.shape)
            # print(firstConv[:,109])
            # print(outSpikesList[0][:,109])
            # print(output1[0,0,:,109])
            print("InputFlitNum",InputFlitNum,"outputFlitNum",TrueOutputFlitNum,"perTimeStepCycle",perTimeStepCycle)

            # 更新NoC中下一层的每一个输入时间
            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)
            # 记录NoC中的一个spine的平均输出flitnum
            compiler.avgFlitNumEachLayer[layerParam.name] = InputFlitNum
            # 拿下一个layer的输入算pooling的输出结果
            # if compiler.layers[layerIndex - 1].name.count("avgpool") > 0:
                # PoolingParam = compiler.layers[layerIndex - 1]
                # compiler.avgFlitNumEachLayer[PoolingParam.name] = InputFlitNum/(PoolingParam.output.shape[-1]*PoolingParam.output.shape[-2]) * math.ceil(math.sqrt(compiler.receptiveField[PoolingParam.name])) * (math.ceil(PoolingParam.input.shape[-1]/PoolingParam.output.shape[-1]))
            # 更新NoC中下一层的每一个输入时间
            compiler.SpikeNumberEachLayer[layerParam.name] = TILE.spikeNum
            compiler.FlitNumberEachLayer[layerParam.name] = InputFlitNum

            totalCycle = perTimeStepCycle[-1]
            if not use_input_time_steps:
                effective_latency_linear = max(0, totalCycle - (TILE.firstSpineArrivalTime or 0))
                spine_bubble_ratio_linear = (TILE.totalBubbleCycle / effective_latency_linear) if effective_latency_linear > 0 else 0.0
                spine_compute_ratio_linear = (TILE.totalComputeCycle / effective_latency_linear) if effective_latency_linear > 0 else 0.0
            else:
                effective_latency_linear = spine_bubble_ratio_linear = spine_compute_ratio_linear = 0.0

            print("point accuracy:",torch.sum(outSpikesList[0] == output1[0,0,:,:])/output1[0,0,:,:].numel())
        
            calculateInfo[layerParam.name] = {}
            calculateInfo[layerParam.name]["InputFlitNum"] = InputFlitNum
            calculateInfo[layerParam.name]["outputFlitNum"] = TrueOutputFlitNum
            calculateInfo[layerParam.name]["perTimeStepCycle"] = perTimeStepCycle
            calculateInfo[layerParam.name]["perLayerLatencyCycle"] = perLayerLatencyCycle
            calculateInfo[layerParam.name]["perLayerBeginCylce"] = perLayerBeginCylce
            calculateInfo[layerParam.name]["numbers of operation"] = layerParam.ops
            calculateInfo[layerParam.name]["numbers of spikes"] = TILE.spikeNum
            calculateInfo[layerParam.name]["one spike operation"] = TILE.spikeNum*TILE.N
            calculateInfo[layerParam.name]["numbers of spikeNumAfterImg2Col"] = spikeNumAfterImg2ColPerTimeStep
            calculateInfo[layerParam.name]["numbers of spikeNumBeforeImg2Col"] = spikeNumBeforeImg2ColPerTimeStep
            calculateInfo[layerParam.name]["sparsity"] = layerParam.inputSparsity
            calculateInfo[layerParam.name]["TileN"] = TILE.tileN
            if not use_input_time_steps:
                calculateInfo[layerParam.name]["spine_bubble_cycles"] = TILE.totalBubbleCycle
                calculateInfo[layerParam.name]["spine_total_compute_cycles"] = TILE.totalComputeCycle
                calculateInfo[layerParam.name]["spine_bubble_ratio"] = spine_bubble_ratio_linear
                calculateInfo[layerParam.name]["spine_compute_ratio"] = spine_compute_ratio_linear
                calculateInfo[layerParam.name]["spine_effective_latency"] = effective_latency_linear
                calculateInfo[layerParam.name]["spine_first_arrival_cycle"] = TILE.firstSpineArrivalTime
        
            CrossbarSwitch.fCount = OutputFlitNum
            RouteEngine.fCount = OutputFlitNum

            calculateInfo = getPEBreakdown(calculateInfo,TILE.PEs,totalCycle,layerParam.name,firstLayer)
            calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name,isConv=False)
                        # begin to calculate PE total area/power
            totalArea = 0.0
            totalEnergy = 0.0
        
            tileArea = TILE.getArea()
            tileEnergy = TILE.calEnergy(totalCycle/200000000)            
        
            totalArea = tileArea + CrossbarSwitch.getArea() + RouteEngine.getArea()
            totalEnergy = tileEnergy + CrossbarSwitch.calEnergy(totalCycle/200000000) + RouteEngine.calEnergy(totalCycle/200000000)
            calculateInfo[layerParam.name]["Total Area"] = totalArea
            calculateInfo[layerParam.name]["Total Energy"] = totalEnergy
            calculateInfo[layerParam.name]["Total Latency"] = totalCycle
            if not use_input_time_steps:
                print(f"[spine pipeline] {layerParam.name.strip()}: bubble_ratio={spine_bubble_ratio_linear:.4f}, compute_ratio={spine_compute_ratio_linear:.4f}, bubble_cycles={TILE.totalBubbleCycle}, effective_latency={effective_latency_linear}, total_latency={totalCycle}")
            firstLayer = False
            # torch.save(outSpikesList,"outSpikesList.pth")
            # torch.save(output1,"output1.pth")
        elif layerParam.type == "pool":
            print(layerParam.printmyself())
            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            InputFlitNum = 0
            input = layerParam.input
            H,W = input.shape[-2],input.shape[-1]
            output = layerParam.output
            stride = (input.shape[-1]//output.shape[-1])
            if layerParam.name.count("avgpool") > 0:
                kernel_size = stride
                padding = 0
                stride = 1
            else:
                if datapath.count("vgg16") > 0:
                    kernel_size = stride
                    padding = 0
                else:
                    kernel_size = 3
                    padding = 1
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}
            T = int(layerParam.input.shape[0]) if use_input_time_steps else Time_step
            compiler.NoC.flitNumber = 0
            # compiler.NoC.reset_router_busy_until()  # 本层内下一 spine 到达时若 router 仍忙则等待
            for t in tqdm(range(T)):
                # if t > 0:
                #     break
                columnFinishCycle[t] = {}
                input2D = input[t].reshape(input[t].shape[1],-1)
                input2Dcycle = torch.zeros((H,W))

                for columnId in columnFinishCycleLast[t].keys():

                    flits = flitGenerator(spikes=input2D[:,columnId], columnId=columnId)
                    # print("columnId:",m,len(flits))
                    curFlitNubmer = 0 #这一个spine需要多少个flit
                    for flit in flits:
                        if len(flit.Payload.rowId) > 0:
                            InputFlitNum = InputFlitNum + 1
                            curFlitNubmer = curFlitNubmer + 1 #更新flit
                
                    SpineInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=columnId, timestep=t, flitNumber=curFlitNubmer)
                    r = columnId//W
                    c = columnId%W
                    input2Dcycle[r,c] = columnFinishCycleLast[t][columnId]
                print(input2Dcycle, "InputFlitNum", InputFlitNum)
                # torch.save(input2Dcycle,"before_maxpooling.pth")
                maxpooling = torch.nn.MaxPool2d(kernel_size=kernel_size,padding=padding,stride=stride)
                outputCycle = maxpooling(input2Dcycle.unsqueeze(0).unsqueeze(0))
                outputCycle = outputCycle.squeeze(0).squeeze(0)
                # torch.save(outputCycle,"after_maxpooling.pth")
                output1DCycle = outputCycle.reshape(-1)
                for columnId in range(output1DCycle.shape[0]):
                    columnFinishCycle[t][columnId] = output1DCycle[columnId].item()
                    # print("output2DCycle[:,columnId]",output2DCycle[:,columnId])
            # print(columnFinishCycle[t])

            # 更新NoC中下一层的每一个输入时间
            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)

    calculateInfo["transmitTraffic"] = compiler.NoC.transmitTraffic
    torch.save(calculateInfo, os.path.join(output_dir, calculate_info_pth))
    torch.save(compiler.NoC.transmitTraffic, os.path.join(output_dir, communication_traffic_pth))
    torch.save(columnFinishCyclePerlayer, os.path.join(output_dir, column_finish_pth))
    SearchInput = {}
    SearchInput["FlitNumberEachLayer"] = compiler.FlitNumberEachLayer
    SearchInput["avgFlitNumEachLayer"] = compiler.avgFlitNumEachLayer
    SearchInput["SpikeNumberEachLayer"] = compiler.SpikeNumberEachLayer
    SearchInput["WidthAfterTiling"] = compiler.WidthAfterTiling
    SearchInput["SRAMNumPerLayer"] = compiler.SRAMNumPerLayer
    SearchInput["VSAWidth"] = compiler.VSAWidth
    SearchInput["totalSRAMNum"] = compiler.totalSRAMNum
    torch.save(SearchInput, os.path.join(output_dir, search_input_final))

    return calculateInfo

