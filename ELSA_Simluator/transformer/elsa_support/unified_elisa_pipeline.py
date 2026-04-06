"""Shared transformer + NoC simulation loop (extracted from VSACompiler)."""
from __future__ import annotations

import os
from copy import deepcopy

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from PETile.Tile import Tile
from processElement.STBIFFunction import STBIFNeuron
from PETile.FlitGenerator import FlitGenerator, FlitCombiner
from router.Switch import SwitchCrossbar
from router.RouteComputer import RouteComputer

from Compilers.VSACompiler import (
    VSACompiler,
    _print_layer_bubble,
    getPEBreakdown,
    getPEBreakdownMulti,
    getRouterBreakdown,
)


def run_unified_transformer_pipeline(
    datapath,
    connectionpath,
    mappingpath,
    occupyPath,
    config_path=None,
    time_step=None,
    output_dir=None,
    *,
    calculate_info_pth: str,
    column_finish_pth: str,
    input_search_pth: str | None = None,
):
    """Run full layer-wise ViT/transformer simulation. ``time_step`` is reserved (data T dominates)."""
    from elsa_support.paths import CONFIG_YAML, TRANSFORMER_ROOT

    if config_path is None:
        config_path = str(CONFIG_YAML)
    if output_dir is None:
        output_dir = str(TRANSFORMER_ROOT / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    compiler = VSACompiler(
        datapath, connectionpath, mappingpath, occupyPath, config_path=config_path
    )
    calculateInfo = {}
    totalCycle = 0
    columnFinishCycle = {}
    columnFinishCycleLast = {}
    columnFinishCyclePerlayer = []
    firstLayer = True
    softmax_stalling_time = 0

    compiler.perTokenBubbleRatios = []
    compiler._bubbleRatioFirstTokenSkipped = False


    for layerIndex, layerParam in enumerate(compiler.layers):
        print(layerIndex)
        # for key in columnFinishCycle.keys():
        #     print(columnFinishCycle[key][0])
        layerParam.printmyself()
        compiler._currentLayerBubbleRatios = []
        if layerParam.type == "linear" and layerParam.name.count("qkv") > 0:
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            print("len(list(columnFinishCycleLast.keys()))",len(list(columnFinishCycleLast.keys())))
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}

            T,B,L,N = layerParam.output.shape
            qkv = layerParam.output.reshape(T,B,L,3,N//3)
            qTrue,kTrue,vTrue = qkv.unbind(dim=3)

            qkv_bias = layerParam.bias.reshape(3,N//3)
            q_bias,k_bias,v_bias = qkv_bias.unbind(dim=0)


            M,K = layerParam.weight.shape
            qkv_weight = layerParam.weight.reshape(3,M//3,K)
            q_weight,k_weight,v_weight = qkv_weight.unbind(dim=0)

            # layerParam.weight = q_weight
            # layerParam.output = q

            bias = layerParam.bias
            weight = layerParam.weight
            input = layerParam.input
            groudtruth = layerParam.output
            neuron_q = STBIFNeuron(threshold=torch.unique(qTrue)[-1],pos_max=6,neg_min=-7,bias=q_bias/torch.unique(qTrue)[-1])
            neuron_k = STBIFNeuron(threshold=torch.unique(kTrue)[-1],pos_max=6,neg_min=-7,bias=k_bias/torch.unique(kTrue)[-1])
            neuron_v = STBIFNeuron(threshold=torch.unique(vTrue)[-1],pos_max=6,neg_min=-7,bias=v_bias/torch.unique(vTrue)[-1])
            # groudtruth = v
            T = layerParam.input.shape[0]

            outputList = []
            outputq = []
            outputqList = []
            outputkList = []
            outputvList = []
            # VAccList = []
            # VSpikeTracer = []
            for t in range(T):    
                wq = torch.nn.functional.linear(input[t], q_weight)
                wk = torch.nn.functional.linear(input[t], k_weight)
                wv = torch.nn.functional.linear(input[t], v_weight)
                # print((wv+v_bias)[0,0,:64])
                outputq = neuron_q(wq)
                outputk = neuron_k(wk)
                outputv = neuron_v(wv)
                output = torch.cat([outputq,outputk,outputv],dim=-1)
                outputqList.append(outputq+0.0)
                outputkList.append(outputk+0.0)
                outputvList.append(outputv+0.0)
                # VAccList.append(neuron_v.q)
                outputList.append(output+0.0)
                # VSpikeTracer.append(neuron_v.acc_q)

            output1 = torch.stack(outputList,dim=0)
            outputq1 = torch.stack(outputqList,dim=0)
            outputk1 = torch.stack(outputkList,dim=0)
            outputv1 = torch.stack(outputvList,dim=0)
            # VAcc = torch.stack(VAccList,dim=0)
            # VSpikeTracer = torch.stack(VSpikeTracer,dim=0)
            print(f"\033[92mloading input accuracy = {((output1 - groudtruth).abs() < 1e-3).sum()/output1.numel()}\033[0m")

            totalCycle = 0
            VthrList = [torch.unique(qTrue)[-1], torch.unique(kTrue)[-1], torch.unique(vTrue)[-1]]
            weightList = [q_weight, k_weight, v_weight]
            biasList = [q_bias, k_bias, v_bias]
            outputList = [outputq1, outputk1, outputv1]
            qkvName = ["q","k","v"]
            name = layerParam.name + ""
        
            for i in range(3):
                compiler._currentLayerBubbleRatios = []
                layerParam.Vthr = VthrList[i]
                layerParam.weight = weightList[i]
                layerParam.bias = biasList[i]
                layerParam.output = outputList[i]
                layerParam.name = name + qkvName[i]
                output1 = outputList[i] + 0.0
                compiler.NoC.flitNumber = 0
                layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)

                TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=False) #move to the east router


                flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
                flitCombiner = FlitCombiner()
                CrossbarSwitch = SwitchCrossbar()
                RouteEngine = RouteComputer()
                InputFlitNum = 0
                OutputFlitNum = 0
                OOutputFlitNum = 0
                perTimeStepCycle = []
                inputTime = 0
                outSpikesList = []
                spikeNumPerTimeStep = []
                for t in tqdm(range(T)):
                    columnFinishCycle[t] = {}
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
                                
                    for m in tqdm(range(InputRows)):
                        # if m > 100:
                        #     break
                        # print("r,c",r,c)
                        rowId = m
                        # if (input2D[:,columnId] == 0).all():
                        #     r,c = TILE.VSAOrderCtrl(r,c)
                        #     continue
                        # print("send spike number:",(torch.abs(input2D[rowId,:])/layerParam.LastVthr1).sum())
                        # print("input2D[rowId,:]",input2D[rowId,:])
                        flits = flitGenerator(spikes=input2D[rowId,:], columnId=rowId, headId=0, ColFirst=False)
                        InputFlitNum = InputFlitNum + len(flits)
                        TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId, timestep=t, flitNumber=len(flits))
                        for flit in flits:
                            spikeNum = spikeNum + len(flit.Payload.columnId)
                            if layerIndex == 2:
                                flit.time = inputTime + 0
                                if len(flit.Payload.columnId) != 0:
                                    inputTime = inputTime + 1                            
                            else:
                                flit.time = TokenInputTime

                        TILE.flitCombiner.tailNum = TILE.PENum - 1

                        output = TILE(flits) # list[Flit]
                                        
                        outFlitsList, outcycle = output
                        if compiler._bubbleRatioFirstTokenSkipped:
                            compiler.perTokenBubbleRatios.append(TILE.lastTokenBubbleRatio)
                            compiler._currentLayerBubbleRatios.append(TILE.lastTokenBubbleRatio)
                        else:
                            compiler._bubbleRatioFirstTokenSkipped = True
                    
                        maxLen = 0
                        # for peid in range(TILE.PENum):
                        #     maxLen = max(maxLen,len(outFlitsList[peid]))
                    
                        # 这里模拟一下4个flit，combine的操作，之后会补上的。
                        spikeNum1 = 0
                        payloadSize = 0
                        for list1 in outFlitsList:
                            for list2 in list1:
                                OOutputFlitNum = OOutputFlitNum + 1
                                spikeNum1 = spikeNum1 + max(len(flit.Payload.columnId),len(flit.Payload.rowId))
                                payloadSize = flit.payloadSize
                        IndexWidth = compiler.cfg["NOC"]["rowColumnBitWidth"]
                        spikesNumberFlit = (payloadSize - IndexWidth)//(IndexWidth+1)
                        OutputFlitNum = OutputFlitNum + math.ceil(spikeNum1/spikesNumberFlit)
                        
                        outSpikeNum = 0
                        for peid in range(TILE.PENum):
                            # OutputFlitNum = OutputFlitNum + len(outFlitsList[peid])
                            output = flitCombiner(outFlitsList[peid])
                            for Flits in outFlitsList[peid]:
                                outSpikeNum = outSpikeNum + len(Flits.Payload.columnId)

                        outSpikes, rowID, maxcycle, colFirst = output
                        if rowID not in columnFinishCycle[t].keys():
                            columnFinishCycle[t][rowID] = maxcycle
                        else:
                            columnFinishCycle[t][rowID] = max(columnFinishCycle[t][rowID],maxcycle)
                                
                        curFlitCycle = maxcycle + 0.0

                        maxOutcycle = max(maxOutcycle,maxcycle)
                            
                        for spike in outSpikes:
                            row_id, column_id, sign = spike
                            outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + layerParam.Vthr if sign == 0 else -layerParam.Vthr

                        # print("column_id",column_id)
                    
                    outSpikesList.append(outSpikes1)
                    # perTimeStepCycle.append(maxOutcycle)
                    spikeNumPerTimeStep.append(spikeNum)
                    print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum()/layerParam.LastVthr1, "inputNum:",TILE.spikeNum)
                
                outSpikesList = torch.stack(outSpikesList,dim=0)
                # mask = (outSpikesList[0,0,:] - output1[0,0,0,:]).abs() > 1e-3
                # print(outSpikesList[0,0,:][mask])
                # print(output1[0,0,0,:][mask])
                # print(outSpikesList.shape)
                # print(output1.shape)
                # print(firstConv[:,109])
                # print(outSpikesList[0][:,109])
                # print(output1[0,0,:,109])
                    
                for keyT in columnFinishCycle.keys():
                    perTimeStepCycle.append(0)

                for keyT in columnFinishCycle.keys():
                    for key in columnFinishCycle[keyT].keys():
                        perTimeStepCycle[keyT] = max(perTimeStepCycle[keyT], columnFinishCycle[keyT][key])

                print("InputFlitNum",InputFlitNum,"InputFlitNumberNoC",compiler.NoC.flitNumber,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
                print("outSpikesList.shape",outSpikesList.shape,"output1.shape",output1.shape)
                print(f"\033[92mpoint accuracy: {torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()}\033[0m")
                compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)

                totalCycle = perTimeStepCycle[-1]
                calculateInfo[layerParam.name+qkvName[i]] = {}
                calculateInfo[layerParam.name+qkvName[i]]["point accuracy"] = torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()
                calculateInfo[layerParam.name+qkvName[i]]["InputFlitNum"] = InputFlitNum
                compiler.InputSearch["InputFlitNum"][layerParam.name+qkvName[i]] = InputFlitNum
                calculateInfo[layerParam.name+qkvName[i]]["outputFlitNum"] = OutputFlitNum
                compiler.InputSearch["OutputFlitNum"][layerParam.name+qkvName[i]] = OutputFlitNum
                compiler.InputSearch["avgFlitNumEachToken"][layerParam.name+qkvName[i]] = InputFlitNum/(layerParam.input.shape[-2])
                print("InputFlitNum",InputFlitNum,"layerParam.input.shape[-2]",layerParam.input.shape[-2])
                calculateInfo[layerParam.name+qkvName[i]]["perTimeStepCycle"] = perTimeStepCycle
                calculateInfo[layerParam.name+qkvName[i]]["numbers of operation"] = layerParam.ops
                calculateInfo[layerParam.name+qkvName[i]]["numbers of spikes"] = spikeNumPerTimeStep
                calculateInfo[layerParam.name+qkvName[i]]["one spike operation"] = TILE.spikeNum*TILE.M
                calculateInfo[layerParam.name+qkvName[i]]["sparsity"] = TILE.spikeNum / (TILE.N*TILE.K)
                print("TILE.N",TILE.N,"TILE.K",TILE.K,"TILE.M",TILE.M)

                CrossbarSwitch.fCount = OutputFlitNum
                RouteEngine.fCount = OutputFlitNum

                calculateInfo = getPEBreakdown(calculateInfo,TILE.PEs,totalCycle,layerParam.name+qkvName[i],1,False)
                calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name+qkvName[i],layerParam.type,isConv=False)
                            # begin to calculate PE total area/power
                totalArea = 0.0
                totalEnergy = 0.0
            
                tileArea = TILE.getArea()
                tileEnergy = TILE.calEnergy(totalCycle/200000000)            
            
                totalArea = tileArea + CrossbarSwitch.getArea() + RouteEngine.getArea()
                totalEnergy = tileEnergy + CrossbarSwitch.calEnergy(totalCycle/200000000) + RouteEngine.calEnergy(totalCycle/200000000)
                calculateInfo[layerParam.name+qkvName[i]]["Total Area"] = totalArea
                calculateInfo[layerParam.name+qkvName[i]]["Total Energy"] = totalEnergy
                calculateInfo[layerParam.name+qkvName[i]]["Total Latency"] = totalCycle            
                _print_layer_bubble(compiler, layerParam.name+qkvName[i])
                firstLayer = False    
        elif layerParam.type == "linear":
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            print("len(list(columnFinishCycleLast.keys()))",len(list(columnFinishCycleLast.keys())))
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}
            compiler.NoC.flitNumber = 0
            layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)
            T,B,L,N = layerParam.output.shape
            bias = layerParam.bias
            weight = layerParam.weight
            input = layerParam.input
            groudtruth = layerParam.output
            vthr = layerParam.Vthr
            if layerParam.name.count("fc1") > 0:
                neuron = STBIFNeuron(threshold=vthr,pos_max=6,neg_min=0,bias=bias/vthr)
            else:
                neuron = STBIFNeuron(threshold=vthr,pos_max=6,neg_min=-7,bias=bias/vthr)

            outputList = []
            for t in range(T):    
                wx = torch.nn.functional.linear(input[t], weight)
                output = neuron(wx)
                outputList.append(output+0.0)

            output1 = torch.stack(outputList,dim=0)

            print(f"\033[92mloading input accuracy = {((output1 - groudtruth).abs() < 1e-3).sum()/output1.numel()}\033[0m")
        
            TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=False) #move to the east router

            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            flitCombiner = FlitCombiner()
            CrossbarSwitch = SwitchCrossbar()
            RouteEngine = RouteComputer()
            InputFlitNum = 0
            OutputFlitNum = 0
            OOutputFlitNum = 0
            perTimeStepCycle = []
            inputTime = 0
            outSpikesList = []
            spikeNumPerTimeStep = []
            for t in tqdm(range(T)):
                columnFinishCycle[t] = {}
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
                            
                for m in tqdm(range(InputRows)):
                    # if m > 100:
                    #     break
                    # print("r,c",r,c)
                    rowId = m
                    # if (input2D[:,columnId] == 0).all():
                    #     r,c = TILE.VSAOrderCtrl(r,c)
                    #     continue
                    # print("send spike number:",(torch.abs(input2D[rowId,:])/layerParam.LastVthr1).sum())
                    flits = flitGenerator(spikes=input2D[rowId,:], columnId=rowId, headId=0, ColFirst=False)
                    InputFlitNum = InputFlitNum + len(flits)
                    TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId, timestep=t, flitNumber=len(flits))
                    for flit in flits:
                        spikeNum = spikeNum + len(flit.Payload.columnId)
                        if layerIndex == 2:
                            flit.time = inputTime + 0
                            if len(flit.Payload.columnId) != 0:
                                inputTime = inputTime + 1                            
                        else:
                            flit.time = TokenInputTime

                    TILE.flitCombiner.tailNum = TILE.PENum - 1

                    output = TILE(flits) # list[Flit]
                                    
                    outFlitsList, outcycle = output
                    if compiler._bubbleRatioFirstTokenSkipped:
                        compiler.perTokenBubbleRatios.append(TILE.lastTokenBubbleRatio)
                        compiler._currentLayerBubbleRatios.append(TILE.lastTokenBubbleRatio)
                    else:
                        compiler._bubbleRatioFirstTokenSkipped = True
                
                    maxLen = 0
                    # for peid in range(TILE.PENum):
                    #     maxLen = max(maxLen,len(outFlitsList[peid]))
                        # 这里模拟一下4个flit，combine的操作，之后会补上的。
                    spikeNum1 = 0
                    payloadSize = 0
                    for list1 in outFlitsList:
                        for list2 in list1:
                            # for flit in list2:
                                OOutputFlitNum = OOutputFlitNum + 1
                                spikeNum1 = spikeNum1 + max(len(flit.Payload.columnId),len(flit.Payload.rowId))
                                payloadSize = flit.payloadSize
                    IndexWidth = compiler.cfg["NOC"]["rowColumnBitWidth"]
                    spikesNumberFlit = (payloadSize - IndexWidth)//(IndexWidth+1)
                    OutputFlitNum = OutputFlitNum + math.ceil(spikeNum1/spikesNumberFlit)   

                    outSpikeNum = 0
                    for peid in range(TILE.PENum):
                        # OutputFlitNum = OutputFlitNum + len(outFlitsList[peid])
                        output = flitCombiner(outFlitsList[peid])
                        for Flits in outFlitsList[peid]:
                            outSpikeNum = outSpikeNum + len(Flits.Payload.columnId)

                    outSpikes, rowID, maxcycle, colFirst = output

                    if rowID not in columnFinishCycle[t].keys():
                        columnFinishCycle[t][rowID] = maxcycle
                    else:
                        columnFinishCycle[t][rowID] = max(columnFinishCycle[t][rowID],maxcycle)
                        
                    curFlitCycle = maxcycle + 0.0

                    maxOutcycle = max(maxOutcycle,maxcycle)
                
                    for spike in outSpikes:
                        row_id, column_id, sign = spike
                        outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + layerParam.Vthr if sign == 0 else -layerParam.Vthr
            
                    # print("column_id",column_id)
                
                outSpikesList.append(outSpikes1)
                # perTimeStepCycle.append(maxOutcycle)
                spikeNumPerTimeStep.append(spikeNum)
                print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum()/layerParam.LastVthr1, "inputNum:",TILE.spikeNum)
            
            outSpikesList = torch.stack(outSpikesList,dim=0)
            # mask = (outSpikesList[0,0,:] - output1[0,0,0,:]).abs() > 1e-3
            # print(outSpikesList[0,0,:][mask])
            # print(output1[0,0,0,:][mask])
            # print(outSpikesList.shape)
            # print(output1.shape)
            # print(firstConv[:,109])
            # print(outSpikesList[0][:,109])
            # print(output1[0,0,:,109])

            for keyT in columnFinishCycle.keys():
                perTimeStepCycle.append(0)

            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    perTimeStepCycle[keyT] = max(perTimeStepCycle[keyT], columnFinishCycle[keyT][key])

            print("InputFlitNum",InputFlitNum,"InputFlitNumberNoC",compiler.NoC.flitNumber,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
            print("outSpikesList.shape",outSpikesList.shape,"output1.shape",output1.shape)
            print(f"\033[92mpoint accuracy: {torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()}\033[0m")

            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)

            totalCycle = perTimeStepCycle[-1]
            calculateInfo[layerParam.name] = {}
            calculateInfo[layerParam.name]["point accuracy"] = torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()
            calculateInfo[layerParam.name]["InputFlitNum"] = InputFlitNum
            compiler.InputSearch["InputFlitNum"][layerParam.name] = InputFlitNum
            calculateInfo[layerParam.name]["outputFlitNum"] = OutputFlitNum
            compiler.InputSearch["OutputFlitNum"][layerParam.name] = OutputFlitNum
            print("InputFlitNum",InputFlitNum,"avgFlitNumEachToken",layerParam.input.shape[-2])
            compiler.InputSearch["avgFlitNumEachToken"][layerParam.name] = InputFlitNum/(layerParam.input.shape[-2])
            calculateInfo[layerParam.name]["perTimeStepCycle"] = perTimeStepCycle
            calculateInfo[layerParam.name]["numbers of operation"] = layerParam.ops
            calculateInfo[layerParam.name]["numbers of spikes"] = spikeNumPerTimeStep
            calculateInfo[layerParam.name]["one spike operation"] = TILE.spikeNum*TILE.M
            calculateInfo[layerParam.name]["sparsity"] = TILE.spikeNum / (TILE.N*TILE.K*T*TILE.head)

            CrossbarSwitch.fCount = OutputFlitNum
            RouteEngine.fCount = OutputFlitNum

            calculateInfo = getPEBreakdown(calculateInfo,TILE.PEs,totalCycle,layerParam.name,1,False)
            calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name,layerParam.type,isConv=False)
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
            _print_layer_bubble(compiler, layerParam.name)
            firstLayer = False    
        elif layerParam.type == "multiplication":
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            print("len(list(columnFinishCycleLast.keys()))",len(list(columnFinishCycleLast.keys())))
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}
            compiler.NoC.flitNumber = 0
            layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)        

            T,B,H,L,N = layerParam.output.shape
            head_rowNum = L
            headNum = H
            bias = None
            weight = layerParam.weight.squeeze(0)
            input = layerParam.input.squeeze(1)
            groudtruth = layerParam.output.squeeze(1).sum(dim=0)
            vthr = layerParam.Vthr
            neuron = STBIFNeuron(threshold=vthr,pos_max=6,neg_min=-7,bias=bias)

            outputList = []
            for t in range(T):    
                wx = input[t]@weight
                output = neuron(wx)
                outputList.append(output+0.0)

            output1 = torch.stack(outputList,dim=0).sum(dim=0)

            print(f"\033[92mloading input accuracy = {((output1 - groudtruth).abs() < 1e-3).sum()/output1.numel()}\033[0m")

            input = layerParam.input.squeeze(1).reshape(T,H*L,L)
            output1 = output1.reshape(H*L,N)

            TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=False) #move to the east router

            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            flitCombiner = FlitCombiner()
            CrossbarSwitch = SwitchCrossbar()
            RouteEngine = RouteComputer()
            InputFlitNum = 0
            OutputFlitNum = 0
            OOutputFlitNum = 0
            perTimeStepCycle = []
            inputTime = 0
            outSpikesList = []
            spikeNumPerTimeStep = [] 
        
            for t in tqdm(range(T)):
                columnFinishCycle[t] = {}
                # if t > 0:
                #     break
                H,W = input.shape[-2],input.shape[-1]
                input2D = input[t].reshape(-1,input[t].shape[-1])
                r = c = 0
                outSpikes1 = torch.zeros(output1.shape[0],output1.shape[1])
                InputRows = input2D.shape[0]
                InputColumns = input2D.shape[1]
                spikeNum = 0
                maxOutcycle = 0
                curFlitCycle = 0
                            
                for m in tqdm(range(head_rowNum)):
                    for h in range(headNum):
                        # if m > 100:
                        #     break
                        # print("r,c",r,c)
                        rowId = m + h*head_rowNum
                        # if (input2D[:,columnId] == 0).all():
                        #     r,c = TILE.VSAOrderCtrl(r,c)
                        #     continue
                        # print("send spike number:",(torch.abs(input2D[rowId,:])/layerParam.LastVthr1).sum())
                        # print("input2D[rowId,:]",input2D[rowId,:])
                        flits = flitGenerator(spikes=input2D[rowId,:], columnId=rowId, headId=0, ColFirst=False)
                        InputFlitNum = InputFlitNum + len(flits)
                        TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId%head_rowNum, timestep=t, flitNumber=len(flits))
                        for flit in flits:
                            spikeNum = spikeNum + len(flit.Payload.columnId)
                            if layerIndex == 2:
                                flit.time = inputTime + 0
                                if len(flit.Payload.columnId) != 0:
                                    inputTime = inputTime + 1                            
                            else:
                                # if rowId < 10:
                                #     print("columnFinishCycleLast[t][rowId]",columnFinishCycleLast[t][rowId])
                                flit.time = TokenInputTime

                        TILE.flitCombiner.tailNum = TILE.PENum - 1

                        output = TILE(flits) # list[Flit]
                                        
                        outFlitsList, outcycle = output
                        if compiler._bubbleRatioFirstTokenSkipped:
                            compiler.perTokenBubbleRatios.append(TILE.lastTokenBubbleRatio)
                            compiler._currentLayerBubbleRatios.append(TILE.lastTokenBubbleRatio)
                        else:
                            compiler._bubbleRatioFirstTokenSkipped = True
                    
                        maxLen = 0
                        # for peid in range(TILE.PENum):
                        #     maxLen = max(maxLen,len(outFlitsList[peid]))

                        spikeNum1 = 0
                        payloadSize = 0
                        for list1 in outFlitsList:
                            for list2 in list1:
                                # for flit in list2:
                                    OOutputFlitNum = OOutputFlitNum + 1
                                    spikeNum1 = spikeNum1 + max(len(flit.Payload.columnId),len(flit.Payload.rowId))
                                    payloadSize = flit.payloadSize
                        IndexWidth = compiler.cfg["NOC"]["rowColumnBitWidth"]
                        spikesNumberFlit = (payloadSize - IndexWidth)//(IndexWidth+1)
                        OutputFlitNum = OutputFlitNum + math.ceil(spikeNum1/spikesNumberFlit)   

                        outSpikeNum = 0
                        for peid in range(TILE.PENum):
                            # OutputFlitNum = OutputFlitNum + len(outFlitsList[peid])
                            output = flitCombiner(outFlitsList[peid])
                            for Flits in outFlitsList[peid]:
                                outSpikeNum = outSpikeNum + len(Flits.Payload.columnId)

                        outSpikes, rowID, maxcycle, colFirst = output
                    
                        rowIDinHead = rowID%head_rowNum
                        if rowIDinHead not in columnFinishCycle[t].keys():
                            columnFinishCycle[t][rowIDinHead] = maxcycle
                        else:
                            columnFinishCycle[t][rowIDinHead] = max(columnFinishCycle[t][rowIDinHead],maxcycle)
                                
                        curFlitCycle = maxcycle + 0.0

                        maxOutcycle = max(maxOutcycle,maxcycle)
                            
                        for spike in outSpikes:
                            row_id, column_id, sign = spike
                            outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + layerParam.Vthr if sign == 0 else -layerParam.Vthr
                
                    # print("column_id",column_id)
                
                outSpikesList.append(outSpikes1)
                # perTimeStepCycle.append(maxOutcycle)
                spikeNumPerTimeStep.append(spikeNum)
                print("spikeNum",spikeNum, "trueNum",torch.abs(input2D).sum()/layerParam.LastVthr1, "inputNum:",TILE.spikeNum)
            
            outSpikesList = torch.stack(outSpikesList,dim=0).sum(dim=0)
            # mask = (outSpikesList[0,0,:] - output1[0,0,0,:]).abs() > 1e-3
            # print(outSpikesList[0,0,:][mask])
            # print(output1[0,0,0,:][mask])
            # print(outSpikesList.shape)
            # print(output1.shape)
            # print(firstConv[:,109])
            # print(outSpikesList[0][:,109])
            # print(output1[0,0,:,109])
            for keyT in columnFinishCycle.keys():
                perTimeStepCycle.append(0)

            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    perTimeStepCycle[keyT] = max(perTimeStepCycle[keyT], columnFinishCycle[keyT][key])

            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    columnFinishCycle[keyT][key] = perTimeStepCycle[keyT] + 0.0

            print("InputFlitNum",InputFlitNum,"InputFlitNumberNoC",compiler.NoC.flitNumber,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
            print("outSpikesList.shape",outSpikesList.shape,"output1.shape",output1.shape)
            print(f"\033[92mpoint accuracy: {torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()}\033[0m")
            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)        

            totalCycle = perTimeStepCycle[-1]
            calculateInfo[layerParam.name] = {}
            calculateInfo[layerParam.name]["point accuracy"] = torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()
            calculateInfo[layerParam.name]["InputFlitNum"] = InputFlitNum
            compiler.InputSearch["InputFlitNum"][layerParam.name] = InputFlitNum
            calculateInfo[layerParam.name]["outputFlitNum"] = OutputFlitNum
            compiler.InputSearch["OutputFlitNum"][layerParam.name] = OutputFlitNum
            compiler.InputSearch["avgFlitNumEachToken"][layerParam.name] = InputFlitNum/(layerParam.input.shape[-2])
            calculateInfo[layerParam.name]["perTimeStepCycle"] = perTimeStepCycle
            calculateInfo[layerParam.name]["numbers of operation"] = layerParam.ops
            calculateInfo[layerParam.name]["numbers of spikes"] = spikeNumPerTimeStep
            calculateInfo[layerParam.name]["one spike operation"] = TILE.spikeNum*TILE.M
            calculateInfo[layerParam.name]["sparsity"] = TILE.spikeNum / (TILE.N*TILE.K*T*TILE.head)

            CrossbarSwitch.fCount = OutputFlitNum
            RouteEngine.fCount = OutputFlitNum

            calculateInfo = getPEBreakdown(calculateInfo,TILE.PEs,totalCycle,layerParam.name,1,False)
            calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name,layerParam.type,isConv=False)
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
            _print_layer_bubble(compiler, layerParam.name)
            firstLayer = False        
        elif layerParam.type == "addition":
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            print("len(list(columnFinishCycleLast.keys()))",len(list(columnFinishCycleLast.keys())))
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}
            compiler.NoC.flitNumber = 0
            layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)
        
            if layerParam.name.count("addition1") > 0:
                prefix = layerParam.name.split(".")[0] + "." + layerParam.name.split(".")[1]
                for thislayerIndex, thislayerParam in enumerate(compiler.layers):     
                    if thislayerParam.name == prefix+".norm2":
                        layerParam2 = thislayerParam
            if layerParam.name.count("addition2") > 0:
                layerParam2 = compiler.layers[layerIndex + 1]
                
            T,B,N,C = layerParam.input.shape
            input1 = layerParam.input.squeeze(1)
            input2 = layerParam.input2.squeeze(1)
            midVthr = layerParam.Vthr
            vthr = layerParam2.Vthr
            groudtruth = layerParam2.output.squeeze(1)
            neuron1 = STBIFNeuron(threshold=midVthr,pos_max=6,neg_min=-7,bias=None)
            neuron2 = STBIFNeuron(threshold=vthr,pos_max=6,neg_min=-7,bias=None)
            weight = layerParam2.weight
            bias = layerParam2.bias
            layernorm = nn.LayerNorm(C)
            layernorm.weight.data = weight
            layernorm.bias.data = bias
        
            outputList = []
            for t in range(T):    
                wx = input1[t] + input2[t]
                output = neuron1(wx)
                for n in range(N):
                    if t == 0:
                        output[n,:] = layernorm(neuron1.acc_q[n,:]*midVthr)
                    else:
                        output[n,:] = layernorm(neuron1.acc_q[n,:]*midVthr) - layernorm(neuron1.acc_q[n,:]*midVthr - output[n,:])
                output = neuron2(output)
                outputList.append(output+0.0)

            output1 = torch.stack(outputList,dim=0)

            print(f"\033[92mloading addition input accuracy = {((output1 - groudtruth).abs() < 1e-3).sum()/output1.numel()}\033[0m")    

            # create layerParam
            layerParam.weight = layerParam2.weight
            layerParam.bias = layerParam2.bias
            layerParam.output = layerParam2.output
            layerParam.midVthr = midVthr
            layerParam.Vthr = vthr
            layerParam.type = "residual_norm"
        
            TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=False) #move to the east router

            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            flitCombiner = FlitCombiner()
            flitCombiner.PENums = 1
            CrossbarSwitch = SwitchCrossbar()
            RouteEngine = RouteComputer()
            InputFlitNum = 0
            OutputFlitNum = 0
            OOutputFlitNum = 0
            perTimeStepCycle = []
            inputTime = 0
            outSpikesList = []
            spikeNumPerTimeStep = [] 

            for t in tqdm(range(T)):
                columnFinishCycle[t] = {}
                # if t > 0:
                #     break
                H,W = input1.shape[-2],input1.shape[-1]
                input2D1 = input1[t].reshape(-1,input1[t].shape[-1])
                input2D2 = input2[t].reshape(-1,input2[t].shape[-1])
                r = c = 0
                outSpikes1 = torch.zeros(output1.shape[-2],output1.shape[-1])
                InputRows = input2D1.shape[0]
                InputColumns = input2D1.shape[1]
                spikeNum = 0
                maxOutcycle = 0
                curFlitCycle = 0
                            
                for m in tqdm(range(InputRows)):
                    rowId = m
                    flits1 = flitGenerator(spikes=input2D1[rowId,:], columnId=rowId, headId=0, ColFirst=False)
                    flits2 = flitGenerator(spikes=input2D2[rowId,:], columnId=rowId, headId=0, ColFirst=False)
                    InputFlitNum = InputFlitNum + len(flits1)
                    InputFlitNum = InputFlitNum + len(flits2)

                    TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId, timestep=t, flitNumber=len(flits1))
                    for flit in flits1:
                        spikeNum = spikeNum + len(flit.Payload.columnId)
                        if layerIndex == 2:
                            flit.time = inputTime + 0
                            if len(flit.Payload.columnId) != 0:
                                inputTime = inputTime + 1                            
                        else:
                            flit.time = TokenInputTime

                    TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId, timestep=t, flitNumber=len(flits2))
                    for flit in flits2:
                        spikeNum = spikeNum + len(flit.Payload.columnId)
                        if layerIndex == 2:
                            flit.time = inputTime + 0
                            if len(flit.Payload.columnId) != 0:
                                inputTime = inputTime + 1                            
                        else:
                            flit.time = TokenInputTime

                    TILE.flitCombiner.tailNum = TILE.PENum - 1
                    TILE.flitCombiner.PENums = TILE.PENum

                    TILE(flits1, timeStep=t) # First Input
                    output = TILE(flits2, timeStep=t) # Second Input
                                    
                    outFlitsList, outcycle = output
                    if compiler._bubbleRatioFirstTokenSkipped:
                        compiler.perTokenBubbleRatios.append(TILE.lastTokenBubbleRatio)
                        compiler._currentLayerBubbleRatios.append(TILE.lastTokenBubbleRatio)
                    else:
                        compiler._bubbleRatioFirstTokenSkipped = True
                
                    maxLen = 0
                    # for peid in range(TILE.PENum):
                    #     maxLen = max(maxLen,len(outFlitsList[peid]))
                    spikeNum1 = 0
                    payloadSize = 0
                    for list1 in outFlitsList:
                        for list2 in list1:
                            # for flit in list2:
                                OOutputFlitNum = OOutputFlitNum + 1
                                spikeNum1 = spikeNum1 + max(len(flit.Payload.columnId),len(flit.Payload.rowId))
                                payloadSize = flit.payloadSize
                    IndexWidth = compiler.cfg["NOC"]["rowColumnBitWidth"]
                    spikesNumberFlit = (payloadSize - IndexWidth)//(IndexWidth+1)
                    OutputFlitNum = OutputFlitNum + math.ceil(spikeNum1/spikesNumberFlit)  

                    outSpikeNum = 0
                    for peid in range(TILE.PENum):
                        # OutputFlitNum = OutputFlitNum + len(outFlitsList[peid])
                        output = flitCombiner(outFlitsList[peid])
                        for Flits in outFlitsList[peid]:
                            outSpikeNum = outSpikeNum + len(Flits.Payload.columnId)
                            # print(Flits.printmyself())

                    outSpikes, rowID, maxcycle, colFirst = output
                    if rowID not in columnFinishCycle[t].keys():
                        columnFinishCycle[t][rowID] = maxcycle
                    else:
                        columnFinishCycle[t][rowID] = max(columnFinishCycle[t][rowID],maxcycle)

                    # print("TokenInputTime",TokenInputTime,"columnFinishCycle[t][rowID]",columnFinishCycle[t][rowID],"maxcycle",maxcycle)
                
                    curFlitCycle = maxcycle + 0.0

                    maxOutcycle = max(maxOutcycle,maxcycle)
                        
                    for spike in outSpikes:
                        row_id, column_id, sign = spike
                        # print(row_id, column_id, sign)
                        outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + layerParam.Vthr if sign == 0 else -layerParam.Vthr
            
                    # print("column_id",column_id)
                
                outSpikesList.append(outSpikes1)
                # perTimeStepCycle.append(maxOutcycle)
                spikeNumPerTimeStep.append(spikeNum)
                print("spikeNum",spikeNum, "trueNum",torch.abs(input2D1).sum()/layerParam.LastVthr1 + torch.abs(input2D2).sum()/layerParam.LastVthr2, "inputNum:",TILE.spikeNum)
            
            outSpikesList = torch.stack(outSpikesList,dim=0)
            # print("output1.shape",output1.shape,"outSpikesList.shape",outSpikesList.shape)
            # print("output1[1,1,:64]",output1[1,1,:64])
            # print("outSpikesList[1,1,:64]",outSpikesList[1,1,:64])
            # print((outSpikesList[1,1,:64] - output1[1,1,:64]).abs() < 1e-3)
            # mask = (outSpikesList[0,0,:] - output1[0,0,0,:]).abs() > 1e-3
            # print(outSpikesList[0,0,:][mask])
            # print(output1[0,0,0,:][mask])
            # print(outSpikesList.shape)
            # print(output1.shape)
            # print(firstConv[:,109])
            # print(outSpikesList[0][:,109])
            # print(output1[0,0,:,109])
            for keyT in columnFinishCycle.keys():
                perTimeStepCycle.append(0)

            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    perTimeStepCycle[keyT] = max(perTimeStepCycle[keyT], columnFinishCycle[keyT][key])

            print("InputFlitNum",InputFlitNum,"InputFlitNumberNoC",compiler.NoC.flitNumber,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
            print("outSpikesList.shape",outSpikesList.shape,"output1.shape",output1.shape)
            print(f"\033[92mpoint accuracy: {torch.sum(outSpikesList == output1)/output1.numel()}\033[0m")

            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)
            # print(f"\033[92mpoint accuracy: {torch.sum(outSpikesList[0] == output1[0])/output1.numel()}\033[0m")
            # print(f"\033[92mpoint accuracy: {torch.sum(outSpikesList[1] == output1[1])/output1.numel()}\033[0m")
        
            totalCycle = perTimeStepCycle[-1]
            calculateInfo[layerParam.name] = {}
            calculateInfo[layerParam.name]["point accuracy"] = torch.sum(outSpikesList == output1.squeeze(1))/output1.squeeze(1).numel()
            calculateInfo[layerParam.name]["InputFlitNum"] = InputFlitNum
            compiler.InputSearch["InputFlitNum"][layerParam.name] = InputFlitNum
            calculateInfo[layerParam.name]["outputFlitNum"] = OutputFlitNum
            compiler.InputSearch["OutputFlitNum"][layerParam.name] = OutputFlitNum
            compiler.InputSearch["avgFlitNumEachToken"][layerParam.name] = InputFlitNum/(layerParam.input.shape[-2])
            calculateInfo[layerParam.name]["perTimeStepCycle"] = perTimeStepCycle
            calculateInfo[layerParam.name]["numbers of operation"] = layerParam.ops
            calculateInfo[layerParam.name]["numbers of spikes"] = spikeNumPerTimeStep
            calculateInfo[layerParam.name]["one spike operation"] = TILE.spikeNum
            calculateInfo[layerParam.name]["sparsity"] = TILE.spikeNum / (TILE.N*TILE.K*T*TILE.head)

            CrossbarSwitch.fCount = OutputFlitNum
            RouteEngine.fCount = OutputFlitNum

            # calculateInfo = getPEBreakdown(calculateInfo,TILE.PEs,totalCycle,layerParam.name,1,False)
            calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name,layerParam.type,isConv=False)
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
            _print_layer_bubble(compiler, layerParam.name)
            firstLayer = False        
            # exit(0)
        elif layerParam.type == "multiplication_softmax":
            columnFinishCycleLast = deepcopy(columnFinishCycle)
            print("len(list(columnFinishCycleLast.keys()))",len(list(columnFinishCycleLast.keys())))
            columnFinishCyclePerlayer.append({layerParam.name : deepcopy(columnFinishCycle)})
            columnFinishCycle = {}

            compiler.NoC.flitNumber = 0
            layerParam.mapLayerNum = compiler.NoC.getMapLayerNum(layerParam.name)

            T,B,H,L,N = layerParam.output.shape
            headNum = H
            head_rowNum = L
            bias = None
            # weight1 = layerParam.weight.squeeze(0)
            input1 = layerParam.input.squeeze(1)
            # weight2 = layerParam.weight2.squeeze(0)
            input2 = layerParam.input2.squeeze(1)
            input1_tracer = torch.zeros(input1[0].shape)
            input2_tracer = torch.zeros(input2[0].shape)
            groudtruth = layerParam.output.squeeze(1)
            vthr = layerParam.Vthr
            midVthr = layerParam.midVthr
            LastVthr1 = layerParam.LastVthr1
            LastVthr2 = layerParam.LastVthr2
            neuron = STBIFNeuron(threshold=midVthr,pos_max=6,neg_min=-7,bias=bias)
            neuron_softmax = STBIFNeuron(threshold=vthr,pos_max=6,neg_min=0,bias=bias)

            outputList = []
            test_output = None
            test_output1 = None
            print(input1.shape,input2.shape,input2_tracer.shape,input1_tracer.shape )
            for t in range(T):    
                wx = input1[t]@input2_tracer.transpose(1,2)
                input1_tracer = input1_tracer + input1[t]
                wx = wx + input1_tracer@input2[t].transpose(1,2)
                input2_tracer = input2_tracer + input2[t]
                # if t == 1:
                #     output = neuron(wx,verbose=True)
                # else:
                output = neuron(wx)
                # if t == 1:
                #     print(wx[5,2,:])
                #     print(output[5,2,:])
            
                for h in range(H):
                    for n in range(L):
                        if t == 0:
                            output[h,n,:] = torch.nn.functional.softmax(neuron.acc_q[h,n,:]*midVthr)
                        else:
                            output[h,n,:] = torch.nn.functional.softmax(neuron.acc_q[h,n,:]*midVthr) - torch.nn.functional.softmax(neuron.acc_q[h,n,:]*midVthr - output[h,n,:])        

                # if t == 1:
                #     print(output[5,2,:])
                #     print("neuron_softmax.q",neuron_softmax.q[5,2,:]*vthr)
                output = neuron_softmax(output)
                # if t == 1:
                #     print(output[5,2,:])
                    # test_output = output + 0.0
            
                outputList.append(output+0.0)

            output1 = torch.stack(outputList,dim=0)

            print(f"\033[92mloading input accuracy = {((output1 - groudtruth).abs() < 1e-3).sum()/output1.numel()}\033[0m")    

            TILE = Tile(layerParam=layerParam,routeInfo=(1,0),first=False) #move to the east router

            flitGenerator = FlitGenerator(RouteInfo=(1,0),PEId=0, TileValue=0)
            flitCombiner = FlitCombiner()
            CrossbarSwitch = SwitchCrossbar()
            RouteEngine = RouteComputer()
            InputFlitNum = 0
            OutputFlitNum = 0
            OOutputFlitNum = 0
            perTimeStepCycle = []
            inputTime = 0
            outSpikesList = []
            spikeNumPerTimeStep = []
            for t in tqdm(range(T)):
                columnFinishCycle[t] = {}
                # if t > 0:
                #     break
                # H,W = input1.shape[-2],input1.shape[-1]
                input2D1 = input1[t].reshape(-1,input1[t].shape[-1])
                input2D2 = input2[t].reshape(-1,input2[t].shape[-1]).T
                r = c = 0
                outSpikes1 = torch.zeros(output1.shape[-3]*output1.shape[-2],output1.shape[-1])
                InputRows = input2D1.shape[0]
                InputColumns = input2D1.shape[1]
                # print(InputRows,InputColumns)
                spikeNum = 0
                maxOutcycle = 0
                curFlitCycle = 0
                if input1[t].abs().sum() > 0 or input2[t].abs().sum() > 0:
                    softmax_stalling_time = softmax_stalling_time + 1

            
                for m in tqdm(range(InputRows//headNum)):
                    for h in range(headNum):
                        rowId = m + h*head_rowNum

                        flits1 = flitGenerator(spikes=input2D1[rowId,:], columnId=rowId, headId=0, ColFirst=False)
                        flits2 = flitGenerator(spikes=input2D2[:,rowId], columnId=rowId, headId=0, ColFirst=True)
                        InputFlitNum = InputFlitNum + len(flits1)
                        InputFlitNum = InputFlitNum + len(flits2)
                        # print("==================================VSACompiler========================================")
                        TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId%head_rowNum, timestep=t, flitNumber=len(flits1))
                        for flit in flits1:
                            spikeNum = spikeNum + len(flit.Payload.columnId)
                            if layerIndex == 2:
                                flit.time = inputTime + 0
                                if len(flit.Payload.columnId) != 0:
                                    inputTime = inputTime + 1                            
                            else:
                                if m in columnFinishCycleLast[t].keys():
                                        flit.time = columnFinishCycleLast[t][m]
                                else:
                                    flit.time = TokenInputTime                            

                            # flit.printmyself()
                        # print("=================================VSACompiler======================================")
                        TokenInputTime = compiler.NoC(curLayerId=layerParam.name, spineTokenId=rowId%head_rowNum, timestep=t, flitNumber=len(flits2))
                        for flit in flits2:
                            spikeNum = spikeNum + len(flit.Payload.columnId)
                            if layerIndex == 0:
                                flit.time = inputTime + 0
                                if len(flit.Payload.columnId) != 0:
                                    inputTime = inputTime + 1                            
                            else:
                                flit.time = TokenInputTime                            

                            # flit.printmyself()
                        
                        TILE.flitCombiner.tailNum = TILE.PENum - 1
                        TILE.flitCombiner.PENums = TILE.PENum
                        TILE(flits1) # First Input
                        if compiler._bubbleRatioFirstTokenSkipped:
                            compiler.perTokenBubbleRatios.append(TILE.lastTokenBubbleRatio)
                            compiler._currentLayerBubbleRatios.append(TILE.lastTokenBubbleRatio)
                        else:
                            compiler._bubbleRatioFirstTokenSkipped = True

                        TILE.flitCombiner.tailNum = TILE.PENum - 1
                        TILE.flitCombiner.PENums = TILE.PENum
                        TILE(flits2) # Second Input
                        if compiler._bubbleRatioFirstTokenSkipped:
                            compiler.perTokenBubbleRatios.append(TILE.lastTokenBubbleRatio)
                            compiler._currentLayerBubbleRatios.append(TILE.lastTokenBubbleRatio)
                        else:
                            compiler._bubbleRatioFirstTokenSkipped = True
                
                
                    row_id_list = []
                for m in tqdm(range(InputRows)):
                    flitCombiner.PENums = int(math.sqrt(TILE.PENum))
                    output = TILE(None, Outupdate=True, OutRowId = m, timeStep=t) 
                
                    outFlitsList, outcycle = output
                
                    maxLen = 0
                    # for peid in range(TILE.PENum):
                    #     maxLen = max(maxLen,len(outFlitsList[peid]))
                    spikeNum1 = 0
                    payloadSize = 0
                    for list1 in outFlitsList:
                        for list2 in list1:
                            # for flit in list2:
                            OOutputFlitNum = OOutputFlitNum + 1
                            spikeNum1 = spikeNum1 + max(len(flit.Payload.columnId),len(flit.Payload.rowId))
                            payloadSize = flit.payloadSize
                    IndexWidth = compiler.cfg["NOC"]["rowColumnBitWidth"]
                    spikesNumberFlit = (payloadSize - IndexWidth)//(IndexWidth+1)
                    OutputFlitNum = OutputFlitNum + math.ceil(spikeNum1/spikesNumberFlit)  

                    outSpikeNum = 0
                    for peid in range(int(math.sqrt(TILE.PENum))):
                        OutputFlitNum = OutputFlitNum + len(outFlitsList[peid])
                        output = flitCombiner(outFlitsList[peid])
                        for Flits in outFlitsList[peid]:
                            outSpikeNum = outSpikeNum + len(Flits.Payload.columnId)

                    outSpikes, rowID, maxcycle, colFirst = output
                    rowIDinHead = rowID%head_rowNum
                    if rowIDinHead not in columnFinishCycle[t].keys():
                        columnFinishCycle[t][rowIDinHead] = maxcycle
                    else:
                        columnFinishCycle[t][rowIDinHead] = max(columnFinishCycle[t][rowIDinHead],maxcycle)
                    # print(outSpikes)
                    curFlitCycle = maxcycle + 0.0

                    maxOutcycle = max(maxOutcycle,maxcycle)
                
                    for spike in outSpikes:
                        row_id, column_id, sign = spike
                        # if len(row_id_list) == 0:
                        #     row_id_list.append(row_id)
                        # elif row_id != row_id_list[-1]:
                        #     row_id_list.append(row_id)
                        outSpikes1[row_id][column_id] = outSpikes1[row_id][column_id] + layerParam.Vthr if sign == 0 else -layerParam.Vthr
            
                    # print("column_id",column_id)
                    
                outSpikesList.append(outSpikes1)
                # perTimeStepCycle.append(maxOutcycle)
                # if t == 1:
                #     test_output1 = outSpikes1 + 0.0
                #     T,B,H,L,N = layerParam.output.shape
                #     test_output1 = test_output1.reshape(H,L,N)
                    # mask = (test_output - test_output1).abs() > 1e-3
                    # print(test_output[5][1])
                    # print(test_output1[5][1])
                    # print((test_output[5][2] - test_output1[5][2]).abs() < 1e-3)
                    # print(outSpikes1[2,:])
                    # print(row_id_list)
                spikeNumPerTimeStep.append(spikeNum)
                # print("spikeNum",spikeNum, "trueNum",torch.abs(input2D1).sum()/layerParam.LastVthr1 + torch.abs(input2D2).sum()/layerParam.LastVthr2, "inputNum:",TILE.spikeNum)
            
            outSpikesList = torch.stack(outSpikesList,dim=0)
            T,B,H,L,N = layerParam.output.shape
            outSpikesList = outSpikesList.reshape(T,H,L,N)
            # mask = (test_output - test_output1).abs() > 1e-3
            # print(test_output[mask][0:64])
            # print(test_output1[mask][0:64])
            # print(outSpikesList.shape)
            # print(output1.shape)
            # print(firstConv[:,109])
            # print(outSpikesList[0][:,109])
            # print(output1[0,0,:,109])
        
            for keyT in columnFinishCycle.keys():
                perTimeStepCycle.append(0)

            # softmax stalling
            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    perTimeStepCycle[keyT] = max(perTimeStepCycle[keyT], columnFinishCycle[keyT][key])

            for keyT in columnFinishCycle.keys():
                for key in columnFinishCycle[keyT].keys():
                    columnFinishCycle[keyT][key] = perTimeStepCycle[keyT] + 0.0

            print("InputFlitNum",InputFlitNum,"InputFlitNumberNoC",compiler.NoC.flitNumber,"outputFlitNum",OutputFlitNum,"perTimeStepCycle",perTimeStepCycle)
            # print("outSpikesList.shape",outSpikesList.shape,"output1.shape",output1.shape)
            print(f"\033[92mpoint accuracy: {torch.sum((outSpikesList - output1).abs() < 1e-3)/output1.numel()}\033[0m")
            compiler.NoC.updateLastLayersTime(curLayerId=layerParam.name,spineTokenTime=columnFinishCycle)

            totalCycle = perTimeStepCycle[-1]
            calculateInfo[layerParam.name] = {}
            calculateInfo[layerParam.name]["point accuracy"] = torch.sum(outSpikesList == output1)/output1.squeeze(1).numel()
            calculateInfo[layerParam.name]["InputFlitNum"] = InputFlitNum
            compiler.InputSearch["InputFlitNum"][layerParam.name] = InputFlitNum
            calculateInfo[layerParam.name]["outputFlitNum"] = OutputFlitNum
            compiler.InputSearch["OutputFlitNum"][layerParam.name] = OutputFlitNum
            compiler.InputSearch["avgFlitNumEachToken"][layerParam.name] = InputFlitNum/(layerParam.input.shape[-2])
            calculateInfo[layerParam.name]["perTimeStepCycle"] = perTimeStepCycle
            calculateInfo[layerParam.name]["numbers of operation"] = layerParam.ops
            calculateInfo[layerParam.name]["numbers of spikes"] = spikeNumPerTimeStep
            calculateInfo[layerParam.name]["one spike operation"] = TILE.spikeNum*TILE.M
            calculateInfo[layerParam.name]["sparsity"] = TILE.spikeNum / (TILE.N*TILE.K*T*TILE.head)

            CrossbarSwitch.fCount = OutputFlitNum
            RouteEngine.fCount = OutputFlitNum

            calculateInfo = getPEBreakdownMulti(calculateInfo,TILE.PEs,totalCycle,layerParam.name,1,False)
            calculateInfo = getRouterBreakdown(calculateInfo, flitGenerator,TILE,CrossbarSwitch, RouteEngine, totalCycle,layerParam.name,layerParam.type,isConv=False)
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
            _print_layer_bubble(compiler, layerParam.name)
            firstLayer = False        
        
        
            # print(f"\033[92mCompare Addition Output and Norm input = {((output1 - layerParam2.input.squeeze(1)).abs() < 1e-3).sum()/output1.numel()}\033[0m")    
        
        

    # print(calculateInfo)

    print("softmax_stalling_time", softmax_stalling_time)
    # 每个 token 的 bubble 占流水比例（不包含冷启动）
    if compiler.perTokenBubbleRatios:
        arr = np.array(compiler.perTokenBubbleRatios)
        print(
            "perTokenBubbleRatio (excl. cold start): count=%d, mean=%.4f, min=%.4f, max=%.4f"
            % (len(arr), float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)))
        )
        calculateInfo["perTokenBubbleRatios"] = compiler.perTokenBubbleRatios
    else:
        print("perTokenBubbleRatio: no tokens recorded")
    calculateInfo["transmitTraffic"] = compiler.NoC.transmitTraffic
    torch.save(calculateInfo, os.path.join(output_dir, calculate_info_pth))
    torch.save(columnFinishCyclePerlayer, os.path.join(output_dir, column_finish_pth))
    if input_search_pth:
        torch.save(compiler.InputSearch, os.path.join(output_dir, input_search_pth))
    return calculateInfo
