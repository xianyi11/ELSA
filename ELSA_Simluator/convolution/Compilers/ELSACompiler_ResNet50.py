import sys
import os
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
import re
from PETile.Tile import Tile
from processElement.STBIFFunction import STBIFNeuron
from PETile.FlitGenerator import FlitGenerator, FlitCombiner
from tqdm import tqdm
from router.Switch import SwitchCrossbar
from router.RouteComputer import RouteComputer
from NetworkOnChip.Router import NOC
import yaml
import math
from fnmatch import fnmatch
import argparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import elsa_support  # noqa: F401 — legacy ``partition`` shim for torch.load

# repeat:
    # find the input, output, weight, bias, M and N for one layer
    # read the input shape and generate the im2col TLB
    # distribute a tile for the layer 

class OneLayerCal():
    def __init__(self):
        self.weight = None
        self.bias = None
        self.input = None
        self.inputSparsity = None
        self.input2 = None
        self.input2Sparsity = None
        self.output = None
        self.M = None
        self.N = None
        self.name = None # name in network
        self.type = None # conv, linear, average pooling or residual addition
        self.ops = 0
    
    def printmyself(self):
        print(f"=========================={self.name[:-1]}============================")
        if self.type is not None:
            print("self.type",self.type)
        if self.weight is not None:
            print("self.weight.shape",self.weight.shape)
        if self.bias is not None:
            print("self.bias.shape",self.bias.shape)
        if self.input is not None:
            print("self.input.shape",self.input.shape)
            print("self.inputSparsity",self.inputSparsity)
            print("negative spike percent:", torch.sum(self.input < 0)/torch.sum(torch.abs(self.input)))
        if self.input2 is not None:
            print("self.input2.shape",self.input2.shape)
            print("self.input2Sparsity",self.input2Sparsity)
        if self.output is not None:
            print("self.output.shape",self.output.shape)
        if self.M is not None:
            print("self.M, self.N",self.M, self.N)
        # print("self.ops",self.ops)

class VSACompiler(nn.Module):
    def __init__(self, networkDir, connection_path, mapping_path, tranOccupy_path, config_path=None):
        super(VSACompiler,self).__init__()
        
        from elsa_support.paths import CONFIG_YAML
        if config_path is None:
            config_path = str(CONFIG_YAML)
        self.cfg = None
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        self.dir = networkDir
        self.layers = []
        self.NoC = NOC(connection_path, mapping_path, tranOccupy_path)
        self.connection_path = connection_path

        # 用来做贝叶斯优化的datasheet
        # 算法端：
        self.FlitNumberEachLayer = {}
        self.SpikeNumberEachLayer = {}
        self.WidthAfterTiling = {}
        self.SRAMNumPerLayer = {}
        self.avgFlitNumEachLayer = {}
        self.kernelStridePerLayer = {}
        self.receptiveField = {}
        # 硬件端：
        self.VSAWidth = self.cfg["processElement"]["accumulator"]["adderNum"]
        self.totalSRAMNum = self.cfg["TILE"]["SRAMTotalNum"]

        self.collectLayers()
        for layer in self.layers:
            layer.printmyself()
    
    def allocatePETile(self):
        
        pass
    
    def collectLayers(self):
        # fileHandler = open(f"{self.dir}/calculationOrder.txt",  "r")
        connection = torch.load(self.connection_path)
        # weightlist.append(weightlist[2])
        # del weightlist[2]
        for i, name in enumerate(list(connection.keys())):
            
            # get the all the file position for one layer
            onelayerfiles = glob(self.dir+f"/*{name[:-1]}_*") + glob(self.dir+f"/*{name[:-1]}.*")
            # print(onelayerfiles)
            # read the input, output, weight, bias, M and N for one layer
            curlayerParam = OneLayerCal()          
            curlayerParam.name = name

            for file in onelayerfiles:
                print(file)
                if file.count(".in") > 0: # input feature map
                    input = np.fromfile(file,dtype=np.int8, count=-1, offset=0)
                    shapestr = re.findall("=\d+", file.split(".")[-2])
                    if len(shapestr) == 5:
                        input.shape = int(shapestr[0][1:]), int(shapestr[1][1:]), int(shapestr[2][1:]), int(shapestr[3][1:]), int(shapestr[4][1:])
                    elif len(shapestr) == 3:
                        input.shape = int(shapestr[0][1:]), int(shapestr[1][1:]), int(shapestr[2][1:])
                    torch_input  = torch.from_numpy(input).type(torch.float)
                    if file.count(".input2") > 0: # residual add
                        curlayerParam.input2 = torch_input
                        curlayerParam.input2Sparsity = torch_input.abs().sum()/torch_input.numel()
                    else:
                        curlayerParam.input = torch_input
                        curlayerParam.inputSparsity = torch_input.abs().sum()/torch_input.numel()
                    # print(torch_input)
                    # print(curlayerParam.input[3].abs().sum(),curlayerParam.input[4].abs().sum())

                elif file.count(".out") > 0: # output feature map
                    output = np.fromfile(file,dtype=np.int8, count=-1, offset=0)
                    shapestr = re.findall("\d+", file.split(".")[-2])
                    if len(shapestr) == 5:
                        output.shape = int(shapestr[0]), int(shapestr[1]), int(shapestr[2]), int(shapestr[3]), int(shapestr[4])
                    elif len(shapestr) == 3:
                        output.shape = int(shapestr[0]), int(shapestr[1]), int(shapestr[2])
                    torch_output  = torch.from_numpy(output).type(torch.float)
                    curlayerParam.output = torch_output
                
                elif file.count("weight") > 0: # synaptic weight
                    weight = np.fromfile(file,dtype=np.int8, count=-1, offset=0)
                    shapestr = re.findall("=\d+", file.split(".")[-2])
                    if len(shapestr) == 4:
                        weight.shape = int(shapestr[0][1:]), int(shapestr[1][1:]), int(shapestr[2][1:]), int(shapestr[3][1:])
                    elif len(shapestr) == 2:
                        weight.shape = int(shapestr[0][1:]), int(shapestr[1][1:])
                    torch_weight  = torch.from_numpy(weight).type(torch.float)
                    curlayerParam.weight = torch_weight

                elif file.count("bias") > 0: # synaptic bias
                    bias = np.fromfile(file,dtype=np.int8, count=-1, offset=0)
                    shapestr = re.findall("=\d+", file.split(".")[-2])
                    bias.shape = int(shapestr[0][1:])
                    torch_bias  = torch.from_numpy(bias).type(torch.float)
                    curlayerParam.bias = torch_bias

                elif file.count("M") > 0: # quantization M
                    M = np.fromfile(file,dtype=np.int8, count=-1, offset=0)
                    M = 256 + torch.from_numpy(M).type(torch.int32)[-1]
                    curlayerParam.M = M 
                    
                elif file.count("N") > 0: # quantization M
                    N = np.fromfile(file,dtype=np.int8, count=-1, offset=0)
                    N = torch.from_numpy(N).type(torch.int32)
                    curlayerParam.N = N

            if name.count("pool") > 0:
                curlayerParam.type = "pool"
            elif name.count("spikeResidual") > 0:
                curlayerParam.type = "residual"
            elif name.count("fc") > 0 or name.count("classifier") > 0:
                curlayerParam.type = "linear"
            elif name.count("downsample") > 0 and curlayerParam.weight is None:
                curlayerParam.type = "quantize"
            else:
                if curlayerParam.weight is None:
                    curlayerParam.type = "pool"
                else:
                    curlayerParam.type = "conv"

            # curlayerParam.printmyself()
            if curlayerParam.type == 'conv':
                N = curlayerParam.weight.shape[0]
                K = curlayerParam.weight.shape[1]*curlayerParam.weight.shape[2]*curlayerParam.weight.shape[3]
                M = curlayerParam.output.shape[3]*curlayerParam.output.shape[4]
                self.WidthAfterTiling[curlayerParam.name] = N//self.cfg["TILE"]["PENum"]
                curlayerParam.ops = N*K*M
                # calculate the number of element in a SRAM
                WeightSRAMElementNum = self.cfg["processElement"]["weightBuffer"]["height"]*self.cfg["processElement"]["weightBuffer"]["inoutWidth"]
                MembraneSRAMElementNum = self.cfg["processElement"]["membrane"]["height"]*self.cfg["processElement"]["membrane"]["inoutWidth"]
                SpikeTracerSRAMElementNum = self.cfg["processElement"]["spikeTracer"]["height"]*self.cfg["processElement"]["spikeTracer"]["inoutWidth"]
                WeightRatio = 1
                MembraneRatio = self.cfg["processElement"]["membrane"]["bitwitdh"]/self.cfg["processElement"]["weightBuffer"]["bitwitdh"]
                SpiketracerRatio = self.cfg["processElement"]["spikeTracer"]["bitwitdh"]/self.cfg["processElement"]["weightBuffer"]["bitwitdh"]
                # the number of weight SRAM
                self.SRAMNumPerLayer[curlayerParam.name] = math.ceil(N*K*WeightRatio/WeightSRAMElementNum)
                # the number of Membrane SRAM
                self.SRAMNumPerLayer[curlayerParam.name] = self.SRAMNumPerLayer[curlayerParam.name] + math.ceil(N*M*MembraneRatio/MembraneSRAMElementNum)
                # the number of SpikeTracer SRAM
                self.SRAMNumPerLayer[curlayerParam.name] = self.SRAMNumPerLayer[curlayerParam.name] + math.ceil(N*M*SpiketracerRatio/SpikeTracerSRAMElementNum)
                # the kernel stride of convolution:
                self.kernelStridePerLayer[curlayerParam.name] = [curlayerParam.weight.shape[-1],curlayerParam.input.shape[-1]//curlayerParam.output.shape[-1]]
            elif curlayerParam.type == 'pool':
                N = curlayerParam.output.shape[2]
                M = curlayerParam.output.shape[3]*curlayerParam.output.shape[4]
                kernelSize = curlayerParam.input.shape[3]/curlayerParam.output.shape[3]
                K = curlayerParam.output.shape[2]*kernelSize*kernelSize
                curlayerParam.ops = N*K*M
                # the kernel stride of pooling:
                if curlayerParam.name.count("maxpool") > 0:
                    self.kernelStridePerLayer[curlayerParam.name] = [3,curlayerParam.input.shape[-1]//curlayerParam.output.shape[-1]]
                elif curlayerParam.name.count("avgpool") > 0:
                    self.kernelStridePerLayer[curlayerParam.name] = [1,curlayerParam.input.shape[-1]//curlayerParam.output.shape[-1]]
            # elif curlayerParam.type == "residual" or curlayerParam.type == "quantize":
            #     curlayerParam.ops = curlayerParam.input.shape[2]*curlayerParam.output.shape[3]*curlayerParam.output.shape[4]
            elif curlayerParam.type == "linear":
                M = curlayerParam.input.shape[1]
                K = curlayerParam.weight.shape[1]
                N = curlayerParam.weight.shape[0]
                self.WidthAfterTiling[curlayerParam.name] = N//self.cfg["TILE"]["PENum"]
                curlayerParam.ops = N*K*M
                # calculate the number of element in a SRAM
                WeightSRAMElementNum = self.cfg["processElement"]["weightBuffer"]["height"]*self.cfg["processElement"]["weightBuffer"]["inoutWidth"]
                MembraneSRAMElementNum = self.cfg["processElement"]["membrane"]["height"]*self.cfg["processElement"]["membrane"]["inoutWidth"]
                SpikeTracerSRAMElementNum = self.cfg["processElement"]["spikeTracer"]["height"]*self.cfg["processElement"]["spikeTracer"]["inoutWidth"]
                WeightRatio = 1
                MembraneRatio = self.cfg["processElement"]["membrane"]["bitwitdh"]/self.cfg["processElement"]["weightBuffer"]["bitwitdh"]
                SpiketracerRatio = self.cfg["processElement"]["spikeTracer"]["bitwitdh"]/self.cfg["processElement"]["weightBuffer"]["bitwitdh"]
                # the number of weight SRAM
                self.SRAMNumPerLayer[curlayerParam.name] = math.ceil(N*K*WeightRatio/WeightSRAMElementNum)
                # the number of Membrane SRAM
                self.SRAMNumPerLayer[curlayerParam.name] = self.SRAMNumPerLayer[curlayerParam.name] + math.ceil(N*M*MembraneRatio/MembraneSRAMElementNum)
                # the number of SpikeTracer SRAM
                self.SRAMNumPerLayer[curlayerParam.name] = self.SRAMNumPerLayer[curlayerParam.name] + math.ceil(N*M*SpiketracerRatio/SpikeTracerSRAMElementNum)
                # the kernel stride of linear:
                self.kernelStridePerLayer[curlayerParam.name] = [1,1]
            self.layers.append(curlayerParam)

# def test_VSACompiler():
#     compiler = VSACompiler(r"D:\tools\HPCA2025\simulator\output_bin_snn")
#     compiler.collectLayers()
#     print("compiler.layers[10]",compiler.layers[10].output.sum())
#     compiler.layers[10].printmyself()
#     torch.save(compiler.layers[10],"onelayerParamForTest.pth")

def get_Receptive_Field(compiler):
    """
    Propagate receptive field along ``connection`` (reversed layer order).

    Legacy VGG scripts iterate the full reversed ``keyList`` (no ``keyList[1:]``).
    Skip edges whose endpoint is missing ``kernelStridePerLayer``. Before using
    ``receptiveField[key]`` on an edge, ensure ``key`` is seeded: if it was never
    filled by a downstream layer (unusual graph order / skipped nodes), initialize
    it with the same local RF as the original ``i == 0`` branch so we never
    KeyError on ``receptiveField[key]``.
    """
    connection = torch.load(compiler.connection_path)
    keyList = list(connection.keys())
    keyList.reverse()
    for key in keyList:
        if key not in compiler.kernelStridePerLayer:
            continue
        if key not in compiler.receptiveField:
            compiler.receptiveField[key] = compiler.kernelStridePerLayer[key][1] * (1 - 1) + compiler.kernelStridePerLayer[key][0]
        for lastkey in connection[key]:
            if lastkey not in compiler.kernelStridePerLayer:
                continue
            if lastkey not in compiler.receptiveField:
                compiler.receptiveField[lastkey] = compiler.kernelStridePerLayer[lastkey][1] * (compiler.receptiveField[key] - 1) + compiler.kernelStridePerLayer[lastkey][0]
    print("compiler.receptiveField", compiler.receptiveField)

def getRouterBreakdown(calculateInfo, FlitGenerator ,TILE, CrossbarSwitch, RouteEngine, totalCycle, name, isConv=True):  
    latency = totalCycle/200000000
    if isConv:
        AysnImg2ColArea = TILE.Im2ColUnit.getArea()
        AysnImg2ColEnergy = TILE.Im2ColUnit.calEnergy(latency)
        VSAOrderControllerArea = TILE.VSAOrderCtrl.getArea()
        VSAOrderControllerEnergy = TILE.VSAOrderCtrl.calEnergy(latency)
        VSAUpdateArbiterArea = TILE.VSAOrderArbiter.getArea()
        VSAUpdateArbiterEnergy = TILE.VSAOrderArbiter.calEnergy(latency)
    FlitGeneratorArea = FlitGenerator.getArea()
    FlitGeneratorEnergy = 0.0
    for flitGenerator in TILE.flitGenerators:        
        FlitGeneratorEnergy = FlitGeneratorEnergy + flitGenerator.calEnergy(latency)
    FlitDecoderArea = TILE.flitCombiner.getArea()
    FlitDecoderEnergy = TILE.flitCombiner.calEnergy(latency)
    RoutingEngineArea = RouteEngine.getArea()
    RoutingEngineEnergy = RouteEngine.calEnergy(latency)
    CrossbarSwitchArea = CrossbarSwitch.getArea()
    CrossbarSwitchEnergy = CrossbarSwitch.calEnergy(latency)
    
    if isConv:
        calculateInfo[name]["AysnImg2ColArea"] = AysnImg2ColArea
        calculateInfo[name]["AysnImg2ColEnergy"] = AysnImg2ColEnergy
        calculateInfo[name]["VSAOrderControllerArea"] = VSAOrderControllerArea
        calculateInfo[name]["VSAOrderControllerEnergy"] = VSAOrderControllerEnergy
        calculateInfo[name]["VSAUpdateArbiterArea"] = VSAUpdateArbiterArea
        calculateInfo[name]["VSAUpdateArbiterEnergy"] = VSAUpdateArbiterEnergy
    calculateInfo[name]["FlitGeneratorArea"] = FlitGeneratorArea
    calculateInfo[name]["FlitGeneratorEnergy"] = FlitGeneratorEnergy
    calculateInfo[name]["FlitDecoderArea"] = FlitDecoderArea
    calculateInfo[name]["FlitDecoderEnergy"] = FlitDecoderEnergy
    calculateInfo[name]["RoutingEngineArea"] = RoutingEngineArea
    calculateInfo[name]["RoutingEngineEnergy"] = RoutingEngineEnergy    
    calculateInfo[name]["CrossbarSwitchArea"] = CrossbarSwitchArea
    calculateInfo[name]["CrossbarSwitchEnergy"] = CrossbarSwitchEnergy    
    
    return calculateInfo

    
def getPEBreakdown(calculateInfo, PEs, totalCycle, name, firstLayer):
    membraneArea = 0.0
    mambraneEnergy = 0.0
    spikeTracerArea = 0.0
    spikeTracerEnergy = 0.0
    weightBufferArea = 0.0
    weightBufferEnergy = 0.0
    inputBufferArea = 0.0
    inputBufferEnergy = 0.0
    adderVectorArea = 0.0
    adderVectorEnergy = 0.0
    fireComponentArea = 0.0
    fireComponentEnergy = 0.0

    for PE in PEs:
        membraneArea = membraneArea + PE.membrane.getArea()
        mambraneEnergy = mambraneEnergy + PE.membrane.calEnergy(totalCycle/200000000)
        
        spikeTracerArea = spikeTracerArea + PE.spikeTracer.getArea()
        spikeTracerEnergy = spikeTracerEnergy + PE.spikeTracer.calEnergy(totalCycle/200000000)
        
        weightBufferArea = weightBufferArea + PE.weightBuffer.getArea()
        weightBufferEnergy = weightBufferEnergy + PE.weightBuffer.calEnergy(totalCycle/200000000)
        
        inputBufferArea = inputBufferArea + PE.inputBuffer.getArea()
        inputBufferEnergy = inputBufferEnergy + PE.inputBuffer.calEnergy(totalCycle/200000000)    

        adderVectorArea = adderVectorArea + (PE.adder.getArea() if firstLayer else PE.adderTree.getArea())
        adderVectorEnergy = adderVectorEnergy + (PE.adder.calEnergy(totalCycle/200000000) if firstLayer else PE.adderTree.calEnergy(totalCycle/200000000))

        fireComponentArea = fireComponentArea + PE.fireComponent.getArea()
        fireComponentEnergy = fireComponentEnergy + PE.fireComponent.calEnergy(totalCycle/200000000)    

    # print("PE.adder.fcount",PE.adder.fCount)
    # print("PE.fireComponent.fcount",PE.fireComponent.fCount)
    
    readCount = 0
    writeCount = 0
    for sram in PE.membrane.srams:
        readCount = readCount + sram.readCount
        writeCount = writeCount + sram.writeCount
    # print("PE.membrane.readCount",readCount)
    # print("PE.membrane.writeCount",writeCount)

    readCount = 0
    writeCount = 0
    for sram in PE.spikeTracer.srams:
        readCount = readCount + sram.readCount
        writeCount = writeCount + sram.writeCount
    # print("PE.spikeTracer.readCount",readCount)
    # print("PE.spikeTracer.writeCount",writeCount)

    readCount = 0
    writeCount = 0
    for sram in PE.weightBuffer.srams:
        readCount = readCount + sram.readCount
        writeCount = writeCount + sram.writeCount
    # print("PE.weightBuffer.readCount",readCount)
    # print("PE.weightBuffer.writeCount",writeCount)


    readCount = 0
    writeCount = 0
    for sram in PE.inputBuffer.srams:
        readCount = readCount + sram.readCount
        writeCount = writeCount + sram.writeCount
    # print("PE.inputBuffer.readCount",readCount)
    # print("PE.inputBuffer.writeCount",writeCount)

    calculateInfo[name]["membraneArea"] = membraneArea
    calculateInfo[name]["mambraneEnergy"] = mambraneEnergy
    calculateInfo[name]["spikeTracerArea"] = spikeTracerArea
    calculateInfo[name]["spikeTracerEnergy"] = spikeTracerEnergy
    calculateInfo[name]["weightBufferArea"] = weightBufferArea
    calculateInfo[name]["weightBufferEnergy"] = weightBufferEnergy
    calculateInfo[name]["inputBufferArea"] = inputBufferArea
    calculateInfo[name]["inputBufferEnergy"] = inputBufferEnergy
    calculateInfo[name]["adderVectorArea"] = adderVectorArea
    calculateInfo[name]["adderVectorEnergy"] = adderVectorEnergy
    calculateInfo[name]["fireComponentArea"] = fireComponentArea
    calculateInfo[name]["fireComponentEnergy"] = fireComponentEnergy

    return calculateInfo


def run_resnet50_vary_t(
    datapath,
    connectionpath,
    mappingpath,
    occupyPath,
    config_path=None,
    time_step=24,
    output_dir=None,
):
    from elsa_support.unified_elisa_pipeline import run_unified_elisa_pipeline

    return run_unified_elisa_pipeline(
        datapath,
        connectionpath,
        mappingpath,
        occupyPath,
        config_path=config_path,
        time_step=time_step,
        output_dir=output_dir,
        search_input_simple="SearchInput_ResNet50_a4w4_with_Noc_simple.pth",
        calculate_info_pth="calculateInfoConv_ResNet50_a4w3_with_Noc_rep.pth",
        communication_traffic_pth="calculateInfoConv_ResNet50_a4w3_communicationtraffic_rep.pth",
        column_finish_pth="columnFinishCyclePerlayer_ResNet50_a4w3_with_Noc_rep.pth",
        search_input_final="SearchInput_ResNet50_a4w3_with_Noc_rep.pth",
    )


def _parse_args_standalone():
    _d = _ROOT / "datas" / "ResNet50_data"
    parser = argparse.ArgumentParser(description="ResNet50 VSA compiler (vary T)")
    parser.add_argument("--Time_step", type=int, default=24)
    parser.add_argument(
        "--datapath",
        default="/data/kang_you/output_bin_snn_resnet50_w4_a3_T16/",
        help="Directory of exported layer binaries (model output)",
    )
    parser.add_argument(
        "--connectionpath",
        default=str(_d / "ResNet50_Connection_Last.pth"),
    )
    parser.add_argument(
        "--mappingpath",
        default=str(_d / "Final_MeshMapping_ResNet50.pth"),
    )
    parser.add_argument(
        "--occupyPath",
        default=str(_d / "TranOccupy_ResNet50.pth"),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Config.yaml (default: convolution/configs/Config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for .pth outputs (default: convolution/outputs)",
    )
    parser.add_argument(
        "--metrics-mode",
        choices=["none", "file", "run"],
        default="run",
        help="none: only simulation; file: load --info-path and print metrics only; run: simulate then print metrics",
    )
    parser.add_argument(
        "--info-path",
        default=None,
        help="calculateInfo *.pth for --metrics-mode file",
    )
    parser.add_argument(
        "--metrics-plots",
        action="store_true",
        help="Show energy/area pie charts (needs matplotlib)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args_standalone()
    if _args.metrics_mode == "file":
        if not _args.info_path:
            raise SystemExit("--info-path is required when --metrics-mode file")
        from elsa_support.resnet50_metrics import compute_resnet50_metrics_from_path

        compute_resnet50_metrics_from_path(_args.info_path, plot=_args.metrics_plots)
    else:
        _info = run_resnet50_vary_t(
            datapath=_args.datapath,
            connectionpath=_args.connectionpath,
            mappingpath=_args.mappingpath,
            occupyPath=_args.occupyPath,
            config_path=_args.config,
            time_step=_args.Time_step,
            output_dir=_args.output_dir,
        )
        if _args.metrics_mode == "run":
            from elsa_support.resnet50_metrics import compute_resnet50_metrics

            compute_resnet50_metrics(_info, plot=_args.metrics_plots)
