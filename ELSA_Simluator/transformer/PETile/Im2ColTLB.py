import numpy
import torch
import torch.nn as nn
from basicModule import VSAModule
import yaml
from elsa_support.paths import CONFIG_YAML
import math
from functions.im2colGroundtruth import im2col_indices
from processElement.STBIFFunction import STBIFNeuron

cfg = None
with open(CONFIG_YAML, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


class VSAOrderController(VSAModule):
    def __init__(self):
        super(VSAOrderController,self).__init__()
        self.fEnergy = cfg["TILE"]["VSAOrderController"]["fEnergy"]
        self.area = cfg["TILE"]["VSAOrderController"]["area"]
        self.staticPower = cfg["TILE"]["VSAOrderController"]["leakage"]

    def forward(self,i,j):
        # given (i,j) output next column position
        self.fCount = self.fCount + 1
        curMaxIndex = max(i,j)
        if i > 0:
            if i == curMaxIndex and j == curMaxIndex:
                return (i-1,j)
            elif i == curMaxIndex:
                return (i,j+1)
            elif j == curMaxIndex:
                return (i-1,j)
        else:
            return (curMaxIndex+1, 0)
            
        
class VSAUpdateArbiter(VSAModule):
    def __init__(self, KH, KW, stride, padding, IH, IW):
        super(VSAUpdateArbiter,self).__init__()
        self.fEnergy = cfg["TILE"]["VSAUpdateArbiter"]["fEnergy"]
        self.area = cfg["TILE"]["VSAUpdateArbiter"]["area"]
        self.staticPower = cfg["TILE"]["VSAUpdateArbiter"]["leakage"]
        self.padding = padding
        self.KH = KH
        self.KW = KW
        self.IH = IH
        self.IW = IW
        self.stride = stride
        self.VSAOrder = VSAOrderController()

    def forward(self,i,j):
        self.fCount = self.fCount + 1
        i = i + self.padding
        j = j + self.padding

        curMaxIndex = max(i,j)
        update = i!=j and (curMaxIndex - i + 1 >= self.KH and i<j or j+1 >= self.KW and i>j) 
            
        outputI = outputJ = 0
        if update and i > j:
            outputI = i - self.KH + 1
            outputJ = j - self.KW + 1
        elif update and i < j:
            outputI = i
            outputJ = j - self.KW + 1

            
        # consider the stride
        update = update and outputI%self.stride == 0 and outputJ%self.stride == 0
        
        outputIList = []
        outputJList = []
        if update:
            outputIList.append(outputI//self.stride)
            outputJList.append(outputJ//self.stride)

        if i == self.padding and j == self.padding + self.IW - 1: # begin calculating padding
            calcuNum = (self.IW + 2*self.padding)*self.padding + (self.IH + 2*self.padding)*self.padding - self.padding*self.padding
            i1,j1 = self.IH+self.padding,0
            for index in range(calcuNum):
                curMaxIndex = max(i1,j1)
                update = i1!=j1 and (curMaxIndex - i1 + 1 >= self.KH and i1<j1 or j1+1 >= self.KW and i1>j1) 
                    
                outputI = outputJ = 0
                if update and i1 > j1:
                    outputI = i1 - self.KH + 1
                    outputJ = j1 - self.KW + 1
                elif update and i1 < j1:
                    outputI = i1
                    outputJ = j1 - self.KW + 1


                # consider the stride
                update = update and outputI%self.stride == 0 and outputJ%self.stride == 0
                # print(f"2: update={update}, i={i1}, j={j1}, curMaxIndex={curMaxIndex}, self.KH={self.KH}, self.KW={self.KW}, outputI={outputI}, outputJ={outputJ}")
                
                if update:
                    outputIList.append(outputI//self.stride)
                    outputJList.append(outputJ//self.stride)
                i1,j1 = self.VSAOrder(i1,j1)
        
        if i == self.padding:
            while i > 0:
                i = i - 1
                curMaxIndex = max(i,j)
                update = i!=j and (curMaxIndex - i + 1 >= self.KH and i<j or j+1 >= self.KW and i>j) 
                    
                # print(f"update={update}, i={i}, j={j}, curMaxIndex={curMaxIndex}, self.KH={self.KH}, self.KW={self.KW}")
                outputI = outputJ = 0
                if update and i > j:
                    outputI = i - self.KH + 1
                    outputJ = j - self.KW + 1
                elif update and i < j:
                    outputI = i
                    outputJ = j - self.KW + 1

                update = update and outputI%self.stride == 0 and outputJ%self.stride == 0
                if update:
                    outputIList.append(outputI//self.stride)
                    outputJList.append(outputJ//self.stride)
                # print(f"2: update={update}, i={i}, j={j}, curMaxIndex={curMaxIndex}, self.KH={self.KH}, self.KW={self.KW}")
        
        # print(f"update={update}, i={i}, j={j}, curMaxIndex={curMaxIndex}, self.KH={self.KH}, self.KW={self.KW}")

        # return update, outputI//self.stride, outputJ//self.stride

        # print(outputIList,outputJList)
        if len(outputIList) == 0:
            return False, [], []
        else:
            # print(f"update=True, outputIList={outputIList}, outputJList={outputJList}")
            return True, outputIList, outputJList
        
    
class Img2ColTLB(VSAModule):
    def __init__(self, stride, KW, KH, IW, IH, padding):
        super(Img2ColTLB,self).__init__()
        self.stride = stride
        self.KW = KW
        self.KH = KH
        self.IW = IW
        self.IH = IH
        self.padding = padding
        self.OH = math.floor((IH+2*padding-(KH-1)-1)/stride + 1)
        self.OW = math.floor((IW+2*padding-(KW-1)-1)/stride + 1)
        # self.IW = self.IW + 2*self.padding
        # self.IH = self.IH + 2*self.padding
        
        self.fEnergy = cfg["TILE"]["Im2ColUnit"]["fEnergy"]
        self.area = cfg["TILE"]["Im2ColUnit"]["area"]
        self.staticPower = cfg["TILE"]["Im2ColUnit"]["leakage"]
        self.VSAOrderCtrl = VSAOrderController()
        
    def forward(self,spike):
        k,i,sign = spike
        self.fCount = self.fCount + 1
        outSpikeInColumn = []

        j = i%self.IW
        i = i//self.IW

        j = j+self.padding
        i = i+self.padding

        IW = self.IW + 2*self.padding
        IH = self.IH + 2*self.padding

        for i_kh in range(0,self.KH):
            for j_kw in range(0,self.KW):
                if i+i_kh >= IH or j+j_kw >= IW or i-self.KH+1+i_kh<0 or j-self.KW+1+j_kw<0 or (i-self.KH+1+i_kh)%self.stride!=0 or (j-self.KW+1+j_kw)%self.stride!=0:
                    continue
                rowI = ((i-self.KH+i_kh+1)*self.OW + j-self.KW+j_kw+1)//self.stride
                colI = (self.KH - i_kh - 1)*self.KH + (self.KW - j_kw - 1) + self.KW * self.KH * k
                # print(f"i={i},i_kh={i_kh},KH={self.KH},j={j},j_kw={j_kw},KW={self.KW},rowI={colI},colI={rowI}")
                outSpikeInColumn.append((colI,rowI,sign))
        
        return outSpikeInColumn
        

def test_VSAOrderController():
    featureMap2D = torch.zeros(4,4)
    VSAOrderCtrl = VSAOrderController()
    i = j = 0
    for k in range(16):
        featureMap2D[i,j] = k
        i,j = VSAOrderCtrl(i,j)

    print(featureMap2D)
    
def test_VSAUpdateArbiter():
    
    featureMap2D = torch.zeros(14,14)
    isUpdate = torch.zeros(14,14)
    VSAOrderCtrl = VSAOrderController()
    VSAOrderArbiter = VSAUpdateArbiter(KH=7,KW=7,stride=2,padding=3,IH=14,IW=14)
    i = j = 0
    for k in range(196):
        featureMap2D[i,j] = k
        update, outputI, outputJ = VSAOrderArbiter(i,j)
        if update:
            print("i,j",i,j,"outputI, outputJ",outputI, outputJ)
        if update:
            isUpdate[i,j] = 1
        i,j = VSAOrderCtrl(i,j)

    print(featureMap2D)
    print(isUpdate)
    
def test_Img2ColTLB():
    
    ori_image = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
    
    C,H,W = ori_image.shape
    image = ori_image.reshape(C,H*W)
    
    stride, KW, KH, padding = 1,2,2,0
    
    OH = math.floor((H+2*padding-(KH-1)-1)/stride + 1)
    OW = math.floor((W+2*padding-(KW-1)-1)/stride + 1)    

    column = torch.zeros((OH*OW,KH*KW*C))
    asynImg2Col = Img2ColTLB(stride, KW, KH, W, H, padding)
    
    for k in range(C):
        for i in range(H*W):
            outcolumns = asynImg2Col((k,i,0))
            for colI,rowI,sign in outcolumns:
                # print(f"rowI={rowI},colI={colI},k={k},i={i},image[k][i]={image[k][i]}")
                column[rowI,colI] = image[k][i]

    column = column.int()
    ori_image = ori_image.unsqueeze(0)
    columnTrue = im2col_indices(ori_image,KH,KW,padding,stride)
    
    print(column.shape)
    print(column.T)
    print(columnTrue.shape)
    print(columnTrue)
    

def test_Tile_convolution_bias():
    from copy import deepcopy

    torch.set_printoptions(profile="full")
    
    layerParam = torch.load(r"D:\tools\HPCA2025\simulator_CARBON\onelayerParamForTest.pth")
    print(layerParam.printmyself())

    M1 = layerParam.M
    N1 = layerParam.N
    bias = layerParam.bias
    weight = layerParam.weight
    input = layerParam.input
    groudtruth = layerParam.output
    neuron = STBIFNeuron(M=M1,N=N1,pos_max=7,neg_min=0,bias=bias.unsqueeze(1))    
    T = layerParam.input.shape[0]
    outputList = []
    stride = layerParam.input.shape[-1]//layerParam.output.shape[-1]
    KW = layerParam.weight.shape[-1]
    K = layerParam.weight.shape[1]*layerParam.weight.shape[2]*layerParam.weight.shape[3]
    firstConv = None
    
    wx = torch.nn.functional.conv2d(input[0], weight, stride=stride, padding=KW//2)
    wx = wx.reshape(wx.shape[1],-1)
    
    stride, KW, KH, padding = 2,7,7,3
    W,H,OW,OH,C = 224,224,112,112,3
    # columnTrue = im2col_indices(input[0],KH,KW,padding,stride)
    column = torch.zeros((OH*OW,KH*KW*C))
    asynImg2Col = Img2ColTLB(stride, KW, KH, W, H, padding)
    input2D = input[0].reshape(C,W*H)
    
    for k in range(C):
        for i in range(H*W):
            outcolumns = asynImg2Col((k,i,0))
            for colI,rowI,sign in outcolumns:
                # print(f"rowI={rowI},colI={colI},k={k},i={i},image[k][i]={image[k][i]}")
                column[rowI,colI] = input2D[k][i]

    weight2D = weight.reshape(weight.shape[0],-1)
    output = weight2D@column.T
    
    print(wx[:,0])
    print(output[:,0])
    
    assert (wx == output).all(), "output1 != groudtruth"

