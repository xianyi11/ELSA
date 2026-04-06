import torch
from functions.im2colGroundtruth import im2col_indices
import math

def VASOrderTable(i,j):
    if i>=j:
        return i*i + j +max((i+j+2)*(j-i), 0)
    else:
        return i*i + i +max((i+j+2)*(j-i), 0)
    
def NormalOrderTable(i,j):
    return i*3+j


def asynImg2Col(i,k, stride, KW, KH, IW, IH, padding):
    '''
    Input:  i,k: are the position in feature, corresponding to (row*column, channel);
            stride: is the stride param in convoluction
            KW, KH: are kernel size
            IW, IH: are input feature map size
    warning: the feature map height should equal to the width
    '''
    assert IW==IH, "the height of the feature map must equal the width."
    # print("i,j,k, stride, KW, KH, IW, IH",i,j,k, stride, KW, KH, IW, IH)
    
    outPositiones = []
    j = i%IW
    i = i//IW

    OH = math.floor((IH+2*padding-(KH-1)-1)/stride + 1)
    OW = math.floor((IW+2*padding-(KW-1)-1)/stride + 1)
    
    i = i+padding
    j = j+padding
    IW = IW + 2*padding
    IH = IH + 2*padding
    
    for i_kh in range(0,KH,stride):
        for j_kw in range(0,KW,stride):
            if i+i_kh >= IH or j+j_kw >= IW or i-KH+1+i_kh<0 or j-KW+1+j_kw<0:
                continue
            rowI = (i-KH+i_kh+1)*OW + j-KW+j_kw+1
            colI = KH - i_kh - 1 + (KW - j_kw - 1) * KH + KW * KH * k
            print(f"i={i},i_kh={i_kh},KH={KH},j={j},j_kw={j_kw},KW={KW},rowI={rowI},colI={colI}")
            outPositiones.append((rowI,colI))
    return outPositiones



# def test_asynImg2Col():
    
#     image = torch.tensor([[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]])
#     print(image.shape)
    
#     C,H,W = image.shape
#     # image.reshape(C,H*W)
    
#     stride, KW, KH = 1,2,2
    
#     OH = OW = 3
    
#     column = torch.zeros((OH*OW,KH*KW*C))
    
#     for k in range(C):
#         for i in range(H):
#             for j in range(W):
#                 outcolumns = asynImg2Col(i,j,k,stride, KW, KH, W,H)
#                 for rowI,colI in outcolumns:
#                     column[rowI,colI] = image[k][i][j]

#     column = column.int()
#     image = image.unsqueeze(0)
#     columnTrue = im2col_indices(image,KH,KW,0,1)
#     columnTrue = columnTrue.T
    
#     print(image.shape,image)
#     print(columnTrue.shape,columnTrue)
#     print(column.shape,column)


def test_asynImg2Col_1():
    
    ori_image = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
    print(ori_image.shape)
    
    C,H,W = ori_image.shape
    image = ori_image.reshape(C,H*W)
    
    stride, KW, KH, padding = 1,3,3,1
    
    OH = math.floor((H+2*padding-(KH-1)-1)/stride + 1)
    OW = math.floor((W+2*padding-(KW-1)-1)/stride + 1)    

    column = torch.zeros((OH*OW,KH*KW*C))
    
    for k in range(C):
        for i in range(H*W):
                outcolumns = asynImg2Col(i,k,stride, KW, KH, W,H, padding)
                for rowI,colI in outcolumns:
                    print(f"rowI={rowI},colI={colI},k={k},i={i}")
                    column[rowI,colI] = image[k][i]

    column = column.int()
    ori_image = ori_image.unsqueeze(0)
    columnTrue = im2col_indices(ori_image,KH,KW,1,1)
    
    print(image)
    print(columnTrue.shape)
    print(columnTrue)
    print(column.shape)
    print(column.T)


# def test_gen_Img2ColTLB():
#     import yaml
#     cfg = None
#     with open('../Config.yaml', 'r', encoding='utf-8') as f:
#         cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
#     M = cfg["processElement"]["input"]["M"]
#     K = cfg["processElement"]["input"]["K"]
#     N = cfg["processElement"]["tile"]["tileN"]
    
    