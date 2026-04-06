"""Reference im2col used for checks against async Img2Col in Im2ColTLB."""
import torch
import torch.nn.functional as F


def im2col_indices(ori_image, KH, KW, padding, stride):
    """
    Unfold spatial dimensions to columns (same layout as F.unfold).

    Parameters
    ----------
    ori_image : Tensor
        Shape (C, H, W) or (1, C, H, W).
    KH, KW : int
        Kernel size.
    padding, stride : int
        Convolution-style padding and stride.

    Returns
    -------
    Tensor of shape (OH * OW, C * KH * KW)
    """
    if ori_image.dim() == 3:
        x = ori_image.unsqueeze(0)
    elif ori_image.dim() == 4:
        x = ori_image
    else:
        raise ValueError(f"ori_image must be 3D or 4D, got shape {tuple(ori_image.shape)}")

    unfolded = F.unfold(x, kernel_size=(KH, KW), padding=padding, stride=stride)
    # (1, C*KH*KW, OH*OW) -> (OH*OW, C*KH*KW)
    return unfolded.squeeze(0).transpose(0, 1)
