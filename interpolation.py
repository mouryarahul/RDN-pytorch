import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from skimage.io import imread
from skimage.transform import resize, rescale
import cv2 as cv




def cv2_resize(img_in: np.ndarray, scale: float, order: int):
    if img_in.ndim == 2:
        M,N = img_in.shape
    elif img_in.ndim == 3 and img_in.shape[2] == 3:
        C,M,N = img_in.shape
    else:
        raise Exception("Input image should be single or three color channels!")

    P,Q = (int(M*scale), int(N*scale))

    if order == 0:
       img_out = cv.resize(img_in, (Q, P), interpolation=cv.INTER_NEAREST)
    elif order == 1:
        img_out = cv.resize(img_in, (Q, P), interpolation=cv.INTER_LINEAR)
    elif order == 3:
        img_out = cv.resize(img_in, (Q, P), interpolation=cv.INTER_CUBIC)
    else:
        raise Exception("Unsupported Interpolation Order.")

    return img_out

def torch_resize(img_in: np.ndarray, scale: float, order: int = 3):
    if img_in.ndim == 2:
        img_in = torch.from_numpy(img_in).unsqueeze(dim=0).unsqueeze(dim=0)
    elif img_in.ndim == 3 and img_in.shape[2] == 3:
        img_in = torch.from_numpy(img_in.transpose(2,0,1)).unsqueeze(dim=0)
    else:
        raise Exception("Input image should be single or three color channels!")
    
    if order == 0:
        img_out = F.interpolate(img_in, size=None, scale_factor=scale, align_corners=False, mode='nearest')
    elif order == 1:
        img_out = F.interpolate(img_in, size=None, scale_factor=scale, align_corners=False, mode='bilinear')
    elif order == 3:
        img_out = F.interpolate(img_in, size=None, scale_factor=scale, align_corners=False, mode='bicubic')
    else:
        raise Exception("Unsupported Interpolation Order.")
    
    img_out = img_out.squeeze().numpy()
    if img_out.ndim == 3:
        img_out = img_out.transpose(1,2,0)
        
    return img_out


def pil_resize(img_in: np.ndarray, scale: float, order: int=3):
    if img_in.ndim == 2:
        M,N = img_in.shape
        C = 1
    elif img_in.ndim == 3 and img_in.shape[2] == 3:
        M,N,C = img_in.shape
    else:
        raise Exception("Input image should be single or three color channels!")
    
    P, Q = (int(M*scale), int(N*scale))

    if order == 0:
        img_out = np.asarray(Image.fromarray(img_in).resize((Q, P), resample=Image.Resampling.NEAREST))
    elif order == 1:
        img_out = np.asarray(Image.fromarray(img_in).resize((Q, P), resample=Image.Resampling.BILINEAR))
    elif order == 3:
        img_out = np.asarray(Image.fromarray(img_in).resize((Q, P), resample=Image.Resampling.BICUBIC))
    else:
        raise Exception("Unsupported Interpolation Order.")

    return img_out

def scikit_resize(img_in: np.ndarray, scale: float, order: int):
    if order == 0 or order == 1 or order == 3:
        img_out = rescale(img_in, scale, order, clip=False, preserve_range=True)
        return img_out
    else:
        raise Exception("Unsupported Interpolation Order.")
