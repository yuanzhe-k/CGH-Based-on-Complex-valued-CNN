import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import Imath
import OpenEXR
from torchvision import transforms

device = torch.device("cuda")


def maskimg(img, depth, n, m, lay_num, i_layer):
    img = torch.from_numpy(img / 255.0).to(device)
    mask = torch.zeros(n, m).to(device)
    depl = 256 / lay_num
    index1 = depth >= 0 + i_layer * depl
    index2 = depth < depl + i_layer * depl
    mask[index1 & index2] = 1
    img_mask = img * mask
    return img_mask


def masktensor(img, depth, n, m, lay_num, i_layer):
    mask = torch.zeros(n, m).to(device)
    depl = 256 / lay_num
    index1 = depth >= 0 + i_layer * depl
    index2 = depth < depl + i_layer * depl
    mask[index1 & index2] = 1
    img_mask = img * mask
    return img_mask


def maskimg_boundgs(img, depth, n, m, lay_num, i_layer):
    depl = 256 / lay_num
    mask = np.zeros((n, m))
    index_layer = np.where((depth >= 0 + i_layer * depl) & (depth < depl + i_layer * depl))
    mask[index_layer[0], index_layer[1]] = 1
    img_mask = img * mask
    img_mask = img_mask.astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_mask, ksize=(9, 9), sigmaX=1.5, sigmaY=1.5)
    index_mask = np.where(mask != 0)
    img_blur[index_mask[0], index_mask[1]] = img_mask[index_mask[0], index_mask[1]]
    img_blur = torch.from_numpy(img_blur / 255.0)
    return img_blur.to(device)


class NPCCLoss(torch.nn.Module):
    def __init__(self):
        super(NPCCLoss, self).__init__()

    def forward(self, orin, recon):
        X0 = orin - torch.mean(orin)
        Y0 = recon - torch.mean(recon)
        X0_norm = torch.sqrt(torch.sum(X0 ** 2))
        Y0_norm = torch.sqrt(torch.sum(Y0 ** 2))
        npcc = -torch.sum(X0 * Y0) / (X0_norm * Y0_norm)
        loss = torch.mean(npcc)
        return loss


def np_gaussian(shape, channel, sigma, reshape_4d=True):
    m = (shape - 1.) / 2
    y, x = np.ogrid[-m:m + 1, -m:m + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    if reshape_4d:
        h = h.reshape([1, channel, shape, shape])
    return h


def low_pass(input, shape, channel, sigma):
    window = np_gaussian(shape=shape, channel=channel, sigma=sigma)
    window = torch.tensor(window, dtype=torch.complex64).to(device)
    lowpass = torch.nn.functional.conv2d(input=input, weight=window, padding=shape//2, groups=1)
    return lowpass


def AADPM(CH, shape, channel, sigma, h, w):
    CH_lp = low_pass(CH, shape=shape, channel=channel, sigma=sigma)
    A_CH_lp = torch.abs(CH_lp)
    A_CH_lp = A_CH_lp / torch.max(A_CH_lp)
    phi_CH_lp = torch.angle(CH_lp)
    P1 = (phi_CH_lp - torch.arccos(A_CH_lp)).squeeze()
    P2 = (phi_CH_lp + torch.arccos(A_CH_lp)).squeeze()
    Phase = torch.zeros(h, w).to(device)
    [n, m] = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w), indexing='ij')
    odd = (m+n) % 2 != 0
    even = (m+n) % 2 == 0
    Phase[odd] = P1[odd]
    Phase[even] = P2[even]
    return Phase


def srgb_to_lin(image):

    thresh = 0.04045

    if torch.is_tensor(image):
        low_val = image <= thresh
        im_out = torch.zeros_like(image)
        im_out[low_val] = 25 / 323 * image[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * image[torch.logical_not(low_val)] + 11)
                                              / 211) ** (12 / 5)
    else:
        im_out = np.where(image <= thresh, image / 12.92, ((image + 0.055) / 1.055) ** (12 / 5))

    return im_out


def loadimg(root_dir, idx, n, m):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (m, n))
    return gray


def loadtensor_RGB(root_dir, idx, channel, n, m):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_channel = img[:, :, channel]
    gray = cv2.resize(gray_channel, (m, n))
    gray = torch.from_numpy(gray / 255.0)
    gray = gray.view(1, 1, n, m).float()
    return gray.to(device)


def loadtensor_GRAY(root_dir, idx, n, m):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (m, n))
    gray = torch.from_numpy(gray / 255.0)
    # gray = gray.view(1, 1, n, m).float()
    return gray.to(device)


def loadtensor_name(root_dir, img_name, channel, n, m):
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    gray_channel = img[:, :, channel]
    gray = cv2.resize(gray_channel, (m, n))
    gray = torch.from_numpy(gray / 255.0)
    gray = gray.view(1, 1, n, m).float()
    return gray.to(device)


def loadimg_name(root_dir, img_name, n, m):
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (m, n))
    return gray


def loadexr_RGB(root_dir, idx, channel, n, m):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    exr_file = OpenEXR.InputFile(img_item_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel = np.frombuffer(exr_file.channel(channel, pt), dtype=np.float32)
    channel.shape = (size[1], size[0])
    exr_arr = np.array(channel)
    exr_arr = cv2.resize(exr_arr, (m, n))
    exr_tensor = torch.from_numpy(exr_arr)
    return exr_tensor.view(1, 1, n, m).to(device)


def loaddep_exr(root_dir, idx, n, m):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    exr_file = OpenEXR.InputFile(img_item_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32)
    channel.shape = (size[1], size[0])
    exr_arr = np.array(channel) * 255
    exr_arr = cv2.resize(exr_arr, (m, n))
    return exr_arr


def loaddisp(root_dir, idx, n, m):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    disp = cv2.imread(img_item_path, cv2.IMREAD_UNCHANGED)
    disp = cv2.resize(disp, (m, n))
    return disp.astype(np.float32) / 8


def PSNR(img_or, img_no):
    img_or = img_or * 255
    img_no = img_no * 255
    diff = img_or - img_no
    mse = torch.mean(torch.square(diff))
    psnr = 10 * torch.log10(255 * 255 / mse)
    return psnr


def padzero(input, num):
    padding = nn.ZeroPad2d(num)
    return padding(input)
