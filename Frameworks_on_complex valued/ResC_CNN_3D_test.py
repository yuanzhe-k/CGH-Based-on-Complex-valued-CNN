import os
import time
import cv2
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from Module_file import CCNN_3DCGH1, CCNN_3DCGH1_woRes
from propagation_ASM import propagation_ASM, propagation_ASM2
from tools import PSNR, masktensor, loadtensor_GRAY, loadtensor_RGB, loadimg, padzero
from ssim import SSIM

device = torch.device("cuda")
n = 1072
m = 1920
z = 100
z_layer = 5
lay_num = 3
pitch = 4.5 * pow(10, -3)
lambda_ = 532 * pow(10, -6)
pad = False
CCNN_3DNet = CCNN_3DCGH1().to(device)
CCNN_3DNet.load_state_dict(torch.load("./module_save_3D/CCNN_3DCGH_iniphase250122_wLDI_l50.pth"))
CCNN_3DNet.eval()

Hbackward = torch.complex(torch.zeros(1, lay_num, n, m), torch.zeros(1, lay_num, n, m)).to(device)
Hforward = torch.complex(torch.zeros(1, lay_num, n, m), torch.zeros(1, lay_num, n, m)).to(device)
for k in range(lay_num):
    Hbackward[:, k, :, :] = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                                            z=-(z + k * z_layer), linear_conv=pad, return_H=True)
    Hforward[:, k, :, :] = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                                           z=(z + k * z_layer), linear_conv=pad, return_H=True)

test_img_root = "./Test_image/img"
test_depth_root = "./Test_image/depth"


def readsize(root_dir, idx):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    h, w, _ = img.shape
    return h, w


test_num = 13
focus = ['n', 'm', 'f']
for test_idx in range(test_num):
    h, w = readsize(root_dir=test_img_root, idx=test_idx)
    img = loadtensor_GRAY(test_img_root, test_idx, n, m).view(1, 1, n, m).float()
    depth = loadtensor_GRAY(test_depth_root, test_idx, n, m).view(1, 1, n, m).float()
    depth_mask = loadimg(test_depth_root, test_idx, n, m)
    start_time = time.time()
    holophase, _ = CCNN_3DNet(img, depth, -z, pad, pitch, lambda_, Hbackward[:, 0, :, :])
    end_time = time.time()
    compute_time = end_time - start_time
    print("用时(s):{}".format(compute_time))
    slm_complex = torch.complex(torch.cos(holophase), torch.sin(holophase))

    # depth_mask = loadimg(root_dir="E:/pythonProject/Test_image", idx=5, n=n, m=m)
    resize_toini = transforms.Resize((h, w), antialias=True)
    recon_full = torch.zeros(1, 1, n, m).to(device)
    for j in range(lay_num):
        recon_field = propagation_ASM2(u_in=slm_complex, z=z+j*z_layer, linear_conv=pad,
                                       feature_size=[pitch, pitch], wavelength=lambda_,
                                       precomped_H=Hforward[:, j, :, :])
        recon = torch.abs(recon_field)
        # remax = torch.max(recon)
        recon_mask = masktensor(recon, depth_mask, n, m, lay_num, j)
        recon_full += recon_mask
        recon = recon / torch.max(recon) * 255
        recon_ini = resize_toini(recon)
        # mask_orisize = torch_resize(mask)
        dic = str(focus[j])
        cv2.imwrite('./result_for_thesis/multi-depth/recon_ResC/recon_{}_{}.png'.format(test_idx, dic), recon_ini.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())

    recon_full = recon_full / torch.max(recon_full)
    print("重建全焦图像PSNR:{}".format(PSNR(img, recon_full)))
    SSIM_t = SSIM()
    print("重建全焦图像SSIM:{}".format(SSIM_t(img, recon_full)))

    holophase += torch.pi
    Phase_slm = holophase / (2 * torch.pi) * 255.0
    torch_resize = transforms.Resize((h, w), antialias=True)
    Phase_slm = torch_resize(Phase_slm)
    cv2.imwrite('./result_for_thesis/multi-depth/holo_ResC/holo_{}.png'.format(test_idx), Phase_slm.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())


