from matplotlib import pyplot as plt
import time
import cv2
from scipy import io
import torch
import numpy as np
from torch.nn import MSELoss
from torchvision import transforms
from Module_file import CCNN_3DCGH1
from propagation_ASM import propagation_ASM, propagation_ASM2
from tools import masktensor, loadtensor_GRAY, loadimg, AADPM
from ssim import MS_SSIM

device = torch.device("cuda")
n = 1072
m = 1920
z = 100
z_RE = 10
pitch = 4.5 * pow(10, -3)
lambda_ = 532 * pow(10, -6)
pad = False

train_RGBD_img_root = "../MIT-CGH-4K-V2 dataset/train_384_v2_png/img_0"
train_RGBD_depth_root = "../MIT-CGH-4K-V2 dataset/train_384_v2_png/depth_0"
valid_RGBD_img_root = "../MIT-CGH-4K-V2 dataset/validate_384_v2_png/img_0"
valid_RGBD_depth_root = "../MIT-CGH-4K-V2 dataset/validate_384_v2_png/depth_0"

CCNN_3DNet = CCNN_3DCGH1().to(device)
# CCNN_3DNet.load_state_dict(torch.load("./module_save_3D/CCNN_3DCGH_0826_wLDI_l50_wl1.pth"))
# CCNN_3DNet.eval()

Loss_fn1 = MSELoss().to(device)
Loss_fn2 = MS_SSIM().to(device)

learning_rate = 1e-4
optimizer = torch.optim.Adam(CCNN_3DNet.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1, verbose=False)

train_num = 3800
valid_num = 100
num_RE = 3
ph_init = torch.zeros(1, 1, n, m).to(device)
total_train_step = 1
total_valid_step = 1
epoch = 30
[alpha, beta, gamma] = [0.8, 0.2, 0.05]
trainl_re = []
trainl_SLMC = []
validl_re = []
validl_SLMC = []
resize_tonet = transforms.Resize((n, m), antialias=True)
start_time = time.time()

Hbackward = torch.complex(torch.zeros(1, num_RE, n, m), torch.zeros(1, num_RE, n, m)).to(device)
Hforward = torch.complex(torch.zeros(1, num_RE, n, m), torch.zeros(1, num_RE, n, m)).to(device)
for k in range(num_RE):
    Hbackward[:, k, :, :] = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                                            z=-(z + k * z_RE), linear_conv=pad, return_H=True)
    Hforward[:, k, :, :] = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                                           z=(z + k * z_RE), linear_conv=pad, return_H=True)

for i in range(epoch):
    train_loss_epoch_recon = 0
    train_loss_epoch_SLMC = 0
    valid_loss_epoch_recon = 0
    valid_loss_epoch_SLMC = 0
    print("------第{}轮训练开始------".format(i+1))
    # 训练步骤开始
    CCNN_3DNet.train()
    for train_idx in range(train_num):
        optimizer.zero_grad()
        Loss_trainimg = 0
        img = loadtensor_GRAY(train_RGBD_img_root, train_idx, n, m).view(1, 1, n, m).float()
        depth = loadtensor_GRAY(train_RGBD_depth_root, train_idx, n, m).view(1, 1, n, m).float()
        depth_mask = loadimg(train_RGBD_depth_root, train_idx, n, m)
        holophase, complex_RGBD = CCNN_3DNet(img, depth, -z, pad, pitch, lambda_, Hbackward[:, 0, :, :])
        complex_LDI = torch.load("../MIT-CGH-4K-V2 dataset/train_384_v2_complex_51/com_{}.pt".format(train_idx))
        # complex_LDI = SM_LBM(train_LDI_img_root, train_LDI_depth_root, train_idx, Hbackward_SM, Hbackward[:, 0, :, :])
        Loss_train_SLMC_re = 1 - Loss_fn2(complex_LDI.real, complex_RGBD.real)
        Loss_train_SLMC_im = 1 - Loss_fn2(complex_LDI.imag, complex_RGBD.imag)
        Loss_train_SLMC = gamma * (Loss_train_SLMC_re + Loss_train_SLMC_im) / 2
        slm_target = torch.complex(torch.cos(holophase), torch.sin(holophase))

        for j in range(num_RE):
            recon_field = propagation_ASM2(u_in=slm_target, z=z+j*z_RE, linear_conv=pad,
                                           feature_size=[pitch, pitch], wavelength=lambda_,
                                           precomped_H=Hforward[:, j, :, :]).to(device)
            recon = torch.abs(recon_field)
            # recon = recon / torch.max(recon)
            recon_mask = masktensor(recon, depth_mask, n, m, num_RE, j)
            target_mask = masktensor(img, depth_mask, n, m, num_RE, j)
            Loss_recon_mse = Loss_fn1(recon_mask, target_mask)
            Loss_recon_msssim = 1 - Loss_fn2(recon_mask, target_mask)
            Loss_recon = alpha * Loss_recon_mse + beta * Loss_recon_msssim
            Loss_trainimg = Loss_trainimg + Loss_recon
        Loss_train = Loss_trainimg + Loss_train_SLMC
        # train_loss_epoch = train_loss_epoch + Loss_train.item()
        train_loss_epoch_recon = train_loss_epoch_recon + Loss_trainimg.item()
        train_loss_epoch_SLMC = train_loss_epoch_SLMC + Loss_train_SLMC.item()
        Loss_train.backward()
        optimizer.step()

    train_loss_average_recon = train_loss_epoch_recon / train_num
    train_loss_average_SLMC = train_loss_epoch_SLMC / train_num
    trainl_re.append(train_loss_average_recon)
    trainl_SLMC.append(train_loss_average_SLMC)
    print("训练epoch：{}，Loss_SLMC：{}，Loss_recon：{}".format(total_train_step, train_loss_average_SLMC, train_loss_average_recon))
    total_train_step = total_train_step + 1

    # if i % 15 == 0:
    #     scheduler.step()

    # 测试步骤开始
    CCNN_3DNet.eval()
    with torch.no_grad():
        for valid_idx in range(valid_num):
            Loss_validimg = 0
            img = loadtensor_GRAY(valid_RGBD_img_root, valid_idx, n, m).view(1, 1, n, m).float()
            depth = loadtensor_GRAY(valid_RGBD_depth_root, valid_idx, n, m).view(1, 1, n, m).float()
            depth_mask = loadimg(valid_RGBD_depth_root, valid_idx, n, m)
            holophase, complex_RGBD = CCNN_3DNet(img, depth, -z, pad, pitch, lambda_, Hbackward[:, 0, :, :])
            # complex_LDI = SM_LBM(valid_LDI_img_root, valid_LDI_depth_root, valid_idx, wl, Hbackward_SM, Hbackward[:, 0, :, :])
            complex_LDI = torch.load("../MIT-CGH-4K-V2 dataset/valid_384_v2_complex_51/com_{}.pt".format(valid_idx))
            Loss_valid_SLMC_re = 1 - Loss_fn2(complex_LDI.real, complex_RGBD.real)
            Loss_valid_SLMC_im = 1 - Loss_fn2(complex_LDI.imag, complex_RGBD.imag)
            Loss_valid_SLMC = gamma * (Loss_valid_SLMC_re + Loss_valid_SLMC_im) / 2
            slm_target = torch.complex(torch.cos(holophase), torch.sin(holophase))

            for j in range(num_RE):
                recon_field = propagation_ASM2(u_in=slm_target, z=z + j * z_RE, linear_conv=pad,
                                               feature_size=[pitch, pitch], wavelength=lambda_,
                                               precomped_H=Hforward[:, j, :, :]).to(device)
                recon = torch.abs(recon_field)
                # recon = recon / torch.max(recon)
                recon_mask = masktensor(recon, depth_mask, n, m, num_RE, j)
                target_mask = masktensor(img, depth_mask, n, m, num_RE, j)
                Loss_recon_mse = Loss_fn1(recon_mask, target_mask)
                Loss_recon_msssim = 1 - Loss_fn2(recon_mask, target_mask)
                Loss_recon = alpha * Loss_recon_mse + beta * Loss_recon_msssim
                Loss_validimg = Loss_validimg + Loss_recon

            Loss_valid = Loss_validimg + Loss_valid_SLMC
            # valid_loss_epoch = valid_loss_epoch + Loss_valid.item()
            valid_loss_epoch_recon = valid_loss_epoch_recon + Loss_validimg.item()
            valid_loss_epoch_SLMC = valid_loss_epoch_SLMC + Loss_valid_SLMC.item()

        valid_loss_average_recon = valid_loss_epoch_recon / valid_num
        valid_loss_average_SLMC = valid_loss_epoch_SLMC / valid_num
        validl_re.append(valid_loss_average_recon)
        validl_SLMC.append(valid_loss_average_SLMC)
        print("测试集上的平均Loss: Loss_SLMC：{}，Loss_recon：{}".format(valid_loss_average_SLMC, valid_loss_average_recon))
        total_valid_step = total_valid_step + 1

torch.save(CCNN_3DNet.state_dict(), "./module_save_3D/CCNN_3DCGH_wLDI.pth")
valid_loss_re = np.mat(validl_re)
valid_loss_SLMC = np.mat(validl_SLMC)
io.savemat("./module_save_3D/valid_loss_wLDI.mat", {'valid_loss_re': valid_loss_re, 'valid_loss_SLMC': validl_SLMC})
train_loss_re = np.mat(trainl_re)
train_loss_SLMC = np.mat(trainl_SLMC)
io.savemat("./module_save_3D/train_loss_wLDI.mat", {'train_loss_re': train_loss_re, 'train_loss_SLMC': trainl_SLMC})

end_time = time.time()
print((end_time - start_time) / 3600)
