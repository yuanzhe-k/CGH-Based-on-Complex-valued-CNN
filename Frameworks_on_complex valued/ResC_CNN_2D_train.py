import os
import time
from matplotlib import pyplot as plt
from scipy import io
import cv2
import torch
import numpy as np
from torch import nn, fft
from torch.nn import MSELoss
from Module_file import CCNN_CGH, CCNN_CGH_res
from propagation_ASM import propagation_ASM2, propagation_ASM
from ssim import MS_SSIM

device = torch.device("cuda")
n = 1072
m = 1920
z = 100
pitch = 4.5 * pow(10, -3)
lambda_ = 532 * pow(10, -6)
pad = False


def loadimg(root_dir, idx, flip):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    img = cv2.resize(img, (m, n))
    img = cv2.split(img)[1]

    flip = flip % 4
    if flip == 1:
        img = cv2.flip(img, flipCode=1)
    if flip == 2:
        img = cv2.flip(img, flipCode=0)
    if flip == 3:
        img = cv2.flip(img, flipCode=-1)

    img = torch.from_numpy(img / 255.0)
    # img = srgb_to_lin(img)
    return img.to(device)


train_root_dir = "../DIV2K_Dataset/DIV2K_train_HR"
valid_root_dir = "../DIV2K_Dataset/DIV2K_valid_HR"
CCNNCGH_Net = CCNN_CGH_res().to(device)


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


Loss_fn1 = MS_SSIM()
Loss_fn1 = Loss_fn1.to(device)
Loss_fn2 = MSELoss()
Loss_fn2 = Loss_fn2.to(device)
Loss_fn3 = NPCCLoss()
Loss_fn3 = Loss_fn3.to(device)

learning_rate = 1e-3
optimizer = torch.optim.AdamW(CCNNCGH_Net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1, verbose=False)

Hbackward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                            z=-z, linear_conv=pad, return_H=True)
Hforward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                           z=z, linear_conv=pad, return_H=True)
Hbackward = Hbackward.to(device)
Hforward = Hforward.to(device)

# pad_size = 3940
train_num = 800
valid_num = 100
phase_init = torch.zeros(1, 1, n, m).to(device)
total_train_step = 1
total_valid_step = 1
epoch = 50
trainl = []
validl = []
[alpha, beta] = [0.8, 0.2]
# zeropad = nn.ZeroPad2d(2)
# writer = SummaryWriter("./logs_CCNNtrain")
start_time = time.time()
for i in range(epoch):
    train_loss_epoch = 0
    valid_loss_epoch = 0
    print("------第{}轮训练开始------".format(i+1))
    # 训练步骤开始
    CCNNCGH_Net.train()
    for train_idx in range(train_num):
        optimizer.zero_grad()
        flip = np.random.randint(0, 100)
        img = loadimg(train_root_dir, train_idx, flip)
        img = img.view(1, 1, n, m).float()
        # img_flip = torch.flip(img, [1, 2])
        # amp = torch.sqrt(img)
        holophase = CCNNCGH_Net(img, phase_init, -z, pad, pitch, lambda_, Hbackward)
        slm_complex = torch.complex(torch.cos(holophase), torch.sin(holophase))
        recon = propagation_ASM2(u_in=slm_complex, feature_size=[pitch, pitch], wavelength=lambda_,
                                 z=z, precomped_H=Hforward, linear_conv=pad)
        recon = torch.abs(recon)
        # recon = recon / torch.max(recon)
        loss_mse = Loss_fn2(img, recon)
        # loss_msssim = 1 - Loss_fn1(img, recon)
        loss_npcc = (Loss_fn3(img, recon) + 1) / 2
        loss = alpha * loss_mse + beta * loss_npcc
        train_loss_epoch = train_loss_epoch + loss.item()
        loss.backward()
        optimizer.step()

    train_loss_average = train_loss_epoch / train_num
    trainl.append(train_loss_average)
    print("训练epoch：{}， Loss：{}".format(total_train_step, train_loss_average))
    # writer.add_scalar("train_loss", train_loss_average, total_train_step)
    total_train_step = total_train_step + 1

    # if i % 25 == 0:
    #     scheduler.step()

    # 测试步骤开始
    CCNNCGH_Net.eval()
    with torch.no_grad():
        for valid_idx in range(valid_num):
            flip = np.random.randint(0, 100)
            img = loadimg(valid_root_dir, valid_idx, flip)
            img = img.view(1, 1, n, m).float()
            # img_flip = torch.flip(img, [1, 2])
            # amp = torch.sqrt(img)
            holophase = CCNNCGH_Net(img, phase_init, -z, pad, pitch, lambda_, Hbackward)
            slm_complex = torch.complex(torch.cos(holophase), torch.sin(holophase))
            # recon = propgation_Fresnel(u_in=slm_complex, feature_size=[pitch, pitch], wavelength=lambda_,
            #                            z=z, pad_size=pad_size)
            recon = propagation_ASM2(u_in=slm_complex, feature_size=[pitch, pitch], wavelength=lambda_,
                                     z=z, precomped_H=Hforward, linear_conv=pad)
            # recon = propgation_RSC(u_in=slm_complex, feature_size=[pitch, pitch], wavelength=lambda_, z=z)
            recon = torch.abs(recon)
            # recon = recon / torch.max(recon)
            loss_mse = Loss_fn2(img, recon)
            # loss_msssim = 1 - Loss_fn1(img, recon)
            loss_npcc = (Loss_fn3(img, recon) + 1) / 2
            loss = alpha * loss_mse + beta * loss_npcc
            valid_loss_epoch = valid_loss_epoch + loss.item()

        valid_loss_average = valid_loss_epoch / valid_num
        validl.append(valid_loss_average)
        print("测试集上的平均Loss:{}".format(valid_loss_average))
        # writer.add_scalar("valid_loss", valid_loss_average, total_valid_step)
        total_valid_step = total_valid_step + 1

end_time = time.time()
torch.save(CCNNCGH_Net.state_dict(), "./module_save_2D/CCNNCGHres_100mm.pth")
valid_loss = np.mat(validl)
io.savemat("./module_save_2D/valid_loss_100mm.mat", {'valid_loss': valid_loss})
train_loss = np.mat(trainl)
io.savemat("./module_save_2D/train_loss_100mm.mat", {'train_loss': train_loss})
print((end_time - start_time) / 3600)
# writer.close()
