import time
import os
import cv2
import torch
from Module_file import CCNN_CGH, CCNN_CGH_res
from propagation_ASM import propagation_ASM, propagation_ASM2
from ssim import SSIM

device = torch.device("cuda")
n = 1072
m = 1920
z = 100
pitch = 4.5 * pow(10, -3)
lambda_ = 532 * pow(10, -6)
pad = False


def loadimg(root_dir, idx):
    img_path = os.listdir(root_dir)
    img_name = img_path[idx]
    img_item_path = os.path.join(root_dir, img_name)
    img = cv2.imread(img_item_path)
    img = cv2.resize(img, (m, n))
    img = cv2.split(img)[1]
    img = torch.from_numpy(img / 255.0)
    # img = srgb_to_lin(img)
    return img.to(device)


CCNNCGH_Net = CCNN_CGH_res().to(device)
# CCNN1 = CCNN1().to(device)
# CCNN2 = CCNN2_3sample().to(device)
CCNNCGH_Net.load_state_dict(torch.load("./module_save_2D/CCNNCGHres_250413_-z_z_100mm.pth"))
CCNNCGH_Net.eval()
SSIM_ = SSIM()
Hbackward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                            z=-z, linear_conv=pad, return_H=True)
Hforward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                           z=z, linear_conv=pad, return_H=True)
Hbackward = Hbackward.to(device)
Hforward = Hforward.to(device)


def PSNR(img_or, img_no):
    img_or = img_or * 255
    img_no = img_no * 255
    diff = img_or - img_no
    mse = torch.mean(torch.square(diff))
    psnr = 10 * torch.log10(255 * 255 / mse)
    return psnr


test_root_dir = "E:/pythonProject/DIV2K_Dataset/DIV2K_valid_HR"
test_num = 79
# test_idx = 20
compute_time_record = []
recon_PSNR_record = []
recon_SSIM_record = []
phase_init = torch.zeros(1, 1, n, m).to(device)
for test_idx in range(test_num):
    img = loadimg(test_root_dir, test_idx)
    img = img.view(1, 1, n, m).float()
    start_time = time.time()
    holophase = CCNNCGH_Net(img, phase_init, -z, pad, pitch, lambda_, Hbackward)
    end_time = time.time()
    # phase[phase > torch.pi] = phase[phase > torch.pi] - 2 * torch.pi * torch.ceil(
    #             (phase[phase > torch.pi] - torch.pi) / 2 / torch.pi)
    # phase[phase < -torch.pi] = phase[phase < -torch.pi] + 2 * torch.pi * torch.floor(
    #             (torch.pi - phase[phase < -torch.pi]) / 2 / torch.pi)
    slm_complex = torch.complex(torch.cos(holophase), torch.sin(holophase))
    recon = propagation_ASM2(u_in=slm_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                             wavelength=lambda_, precomped_H=Hforward)
    recon = torch.abs(recon).squeeze(0)
    # recon = recon / torch.max(recon)

    # holophase = phase
    Phase_slm = torch.ceil((holophase / torch.pi + 1) / 2 * 255)
    cv2.imwrite('./result_for_thesis/plane/holo_Cres.jpg', Phase_slm.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    cv2.imwrite("./result_for_thesis/plane/recon_Cres.jpg", (recon * 255).permute(1, 2, 0).cpu().detach().numpy())
    cv2.imwrite("./result_for_thesis/plane/img.jpg", (img * 255).squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    compute_time = end_time-start_time
    print("用时(s):{}".format(compute_time))
    # compute_time_record.append(compute_time)
    recon_PSNR = PSNR(img, recon)
    print("重建PSNR:{}".format(recon_PSNR))
    # recon_PSNR_record.append(recon_PSNR)
    recon_SSIM = SSIM_(img, recon.view(1, 1, 1072, 1920))
    print("重建SSIM:{}".format(recon_SSIM))
    # recon_SSIM_record.append(recon_SSIM)

# print(recon_PSNR_record)
# print(recon_SSIM_record)
# print(compute_time_record)
