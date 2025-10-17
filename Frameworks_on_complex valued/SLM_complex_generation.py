import cv2
import torch
from propagation_ASM import propagation_ASM, propagation_ASM2
from tools import masktensor, loadtensor_GRAY, loadimg, loadtensor_name, loadimg_name
from torchvision import transforms
from matplotlib import pyplot as plt

device = torch.device("cuda")
n = 1072
m = 1920
z = 100
z_SM = 0.12
pitch = 4.5 * pow(10, -3)
lambda_ = 532 * pow(10, -6)
pad = False
train_num = 3800
valid_num = 100
num_SM = 51
resize_tonet = transforms.Resize((n, m), antialias=True)


def SM_LBM(img_root, depth_root, idx, H_layer, H_z):
    layerd_scene = torch.zeros(1, num_SM, n, m).to(device)
    scene_mask = torch.ones(1, num_SM, n, m).to(device)
    for img_l in range(5):
        dic = str(4 - img_l)
        img_LDI = loadtensor_GRAY(img_root.format(dic), idx, n, m).view(1, 1, n, m).float()
        depth_LDI = loadimg(depth_root.format(dic), idx, n, m)
        for scene_l in range(num_SM):
            layerd_scene[:, scene_l, :, :] = layerd_scene[:, scene_l, :, :] + scene_mask[:, scene_l, :, :] * \
                                             masktensor(img_LDI, depth_LDI, n, m, num_SM, scene_l)
        index_scene = layerd_scene != 0
        scene_mask[index_scene] = 0

    layerd_slice = torch.complex(torch.zeros(1, num_SM, n, m), torch.zeros(1, num_SM, n, m))
    for scene_l in range(num_SM):
        phi_inital = torch.tensor(2 * torch.pi * (scene_l * z_SM + z) / lambda_)
        layerd_slice[:, scene_l, :, :] = layerd_scene[:, scene_l, :, :] * torch.complex(torch.cos(phi_inital), torch.sin(phi_inital))

        # for scene_l in range(num_SM):
        #     cv2.imwrite('E:/pythonProject/Test_image/layered_scene_{}.png'.format(scene_l), (torch.abs(layerd_scene[:, scene_l, :, :])*255).squeeze(0).cpu().detach().numpy())

    layerd_scene_tonet = torch.complex(torch.zeros(1, num_SM, n, m), torch.zeros(1, num_SM, n, m)).to(device)
    for scene_l in range(num_SM):
        layerd_scene_tonet[:, scene_l, :, :] = resize_tonet(layerd_slice[:, scene_l, :, :].view(1, 1, n, m))

    silhouette_mask = torch.zeros(1, num_SM, n, m).to(device)
    for scene_l in range(num_SM):
        index_unocc = layerd_scene_tonet[:, scene_l, :, :] == 0
        silhouette_mask[:, scene_l, :, :][index_unocc] = 1

    complex_targetplane = layerd_scene_tonet[:, num_SM - 1, :, :]
    for layer in range(num_SM - 1):
        complex_targetplane = layerd_scene_tonet[:, num_SM - 2 - layer, :, :] + \
                              silhouette_mask[:, num_SM - 2 - layer, :, :] * \
                              propagation_ASM2(u_in=complex_targetplane, feature_size=[pitch, pitch],
                                               wavelength=lambda_,
                                               z=-num_SM, linear_conv=pad, precomped_H=H_layer)
        # cv2.imwrite('E:/pythonProject/Test_image/layered_result_{}.png'.format(layer),
        #             (torch.abs(complex_targetplane) * 255).squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    complex_SLMplane = propagation_ASM2(u_in=complex_targetplane, feature_size=[pitch, pitch], wavelength=lambda_,
                                        z=-z, linear_conv=pad, precomped_H=H_z)
    # cv2.imwrite('E:/pythonProject/Test_image/layered_result.png',
    #             (torch.abs(complex_SLMplane) * 255).squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    return complex_SLMplane


Hbackward_1_SLM = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                                  z=-z, linear_conv=pad, return_H=True).to(device)
Hforward_1_SLM = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                                 z=z, linear_conv=pad, return_H=True).to(device)
Hbackward_SM = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch], wavelength=lambda_,
                               z=-z_SM, linear_conv=pad, return_H=True).to(device)

train_LDI_img_root = "../MIT-CGH-4K-V2 dataset/train_384_v2_png/img_{}"
train_LDI_depth_root = "../MIT-CGH-4K-V2 dataset/train_384_v2_png/depth_{}"
valid_LDI_img_root = "../MIT-CGH-4K-V2 dataset/validate_384_v2_png/img_{}"
valid_LDI_depth_root = "../MIT-CGH-4K-V2 dataset/validate_384_v2_png/depth_{}"

for train_idx in range(train_num):
    complex_LDI_train = SM_LBM(train_LDI_img_root, train_LDI_depth_root, train_idx, Hbackward_SM, Hbackward_1_SLM)
    torch.save(complex_LDI_train, "../MIT-CGH-4K-V2 dataset/train_384_v2_complex_51/com_{}.pt".format(train_idx))

for valid_idx in range(valid_num):
    complx_LDI_valid = SM_LBM(valid_LDI_img_root, valid_LDI_depth_root, valid_idx, Hbackward_SM, Hbackward_1_SLM)
    torch.save(complx_LDI_valid, "../MIT-CGH-4K-V2 dataset/valid_384_v2_complex_51/com_{}.pt".format(valid_idx))




