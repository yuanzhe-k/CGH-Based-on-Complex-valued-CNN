import time

import torch
from torch import nn, fft
import numpy as np
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
from propagation_ASM import propagation_ASM2, propagation_ASM
from tools import maskimg, padzero, maskimg_boundgs, masktensor

device = torch.device("cuda")


class Down_sampling(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Down_sampling, self).__init__()
        self.C_Conv2d = ComplexConv2d(in_chans, out_chans, 3, stride=2, padding=1)

    def forward(self, x):
        x_conv = self.C_Conv2d(x)
        x_output = complex_relu(x_conv)
        return x_output


class Down_sampling_res(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Down_sampling_res, self).__init__()
        self.C_Conv2d1 = ComplexConv2d(in_chans, out_chans, 3, stride=2, padding=1)
        self.C_conv2d2 = ComplexConv2d(in_chans, out_chans, 1, stride=2)

    def forward(self, x):
        x_conv = self.C_Conv2d1(x)
        x_relu = complex_relu(x_conv)
        x_output = self.C_conv2d2(x) + x_relu
        return x_output


class Down_sampling_res_3D(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Down_sampling_res_3D, self).__init__()
        self.C_Conv2d1 = ComplexConv2d(in_chans, out_chans//2, 3, stride=2, padding=1)
        self.C_Conv2d2 = ComplexConv2d(out_chans//2, out_chans, 3, stride=1, padding=1)
        self.C_conv2d3 = ComplexConv2d(in_chans, out_chans, 1, stride=2)

    def forward(self, x):
        x_relu1 = complex_relu(self.C_Conv2d1(x))
        x_relu2 = complex_relu(self.C_Conv2d2(x_relu1))
        x_output = self.C_conv2d3(x) + x_relu2
        return x_output


class Up_sampling1(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Up_sampling1, self).__init__()
        self.C_Conv2d = ComplexConvTranspose2d(in_chans, out_chans, 4, stride=2, padding=1)

    def forward(self, x):
        x_conv = self.C_Conv2d(x)
        x_output = complex_relu(x_conv)
        return x_output


class Up_sampling1_res(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Up_sampling1_res, self).__init__()
        self.C_Conv2d1 = ComplexConvTranspose2d(in_chans, out_chans, 4, stride=2, padding=1)
        self.C_Conv2d2 = ComplexConvTranspose2d(in_chans, out_chans, 2, stride=2)

    def forward(self, x):
        x_conv = self.C_Conv2d1(x)
        x_relu = complex_relu(x_conv)
        x_output = self.C_Conv2d2(x) + x_relu
        return x_output


class Up_sampling1_res_3D(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Up_sampling1_res_3D, self).__init__()
        self.C_Convtrans2d1 = ComplexConvTranspose2d(in_chans, out_chans*2, 4, stride=2, padding=1)
        self.C_Conv2d = ComplexConv2d(out_chans*2, out_chans, 3, stride=1, padding=1)
        self.C_Convtrans2d3 = ComplexConvTranspose2d(in_chans, out_chans, 2, stride=2)

    def forward(self, x):
        x_relu1 = complex_relu(self.C_Convtrans2d1(x))
        x_relu2 = complex_relu(self.C_Conv2d(x_relu1))
        x_output = self.C_Convtrans2d3(x) + x_relu2
        return x_output


class Up_sampling2(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Up_sampling2, self).__init__()
        self.C_Conv2d = ComplexConvTranspose2d(in_chans, out_chans, 4, stride=2, padding=1)

    def forward(self, x):
        x_output = self.C_Conv2d(x)
        return x_output


class Up_sampling2_res(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Up_sampling2_res, self).__init__()
        self.C_Conv2d1 = ComplexConvTranspose2d(in_chans, out_chans, 4, stride=2, padding=1)
        self.C_Conv2d2 = ComplexConvTranspose2d(in_chans, out_chans, 2, stride=2)

    def forward(self, x):
        x_conv1 = self.C_Conv2d1(x)
        x_output = self.C_Conv2d2(x)
        return x_output


class CCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling(1, 4)
        self.down2 = Down_sampling(4, 8)
        self.down3 = Down_sampling(8, 16)
        self.down4 = Down_sampling(16, 32)

        self.up1 = Up_sampling1(32, 16)
        self.up2 = Up_sampling1(16, 8)
        self.up3 = Up_sampling1(8, 4)
        self.up4 = Up_sampling2(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        output = torch.atan2(x_u4.imag, x_u4.real)

        return output


class CCNN1_2C(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling(1, 4)
        self.down2 = Down_sampling(4, 8)
        self.down3 = Down_sampling(8, 16)
        self.down4 = Down_sampling(16, 32)

        self.up1 = Up_sampling1(32, 16)
        self.up2 = Up_sampling1(16, 8)
        self.up3 = Up_sampling1(8, 4)
        self.up4 = Up_sampling2(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        return x_u4


class CCNN1_2C_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling_res(1, 4)
        self.down2 = Down_sampling_res(4, 8)
        self.down3 = Down_sampling_res(8, 16)
        self.down4 = Down_sampling_res(16, 32)

        self.up1 = Up_sampling1_res(32, 16)
        self.up2 = Up_sampling1_res(16, 8)
        self.up3 = Up_sampling1_res(8, 4)
        self.up4 = Up_sampling2_res(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        return x_u4


class CCNN1_2C_res_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling_res_3D(1, 4)
        self.down2 = Down_sampling_res_3D(4, 16)
        self.down3 = Down_sampling_res_3D(16, 64)
        self.down4 = Down_sampling_res_3D(64, 512)

        self.up1 = Up_sampling1_res_3D(512, 64)
        self.up2 = Up_sampling1_res_3D(64, 16)
        self.up3 = Up_sampling1_res_3D(16, 4)
        self.up4 = Up_sampling1_res_3D(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        return x_u4


class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling(1, 4)
        self.down2 = Down_sampling(4, 8)
        self.down3 = Down_sampling(8, 16)
        self.down4 = Down_sampling(16, 32)

        self.up1 = Up_sampling1(32, 16)
        self.up2 = Up_sampling1(16, 8)
        self.up3 = Up_sampling1(8, 4)
        self.up4 = Up_sampling2(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        output = torch.atan2(x_u4.imag, x_u4.real)

        return output


class CCNN2_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling_res(1, 4)
        self.down2 = Down_sampling_res(4, 8)
        self.down3 = Down_sampling_res(8, 16)
        self.down4 = Down_sampling_res(16, 32)

        self.up1 = Up_sampling1_res(32, 16)
        self.up2 = Up_sampling1_res(16, 8)
        self.up3 = Up_sampling1_res(8, 4)
        self.up4 = Up_sampling2_res(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        output = torch.atan2(x_u4.imag, x_u4.real)

        return output


class CCNN2_res_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down_sampling_res_3D(1, 4)
        self.down2 = Down_sampling_res_3D(4, 16)
        self.down3 = Down_sampling_res_3D(16, 64)
        self.down4 = Down_sampling_res_3D(64, 512)

        self.up1 = Up_sampling1_res_3D(512, 64)
        self.up2 = Up_sampling1_res_3D(64, 16)
        self.up3 = Up_sampling1_res_3D(16, 4)
        self.up4 = Up_sampling1_res_3D(4, 1)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x_u1 = self.up1(x_d4)
        x_u2 = self.up2(x_u1 + x_d3)
        x_u3 = self.up3(x_u2 + x_d2)
        x_u4 = self.up4(x_u3 + x_d1)

        output = torch.atan2(x_u4.imag, x_u4.real)

        return output


class CCNN_CGH(nn.Module):
    def __init__(self):
        super().__init__()
        self.ccnn1 = CCNN1()
        self.ccnn2 = CCNN2()

    def forward(self, amp, ph_init, z, pad, pitch, lambda_, H):
        target_complex = torch.complex(amp * torch.cos(ph_init), amp * torch.sin(ph_init))
        predict_phase = self.ccnn1(target_complex)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))
        slmfield = propagation_ASM2(u_in=predict_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                    wavelength=lambda_, precomped_H=H)
        holophase = self.ccnn2(slmfield)

        return holophase


class CCNN_CGH_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.ccnn1 = CCNN1_2C_res()
        self.ccnn2 = CCNN2_res()

    def forward(self, amp, ph_init, z, pad, pitch, lambda_, H):
        target_complex = torch.complex(amp * torch.cos(ph_init), amp * torch.sin(ph_init))
        predict_complex = self.ccnn1(target_complex)
        slmfield = propagation_ASM2(u_in=predict_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                    wavelength=lambda_, precomped_H=H)
        holophase = self.ccnn2(slmfield)

        return holophase


class CCNN_3DCGH1(nn.Module):
    def __init__(self):
        super(CCNN_3DCGH1, self).__init__()
        self.ccnn1 = CCNN1_2C_res_3D()
        self.ccnn2 = CCNN2_res_3D()

    def forward(self, img, depth, z, pad, pitch, lambda_, H):
        input = torch.complex(img, depth)
        target_complex = self.ccnn1(input)
        slm_complex = propagation_ASM2(u_in=target_complex, z=z, linear_conv=pad,
                                       feature_size=[pitch, pitch], wavelength=lambda_, precomped_H=H)
        holophase = self.ccnn2(slm_complex)
        return holophase, slm_complex


class CCNN_3DCGH1_woRes(nn.Module):
    def __init__(self):
        super(CCNN_3DCGH1_woRes, self).__init__()
        self.ccnn1 = CCNN1_2C()
        self.ccnn2 = CCNN2()

    def forward(self, img, depth, z, pad, pitch, lambda_, H):
        input = torch.complex(img, depth)
        target_complex = self.ccnn1(input)
        slm_complex = propagation_ASM2(u_in=target_complex, z=z, linear_conv=pad,
                                       feature_size=[pitch, pitch], wavelength=lambda_, precomped_H=H)
        holophase = self.ccnn2(slm_complex)
        return holophase, slm_complex


class CCNN_3DCGH2(nn.Module):
    def __init__(self):
        super(CCNN_3DCGH2, self).__init__()
        self.ccnn1 = CCNN1_2C_res_3D()
        self.ccnn2 = CCNN2_res_3D()

    def forward(self, img, depth, z, z_layer, pad, pitch, lambda_, lay_num, H):
        input = torch.complex(img, depth)
        slm_complex = torch.zeros(img.shape, dtype=torch.complex64).to(device)
        target_complex = self.ccnn1(input)
        for j in range(lay_num):
            slm_complex += propagation_ASM2(u_in=target_complex, z=z+j*z_layer, linear_conv=pad,
                                            feature_size=[pitch, pitch], wavelength=lambda_, precomped_H=H[:, j, :, :])
        holophase = self.ccnn2(slm_complex)
        return holophase