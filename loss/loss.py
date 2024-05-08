import kornia.losses
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from time import time
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from cv2.ximgproc import guidedFilter
import numpy as np
from torchvision.transforms import Compose, ToPILImage, CenterCrop, ToTensor, Resize


class FusionLoss(nn.Module):
    def __init__(self, alpha, beta, theta):
        super(FusionLoss, self).__init__()

        # self.ms_ssim = MSSSIM()          # 原来的SSIM损失
        # self.l1_loss = F.l1_loss()
        # self.l2_loss = nn.MSELoss()
        self.grad_loss = JointGrad()

        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, im_fus, im_ir, im_vi, map_ir, map_vi):
        # ms_ssim_loss = (1 - self.ms_ssim(im_fus, im_ir)) + (1 - self.ms_ssim(im_fus, im_vi))      # 原来基本的SSIM损失，不知道为什么实现不了
        # ms_ssim_loss = (1 - self.ms_ssim(im_fus, (map_ir * im_ir + map_vi * im_vi)))
        ms_ssim_loss = ssim_lwy(im_fus, im_ir) + ssim_lwy(im_fus, im_vi)  # 重写的SSIM损失

        # l1_loss = self.l1_loss(im_fus, (map_ir * im_ir + map_vi * im_vi))                           # 梯度损失,跟原文的不一样
        l1_loss = F.l1_loss(torch.maximum(map_ir, map_vi), im_fus)  # 按照公式重写的梯度损失

        grad_loss = self.grad_loss(im_fus, im_ir, im_vi)  # SVS显著性矩阵造的损失
        fuse_loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss
        # fuse_loss = self.alpha * ms_ssim_loss + self.beta * l1_loss

        return fuse_loss


class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):
        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        fus_grad = torch.abs(self.laplacian(im_fus, 3))

        loss_JGrad = self.l1_loss(torch.max(ir_grad, vi_grad), fus_grad)

        return loss_JGrad


# class MSSSIM(torch.nn.Module):
#     def __init__(self, size_average=True, max_val=255):
#         super(MSSSIM, self).__init__()
#         self.size_average = size_average
#         self.channel = 1
#         self.max_val = max_val
#
#     def _ssim(self, img1, img2, size_average=True):
#
#         _, c, w, h = img1.size()
#         window_size = min(w, h, 11)
#         sigma = 1.5 * window_size / 11
#         window = create_window(window_size, sigma, self.channel).cuda()
#         mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
#         mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)
#
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2
#
#         sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
#         sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2
#
#         C1 = (0.01 * self.max_val) ** 2
#         C2 = (0.03 * self.max_val) ** 2
#         V1 = 2.0 * sigma12 + C2
#         V2 = sigma1_sq + sigma2_sq + C2
#         ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
#         mcs_map = V1 / V2
#         if size_average:
#             return ssim_map.mean(), mcs_map.mean()


# def create_window(window_size, sigma, channel):
#     _1D_window = gaussian(window_size, sigma).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window                                                                               # 原来SSIM损失的一部分，替换成如下



def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def ssim_lwy(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = ssim_map.mean()

    return 1 - ret


# # L1损失加梯度损失
class l1_grad_loss(nn.Module):
    def __init__(self):
        super(l1_grad_loss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, generate_img, image_ir, image_vis):
    # def forward(self, vi_ycbcr):
    #     image_y = image_vis[:, 0:1, :, :]
        image_y = image_vis
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        return loss_in, loss_grad

# 这个是计算梯度的索贝尔算子
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)






# 这个是以高斯滤波器为基础的损失
class filter_loss(nn.Module):

    def __init__(self, kernel_size, sigma):
        super(filter_loss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, general_img, ir_img, vi_img):
        # print(general_img.shape)
        ir_fliter = kornia.filters.gaussian_blur2d(ir_img, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
        vi_fliter = kornia.filters.gaussian_blur2d(vi_img, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
        general_fliter = kornia.filters.gaussian_blur2d(general_img, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
        # print(ir_fliter.shape)
        flit_gard = torch.max(ir_fliter, vi_fliter)
        loss_gard = F.l1_loss(general_fliter, flit_gard)
        return loss_gard



def guided_filter_batch(images: torch.Tensor, guide: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """
    Apply guided filter to a batch of images.
    :param images: Input images, shape (N, C, H, W)
    :param guide: Guide image, shape (N, C, H, W)
    :param radius: Radius of the filter
    :param eps: Regularization parameter
    :return: Filtered images
    """
    filtered_images = []
    for image, guide_image in zip(images, guide):
        image = np.array(to_pil_image(image.cpu()))
        guide_image = np.array(to_pil_image(guide_image.cpu()))
        # image = image.numpy()
        # guide_image = guide_image.numpy()
        # print(image.shape)
        filtered_image = guidedFilter(guide_image, image, radius, eps)
        filtered_image = to_tensor(filtered_image).to(images.device)
        filtered_images.append(filtered_image)
    return torch.stack(filtered_images)


def batch_guided_filter(input, guidance, radius=5, eps=1e-6):
    """
    Applies guided filter to a batch of images.
    :param input: Input tensor of shape (N, C, H, W)
    :param guidance: Guidance tensor of shape (N, C, H, W)
    :param radius: Radius of the filter
    :param eps: Regularization parameter
    :return: Filtered tensor of shape (N, C, H, W)
    """
    assert input.shape == guidance.shape
    N, C, H, W = input.shape

    # mean_I = F.avg_pool2d(I, kernel_size=radius*2+1)
    r = radius
    k = r * 2 + 1
    mean_I = torch.nn.functional.avg_pool2d(guidance, k, stride=1, padding=r)

    # mean_p = F.avg_pool2d(p, kernel_size=radius*2+1)
    mean_p = torch.nn.functional.avg_pool2d(input, k, stride=1, padding=r)

    # corr_I = F.avg_pool2d(I*I, kernel_size=radius*2+1)
    corr_I = torch.nn.functional.avg_pool2d(guidance * guidance, k, stride=1, padding=r)

    # corr_Ip = F.avg_pool2d(I*p, kernel_size=radius*2+1)
    corr_Ip = torch.nn.functional.avg_pool2d(guidance * input, k,stride=1,padding=r)

    # var_I = corr_I - mean_I * mean_I
    var_I = corr_I - mean_I * mean_I

    # cov_Ip = corr_Ip - mean_I * mean_p
    cov_Ip = corr_Ip - mean_I * mean_p

    # a = cov_Ip / (var_I + eps)
    a = cov_Ip / (var_I + eps)

    # b = mean_p - a * mean_I
    b = mean_p - a * mean_I

    # mean_a = F.avg_pool2d(a, kernel_size=radius*2+1)
    mean_a = torch.nn.functional.avg_pool2d(a,k,stride=1,padding=r)

    # mean_b = F.avg_pool2d(b,kernel_size=radius*2+1)
    mean_b = torch.nn.functional.avg_pool2d(b,k,stride=1,padding=r)

    q = mean_a * guidance + mean_b
    return input - q

class loss_guided_cpu(nn.Module):
    def __init__(self, radius, eps):
        super(loss_guided_cpu, self).__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, general_img, ir_img, vi_img):
        # ir_fliter_ = batch_guided_filter(ir_img, ir_img, self.radius, self.eps).cuda()
        # vi_fliter_ = batch_guided_filter(vi_img, vi_img, self.radius, self.eps).cuda()
        # general_fliter_ = batch_guided_filter(general_img, general_img, self.radius, self.eps).cuda()

        ir_fliter_ = guided_filter_batch(ir_img, ir_img, self.radius, self.eps).cuda()
        vi_fliter_ = guided_filter_batch(vi_img, vi_img, self.radius, self.eps).cuda()
        general_fliter_ = guided_filter_batch(general_img, general_img, self.radius, self.eps).cuda()


        flit_gard = torch.max(abs(ir_fliter_), abs(vi_fliter_))
        loss_gard = F.l1_loss(general_fliter_, flit_gard)
        return loss_gard

class edgeLoss(nn.Module):
    def __init__(self):
        super(edgeLoss, self).__init__()
    def forward(self, prediction, label):
        label = label.long()
        mask = (label != 0).float()
        num_positive = torch.sum(mask).float()       # 边缘像素点的个数
        num_negative = mask.numel() - num_positive   # 剩下的像素点的个数
        # print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), label.float(), weight=mask, reduce=False)
        return torch.sum(cost)

class loss_guided_gpu(nn.Module):
    def __init__(self, radius, eps):
        super(loss_guided_gpu, self).__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, general_img, ir_img, vi_img):
        ir_fliter_ = batch_guided_filter(ir_img, ir_img, self.radius, self.eps).cuda()
        vi_fliter_ = batch_guided_filter(vi_img, vi_img, self.radius, self.eps).cuda()
        general_fliter_ = batch_guided_filter(general_img, general_img, self.radius, self.eps).cuda()

        # ir_fliter_ = guided_filter_batch(ir_img, ir_img, self.radius, self.eps).cuda()
        # vi_fliter_ = guided_filter_batch(vi_img, vi_img, self.radius, self.eps).cuda()
        # general_fliter_ = guided_filter_batch(general_img, general_img, self.radius, self.eps).cuda()


        flit_gard = torch.max(abs(ir_fliter_), abs(vi_fliter_))
        loss_gard = F.l1_loss(general_fliter_, flit_gard)
        return loss_gard


class Loss_contrast(nn.Module):
    def __init__(self):
        super(Loss_contrast, self).__init__()
        self.L1 = nn.L1Loss()
        # Resize the HRs and fused image for reduce the computation cost
        self.H, self.W = 10, 10
        self.trans = Compose([# ToPILImage(),
                              CenterCrop((100, 100)),
                              Resize((self.H, self.W)),
                              #ToTensor()
        ])

    def L_contrast(self, HR_a, HR_b, sr_):
        Iavg = (HR_a + HR_b) / 2

        A, B, SR, IAVG = HR_a.clone().cpu(), HR_b.clone().cpu(), sr_.clone().cpu(), Iavg.clone().cpu()
        # A, B, SR, IAVG = SizeTransform(A, self.trans), SizeTransform(B, self.trans), SizeTransform(SR, self.trans), SizeTransform(IAVG, self.trans)
        A, B, SR, IAVG = self.trans(A), self.trans(B), self.trans(SR), self.trans(IAVG)

        D = torch.norm(SR - IAVG, p='fro').pow(2)  # F范数的平方

        def denominator(xp, yp, xq, yq):
            d_position = ((xp - xq) ** 2 + (yp - yq) ** 2) ** 0.5
            d_intensity = torch.abs(A[:, :, xp, yp] - A[:, :, xq, yq]) + torch.abs(B[:, :, xp, yp] - B[:, :, xq, yq])
            d_intensity = 1 - torch.tanh(d_intensity / 2)

            return d_position * d_intensity

        C = 0
        xy = list(map(lambda i: divmod(i, self.W), [x for x in range(self.H * self.W)]))

        for i in range(self.H * self.W):
            for j in range(self.H * self.W):
                if i == j:
                    continue
                molecule = torch.abs(SR[:, :, xy[i][0], xy[i][1]] - SR[:, :, xy[j][0], xy[j][1]])
                denominate = denominator(xy[i][0], xy[i][1], xy[j][0], xy[j][1]) + 0.0001  # (xp, yp, xq, yq)
                temp = torch.sum(molecule / denominate)
                C = C + temp.data

        # print('L_contrast over, cost time =', time()-start)
        return torch.abs(D - C).to(D.device)

    def L_pixel(self, HR_a, HR_b, sr_a, sr_b, sr_fused):
        L_pixel_a = self.L1(sr_a, HR_a)  # (input, target)
        L_pixel_b = self.L1(sr_b, HR_b)
        L_pixel_a_fuse = self.L1(sr_fused, HR_a)
        L_pixel_b_fuse = self.L1(sr_fused, HR_b)

        return (L_pixel_a + L_pixel_b), (L_pixel_a_fuse + L_pixel_b_fuse)

    def forward(self, HR_a, HR_b, sr_fused):
        hr_a, hr_b = HR_a.cuda(), HR_b.cuda()  # 复制到同一个device
        l_contrast = self.L_contrast(hr_a, hr_b, sr_fused)  # * 2 # 最后才计算，因为中间会把HR_ab给缩小
        return l_contrast
