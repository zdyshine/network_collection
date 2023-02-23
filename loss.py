import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class AdvancedSobelLoss(nn.Module):
    # Image Demoireing with Learnable Bandpass Filters
    # 改进版本的soble loss，配合提取sobel 高频混合使用
    def __init__(self, gpu):
        super(AdvancedSobelLoss, self).__init__()

        self.kernel_h = torch.FloatTensor([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1, -2, -1]])
        self.sobel_filter_h = self.kernel_h.expand(1, 3, 3, 3).cuda(gpu)  # 原版

        self.kernel_v = torch.FloatTensor([[1, 0, -1],
                                           [2, 0, -2],
                                           [1, 0, -1]])
        self.sobel_filter_v = self.kernel_v.expand(1, 3, 3, 3).cuda(gpu)  # 原版

        self.kernel_hv = torch.FloatTensor([[0, 1, 2],
                                            [-1, 0, 1],
                                            [-2, -1, 0]])
        self.sobel_filter_hv = self.kernel_hv.expand(1, 3, 3, 3).cuda(gpu)

        self.kernel_vh = torch.FloatTensor([[2, 1, 0],
                                            [1, 0, -1],
                                            [0, -1, -2]])
        self.sobel_filter_vh = self.kernel_vh.expand(1, 3, 3, 3).cuda(gpu)

    def forward(self, inputs, targets):
        n, _, h, w = inputs.shape
        n_pixel = n * h * w
        # inputs
        inputs_grad_h = F.conv2d(inputs, self.sobel_filter_h)
        inputs_grad_v = F.conv2d(inputs, self.sobel_filter_v)
        inputs_grad_hv = F.conv2d(inputs, self.sobel_filter_hv)
        inputs_grad_vh = F.conv2d(inputs, self.sobel_filter_vh)
        # targets
        targets_grad_h = F.conv2d(targets, self.sobel_filter_h)
        targets_grad_v = F.conv2d(targets, self.sobel_filter_v)
        targets_grad_hv = F.conv2d(targets, self.sobel_filter_hv)
        targets_grad_vh = F.conv2d(targets, self.sobel_filter_vh)
        loss = torch.sum(abs(inputs_grad_h - targets_grad_h) + abs(inputs_grad_v - targets_grad_v) +
                         abs(inputs_grad_hv - targets_grad_hv) + abs(inputs_grad_vh - targets_grad_vh)) / n_pixel
        return loss


class AdvancedSobelLossRRN(nn.Module):
    # Image Demoireing with Learnable Bandpass Filters
    # 改进版本的soble loss，配合提取sobel 高频混合使用
    def __init__(self, gpu, nframes=10):
        super(AdvancedSobelLossRRN, self).__init__()

        self.kernel_h = torch.FloatTensor([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1, -2, -1]])
        self.sobel_filter_h = self.kernel_h.expand(1, 3*6, 3, 3).cuda(gpu)  # 原版

        self.kernel_v = torch.FloatTensor([[1, 0, -1],
                                           [2, 0, -2],
                                           [1, 0, -1]])
        self.sobel_filter_v = self.kernel_v.expand(1, 3*6, 3, 3).cuda(gpu)  # 原版

        self.kernel_hv = torch.FloatTensor([[0, 1, 2],
                                            [-1, 0, 1],
                                            [-2, -1, 0]])
        self.sobel_filter_hv = self.kernel_hv.expand(1, 3*6, 3, 3).cuda(gpu)

        self.kernel_vh = torch.FloatTensor([[2, 1, 0],
                                            [1, 0, -1],
                                            [0, -1, -2]])
        self.sobel_filter_vh = self.kernel_vh.expand(1, 3*6, 3, 3).cuda(gpu)

    def forward(self, inputs, targets):
        n, _, _, h, w = inputs.shape
        inputs = inputs.view(n, -1, h, w)
        targets = targets.view(n, -1, h, w)
        n_pixel = n * h * w
        # inputs
        inputs_grad_h = F.conv2d(inputs, self.sobel_filter_h)
        inputs_grad_v = F.conv2d(inputs, self.sobel_filter_v)
        inputs_grad_hv = F.conv2d(inputs, self.sobel_filter_hv)
        inputs_grad_vh = F.conv2d(inputs, self.sobel_filter_vh)
        # targets
        targets_grad_h = F.conv2d(targets, self.sobel_filter_h)
        targets_grad_v = F.conv2d(targets, self.sobel_filter_v)
        targets_grad_hv = F.conv2d(targets, self.sobel_filter_hv)
        targets_grad_vh = F.conv2d(targets, self.sobel_filter_vh)
        loss = torch.sum(abs(inputs_grad_h - targets_grad_h) + abs(inputs_grad_v - targets_grad_v) +
                         abs(inputs_grad_hv - targets_grad_hv) + abs(inputs_grad_vh - targets_grad_vh)) / n_pixel
        return loss

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class VIF_Loss(nn.Module):
    def __init__(self, device, sigma_nsq=2, eps=1e-10):
        super(VIF_Loss, self).__init__()
        self.sigma_nsq = sigma_nsq
        self.eps = eps
        self.device = device


    def _fspecial_gauss(self, size, sigma):
        '''
        Function to mimic the 'fspecial' gaussian MATLAB function
        '''
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=0)
        x_data = np.expand_dims(x_data, axis=0)

        y_data = np.expand_dims(y_data, axis=0)
        y_data = np.expand_dims(y_data, axis=0)

        x = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_data, dtype=torch.float32).to(self.device)

        guass = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return guass / torch.sum(guass)


    def forward(self, sr_image, hr_image):
        '''
        Visual information fidelity loss.
        '''
        self.hr = hr_image[:, 0:1, :, :]
        self.sr = sr_image[:, 0:1, :, :]

        num = []
        den = []
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            weiht_kernel = self._fspecial_gauss(5, sd)

            if (scale > 1):
                self.hr = F.conv2d(self.hr, weiht_kernel, stride=[1, 1], padding=0)
                self.sr = F.conv2d(self.sr, weiht_kernel, stride=[1, 1], padding=0)

                self.hr = F.avg_pool2d(self.hr, kernel_size=2, stride=2, padding=0)
                self.sr = F.avg_pool2d(self.sr, kernel_size=2, stride=2, padding=0)

            mu1 = F.conv2d(self.hr, weiht_kernel, stride=[1, 1], padding=0)
            mu2 = F.conv2d(self.sr, weiht_kernel, stride=[1, 1], padding=0)

            mu1_sq =mu1.pow(2)  # 均值平方
            mu2_sq = mu2.pow(2)  # 均值平方
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(self.hr * self.hr, weiht_kernel, stride=[1, 1]) - mu1_sq
            sigma2_sq = F.conv2d(self.sr * self.sr, weiht_kernel, stride=[1, 1]) - mu2_sq
            sigma12 = F.conv2d(self.hr * self.sr, weiht_kernel, stride=[1, 1]) - mu1_mu2

            mask_0 = torch.ge(sigma1_sq, 0.0 * torch.ones_like(sigma1_sq))
            sigma1_sq = torch.mul(sigma1_sq, mask_0.float().to(device=hr_image.get_device()))
            sigma2_sq = torch.mul(sigma2_sq, mask_0.float().to(device=hr_image.get_device()))

            g = sigma12 / (sigma1_sq + self.eps)
            sv_sq = sigma2_sq - g * sigma12

            # mask 1
            mask_eps_greater_equal_1 = torch.ge(sigma1_sq, self.eps * torch.ones_like(sigma1_sq))
            mask_eps_less = torch.lt(sigma1_sq, self.eps * torch.ones_like(sigma1_sq))
            g = torch.mul(g, mask_eps_greater_equal_1.float().to(device=hr_image.get_device()))
            sv_sq = torch.mul(sv_sq, mask_eps_greater_equal_1.float().to(device=hr_image.get_device())) + \
                    torch.mul(sigma2_sq, mask_eps_less.float().to(device=hr_image.get_device()))
            sigma1_sq = torch.mul(sigma1_sq, mask_eps_greater_equal_1.float().to(device=hr_image.get_device()))

            # mask 2
            mask_eps_greater_equal_2 = torch.ge(sigma2_sq, self.eps * torch.ones_like(sigma2_sq))
            g = torch.mul(g, mask_eps_greater_equal_2.float().to(device=hr_image.get_device()))
            sv_sq = torch.mul(sv_sq, mask_eps_greater_equal_2.float().to(device=hr_image.get_device()))

            # mask 3
            mask_g_greater_equal = torch.ge(g, 0.0 * torch.ones_like(g))
            mask_g_less = torch.lt(g, 0.0 * torch.ones_like(g))
            sv_sq = torch.mul(sv_sq, mask_g_greater_equal.float().to(device=hr_image.get_device())) + \
                    torch.mul(sigma2_sq, mask_g_less.float().to(device=hr_image.get_device()))
            g = torch.mul(g, mask_g_greater_equal.float().to(device=hr_image.get_device()))

            # mask 4
            mask_sv_sq_greater = torch.gt(sv_sq, self.eps * torch.ones_like(sv_sq))
            mask_sv_sq_less_equal = torch.le(sv_sq, self.eps * torch.ones_like(sv_sq))
            sv_sq = torch.mul(sv_sq, mask_sv_sq_greater.float().to(device=hr_image.get_device())) + \
                    torch.mul(self.eps * torch.ones_like(sv_sq),
                              mask_sv_sq_less_equal.float().to(device=hr_image.get_device()))

            num.append(torch.sum(torch.log(1 + g * g * sigma1_sq / (sv_sq + self.sigma_nsq)) / torch.log(
                10 * torch.ones([1]).to(hr_image.get_device()))))
            den.append(torch.sum(
                torch.log(1 + sigma1_sq / self.sigma_nsq) / torch.log(10 * torch.ones([1]).to(hr_image.get_device()))))
        # return num, den

        vifp = torch.stack(num, dim=0) / torch.stack(den, dim=0)
        return torch.mean(vifp)

class SSIM_Loss(nn.Module):
    def __init__(self, device, cs_map=False, mean_metric=True, size = 11, sigma = 1.5):
        super(SSIM_Loss, self).__init__()
        '''
        The structural similarity loss
        '''
        self.cs_map = cs_map
        self.mean_metric = mean_metric
        self.size = size
        self.sigma = sigma
        self.device = device
    def _fspecial_gauss(self, size, sigma):
        '''
        Function to mimic the 'fspecial' gaussian MATLAB function
        '''
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=0)
        x_data = np.expand_dims(x_data, axis=0)

        y_data = np.expand_dims(y_data, axis=0)
        y_data = np.expand_dims(y_data, axis=0)

        x = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_data, dtype=torch.float32).to(self.device)

        guass = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return guass / torch.sum(guass)

    def forward(self, sr_image, hr_image):
        self.hr = hr_image[:, 0:1, :, :]
        self.sr = sr_image[:, 0:1, :, :]

        window = self._fspecial_gauss(size=self.size, sigma=self.sigma)
        K1 = 0.01
        K2 = 0.03

        # depth of image(255 in case the image has a different scale)
        L = 1
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        mu1 = F.conv2d(self.hr, window, stride=[1, 1], padding=0)
        mu2 = F.conv2d(self.sr, window, stride=[1, 1], padding=0)

        mu1_sq = mu1.pow(2)  # 均值平方
        mu2_sq = mu2.pow(2)  # 均值平方
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(self.hr * self.hr, window, stride=[1, 1]) - mu1_sq
        sigma2_sq = F.conv2d(self.sr * self.sr, window, stride=[1, 1]) - mu2_sq
        sigma12 = F.conv2d(self.hr * self.sr, window, stride=[1, 1]) - mu1_mu2

        if self.cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.mean_metric:
            value = torch.mean(value)

        return value

class MSSSIM_Loss(nn.Module):
    def __init__(self, device, mean_metric=True, level=3): # default=4
        super(MSSSIM_Loss, self).__init__()
        self.device = device
        self.mean_metric = mean_metric
        self.level = level
        self.ssim_loss = SSIM_Loss(device=device, cs_map=True, mean_metric=False)

    def forward(self, sr_image, hr_image):
        self.hr = hr_image[:, 0:1, :, :]
        self.sr = sr_image[:, 0:1, :, :]
        weight = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        weight = weight.to(self.device)
        # weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        mssim = []
        mcs = []
        for l in range(self.level):
            ssim_map, cs_map = self.ssim_loss(self.hr, self.sr)
            mssim.append(torch.mean(ssim_map))
            mcs.append(torch.mean(cs_map))
            filtered_im1 = F.avg_pool2d(self.hr, kernel_size=2, stride=2, padding=0)
            filtered_im2 = F.avg_pool2d(self.sr, kernel_size=2, stride=2, padding=0)
            self.hr = filtered_im1
            self.sr = filtered_im2

        mssim = torch.stack(mssim, dim=0)  # tf.stack(mssin, axis=0)
        mcs = torch.stack(mcs, dim=0)
        mssim = mssim.to(self.device)
        mcs = mcs.to(self.device)
        value = (torch.prod(mcs[0:self.level - 1] ** weight[0:self.level - 1]) * mssim[self.level - 1] ** weight[self.level - 1])

        if self.mean_metric:
            value = torch.mean(value)
        return value
# weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
# # weight = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
#
# print(weight)