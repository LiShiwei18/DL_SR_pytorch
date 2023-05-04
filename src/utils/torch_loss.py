import torch
import torch.nn.functional as F


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    ssim_loss = ssim_para * (1 - F.ssim(x, y, data_range=1, size_average=True))
    mse_loss = mse_para * F.mse_loss(y, x)

    return mse_loss + ssim_loss


def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    mae_loss = mae_para * F.l1_loss(x, y)
    mse_loss = mse_para * F.mse_loss(y, x)

    return mae_loss + mse_loss
