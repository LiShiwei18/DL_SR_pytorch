import torch
import torch.nn.functional as F
from .ssim import SSIM


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1
    mse_para = 1

    # normalization
    x = y_true
    y = y_pred
    norm_x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-9)
    norm_y = (y - torch.min(y)) / (torch.max(y) - torch.min(y) + 1e-9)
    ssim = SSIM()
    ssim_loss = ssim_para * (1 - ssim(norm_x, norm_y))
    mse_loss = mse_para * F.mse_loss(norm_y, norm_x)

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

if __name__ == "__main__":
    # Generate random tensors with torch.seed()
    torch.manual_seed(42)
    y_true = torch.rand(5, 3, 256, 256)
    y_pred = torch.rand(5, 3, 256, 256)
    # Call the loss_mse_ssim function with the random tensors
    loss = loss_mse_ssim(y_true, y_pred)
    print(loss)

    # result : 0.2644
