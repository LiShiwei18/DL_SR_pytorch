import tensorflow as tf
from keras import backend as K


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss


def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    mae_loss = mae_para * K.mean(K.abs(x-y))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mae_loss + mse_loss

if __name__ == "__main__":
    import torch
    # Generate random tensors with torch.seed()
    torch.manual_seed(42)
    y_true = torch.rand(5, 3, 256, 256).permute(0,2,3,1).numpy()
    y_pred = (torch.rand(5, 3, 256, 256)).permute(0,2,3,1).numpy()
    # Call the loss_mse_ssim function with the random tensors
    loss_model = loss_mse_ssim(y_true, y_pred)
    sess = tf.Session()
    loss = sess.run(loss_model)
    sess.close()
    print(loss)

    # result : 0.265905