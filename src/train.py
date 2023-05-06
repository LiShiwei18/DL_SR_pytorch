import argparse
import matplotlib.pyplot as plt
import numpy as np
import datetime
import glob
import os
from models import DFCAN
from utils.torch_lr_controller import ReduceLROnPlateau
from utils.data_loader import data_loader, data_loader_multi_channel
from utils.utils import img_comp
from utils.torch_loss import loss_mse_ssim
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.8)
parser.add_argument("--mixed_precision_training", type=int, default=1)
parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/CCPs")
parser.add_argument("--save_weights_dir", type=str, default="/root/autodl-tmp/trained_models")
parser.add_argument("--model_name", type=str, default="DFCAN")
parser.add_argument("--patch_height", type=int, default=128)
parser.add_argument("--patch_width", type=int, default=128)
parser.add_argument("--input_channels", type=int, default=1)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--norm_flag", type=int, default=1)
parser.add_argument("--iterations", type=int, default=1000000)
parser.add_argument("--sample_interval", type=int, default=1000)
parser.add_argument("--validate_interval", type=int, default=2000)
parser.add_argument("--validate_num", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--load_weights", type=int, default=0)
parser.add_argument("--optimizer_name", type=str, default="adam")

args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
data_dir = args.data_dir
save_weights_dir = args.save_weights_dir
validate_interval = args.validate_interval
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
patch_height = args.patch_height
patch_width = args.patch_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
validate_num = args.validate_num
iterations = args.iterations
# load_weights = args.load_weights
# optimizer_name = args.optimizer_name
model_name = args.model_name
sample_interval = args.sample_interval

if torch.cuda.is_available():
    device = torch.device("cuda")          # 使用 GPU
else:
    device = torch.device("cpu")           # 使用 CPU

data_name = data_dir.split('/')[-1]
if input_channels == 1:
    save_weights_name = model_name + '-SISR_' + data_name
    cur_data_loader = data_loader
    train_images_path = data_dir + '/training_wf/'
    validate_images_path = data_dir + '/validate_wf/'
else:
    pass
    # save_weights_name = model_name + '-SIM_' + data_name
    # cur_data_loader = data_loader_multi_channel
    # train_images_path = data_dir + '/training/'
    # validate_images_path = data_dir + '/validate/'
save_weights_path = save_weights_dir + '/' + save_weights_name + '/'
train_gt_path = data_dir + '/training_gt/'
validate_gt_path = data_dir + '/validate_gt/'
sample_path = save_weights_path + 'sampled_img/'

if not os.path.exists(save_weights_path):
    os.makedirs(save_weights_path)
if not os.path.exists(sample_path):
    os.makedirs(sample_path)

# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'DFCAN': DFCAN.DFCAN}
modelFN = modelFns[model_name]

# --------------------------------------------------------------------------------
#                              define combined model
# --------------------------------------------------------------------------------
g = modelFN(input_channels)
g = g.to(device)  # assuming device is defined as the target device (e.g., "cuda" or "cpu")
optimizer_g = optim.Adam(params=g.parameters(), lr=start_lr, betas=(0.9, 0.999))
loss_fn = loss_mse_ssim  # assuming loss_mse_ssim is defined elsewhere
lr_controller = ReduceLROnPlateau(model=g, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                  cooldown=0, min_lr=start_lr * 0.1, verbose=True)

# --------------------------------------------------------------------------------
#                                 about Tensorboard
# --------------------------------------------------------------------------------
log_path = save_weights_path + 'graph'
if not os.path.exists(log_path):
    os.mkdir(log_path)
# writer = SummaryWriter(log_path)
train_names = 'training_loss'
val_names = ['val_MSE', 'val_SSIM', 'val_PSNR', 'val_NRMSE']


def write_log(writer, names, logs, batch_no):
    for name, log in zip(names, logs):
        writer.add_scalar(name, log, batch_no)
    writer.flush()

# --------------------------------------------------------------------------------
#                             Sample and validate
# --------------------------------------------------------------------------------
def Validate(iter, sample=0):
    validate_path = glob.glob(validate_images_path + '*')
    validate_path.sort()
    if sample == 1:
        r, c = 3, 3
        mses, nrmses, psnrs, ssims = [], [], [], []
        img_show, gt_show, output_show = [], [], []
        validate_path = np.random.choice(validate_path, size=r)
        for path in validate_path:
            [img, gt] = cur_data_loader([path], validate_images_path, validate_gt_path, patch_height,
                                        patch_width, 1, norm_flag=norm_flag, scale=scale_factor)
            # img = img.to(device)
            # gt = gt.to(device)
            output = g(torch.Tensor(img).to(device)).squeeze().detach().cpu().numpy()
            mses, nrmses, psnrs, ssims = img_comp(gt, output, mses, nrmses, psnrs, ssims)
            img_show.append(np.squeeze(np.mean(img, 1)))
            gt_show.append(np.squeeze(gt))
            output_show.append(output)
            # show some examples
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            axs[row, 1].set_title('MSE=%.4f, SSIM=%.4f, PSNR=%.4f' % (mses[row], ssims[row], psnrs[row]))
            for col, image in enumerate([img_show, output_show, gt_show]):
                axs[row, col].imshow(np.squeeze(image[row]))
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig(sample_path + '%d.png' % iter)
        plt.close()
    else:
        if validate_num < validate_path.__len__():
            validate_path = validate_path[0:validate_num]
        mses, nrmses, psnrs, ssims = [], [], [], []
        for path in validate_path:
            [img, gt] = cur_data_loader([path], validate_images_path, validate_gt_path, patch_height,
                                        patch_width, 1, norm_flag=norm_flag, scale=scale_factor)
            output = g(torch.Tensor(img).to(device)).squeeze().detach().cpu().numpy()
            mses, nrmses, psnrs, ssims = img_comp(gt, output, mses, nrmses, psnrs, ssims)

        # if best, save weights.best
        torch.save(g.state_dict(), save_weights_path + 'weights.pth')
        if min(validate_nrmse) > np.mean(nrmses):
            torch.save(g.state_dict(), save_weights_path + 'weights.pth')

        validate_nrmse.append(np.mean(nrmses))
        curlr = lr_controller.on_epoch_end(iter, np.mean(nrmses))
        # write_log(callback, val_names[0], np.mean(mses), iter)

# --------------------------------------------------------------------------------
#                                    training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_record = []
validate_nrmse = [np.Inf]
lr_controller.on_train_begin()
images_path = glob.glob(train_images_path + '/*')
for it in tqdm(range(iterations)):
    # ------------------------------------
    #         train generator
    # ------------------------------------
    input_g, gt_g = cur_data_loader(images_path, train_images_path, train_gt_path, patch_height, patch_width,
                                    batch_size, norm_flag=norm_flag, scale=scale_factor)
    input_g = torch.from_numpy(input_g).float().to(device)
    gt_g = torch.from_numpy(gt_g).float().to(device)
    
    # zero gradients
    optimizer_g.zero_grad()
    
    # forward pass
    output = g(input_g)
    
    # calculate loss
    loss_generator = loss_fn(output, gt_g)
    loss_record.append(loss_generator.item())

    # backward pass
    loss_generator.backward()
    
    # update parameters
    optimizer_g.step()

    elapsed_time = datetime.datetime.now() - start_time
    print("%d iteration: time: %s, g_loss = %s" % (it + 1, elapsed_time, loss_generator.item()))

    if (it + 1) % sample_interval == 0:
        images_path = glob.glob(train_images_path + '/*')
        Validate(it + 1, sample=1)

    if (it + 1) % validate_interval == 0:
        Validate(it + 1, sample=0)
        # write_log(callback, train_names, np.mean(loss_record), it + 1)
        loss_record = []

    # lr_controller.on_epoch_end(iter, )    #更新学习率