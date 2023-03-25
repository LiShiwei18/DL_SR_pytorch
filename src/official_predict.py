import argparse
import glob
import numpy as np
from PIL import Image
from keras import optimizers
import imageio
import os
import tensorflow as tf
from models import DFCAN16,DFGAN50
from utils.utils import prctile_norm, rm_outliers
from tqdm import tqdm


parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="../dataset/test/F-actin")
# parser.add_argument("--folder_test", type=str, default="input_wide_field_images")
parser.add_argument("--gpu_id", type=str, default="0")  #只有一块GPU，默认0
parser.add_argument("--gpu_memory_fraction", type=float, default=0.8) #占用GPU显存的比例
parser.add_argument("--model_name", type=str, default="DFCAN")
parser.add_argument("--model_weights", type=str, default="model_weights.h5")
parser.add_argument("--input_height", type=int, default=502)
parser.add_argument("--input_width", type=int, default=502)
parser.add_argument("--scale_factor", type=int, default=2)


args = parser.parse_args()
gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
# data_dir = args.data_dir
# folder_test = args.folder_test
model_name = args.model_name
model_weights = args.model_weights
input_width = args.input_width
input_height = args.input_height
scale_factor = args.scale_factor

# output_name = 'output_' + model_name + '-'
# test_images_path = data_dir + '/' + folder_test
# output_dir = data_dir + '/' + output_name

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                              glob test data path
# --------------------------------------------------------------------------------
# img_path = glob.glob(test_images_path + '/*.tif')
# img_path.sort()
# if not img_path:
#     flag_recon = 1
#     img_path = glob.glob(test_images_path + '/*')
#     img_path.sort()
#     n_channel = len(glob.glob(img_path[0] + '/*.tif'))
#     output_dir = output_dir + 'SIM'
# else:
#     flag_recon = 0  #置为0表示进行SISR超分辨率重建，为1表示进行SIM重建
#     n_channel = 1
#     output_dir = output_dir + 'SISR'

# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# --------------------------------------------------------------------------------
#                          select models and load weights
# --------------------------------------------------------------------------------
# modelFns = {'DFGAN': DFGAN50.Generator, 'DFCAN': DFCAN16.DFCAN}
n_channel = 1
modelFN = DFCAN16.DFCAN
optimizer = optimizers.adam(lr=1e-5, decay=0.5)
m = modelFN((input_height, input_width, n_channel), scale=scale_factor)
m.load_weights(model_weights)
m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

print('Processing ' +  '...')
#生成固定输入
import torch
torch.manual_seed(123)
img = torch.randn(1, 1, 502, 502).permute(0,2,3,1).numpy()

pr = m.predict(img)  #预测结果归一化，去除异常值

keras_output_torch_tensor = torch.Tensor(pr).permute(0, 3, 1, 2)
pass
