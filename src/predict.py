'''torch的predict函数'''
'''2023-05-03 16:45'''

import torch
import os
import glob
from models.DFCAN import DFCAN
from tqdm import tqdm
import imageio
import numpy as np
from utils.utils import prctile_norm, rm_outliers
from utils.zero_padding import zero_padding
from PIL import Image

test_types = "F-actin" #测试的样品
data_dir = "dataset/test/{}".format(test_types)
test_images_in_folder = os.path.join(data_dir, "input_wide_field_images")
test_images_out_folder = os.path.join(data_dir, "torch_output")
n_channel = 1
scale = 2
size_psc = 128
weight_file = "trained_models/DFCAN-SISR_{}/weights.pth".format(test_types)
padding_size = 512


#所有输入图片路径list
img_path = glob.glob(test_images_in_folder + '/*.tif')
img_path.sort()
#创建输出路径
os.makedirs(test_images_out_folder, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#实例化对象
model = DFCAN(n_channel, scale, size_psc)
model.load_state_dict(torch.load(weight_file))
model.to(device)

im_count = 0
for img_path_file in tqdm(img_path):
    # img = np.array(imageio.imread(img_path_file).astype(np.float))    #17:20 2023/4/1 改成下列
    img = np.array(imageio.imread(img_path_file).astype(np.float32))
    origin_h, origin_w = img.shape[-2:]
    # img = img[np.newaxis, :, :, np.newaxis]   #16:19 2023/4/1 改成下列
    img = img[np.newaxis, np.newaxis, :, :]

    img = prctile_norm(img) #所有像素归一化到0-1之间
    # img = TF.to_tensor(img) #1 6:20 2023/4/1 改成下列
    img = torch.Tensor(img)
    #填充为512大小
    img = zero_padding(img, padding_size)
    #预测
    img = img.to(device)
    with torch.no_grad():
        predict_result = model(img).squeeze().cpu().numpy()[:origin_h*scale, :origin_w*scale] #截取
        pr = rm_outliers(prctile_norm(predict_result))  #预测结果归一化，去除异常值
    outName = img_path_file.replace(test_images_in_folder, test_images_out_folder)
    if not outName[-4:] == '.tif':
        outName = outName + '.tif'
    img = Image.fromarray(np.uint16(pr * 65535))
    im_count = im_count + 1
    img.save(outName)

#16:33 2023/4/1 在2080ti上cuda内存溢出。执行不了
#cuda 版本不一致。似乎得重做环境了
#17:40 2023/4/1 在A5000上能正常跑，但是效果不如原版DFCAN

