from model.DFCAN import fftshift2d as torch_fftshift2d
import pickle
import tensorflow as tf
import torch

with open("img.pkl", 'rb') as f:
    torch_img:torch.Tensor = pickle.load(f)
a = pickle 
#torch的运算结果
torch_fftshift_result = torch_fftshift2d(torch_img)  
# with open("torch_fftshift.pkl", 'wb') as f:
#     img = pickle.dump(torch_fftshift, f)


#DFCAN官方的fftshift2d
def fftshift2d(input, size_psc=128):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]
    output = tf.concat([tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)], axis=2)
    output = tf.image.resize(output, (size_psc, size_psc), method=tf.image.ResizeMethod.BILINEAR)
    return output
#
tf_img = tf.convert_to_tensor(torch_img.detach().numpy())
#注意维度转换
tf_img = tf.transpose(tf_img, perm=[0, 2, 3, 1])
tf_fftshift_result:tf.Tensor = fftshift2d(tf_img)
tf_fftshift_result = tf.transpose(tf_fftshift_result, perm=[0, 3, 1, 2])
#比较两者是否相同
torch_fftshift_result_numpy = torch_fftshift_result.detach().numpy()
tf_fftshift_result_numpy = tf_fftshift_result.numpy()
assert (torch_fftshift_result_numpy == tf_fftshift_result_numpy).all()
print("测试通过！")
pass