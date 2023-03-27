'''测试第9层（也就是FCAlayer倒数第2个激活层，也就是relu层）'''
'''2023-03-26 21:44'''
import keras
from models.DFCAN16 import DFCAN as keras_DFCAN
from model.DFCAN import DFCAN as torch_DFCAN
input_height, input_width, n_channel = 502, 502, 1

weight_file = "model_weights.h5"

keras_model = keras_DFCAN((input_height, input_width, n_channel))
keras_model.load_weights(weight_file)

#加载keras中可以学习的层的权重
keras_layers_name = []
keras_layers_weight = []
for layer in keras_model.layers:
    if layer.trainable == True:
        keras_layers_name.append(layer.name)
        keras_layers_weight.append(layer.weights)

#加载pytorch权重
torch_layers_names = []
torch_layers_params = []
torch_model =  torch_DFCAN((n_channel))
for name, param in torch_model.named_parameters():
    torch_layers_names.append(name)
    if param.requires_grad:
        # print(name, param.data)
        torch_layers_params.append(param)

import torch

#赋值所有层的权重
for indx, weight in enumerate(keras_model.get_weights()):
    # weight = keras_model.get_weights()[0]   #keras第一层权重：numpy.ndarray
    if len(weight.shape) == 4:  #卷积核
        to_torch_weight = torch.Tensor(weight).permute(3,2,0,1) 
    elif len(weight.shape) == 1:  #偏置
        to_torch_weight = torch.Tensor(weight)
    else:
        raise BaseException
    # to_torch_weight = torch.Tensor(weight).permute(3,2,1,0)   
    [k for k in torch_model.named_parameters()][indx][1].data = to_torch_weight
    # print(to_torch_weight[0,0,:,:])

#生成固定输入
torch.manual_seed(123)
x0 = torch.rand(1, 1, 502, 502)

#定义用于pytorch打印的hook函数
torch_inter_layer_out = None
torch_inter_layer_in = None
def get_layer_output(layer, input, output):
    global torch_inter_layer_out
    torch_inter_layer_out = output
    global torch_inter_layer_in
    torch_inter_layer_in = input

def get_k_layer_name(indx):
    #indx: 所有的卷积层和激活层一共166个，返回第indx/166个
    #获取keras_model所有层。实际上就是83个卷积层
    trainable_layer_names = [layer.name.split(r"/")[0] for layer in keras_model.trainable_weights]
    from collections import OrderedDict
    ordered_dict = OrderedDict.fromkeys(trainable_layer_names)
    set_trainable_layer_names = list(ordered_dict.keys())
    # indx 为偶数返回卷积层,为奇数返回激活层
    ## 注意这里很不靠谱。只使用与前面3层卷积
    if indx % 2 == 0:
        return set_trainable_layer_names[indx//2]
    else:
        current_layer_name = set_trainable_layer_names[indx//2]
        for i, k in enumerate(keras_model.layers):
            if k.name == current_layer_name:
                next_layer_name = keras_model.layers[i+1].name
        return next_layer_name

#获取torch model要插入hook的hook_layer
#4先前明明正常，为什么又不行了。
#2023-03-25 17:11 3正常
#
pre_index = 9

#torch model的所有卷积层
all_torch_conv_layer_names = [k[0] for k in list(torch_model.named_modules()) if isinstance(k[1], torch.nn.modules.conv.Conv2d)]
#torch model的所有激活层（其实少一层）
all_torch_activ_layer_names = [list(torch_model.named_modules())[[m[0] for m in torch_model.named_modules()].index(k) + 1][0] for k in all_torch_conv_layer_names[:-1]]

import re
s = all_torch_conv_layer_names[pre_index//2] if pre_index%2==0 else all_torch_activ_layer_names[pre_index//2]
s = re.sub(r'\.(\d+)\.', r'[\1].', s)
s = re.sub(r'\.(\d+)$', r'[\1]', s)
# s = re.sub(r'\.[^.]*$', '', s)
print(s)

def get_member(cls, string):
    # 将输入字符串按照`.`分割成列表
    attribute_list = string.split(".")
    
    # 定义变量result，初始值为A
    result = cls
    
    # 遍历列表，访问对应的成员
    for attribute_name in attribute_list:
        if "[" in attribute_name and "]" in attribute_name:
            # 如果元素是以[数字]结尾的形式，就将数字提取出来，并使用getattr()函数访问result的对应成员
            index = int(attribute_name[attribute_name.index("[")+1:attribute_name.index("]")])
            attribute_name = attribute_name[:attribute_name.index("[")]
            result = getattr(result, attribute_name)[index]
        else:
            # 否则，直接使用getattr()函数访问result的对应成员
            result = getattr(result, attribute_name)
    return result

# 检查第layer_index层
layer_index = pre_index   
hook_layer = get_member(torch_model,s)
# hook_layer = model.RGs[0].RCABs[0].conv_gelu1[0]
#pytorch
hook_layer.register_forward_hook(get_layer_output) #添加钩子
torch_model(x0)
torch_out = torch_inter_layer_out

#keras
from keras import backend as K
keras_x0 = K.constant(x0.permute(0,2,3,1).numpy())  #转到keras的输入
keras_x0 = x0.permute(0,2,3,1).numpy()
# print(type(keras_x0))
# print(keras_x0.shape)
from keras.models import Model
layer_name = "conv2d_5"  #keras第
intermediate_layer_model = Model(inputs=keras_model.input, outputs=keras_model.get_layer(layer_name).output)
intermediate_layer_model.set_weights(keras_model.get_weights()[:layer_index+2])
keras_output = intermediate_layer_model.predict(keras_x0)
# print("keras卷积层输出：",type(keras_output))
keras_output_torch_tensor = torch.Tensor(keras_output).permute(0, 3, 1, 2)

#比较二者
# assert torch_out.shape == keras_output_torch_tensor.shape
# print(torch_out == keras_output_torch_tensor)
print("torch模型和keras模型第index:{}层输出shape：".format(pre_index),torch_out.shape, keras_output_torch_tensor.shape)
print("torch模型和keras模型第index:{}层输出截取：".format(pre_index),torch_out[0,0,:3,:3], keras_output_torch_tensor[0,0,:3,:3],sep="\n")

a=2
#torch的conv2d是包含偏置的。
#kerass的conv2d也包含偏置。后面的lambda函数是激活函数
#第 4层出错（从0计数）
#17:18 2023/3/25 torch的第4层输入时正确的（即使第3层输出，大头-0.198
#17:20 2023/3/25 torch的第4层权重卷积核大头-0.0237 0.0469
#17:22 2023/3/25 临时keras model即intermediate_layer_model确实输出是第3个卷积层的输出
#17:26 2023/3/25 临时keras model的卷积核参数和torch model的相同-0.02365266,  0.04694267
#17:27 2023/3/25 临时keras model的卷积层bias参数为-0.00071564,  0.03814552,  0.00582773
#17:29 2023/3/25 不对。为什么torch的卷积层参数：hook_layer.weight没有偏置参数。但是keras model.layers[-1].get_weights()会返回两个
#17:33 2023/3/25 hook_layer也有bias参数，值也为-0.0007,  0.0381,  0.0058，那就不懂为什么了
#17:42 2023/3/25 用以下代码测试，发现torch功能应该是正常。应该是keras model内部功能不正常
# import torch.nn as nn
# conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
# conv2d.weight = hook_layer.weight
# conv2d.bias = hook_layer.bias
# print("torch单层模型结果", conv2d(torch_inter_layer_in[0])[0,0,:3,:3])
#18:05 2023/3/25 将x0范围在0-1之间。还是不行
#18:06 2023/3/25 predict即使重复执行结果也不同。不知道和这个有没有关系
#18:11 2023/3/25 改batch_size为1还是不行，多次执行结果不同的毛病也仍然存在
#TODO：要不定义一个keras单层模型看看功能是否正常
#TODO: predict里有一个statards过程，可能和那个有关。->看看之前功能正常的时候stadard前后有没有变化->也不对呀。不管怎样都是要standard的。要错先前就错了

#21:33 2023/3/25 定义一个单层的keras 卷积模型，怎么既和上面intermediate_layer_model不同，又可torch单层模型不同？？
#定义一个keras单卷积层试试
# from keras.models import Sequential
# from keras.layers import Conv2D,Input
# import tensorflow as tf
# from keras.models import Model
# # tf.set_random_seed(0)
# input_shape = (502, 502, 64)
# inputs = Input(input_shape)
# conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
# conv2d_model = Model(inputs=inputs, outputs=conv)
# conv2d_model.set_weights(keras_model.get_weights()[2:4])
# print("keras单层模型结果",conv2d_model.predict(torch_inter_layer_in[0].permute(0,2,3,1).detach().numpy())[0,:3,:3,0])

#21:48 2023/3/25 上述代码中如果整体重复执行，每执行一次结果会不一样。但是如果只重复执行最后一步predict，结果会保持不变。说明
# 1 可能有参数没初始化
# 2 可能有每次定义的时候都会加上某个结构
#TODO:重启程序，看两次intermediate_layer_model执行结果。有没有差别->不是
#TODO：看predict之前有没有可能没有初始化还，get_weights为空->不是
#14:34 2023/3/26 将程序搬到了WSL中进行测试
#14:34 2023/3/26 test_copy 2.py中测试了shift_2d输出。有轻微差异。
#14:46 2023/3/26 test_copy 3.py中测试了fft2d层输出。相同（要说的话第一个元素有0001差异）。所以应该是shift_2d层有错误
#15:18 2023/3/26 修改了torch的dfgan，和keras的结构一样。但是输出还是和先前一样没有变化。还是有一点差距
#15:20 2023/3/26 两边都去掉了插值层后变得一样。说明是插值的原因
#16:05 2023/3/26 把插值函数用tf的代替的。功能是正常了。但是速度有点慢

a=4

