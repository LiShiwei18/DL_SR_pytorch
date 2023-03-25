'''手动读取.h5文件赋值给torch'''
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
model =  torch_DFCAN((n_channel))
for name, param in model.named_parameters():
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
    [k for k in model.named_parameters()][indx][1].data = to_torch_weight
    # print(to_torch_weight[0,0,:,:])

#生成固定输入
torch.manual_seed(123)
x0 = torch.randn(1, 1, 502, 502)

#定义用于pytorch打印的hook函数
torch_inter_layer_out = None
def get_layer_output(layer, input, output):
    global torch_inter_layer_out
    torch_inter_layer_out = output

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
pre_index = 4

#torch model的所有卷积层
all_torch_conv_layer_names = [k[0] for k in list(model.named_modules()) if isinstance(k[1], torch.nn.modules.conv.Conv2d)]
#torch model的所有激活层（其实少一层）
all_torch_activ_layer_names = [list(model.named_modules())[[m[0] for m in model.named_modules()].index(k) + 1][0] for k in all_torch_conv_layer_names[:-1]]

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
hook_layer = get_member(model,s)
# hook_layer = model.RGs[0].RCABs[0].conv_gelu1[0]
#pytorch
hook_layer.register_forward_hook(get_layer_output) #添加钩子
model(x0)
torch_out = torch_inter_layer_out

#keras
from keras import backend as K
# keras_x0 = K.constant(x0.permute(0,2,3,1).numpy())  #转到keras的输入
keras_x0 = x0.permute(0,2,3,1).numpy()
# print(type(keras_x0))
# print(keras_x0.shape)
from keras.models import Model
layer_name = get_k_layer_name(layer_index)  #keras第
intermediate_layer_model = Model(inputs=keras_model.input, outputs=keras_model.get_layer(layer_name).output)
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