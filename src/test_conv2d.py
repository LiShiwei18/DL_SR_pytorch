'''手动读取.h5文件赋值给torch'''
import keras
from models.DFCAN16 import DFCAN as keras_DFCAN
from model.DFCAN import DFCAN as torch_DFCAN
input_height, input_width, n_channel = 502, 502, 1

weight_file = "model_weights.h5"

keras_model = keras_DFCAN((input_height, input_width, n_channel))
keras_model.load_weights(weight_file)   #只能有一个tensflow后端运行。有多个请关闭

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
# assert len(keras_layers_weight) == len(torch_layers_params)
pass

import torch

#转第一层的权重
weight = keras_model.get_weights()[0]   #keras第一层权重：numpy.ndarray
to_torch_weight = torch.Tensor(weight).permute(3,2,0,1)    
# to_torch_weight = torch.Tensor(weight).permute(3,2,1,0)   
[k for k in model.named_parameters()][0][1].data = to_torch_weight
print(to_torch_weight[0,0,:,:])
#转置偏置的参数
weight = keras_model.get_weights()[1]
to_torch_weight = torch.Tensor(weight)
[k for k in model.named_parameters()][1][1].data = to_torch_weight
weight.shape
to_torch_weight.shape

#生成固定输入
torch.manual_seed(123)
x0 = torch.randn(1, 1, 502, 502)

#检查卷积层输出是否一致
#pytorch
torch_out = model.input[0](x0)

#keras
from keras import backend as K
# keras_x0 = K.constant(x0.permute(0,2,3,1).numpy())  #转到keras的输入
keras_x0 = x0.permute(0,2,3,1).numpy()
print(type(keras_x0))
print(keras_x0.shape)
from keras.models import Model
layer_name = 'conv2d_1'
intermediate_layer_model = Model(inputs=keras_model.input, outputs=keras_model.get_layer(layer_name).output)
keras_output = intermediate_layer_model.predict(keras_x0)
print(type(keras_output))
keras_output_torch_tensor = torch.Tensor(keras_output).permute(0, 3, 1, 2)

#比较二者
assert torch_out.shape == keras_output_torch_tensor.shape
# print(torch_out == keras_output_torch_tensor)
print(torch_out.shape, keras_output_torch_tensor.shape)
