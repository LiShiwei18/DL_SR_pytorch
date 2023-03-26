'''测试整体模型功能是否正常'''
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

##预测
#生成固定输入
torch.manual_seed(123)
x0 = torch.rand(1, 1, 502, 502)
## torch的预测
torch_out = torch_model(x0)
##keras的预测
from keras import backend as K
# keras_x0 = K.constant(x0.permute(0,2,3,1).numpy())  #转到keras的输入
keras_x0 = x0.permute(0,2,3,1).numpy()
keras_output = keras_model.predict(keras_x0)
# print("keras卷积层输出：",type(keras_output))
keras_output_torch_tensor = torch.Tensor(keras_output).permute(0, 3, 1, 2)
print("torch模型和keras模型第index:{}层输出截取：".format("最后1"),torch_out[0,0,:3,:3], keras_output_torch_tensor[0,0,:3,:3],sep="\n")

