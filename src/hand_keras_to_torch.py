'''手动读取.h5文件赋值给torch'''
import keras
from models.DFCAN16 import DFCAN as keras_DFCAN
from model.DFCAN import DFCAN as torch_DFCAN
input_height, input_width, n_channel = 502, 502, 1

weight_file = "model_weights.h5"

keras_model = keras_DFCAN((input_height, input_width, n_channel))
keras_model.load_weights(weight_file)

# layer_names = [layer.name for layer in keras_model.layers]
# weights = keras_model.get_weights()

#加载keras中可以学习的层的权重
keras_layers_name = []
keras_layers_weight = []
for layer in keras_model.layers:
    if layer.trainable == True:
        keras_layers_name.append(layer.name)
        keras_layers_weight.append(layer.weight)

#加载pytorch权重
torch_layers_names = []
torch_layers_params = []
model =  torch_DFCAN((n_channel))
for name, param in model.named_parameters():
    torch_layers_names.append(name)
    if param.requires_grad:
        # print(name, param.data)
        torch_layers_params.append(param)
assert len(keras_layers_weight) == len(torch_layers_params)
pass