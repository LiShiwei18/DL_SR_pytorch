
from models.DFCAN16 import keras_DFCAN
keras_weight = "./weights.best"
input_height, input_width, n_channel = 502, 502, 1

import os
os.chmod(keras_weight, 0o644)
os.chmod('./', 0o777)

# from keras.models import load_model
# model = DFCAN((input_height, input_width, n_channel))
# model = model.load_weights(keras_weight)
# model.save_weights('model_weights.h5')

modelFN = keras_DFCAN
scale_factor = 2
model = modelFN((input_height, input_width, n_channel), scale=scale_factor)
model.load_weights(keras_weight)
model.save_weights('model_weights.h5')

from mmdnn.conversion import keras
from mmdnn.conversion import pytorch

# 从Keras导出IR
keras_parser = keras.Keras2Parser('model_weights.h5')
ir_model = keras_parser.run('ir_model')

# 从IR生成PyTorch代码和权重
pytorch_emitter = pytorch.PytorchEmitter(ir_model)
pytorch_emitter.run('pytorch.py', 'pytorch.pth')



import torch

model = torch.load('pytorch_model.pth')