'''填0函数'''
'''2023-04-02 14:08'''
'''尝试补0'''
'''2023-04-02 14:01'''
import torch

def zero_padding(input, size:int):
    '''将x的最后两个维度也就是h和w填充到size大小，填充为0'''
    # 创建全0 Tensor
    result = torch.zeros(*input.shape[:-2], size, size)
    # 将x Tensor放到y Tensor的左上角
    _, _, h, w = input.shape
    result[..., :h, :w] = input
    return result
