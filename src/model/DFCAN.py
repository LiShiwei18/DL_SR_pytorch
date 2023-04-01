# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import tensorflow as tf

first_time = True

def fftshift2d(img, size_psc=128):
    bs,ch, h, w = img.shape

    
    # choice1
    fs11 = img[:, :, -h // 2:h, -w // 2:w]
    fs12 = img[:, :, -h // 2:h, 0:w // 2]
    fs21 = img[:, :,  0:h // 2, -w // 2:w]
    fs22 = img[:, :,  0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    resized_image_np = F.interpolate(output, size=(size_psc, size_psc), mode='bilinear', align_corners=True)

    #choice2
    # img = img.permute(0, 2, 3, 1)
    # fs11 = img[:, -h // 2:h, -w // 2:w, :]
    # fs12 = img[:, -h // 2:h, 0:w // 2, :]
    # fs21 = img[:, 0:h // 2, -w // 2:w, :]
    # fs22 = img[:, 0:h // 2, 0:w // 2, :]
    # output = torch.cat([torch.cat([fs11, fs21], axis=1), torch.cat([fs12, fs22], axis=1)], axis=2)
    # #用tensorflow创建一个会话来resize
    # resized_output = tf.image.resize_images(output.cpu().detach().numpy(), (size_psc, size_psc), 0)
    # sess = tf.Session()
    # resized_image_np = sess.run(resized_output)
    # sess.close()
    # resized_image_np = torch.Tensor(resized_image_np)
    # resized_image_np = resized_image_np.permute(0, 3, 1, 2)
    # resized_image_np = resized_image_np.to(torch.device("cuda:0"))

    return resized_image_np

def shuffle_up_tf(inputs, scale):
    N, C, H, W = inputs.size()
    inputs = inputs.view(N, scale ** 2, -1, H, W).transpose(2, 1).contiguous()
    inputs = inputs.view(N, C, H, W)
    return F.pixel_shuffle(inputs, scale)

class RCAB(nn.Module):
    def __init__(self): #size_psc：crop_size input_shape：depth
        super().__init__()
        self.conv_gelu1=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                               nn.GELU())
        self.conv_gelu2=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                       nn.GELU())


        self.conv_relu1=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                               nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_relu2=nn.Sequential(nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0),
                               nn.ReLU())
        self.conv_sigmoid=nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0),
                               nn.Sigmoid())

    def forward(self,x,gamma=0.8):
        #结构没问题    2023-03-21 18:18
        #参数没看 
        x0=x
        x  = self.conv_gelu1(x)
        x  = self.conv_gelu2(x)
        x1 = x
        x  = torch.fft.fftn(x,dim=(2,3))
        x  = torch.pow(torch.abs(x)+1e-8, gamma) #abs   #2023-03-21 18:15 这里原来结构没有
        #这里等价原理fft2d的位置
        global first_time
        if first_time:
            print("torch的fft层输出：\n",x[0,0,:3,:3])
            first_time = False
        x  = fftshift2d(x)
        x  = self.conv_relu1(x)
        x  = self.avg_pool(x)                                                                 
        x  = self.conv_relu2(x)
        x  = self.conv_sigmoid(x)
        x  = x1*x
        x  = x0+x
        return x


class ResGroup(nn.Module):
    def __init__(self, n_RCAB=4): #size_psc：crop_size input_shape：depth
        super().__init__()
        RCABs=[]
        for _ in range(n_RCAB):
            RCABs.append(RCAB())
        self.RCABs=nn.Sequential(*RCABs)

    def forward(self,x):
        x0=x
        x=self.RCABs(x)
        x=x0+x
        return x



class DFCAN(nn.Module):
    def __init__(self, input_shape, scale=2, size_psc=128): #size_psc：crop_size input_shape：depth
        super().__init__()
        self.input=nn.Sequential(nn.Conv2d(input_shape, 64, kernel_size=3, stride=1, padding=1),
                                       nn.GELU(),)
        n_ResGroup=4
        ResGroups=[]
        for _ in range(n_ResGroup):
            ResGroups.append(ResGroup(n_RCAB=4))
        self.RGs  =nn.Sequential(*ResGroups)
        self.conv_gelu=nn.Sequential(nn.Conv2d(64, 64*(scale ** 2), kernel_size=3, stride=1, padding=1),
                                       nn.GELU())
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv_sigmoid=nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                       nn.Sigmoid())

    def forward(self,x):
        x=self.input(x)
        x=self.RGs(x)
        x=self.conv_gelu(x)
        x=shuffle_up_tf(x,2) #upsampling   #用下列代替了
        #2023-03-29 23:50 重写了
        # #用tensorflow的pixelshuffle
        # from keras.layers import Lambda
        # from models.common import pixel_shiffle
        # scale = 2
        # #定义一个pixel_shiffle并计算
        # x = x.permute(0, 2, 3, 1)
        # upsampled_layer = Lambda(pixel_shiffle, arguments={'scale': scale})(tf.convert_to_tensor(x.detach().numpy()))
        # sess = tf.Session()
        # upsampled_x = sess.run(upsampled_layer)
        # sess.close()
        # upsampled_x = torch.Tensor(upsampled_x)
        # upsampled_x = upsampled_x.permute(0, 3, 1, 2)

        x=self.conv_sigmoid(x)
        return x
    
    # def __getitem__(self, index):
    #     for k in enumerate(self.named_parameters)


if __name__ == '__main__':
    #x = Variable(torch.rand(2,1,64,64)).cuda()
    x=torch.rand(1,6,128,128)

    #model = UNet().cuda()
    model = DFCAN(input_shape=x.size()[1])
    # model.eval()
    y = model(x)
    print('Output shape:',y.shape)
    # import hiddenlayer as hl