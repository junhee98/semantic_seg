
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from utils.helpers import set_trainable
from math import ceil

class CReLU(nn.Module):

    def __init__(self, inplace=False):
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        x = torch.cat((x,-x),1)
        return F.relu(x,inplace=self.inplace)

class ReLU(nn.Module):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        #x = torch.cat((x,-x),1)
        return F.relu(x,inplace=self.inplace)

class SegNet_LRR(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, freeze_backbone=False, **_):
        super(SegNet_LRR, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained= pretrained)
        encoder = list(vgg_bn.features.children())
        self.act=nn.ReLU(inplace=False)
        #self.act=ReLU(inplace=False)
        self.mask_pool = nn.MaxPool2d(3,stride=1, padding=1)

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)


        decoder = encoder
        encoder[2] = CReLU(inplace=True) # 64 -> 128
        encoder[9] = CReLU(inplace=True) # 128 -> 256
        encoder[16] = CReLU(inplace=True) # 256 -> 512
        
        # Encoder, VGG without any maxpooling & activation
        self.stage1_1_encoder = nn.Sequential(*encoder[:3])
        self.stage1_1_encoder_1conv = nn.Sequential(nn.Conv2d(128,64,kernel_size=1))
        self.stage1_2_encoder = nn.Sequential(*encoder[3:5])
        self.stage2_1_encoder = nn.Sequential(*encoder[7:10])
        self.stage2_1_encoder_1conv = nn.Sequential(nn.Conv2d(256,128,kernel_size=1))
        self.stage2_2_encoder = nn.Sequential(*encoder[10:12])
        self.stage3_1_encoder = nn.Sequential(*encoder[14:17])
        self.stage3_1_encoder_1conv = nn.Sequential(nn.Conv2d(512,256,kernel_size=1))
        self.stage3_2_encoder = nn.Sequential(*encoder[17:22])
        self.stage4_1_encoder = nn.Sequential(*encoder[24:27])
        self.stage4_2_encoder = nn.Sequential(*encoder[27:32])
        self.stage5_encoder = nn.Sequential(*encoder[34:42])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.weighted_sum = nn.Conv2d(1472,64,kernel_size=1)
        # Decoder, same as the encoder but reversed, maxpool will not be used
        #decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:8])
        self.stage2_1_decoder = nn.Sequential(*decoder[9:14])
        self.stage2_2_decoder = nn.Sequential(*decoder[14:17])
        self.stage3_1_decoder = nn.Sequential(*decoder[18:23])
        self.stage3_2_decoder = nn.Sequential(*decoder[23:26])
        self.stage4_1_decoder = nn.Sequential(*decoder[27:29])
        self.stage4_2_decoder = nn.Sequential(*decoder[29:32])
        self.stage5_1_decoder = nn.Sequential(*decoder[33:35])
        self.stage5_2_decoder = nn.Sequential(*decoder[35:38])
        self.stage6_decoder = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_1_decoder, self.stage2_2_decoder, self.stage3_1_decoder, self.stage3_2_decoder, self.stage4_1_decoder, self.stage4_2_decoder, self.stage5_1_decoder, self.stage5_2_decoder, self.stage1_1_encoder_1conv, self.stage2_1_encoder_1conv, self.stage3_1_encoder_1conv)
        
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder, self.stage5_encoder], True)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        res = self.stage1_1_encoder(x)
        res = self.stage1_1_encoder_1conv(res)
        out = self.stage1_2_encoder(res)
        x1_size = out.size()
        out = self.act(out + res).clone()
        out_1 = self.edge_making(out) # residual
        x, indices1 = self.pool(out)

        res = self.stage2_1_encoder(x)
        res = self.stage2_1_encoder_1conv(res)
        out = self.stage2_2_encoder(res)
        x2_size = out.size()
        out = self.act(out + res).clone()
        out_2 = self.edge_making(out) # residual
        x, indices2 = self.pool(out)

        res = self.stage3_1_encoder(x)
        res = self.stage3_1_encoder_1conv(res)
        out = self.stage3_2_encoder(res)
        x3_size = out.size()
        out = self.act(out + res).clone()
        out_3 = self.edge_making(out) # residual
        x, indices3 = self.pool(out)

        res = self.stage4_1_encoder(x)
        out = self.stage4_2_encoder(res)
        x4_size = out.size()
        out = self.act(out + res).clone()
        out_4 = self.edge_making(out) # residual
        x, indices4 = self.pool(out)

        out = self.stage5_encoder(x)
        x5_size = out.size()
        out = self.act(out + x).clone()
        out_5 = self.edge_making(out) # residual
        x, indices5 = self.pool(out)


        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = x + out_5
        out = self.stage1_decoder(x)
        x = self.act(out + x)


        res = self.unpool(x, indices=indices4, output_size=x4_size)
        res = res + out_4
        out = self.stage2_1_decoder(res)
        out = self.act(out + res).clone()
        x = self.stage2_2_decoder(out)


        res = self.unpool(x, indices=indices3, output_size=x3_size)
        res = res + out_3
        out = self.stage3_1_decoder(res)
        out = self.act(out + res).clone()
        x = self.stage3_2_decoder(out)
        

        res = self.unpool(x, indices=indices2, output_size=x2_size)
        res = res + out_2
        out = self.stage4_1_decoder(res)
        out = self.act(out + res).clone()
        x = self.stage4_2_decoder(out)


        res = self.unpool(x, indices=indices1, output_size=x1_size)
        res = res + out_1
        out = self.stage5_1_decoder(res)
        out = self.act(out + res).clone()
        x = self.stage5_2_decoder(out)

        x = self.stage6_decoder(x)

        return x

    def edge_making(self, x):
        softmax_x = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
        pos_pooling_x = self.mask_pool(softmax_x)
        neg_pooling_x = self.mask_pool(-softmax_x)
        sum_pooling = pos_pooling_x + neg_pooling_x

        return sum_pooling

    '''def lrr_module(self,high_feature,lowest_feature):
        upsample = nn.Upsample(size=lowest_feature[2:], mode='bilinear')
        high_feature = upsample(high_feature)
        mask_high_feature = self.edge_making(high_feature)
        return mask_high_feature'''
        

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
