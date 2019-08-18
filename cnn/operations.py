import torch
import torch.nn as nn
import math
import numpy as np

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    # BinarizeConv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    # BinarizeConv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      # BinarizeConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
      #           bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      BinarizeConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=C_in, bias=False),
      BinarizeConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      # nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      BinarizeConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                groups=C_in, bias=False),
      BinarizeConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      BinarizeConv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                groups=C_in, bias=False),
      BinarizeConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    # self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    # self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_1 = BinarizeConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = BinarizeConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


def binarize(tensor, mode='det'):
  if mode == 'det':
    if np.isnan(tensor.cpu().numpy()).any() :
      print("RAISING ALARM..,\n", tensor)
      raise Exception(" naN found here ")
    return torch.sign(tensor)
  else:
    return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(
      0, 1).round().mul_(2).add_(-1)

# class BinarizeLinear(nn.Linear):
#
#   def __init__(self, *kargs, **kwargs):
#     super(BinarizeLinear, self).__init__(*kargs, **kwargs)
#     self.reset_parameters()
#     # torch.nn.init.xavier_normal(self.weight)
#
#   def forward(self, input):
#     ## NCHW format. The first Linear or conv layer we dont binarize, but the subsequent ones we do
#     if input.size(1) != 1024:
#       input.data = binarize(input.data)
#     if not hasattr(self.weight, 'org'):
#       self.weight.org = self.weight.data.clone()
#     self.weight.data = binarize(self.weight.org)
#     out = nn.functional.linear(input, self.weight)
#     if not self.bias is None:
#       self.bias.org = self.bias.data.clone()
#       out += self.bias.view(1, -1).expand_as(out)
#     return out

def init_model(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      m.weight.data.normal_(0, math.sqrt(2. / n))
    # elif isinstance(m, nn.BatchNorm2d):
    #   m.weight.data.fill_(1)
    #   if m.bias is not None:
    #     m.bias.data.zero_()


class BinarizeConv2d(nn.Conv2d):

  def __init__(self, *kargs, **kwargs):
    super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
    #self.reset_parameters()
    n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
    self.weight.data.normal_(0,math.sqrt(2. / n))
    # torch.nn.init.xavier_normal(self.weight)

  ## NCHW format
  def forward(self, input):
    # if input.size(1) != 3:
    #   input.data = binarize(input.data)
    if not hasattr(self.weight, 'org'):
      self.weight.org = self.weight.data.clone()
    self.weight.data = binarize(self.weight.org)
    out = nn.functional.conv2d(input, self.weight, None, self.stride,
                               self.padding, self.dilation, self.groups)
    if not self.bias is None:
      self.bias.org = self.bias.data.clone()
      out += self.bias.view(1, -1, 1, 1).expand_as(out)
    return out
