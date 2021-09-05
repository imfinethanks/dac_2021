import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function, Variable
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#----------------------
#  ReAct
#----------------------
class LearnableBias(nn.Module):
    def __init__(self, in_channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)

    def forward(self, input):
        return input + self.bias.expand_as(input)

class RPReLU(nn.Module):
    '''RPReLU is a PReLU sandwitched by learnable biases'''
    def __init__(self, in_channels):
        super(RPReLU, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        self.shift_y = LearnableBias(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, input):
        input = self.shift_x(input)
        input = self.prelu(input)
        input = self.shift_y(input)
        return input

class RSign(nn.Module):
    '''RSign is a Sign function that shifts the inputs'''
    def __init__(self, in_channels):
        super(RSign, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        #self.binarize = FastSign()
        self.binarize = ReactSign()

    def forward(self, input):
        input = self.shift_x(input)
        input = self.binarize(input)
        return input


#----------------------
#  Quant
#----------------------
class FastSign(nn.Module):
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.3
        out_backward = torch.clamp(input, -t_clip, t_clip)
        return (out_forward.detach() - out_backward.detach() + out_backward)

class ReactSign(nn.Module):
    def __init__(self):
        super(ReactSign, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1.0
        mask2 = x < 0
        mask3 = x < 1.0
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        return (out_forward.detach() - out3.detach() + out3)


#-------------------------
#  Layer
#-------------------------
class BinaryConv2d(nn.Conv2d):
    '''
    A convolutional layer with its weight tensor binarized to {-1, +1}.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                        padding, dilation, groups, bias)
        self.binarize = ReactSign()

    def forward(self, input):
        return F.conv2d(input, self.binarize(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ScalingBinaryConv2d(nn.Conv2d):
    '''
    Calculate the mean of weights kernal as the scalar in the outchannel direction
    '''
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=False):
        super(ScalingBinaryConv2d, self).__init__(in_chn, out_chn, kernel_size, stride, padding, bias)
        
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, input):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(input, binary_weights, stride=self.stride, padding=self.padding, bias=self.bias)
        return y

#-------------------------------
# ReCU
#-------------------------------
class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1,2,3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)
        
        #* binarize
        bw = BinaryQuantize().apply(w2)
        #* 1bit conv
        output = F.conv2d(a, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha 
        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        return grad_input


#-------------------------------
# IR
#-------------------------------
class IRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1) #mean 均值
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1) #std 标准差
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = IRBinaryQuantize().apply(bw, self.k, self.t)
        #ba = IRBinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(a, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class IRBinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = k.to(device)
        t = t.to(device)
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class IRConv2d_float(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d_float, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1) #mean 均值
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1) #std 标准差
        #sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        #bw = IRBinaryQuantize().apply(bw, self.k, self.t)
        #ba = IRBinaryQuantize().apply(a, self.k, self.t)
        #bw = bw * sw
        output = F.conv2d(a, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output