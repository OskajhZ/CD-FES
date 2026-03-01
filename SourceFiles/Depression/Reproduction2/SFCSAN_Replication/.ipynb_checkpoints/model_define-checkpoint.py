import torch
from typing import *

class Conv1dGroup(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding)
        self.batch_norm = torch.nn.BatchNorm1d(out_channel)
        self.selu = torch.nn.SELU()
    def forward(self, x):
        '''
        Shape of x: [batch_size, channel, width]
        '''
        y = self.conv(x)
        y = self.batch_norm(y)
        y = self.selu(y)
        return y

class Attention(torch.nn.Module):

    def __init__(self, in_channel: int):
        super().__init__()
        get_conv_pair = lambda out_channel: Conv1dGroup(in_channel, out_channel, 1, "same")
        self.F_trans = get_conv_pair(in_channel//8)
        self.G_trans = get_conv_pair(in_channel//8)
        self.H_trans = get_conv_pair(in_channel)
        self.O_trans = get_conv_pair(in_channel)
        self.gamma = torch.nn.Parameter(torch.tensor(1e-9))

    def forward(self, x):
        '''
        x should firstly be converted into [bs, in_channel, h, w] [bs, 256, 32]
        '''
        batch_size, channel, width = x.size()
        F = self.F_trans(x).transpose(-2, -1) # shape: [batch_size, width, channel]
        G = self.G_trans(x)  # shape: [batch_size, channel, width]
        H = self.H_trans(x) # shape: [batch_size, in_channel, width] 
        attention_matrix = torch.matmul(F,G)  # shape: [batch_size, width, width]
        attention_matrix = torch.nn.functional.softmax(attention_matrix, dim=-1).transpose(-2,-1)
        O = torch.matmul(H, attention_matrix) # shape: [batch_size, in_channel, width]
        O = self.O_trans(O)
        y = self.gamma * O + x
        return y # shape: [batch_size, in_channel, width]



class SFCSANet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        channel = 64
        get_conv = lambda : torch.nn.Sequential(
                Conv1dGroup(1, channel, 3, "same"),
                Conv1dGroup(channel, 2*channel, 3, "same"),
                Conv1dGroup(2*channel, 4*channel, 3, "same"),
                Attention(4*channel),
                Conv1dGroup(4*channel, 1, 7, "same")) # out shape: [bs, 256, 32]
        self.pcnn_theta = get_conv()
        self.pcnn_alpha = get_conv()
        self.pcnn_beta = get_conv()
        self.pcnn_gamma = get_conv()
        self.readout = torch.nn.Sequential(
                torch.nn.Linear(32*4, 32*4),
                torch.nn.SELU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(32*4, 1),
                torch.nn.Sigmoid())

    def forward(self, x):
        '''
        Return: Logits (without Softmax)
        In: [bs, 4*32] in order (theta, alpha, beta, gamma).
        Out: []
        '''
        theta_out = self.pcnn_theta(x[:,:32].view(-1, 1, 32)).flatten(1) # shape: [bs, 32]
        alpha_out = self.pcnn_alpha(x[:,32:2*32].view(-1, 1, 32)).flatten(1)
        beta_out = self.pcnn_beta(x[:,2*32:3*32].view(-1, 1, 32)).flatten(1)
        gamma_out = self.pcnn_gamma(x[:,3*32:].view(-1, 1, 32)).flatten(1)
        total = torch.cat((theta_out, alpha_out, beta_out, gamma_out), 1) # shape: [bs, 32*4]
        y = self.readout(total)
        return y

