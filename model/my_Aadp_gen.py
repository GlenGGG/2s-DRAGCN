import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.ntu_rgb_d import Graph
from .agcn import unit_gcn


class MyAadpGen(nn.Module):

    def __init__(self, in_channels,
                 Aadp_C,
                 Aadp_out_channels,
                 kernel_size,
                 A,
                 multiply_relative_adjacency=True,
                 distance_relative_adjacency=True,
                 T=300
                 ):
        super().__init__()
        self.Aadp_C = Aadp_C
        self.Aadp_out_channel=Aadp_out_channels
        self.multiply_relative_adjacency = multiply_relative_adjacency
        self.distance_relative_adjacency = distance_relative_adjacency
        if multiply_relative_adjacency and distance_relative_adjacency:
            self.Aadp_adj_gen_param = nn.Parameter(
                torch.FloatTensor([0.0]), requires_grad=True)
        self.Aadp_batchnorm2d = nn.BatchNorm2d(T)
        self.Aadp_gen_batchnorm2d = nn.BatchNorm2d(self.Aadp_C)
        self.Aadp_gen_batchnorm2d_out = nn.BatchNorm2d(self.Aadp_out_channel)
        if self.multiply_relative_adjacency:
            self.gen_mul_Aadp = unit_gcn(in_channels, self.Aadp_C,
                                                      A)
            self.gen_mul_Aadp_out = unit_gcn(
                self.Aadp_C, self.Aadp_out_channel, A)
        self.Aadp_dropout = nn.Dropout(0.5)

    def forward(self, x):

        x_person_a, x_person_b = x.chunk(2, 1)
        x_person_a = torch.squeeze(x_person_a, 1)
        x_person_b = torch.squeeze(x_person_b, 1)
        x = []

        if self.distance_relative_adjacency:
            Aadp_e1_exp = torch.exp(x_person_a)
            Aadp_e2_exp = torch.exp(-x_person_b)
            Aadp_exp = torch.einsum(
                'nctv, nctw->nctvw', Aadp_e1_exp, Aadp_e2_exp)
            Aadp_exp = (torch.log(Aadp_exp+1e-5)).pow(2)
            Aadp_exp = (-1/3)*(torch.einsum('nctvw->ntvw', Aadp_exp))
            Aadp_exp = torch.exp(Aadp_exp)
            # Aadp_exp = self.Aadp_batchnorm2d(Aadp_exp)
        if self.multiply_relative_adjacency:
            Aadp_e1_mul = self.gen_mul_Aadp(x_person_a)
            Aadp_e2_mul = self.gen_mul_Aadp(x_person_b)
            Aadp_e1_mul = self.gen_mul_Aadp_out(Aadp_e1_mul)
            Aadp_e2_mul = self.gen_mul_Aadp_out(Aadp_e2_mul)
            # Aadp_mul_mod = torch.einsum('ntv, ntw->ntvw', 
            #     torch.sqrt(torch.sum(Aadp_e1_mul**2, dim=1)), 
            #     torch.sqrt(torch.sum(Aadp_e2_mul**2, dim=1)))
            Aadp_mul = torch.einsum(
                'nctv, nctw->ntvw', Aadp_e1_mul, Aadp_e2_mul)#/(Aadp_mul_mod+1e-5)
            Aadp_exp = self.Aadp_batchnorm2d(Aadp_exp)
            Aadp_mul = tanh_sigmoid(Aadp_mul)

        if self.multiply_relative_adjacency and self.distance_relative_adjacency:
            Aadp_gen_percent = tanh_sigmoid(self.Aadp_adj_gen_param)
            # Aadp_gen_percent = 0.5
            self.Aadp = Aadp_gen_percent*Aadp_mul+(1-Aadp_gen_percent)*Aadp_exp
            # print(Aadp_gen_percent)
        elif self.multiply_relative_adjacency:
            self.Aadp = Aadp_mul
        else:
            self.Aadp = Aadp_exp

        abnormal = torch.sum(torch.var(x_person_b, dim=3), dim=1)
        abnormal = torch.where(
            abnormal < 1e-4,
            torch.zeros_like(abnormal),
            torch.ones_like(abnormal))
        self.Aadp = torch.einsum('ntvw,nt->ntvw', self.Aadp, abnormal)
        self.Aadp = normalize_Aadp(self.Aadp)

        return self.Aadp


def normalize_Aadp(Aadp):

    Aadp = torch.where(Aadp < 0.5, torch.zeros_like(Aadp).cuda(), Aadp)
    da = torch.sum(Aadp, dim=2)  # sum over column
    db = torch.sum(Aadp, dim=3)  # sum over line
    norm = torch.einsum('ntv,ntw->ntvw', db, da)
    norm = torch.pow((norm+1e-5), -0.5)
    Aadp = Aadp*norm

    return Aadp

def tanh_sigmoid(param):

    return (torch.tanh(param)+1.0)/2.0
