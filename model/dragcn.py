import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .ram_gen_process import RAMGen, scaledTanh, normalizeRAM


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(
                kernel_size, 1), padding=(
                pad, 0), stride=(
                stride, 1))
        self.RAM_conv = nn.Conv2d(
            1, 1, kernel_size=(
                kernel_size, 1), padding=(
                pad, 0), stride=(
                stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.RAM_bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        conv_init(self.RAM_conv)
        bn_init(self.bn, 1)
        bn_init(self.RAM_bn, 1)

    def forward(self, x, RAM):
        x = self.bn(self.conv(x))
        N, T, V, W = RAM.shape
        RAM = RAM.contiguous().view(N, 1, T, V*W)
        RAM = self.RAM_bn(self.RAM_conv(RAM))
        RAM = torch.squeeze(RAM, 1)
        N, T, VW = RAM.shape
        RAM = RAM.contiguous().view(N, T, V, W)
        RAM = normalizeRAM(scaledTanh(RAM))
        return x, RAM

# Plut drgc into agc
class unit_drgcn(nn.Module):
    def __init__(self, in_channels, out_channels,
                 A, coff_embedding=4, num_subset=3):
        super(unit_drgcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(
            torch.from_numpy(
                A.astype(
                    np.float32)),
            requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        self.conv_r = nn.Conv2d(in_channels, out_channels, 1)
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, RAM):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA
        res_x = self.down(x)

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(
                0, 3, 1, 2).contiguous().view(
                N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        # split two people
        x = x.contiguous().view(N//2, 2, C, T, V)
        x_a, x_b = x.chunk(2, 1)
        x_a = torch.squeeze(x_a)
        x_b = torch.squeeze(x_b)
        N, C, T, V = y.shape
        y = y.contiguous().view(N//2, 2, C, T, V)
        y_a, y_b = y.chunk(2, 1)
        y_a = torch.squeeze(y_a)
        y_b = torch.squeeze(y_b)
        # drgc
        x_a = self.conv_r(x_a)
        x_b = self.conv_r(x_b)
        x_a_r = torch.einsum('nctw,ntvw->nctv', (x_b, RAM))
        x_b_r = torch.einsum('nctv,ntvw->nctw', (x_a, RAM))
        # integrate
        y_a = y_a+x_a_r
        y_b = y_b+x_b_r
        y = torch.stack((y_a, y_b), 1)
        y = y.contiguous().view(N, C, T, V)

        y = self.bn(y)
        y += res_x
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_drgcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x, y: (0, 0)

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x, y: (x, y)

        else:
            self.residual = unit_tcn(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride)

    def forward(self, x, RAM):
        # plugged TCN for RAM
        res_x, res_RAM = self.residual(x, RAM)
        x, RAM = self.tcn1(self.gcn1(x, RAM), RAM)
        x = x+res_x
        RAM = RAM+res_RAM
        return self.relu(x), RAM


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        self.RAMGen = RAMGen(
            3, 128, 64, (9, A.shape[0], A.shape[1]), A, 300, True, True)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(
            N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(
            N, M, C, T, V)
        RAM = self.RAMGen(x)
        x = x.contiguous().view(N*M, C, T, V)

        x, RAM = self.l1(x, RAM)
        x, RAM = self.l2(x, RAM)
        x, RAM = self.l3(x, RAM)
        x, RAM = self.l4(x, RAM)
        x, RAM = self.l5(x, RAM)
        x, RAM = self.l6(x, RAM)
        x, RAM = self.l7(x, RAM)
        x, RAM = self.l8(x, RAM)
        x, RAM = self.l9(x, RAM)
        x, RAM = self.l10(x, RAM)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
