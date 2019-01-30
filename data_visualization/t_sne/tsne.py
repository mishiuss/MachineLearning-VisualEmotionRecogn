# -*- coding: utf-8 -*-
import torch
import torch.autograd
from torch import nn


class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        # TODO: реализуйте вычисление матрицы сходства для точек отображения и расстояние Кульбака-Лейблера
        # pij - значения сходства между точками данных
        # i, j - индексы точек
        x = self.logits.weight
        A = x[i.long()]
        B = x[j.long()]
        num = (1. + (A - B).pow(2).sum(1)).pow(-1)
        MA = x.expand(self.n_points, self.n_points, self.n_dim)
        MB = MA.permute(1, 0, 2)
        denom = (1. + (MA - MB).pow(2).sum(2)).pow(-1.0).view(-1).sum()
        denom -=pij.shape[0]

        qij = num/denom
        loss_kld = (pij * (torch.log(pij) - torch.log(qij))).sum()
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
