import torch.nn as nn
import torch.nn.init as init

import utils
from . import common


class BaseNonlinear(common.Module):
    def __init__(self, in_dim, out_dim=None):
        super(BaseNonlinear, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim, self.out_dim = in_dim, out_dim


class FunctionalNonlinear(BaseNonlinear):
    def __init__(self, *args, **kwargs):
        super(FunctionalNonlinear, self).__init__(*args, **kwargs)
        self.linear = common.Linear(self.in_dim, self.out_dim)
        self.func = self.get_func()

    @classmethod
    def get_func(cls):
        raise NotImplementedError()

    def forward_loss(self, x):
        return self.invoke(self.func, x)


class TanhNonlinear(FunctionalNonlinear):
    name = "tanh"

    def get_func(cls):
        return nn.Tanh()


class ReluNonlinear(FunctionalNonlinear):
    name = "relu"

    def get_func(cls):
        return nn.ReLU()


class GatedTanhNonlinear(BaseNonlinear):
    name = "gated-tanh"

    def __init__(self, *args, **kwargs):
        super(GatedTanhNonlinear, self).__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.o_linear = nn.Linear(self.in_dim, self.out_dim)
        self.g_linear = nn.Linear(self.in_dim, self.out_dim)

    def forward_loss(self, x):
        o = self.invoke(self.o_linear, x)
        g = self.invoke(self.g_linear, x)

        return self.tanh(o) * self.sigmoid(g)

    def reset_parameters(self):
        init.xavier_normal_(self.o_linear.weight.detach())
        init.xavier_normal_(self.g_linear.weight.detach())
        self.o_linear.bias.detach().zero_()
        self.g_linear.bias.detach().zero_()


_DEFAULT = TanhNonlinear


def get_default():
    global _DEFAULT
    return _DEFAULT


def set_default(module):
    global _DEFAULT
    _DEFAULT = module


MODULES = [
    TanhNonlinear,
    ReluNonlinear,
    GatedTanhNonlinear
]
