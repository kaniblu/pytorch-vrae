import types
import inspect

import torch
import torch.nn as nn
import torch.nn.init as init

import utils


def resolve_cls(name, module):
    if inspect.isclass(name):
        return name
    else:
        return utils.resolve_obj(module, name)


def recursively_reset_parameters(parent):
    for module in parent.children():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


class Sequential(nn.Sequential):
    def reset_parameters(self):
        recursively_reset_parameters(self)


class ModuleList(nn.ModuleList):
    def reset_parameters(self):
        recursively_reset_parameters(self)


class Linear(nn.Linear):
    def reset_parameters(self):
        init.xavier_normal_(self.weight.detach())

        if self.bias is not None:
            self.bias.detach().zero_()


class Embedding(nn.Embedding):
    def reset_parameters(self):
        init.xavier_normal_(self.weight.detach())


class Module(nn.Module):
    name = None

    def __init__(self):
        super(Module, self).__init__()
        self.loss = None

    def reset_parameters(self):
        recursively_reset_parameters(self)

    def invoke(self, module, *args):
        ret = module(*args)
        if isinstance(ret, dict):
            loss = ret.get("loss")
            if loss is not None:
                if self.loss is not None:
                    self.loss += loss
                else:
                    self.loss = loss
            ret = ret.get("pass")
        return ret

    def forward_loss(self, *input):
        # yield "loss", 0
        # yield "pass", 0
        # return <pass>
        raise NotImplementedError()

    def forward(self, *input):
        self.loss = None
        ret = self.forward_loss(*input)
        if isinstance(ret, types.GeneratorType):
            ret = dict(ret)
            loss = ret.get("loss")
            if loss is not None:
                if self.loss is not None:
                    loss += self.loss
                    print("added")
                return {
                    "pass": ret.get("pass"),
                    "loss": loss
                }
            else:
                return {"pass": ret.get("pass")}
        else:
            if self.loss is None:
                return {"pass": ret}
            else:
                return {
                    "pass": ret,
                    "loss": self.loss
                }


class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Concat(Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *xs):
        return torch.cat(xs, self.dim)


class Parameter(nn.Parameter):
    def reset_parameters(self):
        self.data.detach().zero_()
