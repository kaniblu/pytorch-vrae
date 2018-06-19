import torch


def mask(lens, max_len=None):
    if max_len is None:
        max_len = lens.max().item()
    enum = torch.range(0, max_len - 1).long()
    enum = enum.to(lens.device)
    enum_exp = enum.unsqueeze(0)
    return lens.unsqueeze(1) > enum_exp
