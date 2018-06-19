from . import rnn
from . import encoder
from . import decoder
from . import pooling
from . import nonlinear
from . import vae


MODULE_DICT = None


def get_module_names(group):
    ret = []
    for m in group.MODULES:
        names = m.name
        if isinstance(names, str):
            ret.append(names)
        else:
            ret.extend(names)
    return ret


def init():
    global MODULE_DICT
    if MODULE_DICT is None:
        groups = [
            rnn,
            vae,
            encoder,
            decoder,
            pooling,
            nonlinear,
        ]
        MODULE_DICT = {}
        for group in groups:
            for m in group.MODULES:
                names = m.name
                if not isinstance(names, (tuple, list)):
                    names = (names, )
                for name in names:
                    MODULE_DICT[name] = m


def get(name):
    return MODULE_DICT[name]


init()