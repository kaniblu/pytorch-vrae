import os
import sys
import yaml
import json
import random
import logging
import argparse

import pickle
import collections
import numpy as np
import tqdm as _tqdm

import torch
import torch.nn as N


FLOAT_MIN = float(np.finfo(np.float32).min)


class Vocabulary(object):
    def __init__(self):
        self.f2i = {}
        self.i2f = {}

    def add(self, w, ignore_duplicates=True):
        if w in self.f2i:
            if not ignore_duplicates:
                raise ValueError(f"'{w}' already exists")

            return self.f2i[w]
        idx = len(self.f2i)
        self.f2i[w] = idx
        self.i2f[idx] = w

        return self.f2i[w]

    def remove(self, w):
        """
        Removes a word from the vocab. The indices are unchanged.
        """
        if w not in self.f2i:
            raise ValueError(f"'{w}' does not exist.")

        index = self.f2i[w]
        del self.f2i[w]
        del self.i2f[index]

    def reconstruct_indices(self):
        """
        Reconstruct word indices in case of word removals.
        Vocabulary does not handle empty indices when words are removed,
          hence it need to be told explicity about when to reconstruct them.
        """
        words = list(self.f2i.keys())
        del self.i2f, self.f2i
        self.f2i, self.i2f = {}, {}

        for i, w in enumerate(words):
            self.f2i[w] = i
            self.i2f[i] = w

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2f[item]
        elif isinstance(item, str):
            return self.f2i[item]
        elif hasattr(item, "__iter__"):
            return [self[ele] for ele in item]
        else:
            raise ValueError(f"Unknown type: {type(item)}")

    def __contains__(self, item):
        return item in self.f2i or item in self.i2f

    def __len__(self):
        return len(self.f2i)


def populate_vocab(words, vocab, cutoff=None):
    if cutoff is not None:
        counter = collections.Counter(words)
        words, _ = zip(*counter.most_common(cutoff))

    for w in words:
        vocab.add(w)

    return vocab


def generate_dict_value(dict_list, key):
    for d in dict_list:
        yield d[key]


def generate_sent_words(sents):
    for sent in sents:
        for word in sent.split():
            yield word


def assert_oneof(item, candidates, name=None):
    assert item in candidates, \
        (f"'{name}' " if name is not None else "") + \
        f"is not one of '{candidates}'. item given: {item}"


def reduce_sum(tuples):
    ret = {}
    for k, v in tuples:
        if k not in ret:
            ret[k] = 0
        ret[k] += v
    return ret


def normalize(dic):
    s = sum(dic.values())
    return {k: v / s for k, v in dic.items()}


def flatten(lists):
    for l in lists:
        for e in l:
            yield e


def join_dict(dic, item_dlm, kvp_dlm):
    return item_dlm.join(kvp_dlm.join((k, v)) for k, v in dic.items())


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def get_devices(gpus):
    if len(gpus) < 1:
        devices = [torch.device("cpu")]
    else:
        devices = [torch.device("cuda", g) for g in gpus]
    return devices


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_device(mdl, devices):
    assert len(devices) >= 1, \
        "Must provide at least on device"
    if len(devices) == 1:
        mdl = mdl.to(devices[0])
    else:
        assert all(d.type == "cuda" for d in devices), \
            "cpu and cuda devices can't be mixed."
        main_dev = devices[0]
        mdl = mdl.to(main_dev)
        mdl = N.DataParallel(mdl, [d.index for d in devices],
                             output_device=main_dev.index)
    return mdl


def stringify(var):
    if isinstance(var, int):
        return f"{var:d}"
    elif isinstance(var, float):
        return f"{var:.4f}"
    else:
        return str(var)


def tqdm(iterable=None, **kwargs):
    return _tqdm.tqdm(
        iterable=iterable,
        ascii=False,
        unit_scale=True,
        dynamic_ncols=True,
        **kwargs
    )


def config_basic_logger(args):
    def config_handler(args, handler):
        handler.setLevel(level=args.log_level)
    handlers = []
    shandler = logging.StreamHandler(sys.stdout)
    config_handler(args, shandler)
    handlers.append(shandler)

    if args.log_file:
        path = os.path.join(args.save_dir, args.log_filename)
        fhandler = logging.FileHandler(path, mode="w")
        config_handler(args, fhandler)
        handlers.append(fhandler)

    logging.basicConfig(level=args.log_level, handlers=handlers)


def total_parameters(mdl):
    params = [p for p in mdl.parameters() if p.requires_grad]
    return sum(np.prod(p.size()) for p in params)


def unkval_error(name=None, val=None, choices=None):
    if name is None:
        name = "value"
    if val is None:
        val = ""
    else:
        val = f": {val}"
    if choices is not None:
        choices = ", ".join(f"\"{c}\"" for c in choices)
        choices = f"; possible choices: {choices}"
    else:
        choices = ""
    return ValueError(f"Unrecognized {name}{val}{choices}")


def dump_yaml(obj, f):
    if isinstance(f, str):
        f = open(f, "w")
    yaml.dump(
        data=obj,
        stream=f,
        allow_unicode=True,
        default_flow_style=False,
        indent=4
    )
    f.close()


def get_grouped_namespaces(parser, args):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in
                      group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return arg_groups


def save_args(parser, args, f):
    gargs = get_grouped_namespaces(parser, args)
    gargs = {k: vars(v) for k, v in gargs.items()}
    dump_yaml(gargs, f)


def map_val(key, maps: dict, name=None, ignore_err=False, fallback=None):
    if not ignore_err and key not in maps:
        raise unkval_error(name, key, list(maps.keys()))
    return maps.get(key, fallback)


def add_logging_arguments(parser, default_log_file):
    parser.add_argument("--log-level", type=int, default=10)
    parser.add_argument("--log-file", action="store_true", default=False)
    parser.add_argument("--log-filename", type=str,
                        default=f"{default_log_file}.log")


def merge_dict(a, b):
    ret = {}
    ret.update(a)
    ret.update(b)

    return ret


def resolve_obj(module, name):
    items = module.__dict__
    assert name in items, \
        f"Unrecognized attribute '{name}' in module '{module}'"
    return items[name]


def filter_dict(dic, s):
    return {k: v for k, v in dic.items() if k in s}


def manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def extname(path):
    base = os.path.basename(path)
    if "." not in base:
        return
    return os.path.splitext(base)[-1][1:]


def exclude(items, ex):
    if isinstance(ex, str) or not isinstance(ex, collections.Sequence):
        ex = [ex]
    ex = set(ex)
    for item in items:
        if item in ex:
            continue
        yield item


class UniversalFileReader(object):
    def __init__(self, default_ext=None):
        self.extdict = {}
        self.default_ext = default_ext
        self._load_extdict()
        if not self.extdict:
            raise NotImplementedError()
        if self.default_ext is not None:
            assert default_ext in self.extdict, \
                f"Unrecognized default ext: {self.default_ext}"

    def _load_extdict(self):
        for key in dir(self):
            if not key.startswith("open_"):
                continue
            val = getattr(self, key)
            if not callable(val):
                continue
            ext = key[len("open_"):]
            self.extdict[ext] = val

    def __call__(self, path):
        ext = extname(path)
        if ext is None:
            if self.default_ext is None:
                raise ValueError("file extension unavailable")
            ext = self.default_ext
        reader = map_val(ext, self.extdict, "file type")
        return reader(path)


def parse_args(parser):
    args, unk_args = parser.parse_known_args()
    if unk_args:
        logging.warning(f"Some arguments are unrecognized: {unk_args}")
    return args


def initialize_script(parser):
    args = parse_args(parser)
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(parser, args,
              os.path.join(args.save_dir, args.argparse_filename))
    config_basic_logger(args)
    return args