import os
import io
import gzip
import logging

import torch
import numpy as np

import utils


def add_embed_arguments(parser):
    parser.add_argument("--embed-type", type=str, default=None,
                        choices=list(e.name for e in EMBEDDINGS))
    parser.add_argument("--embed-path", type=str, default=None)
    parser.add_argument("--embed-freeze", action="store_true", default=False)


class Embeddings(object):
    name = None

    @property
    def dim(self):
        raise NotImplementedError()

    def preload(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __getitem__(self, item):
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class GloveFormatReader(utils.UniversalFileReader):
    def open_txt(self, path):
        return open(path, "r")

    def open_gz(self, path):
        return io.TextIOWrapper(gzip.open(path, "r"))


class GloveFormatEmbeddings(Embeddings):
    name = "glove-format"

    def __init__(self, path, dim=300, words=None):
        self.path = os.path.abspath(path)
        self.data = None
        self._dim = dim
        self.vocab = words

    @staticmethod
    def _tqdm(iterable=None):
        return utils.tqdm(
            iterable=iterable,
            desc="loading glove",
            unit="w",
        )

    def preload(self):
        self.data = {}
        dim = self.dim
        reader = GloveFormatReader(default_ext="txt")
        with reader(self.path) as f:
            for line in utils._tqdm.tqdm(f):
                tokens = line.split()
                word = " ".join(tokens[:-dim])
                if self.vocab is not None and word not in self.vocab:
                    continue
                vec = np.array([float(v) for v in tokens[-dim:]])
                self.data[word] = vec

        loaded_words = set(self.data.keys())
        coverage = len(loaded_words & self.vocab) / len(self.vocab)
        logging.info(f"{self.name} embeddings from {self.path} loaded, which"
                     f" covers {coverage * 100:.2f}% of vocabulary")

    def __hash__(self):
        return hash(self.name) * 541 + hash(self.path)

    def __eq__(self, other):
        if not isinstance(other, GloveFormatEmbeddings):
            return False
        return self.name == other.name and self.path == other.path

    @property
    def dim(self):
        return self._dim

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data.items())


class WordEmbeddingManager(object):
    def __init__(self):
        self.embeds = dict()

    def __getitem__(self, item: Embeddings):
        key = hash(item)
        if key not in self.embeds:
            item.preload()
            self.embeds[key] = item
        return self.embeds[key]


EMBEDDINGS = [
    GloveFormatEmbeddings
]


def _load_embeddings(weight, vocab, we):
    for w, v in we:
        if w not in vocab.f2i:
            continue
        idx = vocab.f2i[w]
        weight[idx] = torch.FloatTensor(v)


def get_embeddings(args, vocab):
    return utils.map_val(args.embed_type, {
        "glove-format": GloveFormatEmbeddings(
            path=args.embed_path,
            words=set(vocab.f2i)
        )
    }, "embedding type")


def load_embeddings(args, vocab, params):
    if args.embed_type is None:
        return
    embeddings = get_embeddings(args, vocab)
    embeddings.preload()
    _load_embeddings(params, vocab, embeddings)
