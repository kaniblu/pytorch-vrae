import io
import gzip

import torch
import torch.utils.data as td

import utils


class TextFileReader(utils.UniversalFileReader):
    def __init__(self):
        super(TextFileReader, self).__init__("txt")

    def open_txt(self, path):
        return open(path, "r")

    def open_gz(self, path):
        return io.TextIOWrapper(gzip.open(path, "r"))


def pad_tensor(x, size, pad_idx=0):
    if size <= 0:
        return x

    padding = x.new(size).fill_(pad_idx)
    return torch.cat([x, padding])


def pad_sequences(x, max_len=None, pad_idx=0):
    if max_len is None:
        max_len = max(map(len, x))
    x = [pad_tensor(t, max_len - len(t), pad_idx) for t in x]
    x = torch.stack(x)

    return x


class TextSequenceDataset(td.Dataset):
    FEATURES = {
        "string",
        "tensor"
    }

    def __init__(self, path, feats=None, vocab=None, vocab_limit=None,
                 pad_bos=None, pad_eos=None, unk="<unk>"):
        self.path = path
        self.feats = feats
        self.vocab = vocab
        self.vocab_limit = vocab_limit
        self.pad_eos = pad_eos
        self.pad_bos = pad_bos
        self.unk = unk
        self.unk_idx = None
        self.data = None
        if self.feats is None:
            self.feats = [""]
        for feat in feats:
            utils.assert_oneof(feat, self.FEATURES, "sequence feature")
        self.getdata_map = {
            feat: getattr(self, f"get_{feat}") for feat in self.FEATURES
        }
        for feat in self.getdata_map:
            utils.assert_oneof(feat, self.FEATURES)
        self._load_data()

    def _load_data(self):
        reader = TextFileReader()
        with reader(self.path) as f:
            self.data = [line.rstrip().split() for line in f]
            if self.pad_eos is not None:
                self.data = [sent + [self.pad_eos] for sent in self.data]
            if self.pad_bos is not None:
                self.data = [[self.pad_bos] + sent for sent in self.data]

        if self.vocab is None:
            self.vocab = utils.Vocabulary()
            utils.populate_vocab(
                words=[w for s in self.data for w in s],
                vocab=self.vocab,
                cutoff=self.vocab_limit
            )
            self.vocab.add("<unk>")
        self.unk_idx = self.vocab.f2i.get(self.unk)

    def _word2idx(self, w):
        return self.vocab.f2i.get(w, self.unk_idx)

    def get_string(self, tokens):
        return " ".join(tokens)

    def get_tensor(self, tokens):
        return torch.LongTensor([self._word2idx(w) for w in tokens])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ret = dict()
        tokens = self.data[item]
        for feat in self.feats:
            ret[feat] = self.getdata_map[feat](tokens)
        return ret


class TextSequenceBatchCollator(object):
    FEATURES = {
        "string": "list",
        "tensor": "tensorvar"
    }
    DATA_TYPES = {
        "list",
        "tensor",
        "tensorvar"
    }

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx
        self.collate_map = {
            dt: getattr(self, f"collate_{dt}") for dt in self.DATA_TYPES
        }
        # sanity check
        for dt in self.collate_map:
            utils.assert_oneof(dt, self.DATA_TYPES)
        assert set(self.FEATURES) == set(TextSequenceDataset.FEATURES)

    def collate_list(self, batch):
        return batch

    def collate_tensor(self, batch):
        return torch.stack(batch)

    def collate_tensorvar(self, batch):
        lens = torch.LongTensor(list(map(len, batch)))
        max_len = lens.max().item()
        return pad_sequences(batch, max_len, self.pad_idx), lens

    def __call__(self, batches):
        sample = batches[0]
        ret = dict()
        for feat in sample:
            items = [inst[feat] for inst in batches]
            dt = self.FEATURES[feat]
            collate_fn = self.collate_map[dt]
            ret[feat] = collate_fn(items)
        return ret
