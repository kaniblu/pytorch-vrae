from . import rnn
from . import common


class AbstractSequenceEncoder(common.Module):
    def __init__(self, in_dim, hidden_dim):
        super(AbstractSequenceEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def forward_loss(self, x, lens=None):
        raise NotImplementedError()


class RNNEncoder(AbstractSequenceEncoder):
    def __init__(self, *args, rnn_cls=rnn.BaseRNNCell, **kwargs):
        super(RNNEncoder, self).__init__(*args, **kwargs)
        self.rnn_cls = rnn_cls
        self.rnn = rnn_cls(
            input_dim=self.in_dim,
            hidden_dim=self.hidden_dim
        )

class LastStateRNNEncoder(RNNEncoder):
    name = "last-state-rnn-encoder"

    def forward_loss(self, x, lens=None):
        o, h = self.invoke(self.rnn, x, lens)
        return h


class PooledRNNEncoder(RNNEncoder):
    name = "pooled-rnn-encoder"

    def __init__(self, *args, pool_cls, **kwargs):
        super(PooledRNNEncoder, self).__init__(*args, **kwargs)
        self.pool_cls = pool_cls
        self.pool = pool_cls(self.hid_dim)

    def forward_loss(self, x, lens=None):
        o, h = self.invoke(self.rnn, x, lens)
        return self.invoke(self.pool, o, lens)


MODULES = [
    LastStateRNNEncoder,
    PooledRNNEncoder
]