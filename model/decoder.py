from . import rnn
from . import common


class AbstractSequenceDecoder(common.Module):
    def __init__(self, in_dim, hidden_dim, vocab_size):
        super(AbstractSequenceDecoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward_loss(self, z, x, lens=None):
        raise NotImplementedError()


class RNNDecoder(AbstractSequenceDecoder):
    name = "rnn-decoder"

    def __init__(self, *args, rnn_cls=rnn.BaseRNNCell, **kwargs):
        super(RNNDecoder, self).__init__(*args, **kwargs)
        self.rnn_cls = rnn_cls
        self.rnn = rnn_cls(
            input_dim=self.in_dim,
            hidden_dim=self.hidden_dim
        )
        self.linear = common.Linear(
            in_features=self.hidden_dim,
            out_features=self.vocab_size,
            bias=False
        )

    def forward_loss(self, z, x, lens=None):
        o, h = self.invoke(self.rnn, x, lens, z)
        return self.invoke(self.linear, o)


MODULES = [
    RNNDecoder
]