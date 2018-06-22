import torch

from . import rnn
from . import common
from . import nonlinear


class AbstractSequenceDecoder(common.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(AbstractSequenceDecoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward_loss(self, z, x, lens=None):
        raise NotImplementedError()


class RNNDecoder(AbstractSequenceDecoder):
    name = "rnn-decoder"

    def __init__(self, *args, rnn_cls=rnn.BaseRNNCell, **kwargs):
        super(RNNDecoder, self).__init__(*args, **kwargs)
        self.rnn_cls = rnn_cls
        self.input_nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.in_dim
        )
        self.rnn = rnn_cls(
            input_dim=self.in_dim,
            hidden_dim=self.hidden_dim
        )
        self.output_nonlinear = nonlinear.get_default()(
            in_dim=self.hidden_dim,
            out_dim=self.out_dim
        )

    def forward_loss(self, z, x, lens=None):
        batch_size = z.size(0)
        x = self.invoke(self.input_nonlinear, x)
        h = self.rnn.form_hidden(z)
        o, _, _ = self.invoke(self.rnn, x, lens, h)
        o = o.view(-1, self.hidden_dim)
        o = self.invoke(self.output_nonlinear, o)
        return o.view(batch_size, -1, self.out_dim)


class RNNRecallingDecoder(AbstractSequenceDecoder):
    name = "rnn-recalling-decoder"

    def __init__(self, *args, rnn_cls=rnn.BaseRNNCell, **kwargs):
        super(RNNRecallingDecoder, self).__init__(*args, **kwargs)
        self.rnn_cls = rnn_cls
        self.input_nonlinear = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.in_dim
        )
        self.rnn = rnn_cls(
            input_dim=self.in_dim + self.hidden_dim,
            hidden_dim=self.hidden_dim
        )
        self.output_nonlinear = nonlinear.get_default()(
            in_dim=self.hidden_dim,
            out_dim=self.out_dim
        )

    def forward_loss(self, z, x, lens=None):
        batch_size, seq_len, _ = x.size()
        x = self.invoke(self.input_nonlinear, x)
        z_exp = z.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        x = torch.cat([x, z_exp], 2)
        o, _, _ = self.invoke(self.rnn, x, lens)
        o = o.view(-1, self.hidden_dim)
        o = self.invoke(self.output_nonlinear, o)
        return o.view(batch_size, -1, self.out_dim)


MODULES = [
    RNNDecoder,
    RNNRecallingDecoder
]