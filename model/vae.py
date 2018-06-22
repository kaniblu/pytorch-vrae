import torch

from . import common
from . import encoder
from . import decoder
from . import embedding


class VariationalSentenceAutoencoder(common.Module):
    name = "variational-sentence-autoencoder"

    def __init__(self, z_dim, word_dim, vocab_size,
                 kld_scale=1.0,
                 emb_cls=embedding.BasicEmbedding,
                 enc_cls=encoder.AbstractSequenceEncoder,
                 dec_cls=decoder.AbstractSequenceDecoder):
        super(VariationalSentenceAutoencoder, self).__init__()
        self.z_dim = z_dim
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.kld_scale = kld_scale
        self.emb_cls = emb_cls
        self.enc_cls = enc_cls
        self.dec_cls = dec_cls
        self.input_embed = emb_cls(
            vocab_size=vocab_size,
            dim=word_dim,
        )
        self.mu_linear = common.Linear(
            in_features=z_dim,
            out_features=z_dim
        )
        self.logvar_linear = common.Linear(
            in_features=z_dim,
            out_features=z_dim
        )
        self.encoder = enc_cls(
            in_dim=word_dim,
            hidden_dim=z_dim
        )
        self.decoder = dec_cls(
            in_dim=word_dim,
            hidden_dim=z_dim,
            out_dim=word_dim,
        )
        self.output_embed = emb_cls(
            vocab_size=vocab_size,
            dim=word_dim
        )

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    def apply_output_embed(self, o):
        batch_size, seq_len, _ = o.size()
        weight = self.output_embed.weight.t()
        o = torch.mm(o.view(-1, self.word_dim), weight)
        return o.view(batch_size, seq_len, -1)

    def forward_loss(self, x, lens=None):
        x = self.invoke(self.input_embed, x)
        h = self.invoke(self.encoder, x, lens)
        mu = self.invoke(self.mu_linear, h)
        logvar = self.invoke(self.logvar_linear, h)
        yield "loss", self.kld_loss(mu, logvar) * self.kld_scale
        z = self.sample(mu, logvar)
        if lens is not None:
            lens -= 1
        o = self.invoke(self.decoder, z, x[:, :-1], lens)
        yield "pass", self.apply_output_embed(o)

    def decode(self, z, x, eos_idx=None, max_len=100):
        batch_size = z.size(0)
        has_eos = x.new(batch_size).zero_().byte()
        lens = x.new(batch_size).fill_(x.size(1)).long()
        while has_eos.prod().item() != 1 and lens.max() < max_len + 1:
            x_emb = self.invoke(self.input_embed, x)
            o = self.invoke(self.decoder, z, x_emb, lens)
            o = o[:, -1].unsqueeze(1)
            logits = self.apply_output_embed(o)
            logits = logits.squeeze(1)
            preds = logits.max(1)[1]
            x = torch.cat([x, preds.unsqueeze(1)], 1)
            has_eos = (preds == eos_idx) | has_eos
            lens += (1 - has_eos).long()
        return x, lens + 1


MODULES = [
    VariationalSentenceAutoencoder
]