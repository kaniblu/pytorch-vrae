import torch

from . import common
from . import encoder
from . import decoder


class VariationalSentenceAutoencoder(common.Module):
    name = "variational-sentence-autoencoder"

    def __init__(self, z_dim, word_dim, vocab_size, kld_scale=1.0,
                 emb_cls=common.Embedding,
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
        self.embedding = emb_cls(
            num_embeddings=vocab_size,
            embedding_dim=word_dim
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
            vocab_size=vocab_size
        )

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    def forward_loss(self, x, lens=None):
        x = self.invoke(self.embedding, x)
        h = self.invoke(self.encoder, x, lens)
        mu = self.invoke(self.mu_linear, h)
        logvar = self.invoke(self.logvar_linear, h)
        yield "loss", self.kld_loss(mu, logvar) * self.kld_scale
        z = self.sample(mu, logvar)
        if lens is not None:
            lens -= 1
        yield "pass", self.invoke(self.decoder, z, x[:, :-1], lens)


MODULES = [
    VariationalSentenceAutoencoder
]