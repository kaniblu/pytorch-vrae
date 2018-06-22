import utils
from . import rnn
from . import vae
from . import common
from . import pooling
from . import manager
from . import encoder
from . import decoder
from . import nonlinear
from . import embedding


def add_arguments(parser):
    ModelArgumentConstructor(parser).add_all_arguments()


class ModelArgumentConstructor(object):
    def __init__(self, parser):
        self.parser = parser

    @staticmethod
    def joinargs(parent, name):
        assert name is not None, "name cannot be empty"
        frags = [name]
        if parent is not None:
            frags.insert(0, parent)
        return '-'.join(frags)

    def add(self, name, parent=None, **kwargs):
        self.parser.add_argument(f"--{self.joinargs(parent, name)}", **kwargs)

    def add_module_argument(self, key, module):
        modules = manager.get_module_names(module)
        self.add(key, type=str, default=modules[0], choices=modules)

    def add_nonlinear_argument(self, key):
        self.add_module_argument(key, nonlinear)

    def add_pooling_arguments(self, key):
        self.add_module_argument(key, pooling)

    def add_rnn_arguments(self, key):
        self.add_module_argument(key, rnn)
        self.add("layers", parent=key, type=int, default=1)
        self.add("dynamic", parent=key, action="store_true", default=False)
        self.add("dropout", parent=key, type=float, default=0)

    def add_encoder_arguments(self, key):
        self.add_module_argument(key, encoder)
        self.add_rnn_arguments(self.joinargs(key, "cell"))
        self.add_pooling_arguments(self.joinargs(key, "pooling"))

    def add_decoder_arguments(self, key):
        self.add_module_argument(key, decoder)
        self.add_rnn_arguments(self.joinargs(key, "cell"))

    def add_vsae_arguments(self, key):
        self.add_module_argument(key, vae)
        self.add("z-dim", parent=key, type=int, default=512)
        self.add("word-dim", parent=key, type=int, default=300)
        self.add("kld-scale", parent=key, type=float, default=1.0)
        self.add_encoder_arguments(self.joinargs(key, "encoder"))
        self.add_decoder_arguments(self.joinargs(key, "decoder"))
        self.add("embed-freeze", parent=key, action="store_true", default=False)

    def add_all_arguments(self):
        self.add_nonlinear_argument("nonlinear")
        self.add_vsae_arguments("vae")


class ModelBuilder(object):
    def __init__(self, args, vocab):
        self.args = args
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.bos_idx = vocab.f2i.get(args.bos)
        self.eos_idx = vocab.f2i.get(args.eos)

    def get(self, key, default=None):
        return getattr(self.args, key, default)

    def get_module_cls(self, key, kwargs_map=None, fallback=None):
        if fallback is None:
            fallback = {}
        if kwargs_map is None:
            kwargs_map = {}
        type = self.get(key)
        cls = manager.get(type)
        sub_kwargs = utils.map_val(type, kwargs_map,
                                   ignore_err=True, fallback=fallback)
        def create(*args, **kwargs):
            return cls(*args, **kwargs, **sub_kwargs)
        return create

    def get_nonlinear_cls(self, key):
        return self.get_module_cls(key)

    def get_pooling_cls(self, key):
        return self.get_module_cls(key)

    def get_rnn_cls(self, key):
        return self.get_module_cls(key, fallback=dict(
            dynamic=self.get(f"{key}_dynamic"),
            dropout=self.get(f"{key}_dropout"),
            layers=self.get(f"{key}_layers")
        ))

    def get_encoder_cls(self, key):
        return self.get_module_cls(key, {
            "last-state-rnn-encoder": dict(
                rnn_cls=self.get_rnn_cls(f"{key}_cell")
            ),
            "pooled-rnn-encoder": dict(
                rnn_cls=self.get_rnn_cls(f"{key}_cell"),
                pool_cls=self.get_pooling_cls(f"{key}_pooling")
            )
        })

    def get_decoder_cls(self, key):
        return self.get_module_cls(key, {
            "rnn-decoder": dict(
                rnn_cls=self.get_rnn_cls(f"{key}_cell")
            ),
            "rnn-recalling-decoder": dict(
                rnn_cls=self.get_rnn_cls(f"{key}_cell")
            ),
        })

    def get_embedding_cls(self, key):
        return lambda *args, **kwargs: embedding.FineTunableEmbedding(
            *args, **kwargs,
            allow_padding=True,
            freeze=self.get(f"{key}_embed_freeze"),
            unfrozen_idx=[self.bos_idx, self.eos_idx]
        )

    def get_vsae_cls(self, key):
        return self.get_module_cls(key, {
            "variational-sentence-autoencoder": dict(
                z_dim=self.get(f"{key}_z_dim"),
                word_dim=self.get(f"{key}_word_dim"),
                vocab_size=self.vocab_size,
                kld_scale=self.get(f"{key}_kld_scale"),
                emb_cls=self.get_embedding_cls(key),
                enc_cls=self.get_encoder_cls(f"{key}_encoder"),
                dec_cls=self.get_decoder_cls(f"{key}_decoder")
            )
        })


def build_model(*args, **kwargs):
    builder = ModelBuilder(*args, **kwargs)
    nonlinear.set_default(builder.get_nonlinear_cls("nonlinear"))
    return builder.get_vsae_cls("vae")()