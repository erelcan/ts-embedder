from ts_embedder.embedders.core.seq2seq.EncoderDecoder import EncoderDecoder
from ts_embedder.embedders.core.seq2seq.StackedRecurrentEncoder import StackedRecurrentEncoder
from ts_embedder.embedders.core.seq2seq.StackedRecurrentDecoder import StackedRecurrentDecoder
from ts_embedder.embedders.core.vae.concrete.VAE import VAE
from ts_embedder.embedders.core.vae.concrete.VAEEncoder import VAEEncoder
from ts_embedder.embedders.core.vae.concrete.VAEDecoder import VAEDecoder


def get_custom_layer_class(layer_name):
    return _custom_layer_mappings[layer_name]


_custom_layer_mappings = {
    "EncoderDecoder": EncoderDecoder,
    "StackedRecurrentEncoder": StackedRecurrentEncoder,
    "StackedRecurrentDecoder": StackedRecurrentDecoder,
    "VAE": VAE,
    "VAEEncoder": VAEEncoder,
    "VAEDecoder": VAEDecoder
}