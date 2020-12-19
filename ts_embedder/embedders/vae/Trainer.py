from keras.layers import Input
from keras.models import Model

from ts_embedder.embedders.core.vae.VAETrainerABC import VAETrainerABC
from ts_embedder.embedders.core.vae.concrete.VAE import VAE
from ts_embedder.processors.SelfOrExpandReturner import SelfOrExpandReturner


class Trainer(VAETrainerABC):
    def __init__(self, ts_length, feature_dim, outer_generator, generator_info, model_info, save_info):
        super().__init__(SelfOrExpandReturner(), outer_generator, generator_info, model_info, save_info)

        self._ts_length = ts_length
        self._feature_dim = feature_dim

    def _get_model(self):
        encoder_input = Input((self._ts_length, self._feature_dim), dtype="float64")
        enc_dec = VAE(self._ts_length, self._feature_dim, self._model_info["hidden_length"], self._model_info["encoder_latent_info"], self._model_info["encoder_layer_info"], self._model_info["decoder_layer_info"], self._model_info["inner_loss_info"])
        decoder_output, _ = enc_dec(encoder_input)
        model = Model(inputs=encoder_input, outputs=decoder_output)

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["VAE", "VAEEncoder", "VAEDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["VAEEncoder"]

        return model
