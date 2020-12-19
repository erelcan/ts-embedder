from keras.layers import Input, Dense
from keras.models import Model

from ts_embedder.embedders.core.vae.VAETrainerABC import VAETrainerABC
from ts_embedder.embedders.core.vae.concrete.VAE import VAE
from ts_embedder.processors.DiscriminativeWrapper import DiscriminativeWrapper
from ts_embedder.processors.SelfOrExpandReturner import SelfOrExpandReturner
from ts_embedder.processors.SelfReturner import SelfReturner
from ts_embedder.embedders.core.aux.loss_factory import get_loss_function


class DiscriminativeTrainer(VAETrainerABC):
    def __init__(self, ts_length, feature_dim, outer_generator, generator_info, model_info, save_info, discriminative_info):
        super().__init__(DiscriminativeWrapper(SelfOrExpandReturner(mirror_target=True), SelfReturner()), outer_generator, generator_info, model_info, save_info)

        self._discriminative_info = discriminative_info

        self._ts_length = ts_length
        self._feature_dim = feature_dim

    def _get_model(self):
        encoder_input = Input((self._ts_length, self._feature_dim), dtype="float64")
        enc_dec = VAE(self._ts_length, self._feature_dim, self._model_info["hidden_length"], self._model_info["encoder_latent_info"], self._model_info["encoder_layer_info"], self._model_info["decoder_layer_info"], self._model_info["inner_loss_info"], name="main")
        decoder_output, encoded = enc_dec(encoder_input)
        discriminative_output = Dense(self._discriminative_info["target_dim_length"], activation=self._discriminative_info["activation"], name="discriminative")(encoded)

        model = Model(inputs=encoder_input, outputs=[decoder_output, discriminative_output])

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["VAE", "VAEEncoder", "VAEDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["VAEEncoder"]

        return model

    def _extract_embedder_model(self, model):
        encoder_input = model.layers[0].output
        embedding = model.get_layer("main").get_encoder()(encoder_input)
        embedder_model = Model(inputs=encoder_input, outputs=embedding)
        # Maybe misleading...
        # Consider not providing any info in compile~
        if self._model_info["has_implicit_loss"]:
            embedder_model.compile(optimizer=self._model_info["optimizer"], metrics=self._model_info["metrics"]["discriminative"])
        else:
            embedder_model.compile(optimizer=self._model_info["optimizer"], loss=get_loss_function(self._model_info["loss_info"]), metrics=self._model_info["metrics"]["discriminative"])

        return embedder_model
