from keras.layers import Input, Dense
from keras.models import Model

from ts_embedder.embedders.core.seq2seq.Seq2SeqTrainerABC import Seq2SeqTrainerABC
from ts_embedder.embedders.core.seq2seq.EncoderDecoder import EncoderDecoder
from ts_embedder.processors.DiscriminativeWrapper import DiscriminativeWrapper
from ts_embedder.processors.TsSeq2SeqProcessor import TsSeq2SeqProcessor
from ts_embedder.processors.SelfReturner import SelfReturner
from ts_embedder.embedders.core.aux.loss_factory import get_loss_function


class DiscriminativeTrainer(Seq2SeqTrainerABC):
    def __init__(self, ts_length, feature_dim, processor_info, outer_generator, generator_info, model_info, save_info, discriminative_info):
        super().__init__(DiscriminativeWrapper(TsSeq2SeqProcessor(processor_info["aux_tokens"]), SelfReturner()), outer_generator, generator_info, model_info, save_info)

        self._discriminative_info = discriminative_info

        self._ts_length = ts_length
        self._feature_dim = feature_dim

        self._aux_token_is_on = "start_token" in processor_info["aux_tokens"] and "end_token" in processor_info["aux_tokens"]

    def _get_model(self):
        # Generator returns 3D data always..
        if self._aux_token_is_on:
            encoder_input = Input((self._ts_length, self._feature_dim), dtype="float64")
            decoder_input = Input((self._ts_length + 1, self._feature_dim), dtype="float64")
        else:
            encoder_input = Input((self._ts_length - 2, self._feature_dim), dtype="float64")
            decoder_input = Input((self._ts_length - 1, self._feature_dim), dtype="float64")

        enc_dec = EncoderDecoder(self._feature_dim, self._model_info["embedding_length"], "linear", self._model_info["encoder_info"], self._model_info["decoder_info"], name="main")
        decoder_output, encoded = enc_dec([encoder_input, decoder_input])

        discriminative_output = Dense(self._discriminative_info["target_dim_length"], activation=self._discriminative_info["activation"], name="discriminative")(encoded)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, discriminative_output])

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["EncoderDecoder", "StackedRecurrentEncoder", "StackedRecurrentDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["StackedRecurrentEncoder"]

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
