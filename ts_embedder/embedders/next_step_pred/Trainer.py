from keras.layers import Input
from keras.models import Model

from ts_embedder.embedders.core.seq2seq.Seq2SeqTrainerABC import Seq2SeqTrainerABC
from ts_embedder.embedders.core.seq2seq.EncoderDecoder import EncoderDecoder
from ts_embedder.processors.TsSeq2SeqProcessor import TsSeq2SeqProcessor


class Trainer(Seq2SeqTrainerABC):
    def __init__(self, ts_length, feature_dim, processor_info, outer_generator, generator_info, model_info, save_info):
        super().__init__(TsSeq2SeqProcessor(processor_info["aux_tokens"]), outer_generator, generator_info, model_info, save_info)

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

        enc_dec = EncoderDecoder(self._feature_dim, self._model_info["embedding_length"], "linear", self._model_info["encoder_info"], self._model_info["decoder_info"])

        decoder_output, _ = enc_dec([encoder_input, decoder_input])
        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

        self._main_model_artifacts["custom_objects_info"]["layer_info"] = ["EncoderDecoder", "StackedRecurrentEncoder", "StackedRecurrentDecoder"]
        self._embedder_artifacts["custom_objects_info"]["layer_info"] = ["StackedRecurrentEncoder"]

        return model
