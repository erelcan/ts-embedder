import numpy as np

from ts_embedder.processors.ProcessorABC import ProcessorABC
from ts_embedder.processors.Seq2SeqShifter import Seq2SeqShifter


class TsSeq2SeqProcessor(ProcessorABC):
    def __init__(self, aux_tokens):
        super().__init__()

        self._start_token = aux_tokens["start_token"] if "start_token" in aux_tokens else None
        self._end_token = aux_tokens["end_token"] if "end_token" in aux_tokens else None

        self._shifter = Seq2SeqShifter(self._start_token, self._end_token)

    def process(self, data, training=True):
        # Supporting only 2D and 3D data.
        encoder_input, decoder_input, decoder_target = self._shifter.process(data)
        if len(data.shape) == 2:
            if training:
                return [np.expand_dims(encoder_input, axis=2), np.expand_dims(decoder_input, axis=2)], np.expand_dims(decoder_target, axis=2)
            else:
                return np.expand_dims(encoder_input, axis=2)
        else:
            if training:
                return [encoder_input, decoder_input], decoder_target
            else:
                return encoder_input
