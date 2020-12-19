# Time-Series Embedder

Representing time-series is not straightforward. When deep learning solutions are considered, the number of samples plays a critical role, due to overfitting.

This project aims to provide an umbrella for time-series embedding solutions. First target is to implement the basic/naive approaches. Then, shape them to overcome overfitting.


## Software Engineering Aspects

- Keeps a definite interface for generators (the so-called *inner-generator*) feeding the models.
  - Users may provide their custom generators (the so-called *outer-generator*) to integrate any data source.
  - Thus, we separate the ingestion logic from ML/DL architectures/models.
- Abstracts shared training and embedding tasks so that developers can focus only on the model architectures.
- Currently supports 2 main approaches:
  - Seq2seq learning
    - Re-usable and generic encoder-decoders allowing choice of recurrent cells, their parameters; the number of layers in encoder and decoder etc.
  - VAE learning
    - Enables both continuous and discrete/concerete latent embeddings.
    - Enables users to define encoder and decoder over a config file for high flexiblity.
- Provides discriminative wrappers for both approaches.
- Provides basic processors and layers in addition to the ones in Keras.
- Hopefully, this software engineering approach will yield a better maintainable and flexible solution.


## TO-DO
- Although informal tests are conducted, we need more formal tests.
- Please, be careful with loss usage. Do not use before analyzing the needs, tensor shapes and theory.
- Need to benchmark!
  - On small number of samples, it tends to overfit.
  - Any contribution is kindly welcomed for applying the solution on large number of samples.
- May add existing time-series embedding solutions.
