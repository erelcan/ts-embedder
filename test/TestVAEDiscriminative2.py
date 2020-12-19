from ts_embedder.generators.outer.PandasGenerator import PandasGenerator
from ts_embedder.embedders.vae.DiscriminativeTrainer import DiscriminativeTrainer
from ts_embedder.embedders.core.training.Embedder import Embedder

# Here is an example embedding_info for 1 dimensional time-series embedding (set feature_dim accordingly).
# In this example, cnn-based auto-encoder is implemented.
# For flexibility, we can build encoders and decoders from embedding info.
# Currently a few custom and keras layers are supported (Add new ones with one-liners in layer_creator.py)
# hidden_length is for the hidden layer which we sample from (not exactly, see the code VAEEncoder)~
# In VAE, we have implicit losses. It should be set True when the loss is computed in call method (e.g. with add_loss).
# The loss is composed of reconstruction loss and continuous latent loss (we should weight them).
# If the embedding distribution is discrete use discrete loss rather than continuous loss (or use combination of all).
# As there are 2 outputs and losses (both for main and discriminative parts), we should provide a weighting
# in loss weights,
# Also, if there are imbalance in data, provide class_weights in the loss info for the discriminative part
# as class_weights.
# If you have metrics, you may add them for main and discriminative parts.

# Need to check losses and network architecture. Losses are not stable (may see NaN etc.)

# embedding_length = 8
# embedding_info = {
#     "feature_dim": 1,
#     "generator_info": {
#         "pass_count": None,
#         "use_remaining": False
#     },
#     "model_info": {
#         "hidden_length": 10,
#         "has_implicit_loss": True,
#         "optimizer": "adam",
#         "metrics": {
#             "main": [],
#             "discriminative": []
#         },
#         "loss_weights": {
#             "main": 0.5,
#             "discriminative": 0.5
#         },
#         "loss_info": {
#             "main": None,
#             "discriminative": {
#                 "type": "hinge",
#                 "parameters": {},
#                 "class_weights": [0.25, 0.75]
#             }
#         },
#         "inner_loss_info": {
#             "reconstruction_loss": {
#                 "type": "MSLE",
#                 "parameters": {}
#             },
#             "z_loss_weight": 0.5,
#             "c_loss_weight": 0.0,
#             "r_loss_weight": 0.5
#         },
#         "encoder_latent_info": {
#             "temperature": 0.95,
#             "continuous_latent_length": embedding_length,
#             "discrete_latent_length": None
#         },
#         "encoder_layer_info": [
#             {
#                 "type": "Conv1D",
#                 "parameters": {
#                     "filters": 16,
#                     "kernel_size": 3,
#                     "padding": "same",
#                     "activation": "relu"
#                 }
#             },
#             {
#                 "type": "MaxPooling1D",
#                 "parameters": {
#                     "pool_size": 2,
#                     "padding": "same"
#                 }
#             },
#             {
#                 "type": "Conv1D",
#                 "parameters": {
#                     "filters": 1,
#                     "kernel_size": 3,
#                     "padding": "same",
#                     "activation": "relu"
#                 }
#             },
#             {
#                 "type": "MaxPooling1D",
#                 "parameters": {
#                     "pool_size": 2,
#                     "padding": "same"
#                 }
#             },
#             {
#                 "type": "Flatten",
#                 "parameters": {
#
#                 }
#             }
#         ],
#         "decoder_layer_info": [
#             {
#                 "type": "ExpandLayer",
#                 "parameters": {
#                     "axis": -1
#                 }
#             },
#             {
#                 "type": "Conv1D",
#                 "parameters": {
#                     "filters": 1,
#                     "kernel_size": 3,
#                     "padding": "same",
#                     "activation": "relu"
#                 }
#             },
#             {
#                 "type": "UpSampling1D",
#                 "parameters": {
#                     "size": 2
#                 }
#             },
#             {
#                 "type": "Conv1D",
#                 "parameters": {
#                     "filters": 16,
#                     "kernel_size": 3,
#                     "padding": "same",
#                     "activation": "relu"
#                 }
#             },
#             {
#                 "type": "TDD",
#                 "parameters": {
#                     "units": 1,
#                     "activation": "linear"
#                 }
#             }
#         ],
#         "fit_parameters": {"epochs": 100}
#     },
#     "save_info": {
#         "main_model_path": ".../main_model.h5",
#         "embedder_model_path": ".../embedder_model.h5",
#         "main_model_artifacts_path": ".../main_model_artifacts.pkl",
#         "embedder_artifacts_path": ".../embedder_model_artifacts.pkl"
#     },
#     "discriminative_info": {
#         "target_dim_length": 1,
#         "activation": "sigmoid"
#     }
# }


def train_and_embed(ts_data, labels, batch_size, embedding_info):
    # ts_data: pandas dataframe representing a time series over its columns (should be ordered).
    # target_df: pandas series keeping the label information (there may be a unstable results when a dataframe is
    # used instead of a series).

    train_embedder(ts_data, labels, batch_size, embedding_info)
    embedder = Embedder(embedding_info["save_info"]["embedder_model_path"], embedding_info["save_info"]["embedder_artifacts_path"])
    embeddings = embed(embedder, ts_data)
    return embeddings


def train_embedder(ts_data, labels, batch_size, embedding_info):
    # ts_data: pandas dataframe representing a time series over its columns (should be ordered).
    # target_df: pandas series keeping the label information (there may be a unstable results when a dataframe is
    # used instead of a series).

    # Create a generator from the given pandas dataframe (We may create custom outer generators from any source).
    # All the embedders use InnerGenerator which is a definite interface between the outer generator and the embedders.
    outer_generator = PandasGenerator(ts_data, batch_size, target_df=labels, return_targets=True)
    trainer = DiscriminativeTrainer(len(ts_data.columns), embedding_info["feature_dim"], outer_generator, embedding_info["generator_info"], embedding_info["model_info"], embedding_info["save_info"], embedding_info["discriminative_info"])
    embedder_model, main_model = trainer.train()
    return embedder_model, main_model


def embed(embedder, ts_data):
    # ts_data: pandas dataframe representing a time series over its columns (should be ordered).

    embeddings = embedder.embed(ts_data.to_numpy())
    return embeddings
