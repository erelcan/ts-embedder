from ts_embedder.generators.outer.PandasGenerator import PandasGenerator
from ts_embedder.embedders.next_step_pred.DiscriminativeTrainer import DiscriminativeTrainer
from ts_embedder.embedders.core.training.Embedder import Embedder

# Here is an example embedding_info for 1 dimensional time-series embedding (set feature_dim accordingly).
# It uses lstm encoder-decoder with NSP.
# Using negative cosine similarity loss (Note that we normalize the embeddings when embedding).
# If no aux_tokens is provided, then it uses first and last time-step values as start and end tokens.
# As there are 2 outputs and losses (both for main and discriminative parts), we should provide a weighting
# in loss weights,
# Also, if there are imbalance in data, provide class_weights in the loss info for the discriminative part
# as class_weights.
# If you have metrics, you may add them for main and discriminative parts.


# embedding_info = {
#     "feature_dim": 1,
#     "processor_info": {
#         "aux_tokens": {
#             "start_token": 0,
#             "end_token": 0
#         }
#     },
#     "generator_info": {
#         "pass_count": None,
#         "use_remaining": True
#     },
#     "model_info": {
#         "embedding_length": 8,
#         "has_implicit_loss": False,
#         "optimizer": "adam",
#         "metrics": {
#             "main": [],
#             "discriminative": []
#         },
#         "loss_weights": {
#             "main": 0.5,
#             "discriminative": 0.5,
#         },
#         "loss_info": {
#             "main": {
#                 "type": "MSLE",
#                 "parameters": {
#
#                 }
#             },
#             "discriminative": {
#                 "type": "hinge",
#                 "parameters": {},
#                 "class_weights": [0.25, 0.75]
#             }
#         },
#         "encoder_info": {
#             "num_of_layers": 1,
#             "recurrent_type": "LSTM",
#             "recurrent_parameters": {"activation": "selu", "recurrent_activation": "tanh"},
#             "should_normalize": True
#         },
#         "decoder_info": {
#             "num_of_layers": 1,
#             "recurrent_type": "LSTM",
#             "recurrent_parameters": {"activation": "selu", "recurrent_activation": "tanh"}
#         },
#         "fit_parameters": {
#             "epochs": 100
#         }
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
    # Return targets will return label information along with the features.
    outer_generator = PandasGenerator(ts_data, batch_size, target_df=labels, return_targets=True)
    trainer = DiscriminativeTrainer(len(ts_data.columns), embedding_info["feature_dim"], embedding_info["processor_info"], outer_generator, embedding_info["generator_info"], embedding_info["model_info"], embedding_info["save_info"], embedding_info["discriminative_info"])
    embedder_model, main_model = trainer.train()
    return embedder_model, main_model


def embed(embedder, ts_data):
    # ts_data: pandas dataframe representing a time series over its columns (should be ordered).

    embeddings = embedder.embed(ts_data.to_numpy())
    return embeddings
