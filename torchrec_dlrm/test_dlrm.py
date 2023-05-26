#%%
from dlrm_main import *
from pprint import pprint
# %%
# Set the row numbers for each embedding table
num_embeddings_per_feature = [1000*i for i in range (1, 27)]

eb_configs = [
    EmbeddingBagConfig(
        name=f"t_{feature_name}",
        embedding_dim=64,
        num_embeddings=num_embeddings_per_feature[feauture_idx],
        feature_names=[feature_name],
    )
    for feauture_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
]
# %%
# Let's define DLRM model
dlrm_model = DLRM(
    embedding_bag_collection=EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
    dense_in_features=len(DEFAULT_INT_NAMES),
    dense_arch_layer_sizes=[512, 256, 64],
    over_arch_layer_sizes=[512, 256, 64, 1],
    dense_device="cuda:0",
)

# Wrap DLRM model with DLRMTrain
train_model = DLRMTrain(dlrm_model)

pprint(train_model)
# %%
