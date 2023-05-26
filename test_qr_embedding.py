#%%
import dlrm_s_pytorch as s
import numpy as np
import torch
import tricks.qr_embedding_bag as qr

#%%
dlrm = s.DLRM_Net(
    m_spa=16,  # embedding table dimension
    ln_emb=np.array([100, 100]),  # row size for each embedding table
    ln_bot=np.array([13, 64, 16]),  # bottom MLP arch
    ln_top=np.array([19, 16, 8, 4, 1]),  # top MLP arch
    arch_interaction_op="dot",  # or "cat"
    arch_interaction_itself=False,
    qr_flag=False,
    md_flag=False,
)

qr_emb = qr.QREmbeddingBag(num_categories=100, embedding_dim=16, num_collisions=3)