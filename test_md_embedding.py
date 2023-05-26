#%%
import dlrm_s_pytorch as s
import numpy as np
import torch
import tricks.md_embedding_bag as md

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

md.md_solver(
    torch.Tensor([100, 200, 300, 400]),
    alpha=0.3,
    d0=128,
    round_dim=False
    )

ud_emb = md.PrEmbeddingBag(num_embeddings=400, embedding_dim=32, base_dim=128)
ud_emb

# %%
dense_X = torch.randn(2, 13)
sparse_offset = [torch.LongTensor([0, 4]), torch.LongTensor([0, 4])]
sparse_index = [torch.LongTensor([9, 23, 29, 62, 31, 42, 43, 49]), torch.LongTensor([50, 62, 88, 93, 31])]

dlrm(dense_X, sparse_offset, sparse_index)
# %%
