import dgl
from scipy import sparse as sp
from scipy.sparse.linalg import eigs
import torch
import numpy as np


def positional_encoding(g: dgl.DGLGraph, pos_enc_dim: int) -> torch.Tensor:
    """
    Graph positional encoding v/ Laplacian eigenvectors

    Parameters
    ----------
    g : dgl.DglGraph
        A Dgl Graph
    pos_enc_dim : int
        The Positional Encoding Dimension to be added to Nodes of the graph

    Returns
    -------
    pos_enc : torch.Tensor
        A Dgl Graph with updated node data pos_enc tensor (N_nodes x pos_enc_dim)
    """
    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    EigVal, EigVec = eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    # increasing order
    EigVec = EigVec[:, EigVal.argsort()]
    pos_enc = torch.from_numpy(np.real(EigVec[:, 1:pos_enc_dim+1])).float()
    return pos_enc
