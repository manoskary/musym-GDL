from musym.utils import load_and_save
import torch
from itertools import combinations


def find_onsets(dataset="cad_pac_wtc", data_dir="../data/")
    g, n_classes = load_and_save(dataset, data_dir)
    node_features = g.ndata.pop('feat')
    piece_idx = g.ndata.pop("score_name")
    onsets = node_features[:, 0]
    # Filter out up-beat instances
    mod_onsets = torch.remainder(onsets, 1)
    unique_pieces = torch.unique(piece_idx).to_list()
    combinations = list(combinations(unique_pieces, 12))
    for comb in combinations:
        for scidx in comb:
            pass