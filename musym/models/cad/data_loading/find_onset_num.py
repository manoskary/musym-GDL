from musym.utils import load_and_save
import torch
import itertools


def find_onsets(dataset="cad_pac_wtc", data_dir="../data/", num_comb=12):
    g, n_classes = load_and_save(dataset, data_dir)
    node_features = g.ndata.pop('feat')
    piece_idx = g.ndata.pop("score_name")
    score_duration = node_features[:, 3]
    onsets = node_features[:, 0]
    # Filter out up-beat instances
    mod_onsets = torch.remainder(onsets, 1)
    filter_beats_idx = torch.nonzero(mod_onsets == 0, as_tuple=True)[0]
    piece_idx = piece_idx[filter_beats_idx]
    score_duration = score_duration[filter_beats_idx]
    unique_pieces = torch.unique(piece_idx).tolist()
    combinations = list(itertools.combinations(unique_pieces, num_comb))
    for comb in combinations:
        num_onsets = 0
        for scidx in comb:
            durs = score_duration[piece_idx == scidx]
            num_onsets += len(torch.unique(durs))
        if num_onsets <= 2357:
            print(comb)


if __name__ == "__main__":
    find_onsets()