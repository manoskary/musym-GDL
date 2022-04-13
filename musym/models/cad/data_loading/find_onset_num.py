from musym.utils import load_and_save
import torch
import itertools


def find_onsets(dataset="cad_pac_wtc", data_dir="../data/", num_comb=13):
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
    piece_onset_nums = dict()
    for scidx in unique_pieces:
        durs = score_duration[piece_idx == scidx]
        piece_onset_nums[scidx] = len(torch.unique(durs))
    combinations = list(itertools.combinations(unique_pieces, num_comb))
    for comb in combinations:
        num_onsets = sum(list(map(lambda x: piece_onset_nums[x], comb)))
        if num_onsets == 2357:
            print(comb)

if __name__ == "__main__":
    find_onsets()