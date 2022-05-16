import os.path
import torch
from models import HGVAE, VAELoss
from tqdm import tqdm
import pandas as pd


CHORDS = {
    "M/m": [0, 0, 1, 1, 1, 0],
    "sus4": [0, 1, 0, 0, 2, 0],
	"M7": [0, 1, 2, 1, 1, 1],
	"M7wo5": [0, 1, 0, 1, 0, 1],
	"Mmaj7": [1, 0, 1, 2, 2, 0],
	"Mmaj7maj9" : [1, 2, 2, 2, 3, 0],
	"M9": [1, 1, 4, 1, 1, 2],
	"M9wo5": [1, 1, 2, 1, 0, 1],
	"m7": [0, 1, 2, 1, 2, 0],
	"m7wo5": [0, 1, 1, 0, 1, 0],
	"m9": [1, 2, 2, 2, 3, 0],
	"m9wo5": [1, 2, 1, 1, 1, 0],
	"m9wo7": [1, 1, 1, 1, 2, 0],
	"mmaj7": [1, 0, 1, 3, 1, 0],
	"Maug": [0, 0, 0, 3, 0, 0],
	"Maug7": [1, 0, 1, 3, 1, 0],
	"mdim": [0, 0, 2, 0, 0, 1],
	"mdim7": [0, 0, 4, 0, 0, 2]
}

BASIS_FN = [
	'onset_feature.score_position', 'duration_feature.duration', 'fermata_feature.fermata',
	'grace_feature.n_grace', 'grace_feature.grace_pos', 'onset_feature.onset',
	'polynomial_pitch_feature.pitch', 'grace_feature.grace_note',
	'relative_score_position_feature.score_position', 'slur_feature.slur_incr',
	'slur_feature.slur_decr', 'time_signature_feature.time_signature_num_1',
	'time_signature_feature.time_signature_num_2', 'time_signature_feature.time_signature_num_3',
	'time_signature_feature.time_signature_num_4', 'time_signature_feature.time_signature_num_5',
	'time_signature_feature.time_signature_num_6', 'time_signature_feature.time_signature_num_7',
	'time_signature_feature.time_signature_num_8', 'time_signature_feature.time_signature_num_9',
	'time_signature_feature.time_signature_num_10', 'time_signature_feature.time_signature_num_11',
	'time_signature_feature.time_signature_num_12', 'time_signature_feature.time_signature_num_other',
	'time_signature_feature.time_signature_den_1', 'time_signature_feature.time_signature_den_2',
	'time_signature_feature.time_signature_den_4', 'time_signature_feature.time_signature_den_8',
	'time_signature_feature.time_signature_den_16', 'time_signature_feature.time_signature_den_other',
	'vertical_neighbor_feature.n_total', 'vertical_neighbor_feature.n_above', 'vertical_neighbor_feature.n_below',
	'vertical_neighbor_feature.highest_pitch', 'vertical_neighbor_feature.lowest_pitch',
	'vertical_neighbor_feature.pitch_range'
	]

NOTE_FEATURES = ["int_vec1", "int_vec2", "int_vec3", "int_vec4", "int_vec5", "int_vec6"] + \
    ["interval"+str(i) for i in range(13)] + list(CHORDS.keys()) + \
    ["is_maj_triad", "is_pmaj_triad", "is_min_triad", 'ped_note',
     'hv_7', "hv_5", "hv_3", "hv_1", "chord_has_2m", "chord_has_2M"]

CAD_FEATURES = [
	'perfect_triad', 'perfect_major_triad','is_sus4', 'in_perfect_triad_or_sus4',
	'highest_is_3', 'highest_is_1', 'bass_compatible_with_I', 'bass_compatible_with_I_scale',
	'one_comes_from_7', 'one_comes_from_1', 'one_comes_from_2', 'three_comes_from_4',
	'five_comes_from_5', 'strong_beat', 'sustained_note', 'rest_highest',
	'rest_lowest', 'rest_middle', 'voice_ends', 'v7',
	'v7-3', 'has_7', 'has_9', 'bass_voice',
	'bass_moves_chromatic', 'bass_moves_octave', 'bass_compatible_v-i', 'bass_compatible_i-v',
	'bass_moves_2M']


class MyGraph(object):
    def __init__(self, nodes, edges, name=None):
        self.x = nodes
        self.edge_index = edges
        self.name = name


class MyGraphDataset(object):
    def __init__(self, root_dir="root/dir"):
        self.graphs = self.process(root_dir)


    def process(self, root_dir):
        files = [file for file in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, file))]
        graphs = list()
        features = ["onset", "duration", "voice", "ts"] + BASIS_FN + NOTE_FEATURES + CAD_FEATURES
        for f in files:
            note_df = pd.read_csv(os.path.join(root_dir, f, "nodes.csv"))
            note_df = note_df[features]
            x = torch.Tensor(note_df.to_numpy()).float()
            edge_df = pd.read_csv(os.path.join(root_dir, f, "edges.csv"))
            edge_df.drop(columns=edge_df.columns[0], axis=1 , inplace=True)
            edge_index = torch.Tensor(edge_df.to_numpy()).long().t()
            graphs.append(MyGraph(x, edge_index, f))
        return graphs


def main():
    graphs = MyGraphDataset(root_dir='/home/manos/Desktop/JKU/codes/tonnetzcad/node_classification/cad-feature-wtc').graphs
    in_feats = graphs[0].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGVAE(in_feats, 16, in_feats, 3, 0.5)
    criterion = VAELoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in tqdm(range(50), position=0, desc="Epoch", leave=True):
        for graph in tqdm(graphs[:200], position=1, desc="Graph", leave=True):
            x, edge_index = graph.x, graph.edge_index
            x = x.squeeze().to(device)
            edge_index = edge_index.to(device)
            optimizer.zero_grad()
            out, mu, log_var = model(x, edge_index)
            loss = criterion(out, x, mu, log_var)
            loss.backward()
            optimizer.step()
        print("Epoch {:3d} | Loss {:.4f}".format(epoch, loss.item()))


if __name__ == '__main__':
    main()