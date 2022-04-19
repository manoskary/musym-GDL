# General Imports
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import pandas as pd
from itertools import combinations
# modified perso imports
import partitura
# Local Imports
from data_loading import data_loading

MOZART_STRING_QUARTETS = [
	'k590-01', 'k155-02', 'k156-01', 'k080-02', 'k172-01',
	'k171-01', 'k172-04', 'k157-01', 'k589-01', 'k458-01',
	'k169-01', 'k387-01', 'k158-01', 'k157-02', 'k171-03',
	'k159-02', 'k428-02', 'k173-01', 'k499-03', 'k156-02',
	'k168-01', 'k080-01', 'k421-01', 'k171-04', 'k168-02',
	'k428-01', 'k499-01', 'k172-02', 'k465-04', 'k155-01',
	'k465-01', 'k159-01'
	]


BACH_FUGUES = [
	'wtc1f01', 'wtc1f07', 'wtc1f15', 'wtc1f13',
	'wtc1f06', 'wtc1f03', 'wtc1f02', 'wtc1f18',
	'wtc1f17', 'wtc1f09', 'wtc1f24', 'wtc1f10',
	'wtc1f22', 'wtc1f16', 'wtc1f12', 'wtc1f23',
	'wtc1f19', 'wtc1f05', 'wtc1f14', 'wtc1f04',
	'wtc1f08', 'wtc1f20', 'wtc1f21',
	]

CAD_FEATURES = [
    "int_vec1", "int_vec2", "int_vec3",
    "int_vec4", "int_vec5", "int_vec6",
    "memb_of_triad", "memb_of_sus4", "meb_of_v7",
    "is_maj_triad", "is_min_triad", "is_pmaj_triad",
    "is_dim", "ped_note",
    "hv_7", "hv_3", "hv_1",
    "bass_from_5", "is_onset", "three_from_four",
    "four_from_three", "one_from_seven", "one_from_two",
    "bass_moves_2m", "bass_moves_2M", "chord_has_2m", "chord_has_2M"
    ] + ["interval"+str(i) for i in range(13)]

def chord_to_intervalVector(midi_pitches):
	'''Given a chord it calculates the Interval Vector.
	Parameters
	----------
	midi_pitches : list(int)
		The midi_pitches, is a list of integers < 128.
	Returns
	-------
	intervalVector : list(int)
		The interval Vector is a list of six integer values.
	'''
	intervalVector = [0,0,0,0,0,0]
	PC = set([mp%12 for mp in midi_pitches])
	for p1, p2 in combinations(PC, 2):
		interval = int(abs(p1 - p2))
		if interval <= 6 :
			index = interval
		else :
			index = 12 - interval
		if index != 0:
			index = index-1
			intervalVector[index] += 1
	return intervalVector, list(PC)


def make_cad_features(na):
    ca = np.zeros((len(na), len(CAD_FEATURES)))
    bass_voice = na["voice"].max()
    for i, n in enumerate(na):
        n_onset = na[na["onset_beat"] == n["onset_beat"]]
        n_dur = na[np.where((na["onset_beat"] < n["onset_beat"]) & (na["onset_beat"]+na["duration_beat"] > n["onset_beat"]))]
        n_cons = na[na["onset_beat"]+na["duration_beat"] == n["onset_beat"]]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist())
        pc_class_recentered = sorted(list(map(lambda x : x - min(pc_class), pc_class)))
        memb_of_triad = 1 if int_vec == [0, 0, 1, 1, 1, 0] else 0
        memb_of_sus4 = 1 if int_vec == [0, 1, 0, 0, 2, 0] else 0
        meb_of_v7 = 1 if int_vec[1] > 0 and int_vec[2] > 1 and int_vec[3] > 0 and int_vec[4] > 1 and int_vec[5] > 0 else 0
        is_maj_triad = 1 if memb_of_triad and pc_class_recentered in [[0, 4, 7], [0, 5, 9], [0, 3, 8]] else 0
        is_min_triad = 1 if memb_of_triad and pc_class_recentered in [[0, 3, 7], [0, 5, 8], [0, 4, 9]] else 0
        is_pmaj_triad = 1 if is_maj_triad and 4 in (chord_pitch - chord_pitch.min())%12 and 7 in (chord_pitch - chord_pitch.min())%12 else 0
        # New Featyres
        is_dim = 1 if int_vec == [0, 0, 4, 0, 0, 2] else 0
        ped_note = 1 if n["duration_beat"] >= 3 else 0
        # End of New Features
        hv_7 = 1 if (chord_pitch.max() - chord_pitch.min())%12 == 10 else 0
        hv_3 = 1 if (chord_pitch.max() - chord_pitch.min())%12 in [3, 4] else 0
        hv_1 = 1 if (chord_pitch.max() - chord_pitch.min())%12 == 0 and chord_pitch.max() != chord_pitch.min() else 0
        is_onset = 1 if n["onset_beat"] % 1 == 0 else 0
        chord_has_2m = 1 if n["pitch"] - chord_pitch.min() in [1, -1] else 0
        chord_has_2M = 1 if n["pitch"] - chord_pitch.min() in [2, -2] else 0
        if n_cons.size:
            bass_from_5 = 1 if (chord_pitch.min()%12 - n_cons["pitch"].min()%12)%12 in [5, 7] else 0
            three_from_four = 1 if (n["pitch"] - chord_pitch.min())%12 in [3, 4] and (n["pitch"]+1 in n_cons["pitch"] or n["pitch"]+2 in n_cons["pitch"]) else 0
            four_from_three = 1 if (n["pitch"] - chord_pitch.min())%12 == 5 and (n["pitch"] - 1 in n_cons["pitch"] or n["pitch"] - 2 in n_cons["pitch"]) else 0
            one_from_seven = 1 if (n["pitch"] - chord_pitch.min())%12 == 0 and n["pitch"] != chord_pitch.min() and n["pitch"] - 1 in n_cons["pitch"] else 0
            one_from_two = 1 if (n["pitch"] - chord_pitch.min())%12 == 0 and n["pitch"] != chord_pitch.min() and (n["pitch"] + 1 in n_cons["pitch"] or n["pitch"] + 2 in n_cons["pitch"]) else 0
            bass_moves_2m = 1 if n["pitch"] - n_cons["pitch"].min() in [1, -1] and n["voice"] == bass_voice and bass_voice in n_cons[n_cons["pitch"] == n_cons["pitch"].min()]["voice"] else 0
            bass_moves_2M = 1 if n["pitch"] - n_cons["pitch"].min() in [2, -2] and n["voice"] == bass_voice and bass_voice in n_cons[n_cons["pitch"] == n_cons["pitch"].min()]["voice"] else 0
            # New Features
            n_cons_voice = n_cons[n_cons["voice"] == n["voice"]]
            # same voice intervals
            if n_cons_voice.size:
                intervals = {"interval"+str(i): (1 if i in (n_cons_voice["pitch"] - n["pitch"]) or -i in (n_cons_voice["pitch"] - n["pitch"]) else 0) for i in range(13)}
            else:
                intervals = {"interval"+str(i): 0 for i in range(13)}
            # End of New Features
        else:
            bass_from_5 = 0
            three_from_four = 0
            four_from_three = 0
            one_from_seven = 0
            one_from_two = 0
            bass_moves_2m = 0
            bass_moves_2M = 0
            intervals = {"interval" + str(i): 0 for i in range(13)}
        ca[i] = np.array(int_vec +
                         [memb_of_triad, memb_of_sus4, meb_of_v7,
                          is_maj_triad, is_min_triad, is_pmaj_triad,
                          is_dim, ped_note,
                          hv_7, hv_3, hv_1, bass_from_5, is_onset,
                          three_from_four, four_from_three, one_from_seven,
                          one_from_two, bass_moves_2m, bass_moves_2M, chord_has_2m, chord_has_2M]+list(intervals.values()))
    feature_fn = CAD_FEATURES
    ca = np.array([tuple(x) for x in ca], dtype=[(x, '<f8') for x in feature_fn])
    return ca, feature_fn


def align_feature(na, ba, ca):
    pitch_norm = na["pitch"] / 127.
    if np.all(np.isclose(pitch_norm, ba["polynomial_pitch_feature.pitch"])):
        return rfn.merge_arrays([na, ca, ba], flatten=True, usemask=False)
    else:
        print(np.nonzero(np.isclose(pitch_norm, ba["polynomial_pitch_feature.pitch"]))[0].shape)
        raise ValueError


def graph_csv_from_na(na, ra, t_sig, labels, feature_fn=None, norm2bar=True):
    '''Turn note_array to homogeneous graph dictionary.

    Parameters
    ----------
    na : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    ra : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.

    '''
    note_dict = {
        "nid": np.array(range(len(na))),
        "pitch": na["pitch"],
        "onset": na["onset_beat"],
        "duration": na["duration_beat"],
        "voice": na["voice"],
        "ts": np.array(list(map(lambda x: select_ts(x, t_sig), na))),
        "label": labels
    }
    for k in feature_fn:
        note_dict[k] = na[k]
    note = pd.DataFrame(note_dict)

    re = list()
    max_onset = np.max(na["onset_beat"])
    for r in ra:
        u = \
        np.where(((na["onset_beat"] >= r["onset_beat"]) & (na["onset_beat"] < r["onset_beat"] + r["duration_beat"])))[0]
        v = \
        np.where(((na["onset_beat"] < r["onset_beat"]) & (na["onset_beat"] + na["duration_beat"] > r["onset_beat"])))[0]
        if u.size == 0 and v.size == 0 and r["onset_beat"] < max_onset and r["duration_beat"] != 0:
            re.append(r)
    re = np.unique(np.array(re, dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4'), ('voice', '<i4')]))
    filter_inds = [i for i in np.where(re["onset_beat"] + re["duration_beat"] == r["onset_beat"])[0] for r in re]
    if filter_inds:
        for ind in filter_inds:
            re[ind]["duration_beat"] = re[ind + 1]["duration_beat"]
        re = np.delete(re, filter_inds)

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(na)
    for i, x in enumerate(na):
        for j in np.where((np.isclose(na["onset_beat"], x["onset_beat"]) == True) & (na["pitch"] != x["pitch"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(np.isclose(na["onset_beat"], x["onset_beat"] + x["duration_beat"]) == True)[0]:
            edg_src.append(i)
            edg_dst.append(j)

        if re.size > 0:
            for j in np.where(np.isclose(re["onset_beat"], x["onset_beat"] + x["duration_beat"]) == True)[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)

        for j in \
        np.where((x["onset_beat"] < na["onset_beat"]) & (x["onset_beat"] + x["duration_beat"] > na["onset_beat"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

    if re.size > 0:
        for i, r in enumerate(re):
            for j in np.where(np.isclose(na["onset_beat"], r["onset_beat"] + r["duration_beat"]) == True)[0]:
                edg_src.append(start_rest_index + i)
                edg_dst.append(j)

        rest_dict = {
            "nid": np.array(range(len(re))) + start_rest_index,
            "pitch": np.zeros(len(re)),
            "onset": re["onset_beat"],
            "duration": re["duration_beat"],
            "voice": re["voice"],
            "ts": np.array(list(map(lambda x: select_ts(x, t_sig), re))),
            "label": np.zeros(len(re))
        }
        for k in feature_fn:
            rest_dict[k] = np.zeros(len(re))
        rest = pd.DataFrame(rest_dict

                            )
    else:
        rest = pd.DataFrame()
    edges = pd.DataFrame(
        {
            "src": edg_src,
            "dst": edg_dst
        }
    )
    # Resize Onset Beat to bar
    if norm2bar:
        note["onset"] = np.mod(note["onset"], note["ts"])
        if re.size > 0:
            rest["onset"] = np.mod(rest["onset"], rest["ts"])

    nodes = pd.concat([note, rest], ignore_index=True)
    return nodes, edges


def select_ts(x, t_sig):
    for y in t_sig:
        if (x["onset_beat"] < y["end_beat"] or y["end_beat"] == -1) and x["onset_beat"] >= y["onset_beat"]:
            return y["nominator"]
    print(t_sig, x)


def filter_ts_end(ts, part):
    if ts.end:
        return part.beat_map(ts.end.t)
    else:
        return -1


def create_data(args):
    """
    Create a Toy Dataset from Mozart String Quartets.

    The Dataset searches >3 consecutive notes less than beat value.

    Parameters
    ----------
    args.save_dir : str
        The path of the save directory.
    """
    if not args.save_dir:
        args.save_dir = os.path.dirname(__file__)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    scores, annotations = data_loading(args)
    for key, fn in scores.items():
        if not os.path.exists(os.path.join(args.save_dir, key)):
            print(key)
            if fn.endswith(".musicxml"):
                part = partitura.load_musicxml(fn, force_note_ids=True)
            elif fn.endswith(".krn"):
                part = partitura.load_kern(fn)
            else:
                raise ValueError("The score {} format is not recognized".format(fn))
            # Assign Concistently Voices
            if isinstance(part, list):
                for i in range(len(part)):
                    for note in part[i].iter_all(partitura.score.Note):
                        note.voice = i
                    for rest in part[i].iter_all(partitura.score.Rest):
                        rest.voice = i

            part = partitura.score.merge_parts(part)
            # Not sure If I have to unfold
            part = partitura.score.unfold_part_maximal(part)

            rest_array = np.array(
                [(part.beat_map(r.start.t), part.beat_map(r.duration), r.voice) for r in part.iter_all(partitura.score.Rest)],
                dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4'), ('voice', '<i4')])
            time_signature = np.array(
                [
                    (part.beat_map(ts.start.t), filter_ts_end(ts, part), ts.beats, ts.beat_type) for ts in
                    part.iter_all(partitura.score.TimeSignature)
                ],
                dtype=[('onset_beat', '<f4'), ('end_beat', '<f4'), ("nominator", "<i4"), ("denominator", "<i4")]
            )
            na = partitura.utils.ensure_notearray(part)
            ca, cad_features = make_cad_features(na)
            ba, feature_fn = partitura.musicanalysis.make_note_feats(part, "all")
            ba = np.array([tuple(x) for x in ba], dtype=[(x, '<f8') for x in feature_fn])
            note_array = align_feature(na, ba, ca)
            labels = np.zeros(note_array.shape[0])
            feature_fn = feature_fn + cad_features
            # In case annotation are of the form Bar & Beat transform to global beat.
            if len(annotations[key]) >0:
                if isinstance(annotations[key][0], tuple):
                    measures = dict()
                    for m in part.iter_all(partitura.score.Measure):
                        # Account for repeat unfolding
                        if m.number in measures.keys():
                            measures[m.number].append(part.beat_map(m.start.t))
                        else:
                            measures[m.number] = [part.beat_map(m.start.t)]
                    tmp = [b_map + onset for bar, onset in annotations[key] for b_map in measures[bar]]
                    # transform to list of beats
                    annotations[key] = tmp

                if (not np.all(na["onset_beat"] >= 0)) and (args.source != "mps"):
                    annotations[key] += min(na["onset_beat"])

                if key in MOZART_STRING_QUARTETS:
                    args.source = "msq"
                elif key in BACH_FUGUES:
                    args.source = "wtc"
                else:
                    pass

                # Corrections of annotations with respect to time signature.
                if time_signature["denominator"][0] == 2 and args.source in ["wtc", "msq"]:
                    annotations[key] = list(map(lambda x: x/2, annotations[key]))
                elif time_signature["denominator"][0] == 8 and args.source in ["wtc", "msq"]:
                    annotations[key] = list(map(lambda x: x*2, annotations[key]))
                else:
                    pass

            for cad_onset in annotations[key]:
                labels[np.where(note_array["onset_beat"] == cad_onset)] = 1
                # check for false annotation that does not have match
                if np.all((note_array["onset_beat"] == cad_onset) == False):
                    raise IndexError("Annotated beat {} does not match with any score beat in score {}.".format(cad_onset, key))

            nodes, edges = graph_csv_from_na(note_array, rest_array, time_signature, labels, feature_fn=feature_fn,
                                             norm2bar=args.norm2bar)
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.makedirs(os.path.join(args.save_dir, key))
            nodes.to_csv(os.path.join(args.save_dir, key, "nodes.csv"))
            edges.to_csv(os.path.join(args.save_dir, key, "edges.csv"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Which Dataset to Create')
    parser.add_argument("-s", '--source', type=str, default="msq",
                        choices=['msq', 'mps', "mozart piano sonatas", "mozart string quartets", "mix", "quartets", "hsq", "wtc", "piano", "mozart"],
                        help='Select from which dataset to create graph dataset.')
    parser.add_argument("--norm2bar", action="store_true", help="Resize Onset Beat relative to Bar Beat.")
    parser.add_argument("--save-name", default="cad-feature-homo", help="Save Name in the Tonnetz Cadence.")
    parser.add_argument("--cad-type", default="all", choices=["all", "pac", "riac", "hc"], help="Choose type of Cadence to parse in dataset.")
    args = parser.parse_args()

    args.save_name = "cad-{}-{}".format(args.cad_type, args.source) if args.cad_type != "all" else "cad-feature-{}".format(args.source)

    dirname = os.path.abspath(os.path.dirname(__file__))
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    args.par_dir = par(par(dirname))

    import platform
    if platform.system() == "Windows":
        args.save_dir = os.path.join("C:\\Users\\melki\\Desktop\\codes", args.save_name)
    elif platform.system() == "Linux":
        args.save_dir = os.path.join("/home/manos/Desktop/JKU/codes/tonnetzcad/node_classification", args.save_name)
    data = create_data(args)
