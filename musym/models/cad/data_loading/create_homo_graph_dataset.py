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

NOTE_FEATURES = ["int_vec1", "int_vec2", "int_vec3", "int_vec4", "int_vec5", "int_vec6"] + \
    ["interval"+str(i) for i in range(13)] + list(CHORDS.keys()) + \
    ["is_maj_triad", "is_pmaj_triad", "is_min_triad", 'ped_note',
     'hv_7', "hv_5", "hv_3", "hv_1", "chord_has_2m", "chord_has_2M"]


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
    features = list()
    bass_voice = na["voice"].max() if na["voice" == na["voice"].max()]["pitch"].mean() < na["voice" == na["voice"].min()]["pitch"].mean() else na["voice"].min()
    high_voice = na["voice"].min() if na["voice" == na["voice"].min()]["pitch"].mean() > \
                                      na["voice" == na["voice"].max()]["pitch"].mean() else na["voice"].max()
    for i, n in enumerate(na):
        d = {}
        n_onset = na[na["onset_beat"] == n["onset_beat"]]
        n_dur = na[np.where((na["onset_beat"] < n["onset_beat"]) & (na["onset_beat"] + na["duration_beat"] > n["onset_beat"]))]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist())
        pc_class_recentered = sorted(list(map(lambda x: x - min(pc_class), pc_class)))
        maj_int_vecs = [[0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]
        prev_4beats = na[np.where((na["onset_beat"] < n["onset_beat"]) & (na["onset_beat"] > n["onset_beat"] - 4))][
                          "pitch"] % 12
        prev_8beats = na[np.where((na["onset_beat"] < n["onset_beat"]) & (na["onset_beat"] > n["onset_beat"] - 8))][
                          "pitch"] % 12
        maj_pcs = [[0, 4, 7], [0, 5, 9], [0, 3, 8], [0, 4], [0, 8], [0, 7], [0, 5]]
        scale = [2, 3, 5, 7, 8, 11] if (n["pitch"] + 3) in chord_pitch % 12 else [2, 4, 5, 7, 9, 11]
        v7 = [[0, 1, 2, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 0]]
        next_voice_notes = na[np.where((na["voice"] == n["voice"]) & (na["onset_beat"] > n["onset_beat"]))]
        prev_voice_notes = na[np.where((na["voice"] == n["voice"]) & (na["onset_beat"] < n["onset_beat"]))]
        prev_voice_pitch = prev_voice_notes[prev_voice_notes["onset_beat"] == prev_voice_notes["onset_beat"].max()]["pitch"] if prev_voice_notes.size else 0
        # start Z features
        d["perfect_triad"] = int_vec in maj_int_vecs
        d["perfect_major_triad"] = d["perfect_triad"] and pc_class_recentered in maj_pcs
        d["is_sus4"] = int_vec == [0, 1, 0, 0, 2, 0] or pc_class_recentered == [0, 5]
        d["in_perfect_triad_or_sus4"] = d["perfect_triad"] or d["is_sus4"]
        d["highest_is_3"] = (chord_pitch.max() - chord_pitch.min()) % 12 in [3, 4]
        d["highest_is_1"] = (chord_pitch.max() - chord_pitch.min()) % 12 == 0 and chord_pitch.max() != chord_pitch.min()

        d["bass_compatible_with_I"] = (n["pitch"] + 5) % 12 in prev_4beats and (n["pitch"] + 11) % 12 in prev_4beats if prev_4beats.size else False
        d["bass_compatible_with_I_scale"] = all([(n["pitch"] + ni) % 12 in prev_8beats for ni in scale]) if prev_8beats.size else False
        d["one_comes_from_7"] = 11 in (prev_voice_pitch - chord_pitch.min())%12 and (
                n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["one_comes_from_1"] = 0 in (prev_voice_pitch - chord_pitch.min())%12 and (
                    n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["one_comes_from_2"] = 2 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["three_comes_from_4"] = 5 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min())%12 in [3, 4] if prev_voice_notes.size else False
        d["five_comes_from_5"] = 7 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min()) % 12 == 7 if prev_voice_notes.size else False

        # Make R features
        d["strong_beat"] = (n["ts_beats"] == 4 and n["onset_beat"] % 2 == 0) or (n["onset_beat"] % n['ts_beats'] == 0) # to debug
        d["sustained_note"] = n_dur.size > 0
        if next_voice_notes.size:
            d["rest_highest"] = n["voice"] == high_voice and next_voice_notes["onset_beat"].min() > n["onset_beat"] + n["duration_beat"]
            d["rest_lowest"] = n["voice"] == bass_voice and next_voice_notes["onset_beat"].min() > n["onset_beat"] + n["duration_beat"]
            d["rest_middle"] = n["voice"] != high_voice and n["voice"] != bass_voice and next_voice_notes["onset_beat"].min() > n[
                "onset_beat"] + n["duration_beat"]
            d["voice_ends"] = False
        else:
            d["rest_highest"] = False
            d["rest_lowest"] = False
            d["rest_middle"] = False
            d["voice_ends"] = True

        # start Y features
        d["v7"] = int_vec in v7
        d["v7-3"] = int_vec in v7 and 4 in pc_class_recentered
        d["has_7"] = 10 in pc_class_recentered
        d["has_9"] = 1 in pc_class_recentered or 2 in pc_class_recentered
        d["bass_voice"] = n["voice"] == bass_voice
        if prev_voice_notes.size:
            x = prev_voice_notes[prev_voice_notes["onset_beat"] == prev_voice_notes["onset_beat"].max()]["pitch"]
            d["bass_moves_chromatic"] = n["voice"] == bass_voice and (1 in x - n["pitch"] or -1 in x-n["pitch"])
            d["bass_moves_octave"] = n["voice"] == bass_voice and (12 in x - n["pitch"] or -12 in x - n["pitch"])
            d["bass_compatible_v-i"] = n["voice"] == bass_voice and (7 in x - n["pitch"] or -5 in x - n["pitch"])
            d["bass_compatible_i-v"] = n["voice"] == bass_voice and (-7 in x - n["pitch"] or 5 in x - n["pitch"])
        # X features
            d["bass_moves_2M"] = n["voice"] == bass_voice and (2 in x - n["pitch"] or -2 in x - n["pitch"])
        else:
            d["bass_moves_chromatic"] = d["bass_moves_octave"] = d["bass_compatible_v-i"] = d["bass_compatible_i-v"] = d["bass_moves_2M"] = False
        features.append(tuple(d.values()))
    feat_array = np.array(features, dtype=list(map(lambda x: (x, '<f8'), d.keys())))
    return feat_array, list(d.keys())


def make_general_features(na):
    ca = np.zeros((len(na), len(NOTE_FEATURES)))
    for i, n in enumerate(na):
        n_onset = na[na["onset_beat"] == n["onset_beat"]]
        n_dur = na[np.where((na["onset_beat"] < n["onset_beat"]) & (na["onset_beat"]+na["duration_beat"] > n["onset_beat"]))]
        n_cons = na[na["onset_beat"]+na["duration_beat"] == n["onset_beat"]]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist())
        chords_features = {k: (1 if int_vec == v else 0) for k,v in CHORDS.items()}
        pc_class_recentered = sorted(list(map(lambda x : x - min(pc_class), pc_class)))
        is_maj_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 4, 7], [0, 5, 9], [0, 3, 8]] else 0
        is_min_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 3, 7], [0, 5, 8], [0, 4, 9]] else 0
        is_pmaj_triad = 1 if is_maj_triad and 4 in (chord_pitch - chord_pitch.min())%12 and 7 in (chord_pitch - chord_pitch.min())%12 else 0
        ped_note = 1 if n["duration_beat"] > n["ts_beats"] else 0
        hv_7 = 1 if (chord_pitch.max() - chord_pitch.min())%12 == 10 else 0
        hv_5 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 7 else 0
        hv_3 = 1 if (chord_pitch.max() - chord_pitch.min())%12 in [3, 4] else 0
        hv_1 = 1 if (chord_pitch.max() - chord_pitch.min())%12 == 0 and chord_pitch.max() != chord_pitch.min() else 0
        chord_has_2m = 1 if n["pitch"] - chord_pitch.min() in [1, -1] else 0
        chord_has_2M = 1 if n["pitch"] - chord_pitch.min() in [2, -2] else 0
        intervals = {"interval" + str(i): (1 if i in (n_cons["pitch"] - n["pitch"]) or -i in (
                n_cons["pitch"] - n["pitch"]) else 0) for i in range(13)} if n_cons.size else {"interval" + str(i): 0 for i in range(13)}
        ca[i] = np.array(int_vec + list(intervals.values()) + list(chords_features.values()) +
                         [is_maj_triad, is_pmaj_triad, is_min_triad, ped_note, hv_7, hv_5, hv_3, hv_1, chord_has_2m,
                          chord_has_2M])
    feature_fn = NOTE_FEATURES
    ca = np.array([tuple(x) for x in ca], dtype=[(x, '<f8') for x in feature_fn])
    return ca, feature_fn



def check_annotations(cadences, part, na, key, time_signature, labels, note_array, label_num=1):
    if len(cadences) > 0:
        if isinstance(cadences[0], tuple):
            measures = dict()
            for m in part.iter_all(partitura.score.Measure):
                # Account for repeat unfolding
                if m.number in measures.keys():
                    measures[m.number].append(part.beat_map(m.start.t))
                else:
                    measures[m.number] = [part.beat_map(m.start.t)]
            tmp = [b_map + onset for bar, onset in cadences for b_map in measures[bar]]
            # transform to list of beats
            cadences = tmp

        if (not np.all(na["onset_beat"] >= 0)) and (args.source != "mps"):
            cadences += min(na["onset_beat"])

        if key in MOZART_STRING_QUARTETS:
            args.source = "msq"
        elif key in BACH_FUGUES:
            args.source = "wtc"

        # Corrections of annotations with respect to time signature.
        if time_signature["denominator"][0] == 2 and args.source in ["wtc", "msq"]:
            cadences = list(map(lambda x: x / 2, cadences))
        elif time_signature["denominator"][0] == 8 and args.source in ["wtc", "msq"]:
            if time_signature["nominator"][0] in [6, 9, 12]:
                cadences = list(map(lambda x: 2 * x / 3, cadences))
            else:
                cadences = list(map(lambda x: 2 * x, cadences))

    for cad_onset in cadences:
        labels[np.hstack((np.where(note_array["onset_beat"] == cad_onset)[0], np.where(
            (note_array["onset_beat"] + note_array["duration_beat"] > cad_onset) & (
                    note_array["onset_beat"] < cad_onset))[0]))] = label_num
        # check for false annotation that does not have match
        if np.all((note_array["onset_beat"] == cad_onset) == False):
            raise IndexError(
                "Annotated beat {} does not match with any score beat in score {}.".format(cad_onset,
                                                                                           key))
    return labels


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
            if ind+1 < len(re):
                re[ind]["duration_beat"] = re[ind + 1]["duration_beat"]
        re = np.delete(re, filter_inds)

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(na)
    for i, x in enumerate(na):
        for j in np.where((np.isclose(na["onset_beat"], x["onset_beat"], rtol=1e-04, atol=1e-04) == True) & (na["pitch"] != x["pitch"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(np.isclose(na["onset_beat"], x["onset_beat"] + x["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
            edg_src.append(i)
            edg_dst.append(j)

        if re.size > 0:
            for j in np.where(np.isclose(re["onset_beat"], x["onset_beat"] + x["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)

        for j in \
        np.where((x["onset_beat"] < na["onset_beat"]) & (x["onset_beat"] + x["duration_beat"] > na["onset_beat"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

    if re.size > 0:
        for i, r in enumerate(re):
            for j in np.where(np.isclose(na["onset_beat"], r["onset_beat"] + r["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
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
            if y["denominator"] == 8 and y["nominator"] in [6, 9, 12]:
                return y["nominator"]/3
            else:
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
            # Conform beat for 3/8, 6/8 and 9/8 to be by three.
            part.use_musical_beat({"6/8": 2, "3/8": 3, "9/8": 3, "12/8": 4, "6/4":6})
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
            na = partitura.utils.ensure_notearray(part, include_time_signature=True)
            ca, cad_features = make_cad_features(na)
            fa, other_features = make_general_features(na)
            ba, feature_fn = partitura.musicanalysis.make_note_feats(part, "all")
            ba = np.array([tuple(x) for x in ba], dtype=[(x, '<f8') for x in feature_fn])
            note_array = rfn.merge_arrays([na, ca, ba, fa], flatten=True, usemask=False)
            labels = np.zeros(note_array.shape[0])
            feature_fn = feature_fn + cad_features + other_features
            # Adapted for multiclass
            if isinstance(annotations[key], dict):
                for i, cadences in enumerate(annotations[key].values()):
                    labels = check_annotations(cadences, part, na, key, time_signature, labels, note_array, label_num=i+1)
            else:
                labels = check_annotations(annotations[key], part, na, key, time_signature, labels, note_array)

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
    parser.add_argument("--multiclass", action="store_true", default=False)
    args = parser.parse_args()

    args.save_name = "cadence-{}-{}".format(args.cad_type, args.source) if args.cad_type != "all" else "cad-feature-{}".format(args.source)

    dirname = os.path.abspath(os.path.dirname(__file__))
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    args.par_dir = par(par(dirname))

    import platform
    if platform.system() == "Windows":
        args.save_dir = os.path.join("C:\\Users\\melki\\Desktop\\codes", args.save_name)
    elif platform.system() == "Linux":
        args.save_dir = os.path.join("/home/manos/Desktop/JKU/codes/tonnetzcad/node_classification", args.save_name)
    data = create_data(args)
