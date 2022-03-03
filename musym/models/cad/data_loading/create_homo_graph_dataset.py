# General Imports
import numpy as np
import os
import pandas as pd

# modified perso imports
import partitura
from basismixer.basisfunctions import list_basis_functions, make_basis

# Local Imports
from musym.models.cad.data_loading import data_loading


def join_struct_arrays(arrays):
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def align_basis(na, ba):
    pitch_norm = na["pitch"] / 127.
    if np.all(np.isclose(pitch_norm, ba["polynomial_pitch_basis.pitch"])):
        return join_struct_arrays([na, ba])
    else:
        print(np.nonzero(np.isclose(pitch_norm, ba["polynomial_pitch_basis.pitch"]))[0].shape)
        raise ValueError


def graph_csv_from_na(na, ra, t_sig, labels, basis_fn=None, norm2bar=True):
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
        "ts": np.array(list(map(lambda x: select_ts(x, t_sig), na))),
        "label": labels
    }
    for k in basis_fn:
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
    re = np.unique(np.array(re, dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4')]))
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
            "ts": np.array(list(map(lambda x: select_ts(x, t_sig), re))),
            "label": np.zeros(len(re))
        }
        for k in basis_fn:
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
        print(key)
        part = partitura.load_musicxml(fn)
        part = partitura.score.merge_parts(part)

        rest_array = np.array(
            [(part.beat_map(r.start.t), part.beat_map(r.duration)) for r in part.iter_all(partitura.score.Rest)],
            dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4')])
        time_signature = np.array(
            [
                (part.beat_map(ts.start.t), filter_ts_end(ts, part), ts.beats, ts.beat_type) for ts in
                part.iter_all(partitura.score.TimeSignature)
            ],
            dtype=[('onset_beat', '<f4'), ('end_beat', '<f4'), ("nominator", "<i4"), ("denominator", "<i4")]
        )
        na = partitura.utils.ensure_notearray(part)
        base_fn = [x for x in list_basis_functions() if
                   x not in ["metrical_basis", "metrical_strength_basis", "articulation_basis"]]
        ba, basis_fn = make_basis(part, base_fn)
        ba = np.array([tuple(x) for x in ba], dtype=[(x, '<f8') for x in basis_fn])
        note_array = align_basis(na, ba)
        labels = np.zeros(note_array.shape[0])
        for cad_onset in annotations[key]:
            labels[np.where(note_array["onset_beat"] == cad_onset)] = 1

        nodes, edges = graph_csv_from_na(note_array, rest_array, time_signature, labels, basis_fn=basis_fn,
                                         norm2bar=args.norm2bar)
        if not os.path.exists(os.path.join(args.save_dir, key)):
            os.makedirs(os.path.join(args.save_dir, key))
        nodes.to_csv(os.path.join(args.save_dir, key, "nodes.csv"))
        edges.to_csv(os.path.join(args.save_dir, key, "edges.csv"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Which Dataset to Create')
    parser.add_argument("-s", '--source', type=str, default="msq",
                        choices=['msq', 'mps', "mozart piano sonatas", "mozart string quartets", "mix", "all"],
                        help='Select from which dataset to create graph dataset.')
    parser.add_argument("--norm2bar", action="store_true", help="Resize Onset Beat relative to Bar Beat.")
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    args.par_dir = par(par(dirname))

    args.save_dir = os.path.join(par(args.par_dir), "tonnetzcad", "node_classification", "cad-basis-homo")

    data = create_data(args)
