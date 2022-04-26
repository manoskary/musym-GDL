import os, sys
import yaml, urllib
import urllib.request
import pandas as pd
import numpy as np
import platform


def retrieve_haydn_cad_annotations(annotation_path, cad_type, multiclass=False):
	df = pd.read_csv(annotation_path, encoding='cp1252')
	sub_table = df[np.where(df["Descriptive Information"] == "Cad Cat.")[0][0] : ].to_numpy()
	new_df_keys = sub_table[0, :].tolist()
	new_df_values = [sub_table[1:, i].tolist() for i in range(len(new_df_keys))]
	new_df = pd.DataFrame(data=dict(zip(new_df_keys, new_df_values))).dropna(how="all", axis=1)
	new_df = new_df.dropna(how="all", axis=0)
	if multiclass:
		pac_cads = retrieve_haydn_cad_annotations(annotation_path, "pac", multiclass=False)
		hc_cads = retrieve_haydn_cad_annotations(annotation_path, "hc", multiclass=False)
		return {"pac" : pac_cads, "hc" : hc_cads}
	if cad_type == "pac":
		idx = np.where(new_df["Cad Cat."] == "PAC")[0].tolist()
	elif cad_type == "hc":
		idx = np.where(new_df["Cad Cat."] == "HC")[0].tolist()
	else:
		idx = list(range(len(new_df["Cad Cat."])))
	bars = list(map(lambda x: int(x), new_df["Bar #"][idx]))
	beats = list(map(lambda x: float(x) - 1, new_df["Pulse #"][idx]))
	return list(zip(bars, beats))


def filter_cadences_from_annotations(annotations, cad_type, multiclass):
	"""
	Create a Trainset from annotations and scores.

	Parameters
	----------
	annotations : dataframe
		Read from tsv file with annotations.
	Returns
	-------
	phrase_dict : dictionary
		Keys are piece names, i.e. K279-1.
		Values are lists of floats with the beat positions where cadences occur.
	"""
	# TODO rethink how to correctly extract note positions from annotations.
	annotations["bar"] = annotations["mn"]
	annotations["onset_norm"] = annotations["onset"].apply(lambda x : x.numerator) / (annotations["onset"].apply(lambda x : x.denominator) / annotations["timesig"].astype(str).str[2].astype(float))
	# annotations["cad_pos"] = annotations["timesig"].astype(str).str[0].astype(int) * (annotations["mc"] - 1) + \
	# 						 annotations["onset"].apply(lambda x : x.numerator - 1) / (  / annotations["timesig"].astype(str).str[2].astype(float))
	annotations["filename"] = annotations.index.get_level_values("filename")
	cad_dict = dict()
	for filename in annotations.filename.unique():
		new_df = annotations.loc[annotations["filename"] == filename]
		if multiclass:
			cad_dict[filename] = {
				"pac": list(zip(new_df.loc[new_df["cadence"] == "PAC", "bar"].to_list(),
										  new_df.loc[new_df["cadence"] == "PAC", "onset_norm"].to_list())),
				"hc": list(zip(new_df.loc[new_df["cadence"] == "HC", "bar"].to_list(),
										  new_df.loc[new_df["cadence"] == "HC", "onset_norm"].to_list()))
			}
		else:
			if cad_type == "pac":
				cad_dict[filename] = list(zip(new_df.loc[new_df["cadence"] == "PAC", "bar"].to_list(),
											  new_df.loc[new_df["cadence"] == "PAC", "onset_norm"].to_list()))
			elif cad_type == "hc":
				cad_dict[filename] = list(zip(new_df.loc[new_df["cadence"] == "HC", "bar"].to_list(),
											  new_df.loc[new_df["cadence"] == "HC", "onset_norm"].to_list()))
			else:
				cad_dict[filename] = list(zip(annotations.loc[annotations["filename"] == filename, "bar"].to_list(),
											  annotations.loc[annotations["filename"] == filename, "onset_norm"].to_list()))
	return cad_dict


def data_loading_mps(score_dir, cad_type, multiclass):
	"""Data Loading for Mozart Piano Sonatas.

	Parameters
	----------
	args : argparse Object

	Returns
	-------
	scores : dict
		A dictionary with keys of score names and values of score paths.
	annotations : dict
		A dictionary with keys of score names and values Cadence positions.

	"""
	par_directory = os.path.join(score_dir, "utils")
	tsv_dir = os.path.join(score_dir, "formatted", "-Ce_cadences.tsv")
	sys.path.append(par_directory)
	for p in sys.path:
		print(p)
	from feature_matrices import load_tsv
	annotations = filter_cadences_from_annotations(load_tsv(tsv_dir, stringtype=False), cad_type=cad_type, multiclass=multiclass)
	score_dir = os.path.join(score_dir, "musicxml_scores")
	scores = dict()
	for score_name in os.listdir(score_dir):
		if score_name.endswith(".musicxml"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(score_dir, score_name)
	# TODO Fix the problematic pieces.
	# Remove problematic Pieces
	problematic_pieces = ["K331-2"]
	for key in problematic_pieces:
		scores.pop(key, None)
		annotations.pop(key, None)
	return scores, annotations


def data_loading_msq(score_dir, cad_type, multiclass=False):
	"""Data Loading for Mozart String Quartets.


	Parameters
	----------
	args : argparse Object

	Returns
	-------
	scores : dict
		A dictionary with keys of score names and values of score paths.
	annotations : dict
		A dictionary with keys of score names and values Cadence positions.
	"""
	scores = dict()
	annotations = dict()
	for score_name in os.listdir(score_dir):
		if score_name.endswith(".krn"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(score_dir, score_name)
			fn = key.replace("k0", "k").replace("-0", ".")
			annotation_dir = os.path.join(score_dir, "annotations", fn+"-ref.dez")
			if multiclass:
				cond_pac = lambda x: "PAC" in x["tag"] if "tag" in x.keys() else False
				cond_hc = lambda x: "HC" in x["tag"] if "tag" in x.keys() else False
				with open(annotation_dir, "r") as f:
					l = yaml.safe_load(f)["labels"]
					annotations[key] = {
						"pac": [dv["start"] for dv in l if
								 (dv['type'] == 'Cadence') and cond_pac(dv)],
						"hc" : [dv["start"] for dv in l if
								 (dv['type'] == 'Cadence') and cond_hc(dv)],
					}
			else:
				if cad_type == "pac":
					cond = lambda x: "PAC" in x["tag"] if "tag" in x.keys() else False
				elif cad_type == "HC":
					cond = lambda x: "HC" in x["tag"]  if "tag" in x.keys() else False
				else:
					cond = lambda x: True
				with open(annotation_dir, "r") as f:
					annotations[key] = [dv["start"] for dv in yaml.safe_load(f)["labels"] if
										(dv['type'] == 'Cadence') and cond(dv)]
	return scores, annotations


def data_loading_hsq(score_dir, cad_type, multiclass=False):
	"""Data Loading for Haydn String Quartets.


	Parameters
	----------
	args : argparse Object

	Returns
	-------
	scores : dict
		A dictionary with keys of score names and values of score paths.
	annotations : dict
		A dictionary with keys of score names and values Cadence positions.
	"""
	scores = dict()
	annotations = dict()
	annotation_dir = os.path.join(score_dir, "annotations", "cadences_keys")
	for score_name in os.listdir(os.path.join(score_dir, "kern")):
		if score_name.endswith(".krn"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(score_dir, "kern", score_name)
			annotations[key] = retrieve_haydn_cad_annotations(os.path.join(annotation_dir, key + ".csv"), cad_type, multiclass)
	return scores, annotations


def data_loading_wtc(score_dir, cad_type, multiclass=False):
	"""Data loading for Bach Well Tempered Clavier Fugues.

	Parameters
	----------
	score_dir : The score Directory.

	Returns
	-------
	scores : dict
		A dictionary with keys of score names and values of score paths.
	annotations : dict
		A dictionary with keys of score names and values Cadence positions.
	"""
	scores = dict()
	annotations = dict()
	for score_name in os.listdir(score_dir):
		if score_name.endswith(".krn"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(score_dir, score_name)
			fugue_num = key[-2:]
			fn = "{}-bwv{}-ref.dez".format(fugue_num, 845 + int(fugue_num))
			annotation_dir = os.path.join(score_dir, "annotations", fn)
			if multiclass:
				cond_pac = lambda x: "PAC" in x["tag"] if "tag" in x.keys() else False
				cond_riac = lambda x: "rIAC" in x["tag"] if "tag" in x.keys() else False
				with open(annotation_dir, "r") as f:
					l = yaml.safe_load(f)["labels"]
					annotations[key] = {
						"pac": [dv["start"] for dv in l if
								 (dv['type'] == 'Cadence') and cond_pac(dv)],
						"riac": [dv["start"] for dv in l if
								(dv['type'] == 'Cadence') and cond_riac(dv)]
					}
			else:
				if cad_type == "pac":
					cond = lambda x: "PAC" in x["tag"] if "tag" in x.keys() else False
				elif cad_type == "riac":
					cond = lambda x: "rIAC" in x["tag"] or "PAC" in x["tag"] if "tag" in x.keys() else False
				else:
					cond = lambda x: True
				with open(annotation_dir, "r") as f:
					annotations[key] = [dv["start"] for dv in yaml.safe_load(f)["labels"] if (dv['type'] == 'Cadence') and cond(dv)]
	return scores, annotations


def check_source_name(args):
	if hasattr(args, "source_name"):
		score_dir = os.path.join(args.par_dir, "samples", "mymusicxml_scores", args.source_name)
		if not os.path.exists(score_dir):
			from git import Repo
			Repo.clone_from("https://github.com/melkisedeath/mymusicxml_scores", os.path.join(args.par_dir, "samples", "mymusicxml_scores"))
		return score_dir
	else :
		raise AttributeError("A source Name is not provided")


def data_loading(args):
	if platform.system() == "Linux":
		base_dir = "/home/manos/Desktop/JKU/"
	elif platform.system() == "Windows":
		base_dir = "C:\\Users\\melki\\Desktop"
	if args.source == "msq" or args.source == "mozart string quartets":
		score_dir = os.path.join(base_dir, "data", "mozart_string_quartets", "kern")
		scores, annotations = data_loading_msq(score_dir=score_dir, cad_type=args.cad_type, multiclass=args.multiclass)
	elif args.source == "mps" or args.source == "mozart piano sonatas":
		score_dir = os.path.join(base_dir, "codes", "mozart_piano_sonatas")
		tsv_dir = os.path.join(score_dir, "formatted", "-Ce_cadences.tsv")
		if not os.path.exists(tsv_dir):
			python_script_dir = os.path.join(score_dir, "mozart_loader.py")
			os.chdir(os.path.normpath(score_dir))
			os.system('python '+ python_script_dir + " -C")
		scores, annotations = data_loading_mps(score_dir, cad_type=args.cad_type, multiclass=args.multiclass)
	elif args.source == "hsq":
		score_dir = os.path.join(base_dir, "data", "haydn_string_quartets")
		scores, annotations = data_loading_hsq(score_dir=score_dir, cad_type=args.cad_type, multiclass=args.multiclass)
	elif args.source == "wtc":
		score_dir = os.path.join(base_dir, "data", "wtc-fugues")
		scores, annotations = data_loading_wtc(score_dir=score_dir, cad_type=args.cad_type, multiclass=args.multiclass)
	elif args.source == "mozart":
		args.source = "mps"
		s2, a2 = data_loading(args)
		args.source = "msq"
		s1, a1 = data_loading(args)
		scores = dict(s1, **s2)
		annotations = dict(a1, **a2)
	elif args.source == "quartets":
		args.source = "msq"
		s1, a1 = data_loading(args)
		args.source = "hsq"
		s2, a2 = data_loading(args)
		scores = dict(s1, **s2)
		annotations = dict(a1, **a2)
	elif args.source == "piano":
		args.source = "mps"
		s1, a1 = data_loading(args)
		args.source = "wtc"
		s2, a2 = data_loading(args)
		scores = dict(s1, **s2)
		annotations = dict(a1, **a2)
	elif args.source == "mix":
		args.source = "quartets"
		s1, a1 = data_loading(args)
		args.source = "piano"
		s2, a2 = data_loading(args)
		scores = dict(s1, **s2)
		annotations = dict(a1, **a2)
	else:
		raise ValueError("The Specified Source {} does not exist".format(args.source))

	return scores, annotations

if __name__ == "__main__":
	scores, annotations = data_loading_hsq("/home/manos/Desktop/JKU/data/haydn_string_quartets/", cad_type="pac")
	print(scores.keys())
	print(annotations)