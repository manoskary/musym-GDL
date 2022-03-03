import os, sys
import yaml, urllib
import urllib.request


def filter_cadences_from_annotations(annotations, include_type=False):
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
	annotations["cad_pos"] = annotations["timesig"].astype(str).str[0].astype(int) * (annotations["mc"] - 1) + \
							 annotations["onset"].apply(lambda x : x.numerator) / ( annotations["timesig"].astype(str).str[2].astype(float) / annotations["onset"].apply(lambda x : x.denominator))
	annotations["filename"] = annotations.index.get_level_values("filename")
	cad_dict = dict()
	for filename in annotations.filename.unique():
		if include_type:
			cad_dict[filename] = list(zip(annotations.loc[annotations["filename"] == filename, "cad_pos"].to_list(),
								  annotations.loc[annotations["filename"] == filename, "cadence"].to_list()))
		else:
			cad_dict[filename] = annotations.loc[annotations["filename"] == filename, "cad_pos"].to_list()
	return cad_dict


def data_loading_mps(args):
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
	par_directory = os.path.join(args.par_dir, "mozart_piano_sonatas", "utils")
	sys.path.append(par_directory)
	for p in sys.path:
		print(p)
	from feature_matrices import load_tsv

	annotations = filter_cadences_from_annotations(load_tsv(args.tsv_dir, stringtype=False))
	scores = dict()
	for score_name in os.listdir(args.score_dir):
		if score_name.endswith(".musicxml"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(args.score_dir, score_name)       
	return scores, annotations


def data_loading_msq(args):
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
	for score_name in os.listdir(args.score_dir):
		if score_name.endswith(".musicxml"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(args.score_dir, score_name)
			fn = key.replace("k0", "k").replace("-0", ".")
			link = "https://gitlab.com/algomus.fr/algomus-data/-/raw/master/quartets/mozart/"+ fn +"-ref.dez"           
		with urllib.request.urlopen(link) as url:
			annotations[key] = [dv["start"] for dv in yaml.load(url)["labels"] if dv['type'] == 'Cadence'] 
	return scores, annotations


def data_loading_hsq(args):
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
	for score_name in os.listdir(os.path.join(args.score_dir, "kern")):
		if score_name.endswith(".krn"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(args.score_dir, "kern", score_name)
			annotations[key] = []
	return scores, annotations


def data_loading_wtf(args):
	"""Data loading for Bach Well Tempered Clavier Fugues.


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
	for score_name in os.listdir(args.score_dir):
		if score_name.endswith(".krn"):
			key = os.path.splitext(score_name)[0]
			scores[key] = os.path.join(args.score_dir, score_name)
			fn = key.replace("k0", "k").replace("-0", ".")
			link = "https://gitlab.com/algomus.fr/algomus-data/-/raw/master/fugues/bach-wtc-i/"+ fn +"-ref.dez"
		with urllib.request.urlopen(link) as url:
			annotations[key] = [dv["start"] for dv in yaml.load(url)["labels"] if dv['type'] == 'Cadence']
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
	if args.source == "msq" or args.source == "mozart string quartets":
		args.source_name = "mozart_string_quartets"
		args.score_dir = check_source_name(args)
		scores, annotations = data_loading_msq(args)
	elif args.source == "mps" or args.source == "mozart piano sonatas":
		args.source_name = "mozart_piano_sonatas"
		args.score_dir = check_source_name(args)
		if "mozart_piano_sonatas" not in os.listdir(args.par_dir):
			from git import Repo
			Repo.clone_from("https://github.com/DCMLab/mozart_piano_sonatas.git", os.path.join(args.par_dir, "mozart_piano_sonatas"))
		args.tsv_dir = os.path.join(args.par_dir, "mozart_piano_sonatas", "formatted", "-C_cadences.tsv")
		if not os.path.exists(args.tsv_dir): 	
			python_script_dir = os.path.join(args.par_dir, "mozart_piano_sonatas", "mozart_loader.py")
			os.chdir(os.path.join(args.par_dir, "mozart_piano_sonatas"))
			os.system('python '+ python_script_dir + " -C")
		scores, annotations = data_loading_mps(args)
	elif args.source == "haydn_string_quartets":
		scores, annotation = data_loading_hsq(args)
	elif args.source == "bach_fugues":
		scores, annotation = data_loading_wtf(args)
	elif args.source == "all" or args.source == "mix":
		# Calls data_loading individually for each source
		args.source = "mps"
		s2, a2 = data_loading(args)
		args.source = "msq"
		s1, a1 = data_loading(args)
		scores = dict(s1, **s2)
		annotations = dict(a1, **a2)
	else:
		raise ValueError("The Specified Source {} does not exist".format(args.source))

	return scores, annotations