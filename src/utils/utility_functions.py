import sys, os
from dgl import load_graphs
from dgl.data.utils import load_info


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def load_and_save(name, classname=None):
    data_dir = os.path.abspath("./data/")
    if os.path.exists(os.path.join(data_dir, name)):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(data_dir, name, name + '_graph.bin')
        # Load the Homogeneous Graph
        g = load_graphs(graph_path)[0][0]
        info_path = os.path.join(data_dir, name, name + '_info.pkl')
        n_classes = load_info(info_path)['num_classes']
        return g, n_classes
    else:
        if classname:
            dataset = str_to_class(classname)(save_path=data_dir)
        else:
            dataset = str_to_class(name)(save_path=data_dir)

        dataset.save_data()
        # Load the Homogeneous Graph
        g = dataset[0]
        n_classes = dataset.num_classes
        return g, n_classes