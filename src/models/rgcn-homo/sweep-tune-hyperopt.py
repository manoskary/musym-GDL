from entity_classify_gs import main
import argparse
from ray import tune
from ray.tune.integration.wandb import WandbLogger
import wandb

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", '--dataset', type=str, default='reddit')
    argparser.add_argument("-a", '--gnn', type=str, default='GraphSage')
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    argparser.add_argument("--aggregator-type", type=str, default="pool",
                        help="Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    # argparser.add_argument("--init_eweights", default=True, type=bool, help="Initialize edge weights")
    args = argparser.parse_args()
    
    wandb.login()

    config = vars(args)
    config["wandb"] = {"project" : "Toy-01-Grid-Search"}
    config["lr"] = tune.grid_search([0.1])
    config["num_hidden"] = tune.grid_search([16, 32, 64, 128])
    config["num_layers"] = tune.grid_search([1, 2, 3])
    config["gnn"] = tune.grid_search(["SAGE"])

    analysis = tune.run(
        main,
        loggers=[WandbLogger],  # WandbLogger logs experiment configurations and metrics reported via tune.report() to W&B Dashboard
        resources_per_trial={'gpu': 1},
        config=config)

