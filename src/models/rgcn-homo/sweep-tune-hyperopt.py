import os.path

from gaug_test import main
import argparse
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Gaug_sweep')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", '--dataset', type=str, default='cora')
    argparser.add_argument('--model', type=str, default="GraphSAGE")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-gpu', type=int, default=1)
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--add-self-loop', action='store_true',
                           help="Adding a sef loop to each node.")
    argparser.add_argument("--init-eweights", type=int, default=0, 
        help="Initialize learnable graph weights. Use 1 for True and 0 for false")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    argparser.add_argument("--aggregator-type", type=str, default="pool",
                        help="Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--fan-out', type=str, default='5, 10')
    argparser.add_argument("--quick-test", action="store_true", help="Finish quickly for testing")
    argparser.add_argument("--alpha", type=float, default=1)
    argparser.add_argument("--beta", type=float, default=0.5)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--group", type=str, required=True)
    argparser.add_argument("--job-type", type=str, default="Node-Classification")
    args = argparser.parse_args()

    wandb.login()

    config = vars(args)
    if "toy" in args.dataset:
        if args.dataset =="toy":
            dnum = "00-"
        else:
            dnum = args.dataset[-2:]+"-"
    else :
        raise ValueError("The Dataset is not Set for Optimization")
    config["project_name"] = "Toy-" + dnum + "BenchMark-Frameworks"
    config["lr"] = 0.1
    config["num_hidden"] = tune.choice([8, 16, 32])
    config["fan_out"] = tune.choice([[5], [5, 10]])
    config["batch_size"] = tune.choice([512, 1024])
    config["dropout"] = 0.5
    config["init_eweights"] = 1
    config["shuffle"] = 0
    config["alpha"] = tune.uniform(0, 1)
    config["beta"] = tune.uniform(0, 1)
    config["temperature"] = tune.uniform(0, 1)
    config["data_dir"] = os.path.abspath("./data/")
    config["wandb"] = {"project": config["project_name"],
        "group": args.group,
        "job_type": args.job_type}

    # AsyncHyperBand enables aggressive early stopping of bad trials.
    scheduler = AsyncHyperBandScheduler(grace_period=5, metric="mean_loss", mode="min", reduction_factor=4)
    search_alg = HyperOptSearch(metric="mean_loss", mode="min")
    stopping_criteria = {"training_iteration": 1 if args.quick_test else 9999}
    # WandbLogger logs experiment configurations and metrics reported via tune.report() to W&B Dashboard
    # callback = WandbLoggerCallback if not config["quick_test"] else None # For testing.
    analysis = tune.run(
        # your main function or script.py
        main,
        name="GNN-Benchmark-Frameworks",
        metric="mean_loss",
        mode="min",
        verbose=1,
        resources_per_trial={'gpu': 0.25},
        # Config is a dict with some tune.grid_Searchs or other tune hyparam opt.
        config=config,
        search_alg=search_alg,
        # Early Stopping Scheduler
        scheduler=scheduler,
        stop= stopping_criteria
        # Conditional Search Spaces.
    )

    print("best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))
