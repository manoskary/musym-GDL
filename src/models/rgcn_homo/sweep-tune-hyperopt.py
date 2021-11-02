import os.path

from gaug_test import main as main_gaug
from entity_classify_mp import main as main_mp
from entity_classify import main as main_simple
import argparse
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import wandb

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Gaug_sweep')
    argparser.add_argument('--gpu', type=float, default=0)
    argparser.add_argument('--cpu', type=float, default=2,
                           help="How many cpus per trial to use")
    argparser.add_argument("-d", '--dataset', type=str, default='cora')
    argparser.add_argument('--model', type=str, default="GraphSAGE")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-gpu', type=int, default=1, help="How many gpus per trial to use")
    argparser.add_argument('--shuffle', type=int, default=1)
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--add-self-loop', action='store_true',
                           help="Adding a sef loop to each node.")
    argparser.add_argument("--init-eweights", type=int, default=1,
        help="Initialize learnable graph weights. Use 1 for True and 0 for false")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--num-workers', type=int, default=0,
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
    argparser.add_argument("--unlog", action="store_true", help="Unbinds wandb.")
    args = argparser.parse_args()

    wandb.login()

    config = vars(args)
    if "toy" in args.dataset:
        if args.dataset =="toy":
            dnum = "00-"
        else:
            dnum = args.dataset[-2:]+"-"
        config["project_name"] = "Toy-" + dnum + "BenchMark-Frameworks"
    elif "cad" == args.dataset :
        config["project_name"] = "Cadence Detection Homogeneous"
    else:
        raise ValueError("This dataset is not set for optimization")

    config["num_hidden"] = 16 # tune.grid_search([8, 16, 32])
    config["fan_out"] = [5, 10, 10]
    config["batch_size"] = 256 # tune.grid_search([1024, 2048, 4096])
    # config["alpha"] = tune.grid_search([0.3, 0.5])
    config["beta"] = 0.15
    config["temperature"] = 0.5
    config["data_dir"] = os.path.abspath("./data/")
    config["wandb"] = {
        "project": config["project_name"],
        "group": args.group,
        "job_type": args.job_type
    }

    if args.model == "sage-mp":
        main = main_mp
    elif args.model == "sage":
        main = main_simple
    else:
        main = main_gaug

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
        # num_samples=100,
        resources_per_trial={'gpu': 1},
        config=config,
        # search_alg=HyperOptSearch(),
        # scheduler=AsyncHyperBandScheduler(grace_period=5, reduction_factor=4),
        stop=stopping_criteria
    )

    print("best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))
