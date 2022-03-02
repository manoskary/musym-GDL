from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from cad_learning import main
import argparse
import os


argparser = argparse.ArgumentParser(description='Cadence Learning GraphSMOTE')
argparser.add_argument('--dataset', type=str, default="mozart")
argparser.add_argument('--gpus-per-trial', type=float, default=0.5)
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=50)
argparser.add_argument('--num-hidden', type=int, default=128)
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--lr', type=float, default=0.001123)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument("--weight-decay", type=float, default=5e-4,
                       help="Weight for L2 loss")
argparser.add_argument("--gamma", type=float, default=0.001248,
                       help="weight of decoder regularization loss.")
argparser.add_argument("--ext-mode", type=str, default=None, choices=["lstm", "attention"])
argparser.add_argument("--fan-out", default=[5, 10])
argparser.add_argument('--shuffle', type=int, default=True)
argparser.add_argument("--tune", type=bool, default=True)
argparser.add_argument("--batch-size", type=int, default=2048)
argparser.add_argument("--num-workers", type=int, default=10)
argparser.add_argument("--num-samples", type=int, default=32)
argparser.add_argument('--data-cpu', action='store_true',
                       help="By default the script puts all node features and labels "
                            "on GPU when using it to save time for data copy. This may "
                            "be undesired if they cannot fit in GPU memory at once. "
                            "This flag disables that.")
argparser.add_argument("--data-dir", type=str, default=os.path.abspath("./data/"))
argparser.add_argument("--preprocess", action="store_true", help="Train and store graph embedding")
argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
argparser.add_argument("--load-model", action="store_true", help="Load pretrained model.")
argparser.add_argument("--eval", action="store_true", help="Preview Results on Validation set.")
args = argparser.parse_args()

config = args if isinstance(args, dict) else vars(args)
gpus_per_trial = args.gpus_per_trial
config["fan_out"] = tune.choice(["5, 10", "5,5", "5,5,5", "5,10,15"])
config["lr"] = tune.uniform(0.0001, 0.01)
config["weight_decay"] = tune.uniform(1e-5, 1e-2)
config["gamma"] = tune.uniform(1e-5, 1e-2)
config["batch_size"] = tune.choice([256, 512, 1024, 2048])
config["dropout"] = tune.choice([0.2, 0.5])
config["dropout"] = tune.choice([0.2, 0.5])
config["ext_mode"] = tune.choice(["lstm", "None"])


reporter = CLIReporter(
        parameter_columns=["num_hidden", "fan_out", "lr", "gamma", "batch_size", "dropout"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

scheduler = ASHAScheduler(
        max_t=config["num_epochs"],
        grace_period=20,
        reduction_factor=2)

analysis = tune.run(
        tune.with_parameters(
            main
            ),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=config["num_samples"],
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_{}".format(config["dataset"]))

print("Best hyperparameters found were: ", analysis.best_config)


