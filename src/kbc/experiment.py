import ast
import argparse
from functools import partial
import multiprocessing as mp
import wandb

from kbc.engines import KBCEngine
from kbc.datasets import DATASETS_NAMES
from kbc.models import MODELS_NAMES
from kbc.gekc_models import TSRL_MODELS_NAMES
from kbc.regularizers import REGULARIZERS_NAMES
from kbc.optimizers import OPTIMIZERS_NAMES
from kbc.losses import LOSSES_NAMES


parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Experiments Script"
)
parser.add_argument(
    '--alias', default='',
    help="Alias for the experiment"
)
parser.add_argument(
    '--experiment_id', default='',
    help="Experiment ID which current run belongs to"
)
parser.add_argument(
    '--tboard_dir', default='tboard-runs',
    help="Default Tensorboard output directory"
)
parser.add_argument(
    '--log_parameters', type=ast.literal_eval, default=False,
    help="Whether to log parameters' statistics and gradients on Tensorboard"
)
parser.add_argument(
    '--seed', default=0, type=int,
    help="Seed user for random states"
)
parser.add_argument(
    '--device', default='cuda', type=str,
    help="The device to use"
)
parser.add_argument(
    '--data_path', default='data', type=str,
    help="The data root path"
)
parser.add_argument(
    '--dataset', choices=DATASETS_NAMES,
    help="Dataset in {}".format(DATASETS_NAMES)
)
parser.add_argument(
    '--reciprocal', type=ast.literal_eval, default=False,
    help="Whether to enable reciprocal relation as triples augmentation"
)
parser.add_argument(
    '--model', choices=MODELS_NAMES + TSRL_MODELS_NAMES, required=True,
    help="Model in {}".format(MODELS_NAMES + TSRL_MODELS_NAMES)
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank"
)
parser.add_argument(
    '--rank_r', default=100, type=int,
    help="Factorization rank for the relations only"
)
parser.add_argument(
    '--init_scale', default=1e-3, type=float,
    help="Initial scale for embeddings initialization"
)
parser.add_argument(
    '--regularizer', choices=REGULARIZERS_NAMES, default=REGULARIZERS_NAMES[0],
    help="Regularizer in {}".format(REGULARIZERS_NAMES)
)
parser.add_argument(
    '--lmbda', default=0.0, type=float,
    help="Regularization strength"
)
parser.add_argument(
    '--dropout', default=0.0, type=float,
    help="Dropout rate"
)
parser.add_argument(
    '--optimizer', choices=OPTIMIZERS_NAMES, default=OPTIMIZERS_NAMES[0],
    help="Optimizer in {}".format(OPTIMIZERS_NAMES)
)
parser.add_argument(
    '--num_epochs', default=500, type=int,
    help="Number of epochs"
)
parser.add_argument(
    '--num_valid_step', default=1, type=int,
    help="Number of epochs before doing validation."
)
parser.add_argument(
    '--patience', default=3, type=int,
    help="Stop after a number of validations without improvement on the MRR. Set to negative value to disable"
)
parser.add_argument(
    '--repetition_id', default=0, type=int,
    help="The repetition's identifier"
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size"
)
parser.add_argument(
    '--learning_rate', default=0.1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="Decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="Decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--momentum', default=0.9, type=float,
    help="Momentum parameter to apply when using SGD as optimizer"
)
parser.add_argument(
    '--weight_decay', default=0.0, type=float,
    help="L2 regularisation factor"
)
parser.add_argument(
    '--eval_cache_path', default=None,
    help='The path where to save predictions. Set to None to disable caching'
)
parser.add_argument(
    '--model_cache_path', default='models',
    help="The path where to save models and training statistics. Set to None to disable caching"
)
parser.add_argument(
    '--checkpoint_model', type=ast.literal_eval, default=True,
    help="Whether to checkpoint the model after every improved validation step"
)
parser.add_argument(
    '--checkpoint_opt', type=ast.literal_eval, default=False,
    help="Whether to checkpoint also the optimizer after every improved validation step (useful to restore training)"
)
parser.add_argument(
    '--model_file_id', type=str, default='best_valid',
    help="The filename identifier of the cached models and training statistics"
)
parser.add_argument(
    '--restore_model', type=ast.literal_eval, default=False,
    help="Whether to restore the model's parameters"
)
parser.add_argument(
    '--restore_opt', type=ast.literal_eval, default=False,
    help="Whether to restore the optimizer and other training statistics"
)
parser.add_argument(
    '--distil_model', type=ast.literal_eval, default=False,
    help="Whether to initialize a Squared KBC model using the parameters of the corresponding traditional KBC model"
)
parser.add_argument(
    '--distil_exp_id', type=str, default="PLL",
    help="The traditional KBC model experiment id. It is used together with --distil_model"
)
parser.add_argument(
    '--distil_run', type=str, default="",
    help="The traditional KBC model run name. It is used together with --distil_model"
)
parser.add_argument(
    '--score_rel', type=ast.literal_eval, default=False,
    help="Whether to score over relations"
)
parser.add_argument(
    '--score_lhs', type=ast.literal_eval, default=False,
    help="Whether to score over subjects"
)
parser.add_argument(
    '--score_rhs', type=ast.literal_eval, default=False,
    help="Whether to score over objects"
)
parser.add_argument(
    '--score_ll', type=ast.literal_eval, default=False,
    help="Whether to use the log-likelihoods of triples"
)
parser.add_argument(
    '--w_rel', type=float, default=1.0,
    help="The multiplicative factor of the relations conditional loss"
)
parser.add_argument(
    '--w_lhs', type=float, default=1.0,
    help="The multiplicative factor of the subjects conditional loss"
)
parser.add_argument(
    '--w_rhs', type=float, default=1.0,
    help="The multiplicative factor of the objects conditional loss"
)
parser.add_argument(
    '--w_ll', type=float, default=1.0,
    help="The multiplicative factor of the negative log-likelihood loss"
)
parser.add_argument(
    '--num_workers', type=int, default=2,
    help="The number of workers for the data loader. Set to 0 to use the main thread"
)
parser.add_argument(
    '--persistent_workers', type=ast.literal_eval, default=False,
    help="Whether to keep workers alive after the data loader has been consumed"
)
parser.add_argument(
    '--show_bar', type=ast.literal_eval, default=False,
    help="Whether to show the progress bar for each training epoch"
)
parser.add_argument(
    '--wandb', type=ast.literal_eval, default=False,
    help="Whether to enable W&B"
)
parser.add_argument(
    '--wandb_offline', type=ast.literal_eval, default=False,
    help="Whether to use W&B offline"
)
parser.add_argument(
    '--wandb_project', type=str, default='org/myproject', help="The W&B project name"
)
parser.add_argument(
    '--wandb_dir', type=str, default='runs-wandb', help="Whether to store W&B local logs"
)
parser.add_argument(
    '--wandb_sweep', type=ast.literal_eval, default=False, help="Whether to run W&B hyperparameters sweep"
)
parser.add_argument(
    '--wandb_sweep_id', type=str, default='', help="W&B hyperparameters sweep id"
)
parser.add_argument(
    '--wandb_sweep_trials', type=int, default=10, help="The number of W&B hyperparameters sweep trials"
)


def forked(fn):
    def call(*args, **kwargs):
        ctx = mp.get_context("fork")
        q = ctx.Queue(1)
        is_error = ctx.Value("b", False)
        def target():
            try:
                q.put(fn(*args, **kwargs))
            except BaseException as e:
                is_error.value = True
                q.put(e)
        ctx.Process(target=target).start()
        result = q.get()
        if is_error.value:
            raise result
        return result
    return call


def main_sweep(group: str, conf: dict):
    run = wandb.init(group=group)
    run.name = run.id
    conf = conf.copy()
    conf.update(run.config)
    engine = KBCEngine(conf)
    engine.episode()


@forked
def agent(sweep_id: str, conf: dict):
    group = f"{conf['dataset']}-{conf['model']}"
    if conf['alias']:
        group = f"{group}-{conf['alias']}"
    wandb.agent(
        sweep_id=sweep_id,
        function=partial(
            main_sweep,
            group=group,
            conf=conf
        ),
        count=1
    )


if __name__ == "__main__":
    args = parser.parse_args()
    conf = vars(args)
    if conf['wandb'] and conf['wandb_sweep']:
        assert len(conf['wandb_sweep_id']) > 0
        assert conf['wandb_sweep_trials'] > 0
        for _ in range(conf['wandb_sweep_trials']):
            agent(f"{conf['wandb_project']}/{conf['wandb_sweep_id']}", conf)
    else:
        engine = KBCEngine(conf)
        engine.episode()
