from typing import Iterator, List, Optional
from collections import defaultdict

import json
import itertools
import subprocess
import multiprocessing
import argparse


def expand_hparams_grid(hparams_grid: dict, common_hparams_grid: dict) -> List[dict]:
    grid = common_hparams_grid.copy()
    grid.update(hparams_grid)
    return [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]


def build_command_string(expid: str, dataset: str, model: str, args: dict, repid: int) -> str:
    cmd = 'python -m kbc.experiment'
    cmd = '{} {}'.format(cmd, '--experiment_id {} --dataset {} --model {}'.format(expid, dataset, model))
    for field, value in args.items():
        cmd = '{} {}'.format(cmd, '--{} {}'.format(field, value))
    cmd = '{} {}'.format(cmd, '--repetition_id {}'.format(repid))
    return cmd


device_ids_cycle_g: Optional[Iterator[int]] = None

def device_next_id() -> int:
    return next(device_ids_cycle_g)


parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Grid Search Script"
)
parser.add_argument(
    'config', help="Experiments grid search configuration file"
)
parser.add_argument(
    '--dry-run', action='store_true', help="Whether to just print the commands without executing"
)
parser.add_argument(
    '--num-jobs', type=int, default=1, help="The number of processes to run in parallel (on a single device)"
)
parser.add_argument(
    '--multi-devices', type=str, default="",
    help="The list of device IDs to run in parallel, as an alternative to --n-jobs"
)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as fp:
        config = json.load(fp)
    dry_run = args.dry_run
    num_jobs = args.num_jobs
    multi_devices = args.multi_devices.split()
    assert num_jobs > 0
    if not multi_devices:
        assert num_jobs == 1, "Multiple jobs on multiple devices are not supported yet"
    else:
        device_ids_cycle_g = itertools.cycle(multi_devices)

    common_config = config['common']
    common_keys = ['tboard_dir', 'log_parameters',
                   'num_epochs', 'num_valid_step',
                   'num_workers', 'persistent_workers',
                   'patience', 'reciprocal']
    common_hparams_grid = config['hparams_grid']['common']
    init_repetition = config['init_repetition']
    num_repetitions = config['num_repetitions']
    assert init_repetition >= 0

    # Produce the list of commands
    commands = list()
    for dataset in config['datasets']:
        # Get the hyperparameters grid, based on the dataset
        hparams_grid_datasets = config['hparams_grid']['datasets'].keys()
        selected_ds = next(filter(lambda d: dataset in d.split('|'), hparams_grid_datasets))
        hparams_grid = config['hparams_grid']['datasets'][selected_ds]

        for model in hparams_grid:
            # # Get the list of hyperparameters, based on the model
            hparams = expand_hparams_grid(hparams_grid[model], common_hparams_grid)

            # Get the experimentation settings
            esettings_models = config['settings']['models'].keys()
            selected_es = next(filter(lambda m: model in m.split('|'), esettings_models))
            es = config['settings']['models'][selected_es]

            # Run each experiment in the experimental settings, with possibly multiple repetitions
            for expid, expsett in es.items():
                for hps in hparams:
                    args = hps.copy()
                    args.update(expsett)
                    args.update({k: common_config[k] for k in common_keys})    
                    if common_config['model_cache_path'] is not None:
                        args['model_cache_path'] = common_config['model_cache_path']
                    for repid in range(init_repetition, init_repetition + num_repetitions):
                        cmd = build_command_string(expid, dataset, model, args, repid)
                        device = device_next_id() if multi_devices else common_config['device']
                        commands.append((cmd, device))

    # Run the commands, if --dry-run is not specified
    if (num_jobs == 1 and not multi_devices) or dry_run:
        for cmd, device in commands:
            cmd = f'{cmd} --device {device}'
            print(cmd)
            if not dry_run:
                subprocess.run(cmd.split())
    elif multi_devices:
        def run_multi_commands(device_cmds: List[str], stdout: int = subprocess.DEVNULL):
            for cmd in device_cmds:
                subprocess.run(cmd.split(), stdout=stdout)
        num_devices = len(multi_devices)
        commands_per_device = defaultdict(list)
        for cmd, device in commands:
            commands_per_device[device].append(f'{cmd} --device {device}')
        with multiprocessing.Pool(num_devices) as pool:
            for device, device_cmds in commands_per_device.items():
                pool.apply_async(run_multi_commands, args=[device_cmds])
            pool.close()
            pool.join()
    else:
        with multiprocessing.Pool(num_jobs) as pool:
            for cmd, device in commands:
                cmd = f'{cmd} --device {device}'
                pool.apply_async(
                    subprocess.run, args=[cmd.split()],
                    kwds=dict(stdout=subprocess.DEVNULL)
                )
            pool.close()
            pool.join()
