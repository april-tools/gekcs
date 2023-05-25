import os
from collections import defaultdict
from typing import Tuple, Optional, Union, List

import numpy as np
import pandas
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tueplots import fonts, figsizes, fontsizes

from kbc.engines import setup_model
from kbc.models import KBCModel
from kbc.gekc_models import TractableKBCModel


def setup_tueplots(nrows: int, ncols: int, rel_width: float = 1.0, hw_ratio: Optional[float] = None):
    """Neurips 2022 bundle."""
    font_config = fonts.neurips2022_tex(family='serif')
    kwargs = dict()
    if hw_ratio is not None:
        kwargs['height_to_width_ratio'] = hw_ratio
    size = figsizes.neurips2022(rel_width=rel_width, nrows=nrows, ncols=ncols, **kwargs)
    fontsize_config = fontsizes.neurips2022()
    rc_params = {**font_config, **size, **fontsize_config}
    plt.rcParams.update(rc_params)
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(
            color=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                   "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
        ),
        "patch.facecolor": "#0173B2"
    })


def format_model_name(name: str) -> str:
    if 'Squared' in name:
        if 'Typed' in name:
            name = f"$\\textsc{'{' + 'd-' + name.split('Squared')[1] + '}'}^2$"
        else:
            name = f"$\\textsc{'{' + name.split('Squared')[1] + '}'}^2$"
    elif 'NNeg' in name:
        name = f"$\\textsc{'{' + name.split('NNeg')[1] + '}'}^+$"
    else:
        name = f"$\\textsc{'{' + name + '}'}$"
    return name


def setup_data_loader(*arrays: np.ndarray, batch_size: int = 1) -> DataLoader:
    tensors = list(map(torch.from_numpy, arrays))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size, shuffle=False, drop_last=False)


def load_model(
        models_path: str,
        dataset: str,
        n_entities: int,
        n_relations: int,
        name: str,
        exp_id: str,
        run_name: str,
        model_filename: str,
        checkpoint_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        filter_inverted_relations: bool = False
) -> Union[KBCModel, TractableKBCModel]:
    if 'Rank' in run_name:
        rank = int(next(filter(lambda x: 'Rank' in x, run_name.split('_')))[4:])
    else:
        rank = int(next(filter(lambda x: x[0] == 'R', run_name.split('_')))[1:])
    conf = {'model': name, 'device': 'cpu', 'rank': rank, 'init_loc': 0.0, 'init_scale': 1.0}
    model = setup_model((n_entities, n_relations * (2 if filter_inverted_relations else 1), n_entities), conf)
    filepath = os.path.join(models_path, dataset, name, exp_id, run_name)
    if checkpoint_id is None and isinstance(model, TractableKBCModel):
        checkpoint_id = 'kbc'
    if checkpoint_id is not None:
        filepath = os.path.join(filepath, checkpoint_id)
    filepath = os.path.join(filepath, model_filename)
    state_dict = torch.load(filepath, map_location='cpu')
    model.load_state_dict(state_dict['weights'] if 'weights' in state_dict else state_dict)
    if filter_inverted_relations:
        model.filter_inverted_relations()
    model.to(device)
    model.eval()
    return model


def eval_log_likelihood(
        model: TractableKBCModel,
        triples: np.ndarray,
        batch_size: int = 1,
        device: Optional[Union[torch.device, str]] = None
) -> Tuple[float, float]:
    dataset = TensorDataset(torch.from_numpy(triples))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    lls = torch.empty(len(dataset), dtype=torch.float32, device=device)
    for i, (batch,) in enumerate(data_loader):
        batch = batch.to(device)
        ll, _ = model.log_likelihood(batch)
        lls[i*batch_size:i*batch_size+batch_size] = ll.squeeze(dim=1)
    assert torch.all(torch.isfinite(lls))
    stdev_ll, mean_ll = torch.std_mean(lls)
    return mean_ll.item(), stdev_ll.item()


def from_bytes_to_gib(bytes: int) -> float:
    return bytes / (1024.0 * 1024.0 * 1024.0)


def build_run_id(hparams: dict) -> str:
    r = f"O{hparams['optimizer']}"
    r = f"{r}_LR{hparams['learning_rate']}"
    r = f"{r}_B{int(hparams['batch_size'])}"
    r = f"{r}_G{hparams['regularizer']}"
    r = f"{r}_R{int(hparams['rank'])}"
    return r


def retrieve_best_hparams(
        df: pd.DataFrame,
        dataset: str,
        model: str,
        exp_id: str = 'PLL',
        metric: str = 'mrr',
        distil_model: bool = False,
        group: Optional[List[str]] = None
) -> dict:
    filter_dict = {'dataset': dataset, 'model': model}
    filter_dict['distil_model'] = float(distil_model)
    if exp_id == 'PLL':
        filter_dict['score_lhs'] = True
        filter_dict['score_rhs'] = True
        filter_dict['score_rel'] = True
        filter_dict['score_ll'] = False
    elif exp_id == 'MLE':
        filter_dict['score_lhs'] = False
        filter_dict['score_rhs'] = False
        filter_dict['score_rel'] = False
        filter_dict['score_ll'] = True
    else:
        raise ValueError("Unknown experiment id")
    df['regularizer'].fillna('None', inplace=True)
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    if group is None:
        row = df.loc[df['valid_' + metric].idxmax()]
        return row.to_dict()
    group_df = df.groupby(by=group)['valid_' + metric]
    df = df.loc[group_df.idxmax()]
    records = df.to_dict('records')
    hparams = defaultdict(list)
    for r in records:
        for k, v in r.items():
            hparams[k].append(v)
    return hparams
