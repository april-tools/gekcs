import os
import time
from typing import Optional, Tuple

import torch
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from kbc.datasets import DATASETS_NAMES, Dataset, TypedDataset
from kbc.regularizers import REGULARIZERS_NAMES, setup_regularizer
from kbc.losses import setup_loss
from kbc.optimizers import OPTIMIZERS_NAMES, setup_optimizer
from kbc.utils import average_metrics_entity, set_seed

from kbc.models import MODELS_NAMES, KBCModel, CP, RESCAL, TuckER, ComplEx
from kbc.gekc_models import TSRL_MODELS_NAMES, TypedSquaredKBCModel
from kbc.gekc_models import TractableKBCModel, NNegCP, NNegRESCAL, NNegTuckER, NNegComplEx
from kbc.gekc_models import SquaredCP, SquaredComplEx
from kbc.gekc_models import TypedSquaredCP, TypedSquaredComplEx


def setup_model(size: Tuple[int, int, int], conf: dict) -> KBCModel:
    if conf['model'] == 'ComplEx':
        model = ComplEx(size, conf['rank'], conf['init_scale'])
    elif conf['model'] == 'TuckER':
        model = TuckER(size, conf['rank'], conf['rank_r'], conf['init_scale'], conf['dropout'])
    elif conf['model'] == 'RESCAL':
        model = RESCAL(size, conf['rank'], conf['init_scale'])
    elif conf['model'] == 'CP':
        model = CP(size, conf['rank'], conf['init_scale'])
    elif conf['model'] == 'NNegCP':
        model = NNegCP(size, conf['rank'], init_scale=conf['init_scale'])
    elif conf['model'] == 'NNegRESCAL':
        model = NNegRESCAL(size, conf['rank'], init_scale=conf['init_scale'])
    elif conf['model'] == 'NNegTuckER':
        model = NNegTuckER(size, conf['rank'], conf['rank_r'], init_scale=conf['init_scale'])
    elif conf['model'] == 'NNegComplEx':
        model = NNegComplEx(size, conf['rank'], init_scale=conf['init_scale'])
    elif conf['model'] == 'SquaredCP':
        model = SquaredCP(size, conf['rank'], init_scale=conf['init_scale'])
    elif conf['model'] == 'SquaredComplEx':
        model = SquaredComplEx(size, conf['rank'], init_scale=conf['init_scale'])
    elif conf['model'] == 'TypedSquaredCP':
        model = TypedSquaredCP(size, conf['rank'], init_scale=conf['init_scale'])
    elif conf['model'] == 'TypedSquaredComplEx':
        model = TypedSquaredComplEx(size, conf['rank'], init_scale=conf['init_scale'])
    else:
        raise ValueError("Unknown model called {}".format(conf['model']))
    return model


def setup_exp_alias(conf: dict) -> str:
    suffix = '{}_{}'.format(conf['dataset'], conf['model'])
    suffix = '{}_O{}_LR{}_B{}_G{}'.format(
        suffix, conf['optimizer'], conf['learning_rate'], conf['batch_size'], conf['regularizer']
    )
    suffix = '{}_R{}'.format(suffix, conf['rank'])
    if 'TuckER' in conf['model']:
        suffix = '{}_RP{}'.format(suffix, conf['rank_r'])
    if conf['regularizer'] != 'None':
        suffix = '{}_L{}'.format(suffix, conf['lmbda'])
    if conf['model'] == 'TuckER':
        suffix = '{}_DP{}'.format(suffix, conf['dropout'])
    if conf['alias'] != '':
        suffix = '{}/{}'.format(conf['alias'], suffix)
    return suffix


def setup_short_exp_alias(conf: dict) -> str:
    suffix = '{}-{}-{}'.format(conf['dataset'], conf['model'], conf['experiment_id'])
    if conf['alias'] != '':
        suffix = '{}-{}'.format(suffix, conf['alias'])
    return suffix


def setup_cache_path(root: str, run_dir: str) -> Optional[str]:
    if root is None or len(root) == 0:
        return None
    cache_path = os.path.join(root, run_dir)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    return cache_path


def setup_hparams(conf: dict):
    keys = ['dataset', 'model', 'reciprocal', 'distil_model']
    keys.extend(['optimizer', 'learning_rate', 'batch_size', 'regularizer'])
    keys.extend(['rank', 'rank_r'])
    keys.extend(['dropout', 'lmbda'])
    keys.extend(['score_rhs', 'score_rel', 'score_lhs', 'score_ll'])
    keys.extend(['repetition_id'])
    hparam_domain_discrete = {
        'dataset': DATASETS_NAMES,
        'model': MODELS_NAMES + TSRL_MODELS_NAMES,
        'optimizer': OPTIMIZERS_NAMES,
        'regularizer': REGULARIZERS_NAMES,
        'score_rhs': [False, True],
        'score_rel': [False, True],
        'score_lhs': [False, True],
        'score_ll': [False, True],
        'reciprocal': [False, True],
        'distil_model': [False, True]
    }
    hparams = {k: conf[k] for k in keys}
    return hparams, hparam_domain_discrete


class KBCEngine:
    def __init__(self, conf: dict):
        # Check configuration dictionary
        self.check_config(conf)

        # Set the seed as a function of the given seed and the repetition id
        self.seed = conf['seed']
        self.repetition_id = conf['repetition_id']
        self.seed = self.seed + 123 * self.repetition_id
        set_seed(self.seed)
        self.device = torch.device(conf['device'])

        # Setup alias, run name and the hyperparameters
        self.alias = setup_exp_alias(conf)
        self.short_alias = setup_short_exp_alias(conf)
        self.run_name = '_'.join(self.alias.split('_')[2:])
        self.sub_run_dir = os.path.join(conf['dataset'], conf['model'], conf['experiment_id'], self.run_name)
        self.hparams, self.hparam_domain_discrete = setup_hparams(conf)

        # Setup cache directories and log settings, if specified
        self.eval_cache_path = setup_cache_path(conf['eval_cache_path'], self.sub_run_dir)
        self.model_cache_path = setup_cache_path(conf['model_cache_path'], self.sub_run_dir)
        self.model_file_id = conf['model_file_id']
        self.log_parameters = conf['log_parameters']
        self.checkpoint_model = conf['checkpoint_model']
        self.checkpoint_opt = conf['checkpoint_opt']

        # Set up the dataset
        self.typed_kg = 'typed' in conf['model'].lower()
        dataset_cls = TypedDataset if self.typed_kg else Dataset
        self.dataset = dataset_cls(
            conf['dataset'], conf['device'], conf['reciprocal'],
            data_path=conf['data_path'], seed=self.seed
        )

        # Set up the model
        self.model = setup_model(self.dataset.get_shape(), conf).to(self.device)

        # Setup loss, batch size, regularizer and optimizer
        self.loss_name = 'NLL' if conf['model'] in TSRL_MODELS_NAMES else 'LCWA+ce'
        self.loss = setup_loss(self.loss_name)
        self.batch_size = conf['batch_size']
        self.regularizer = setup_regularizer(conf['regularizer'], conf['lmbda'])
        self.optimizer = setup_optimizer(
            self.model.parameters(), conf['optimizer'], conf['learning_rate'],
            conf['decay1'], conf['decay2'], conf['momentum'], conf['weight_decay']
        )

        # Restore training, if specified
        self.init_epoch = 1
        self.init_best_metrics = {
            'valid_mrr': -0.0, 'test_mrr': -0.0, 'valid_mrr_epoch': -1,
            'test_hits@1': -0.0, 'test_hits@3': -0.0, 'test_hits@10': -0.0,
            'valid_avg_ll': -np.inf, 'test_avg_ll': -np.inf, 'valid_avg_ll_epoch': -1
        }
        self.trial_id = str(round(time.time(), 2))
        self.restored = False
        self.training_time = 0.0
        if conf['restore_model']:
            state_dict = torch.load(os.path.join(self.model_cache_path, f'{self.model_file_id}.pt'))
            self.model.load_state_dict(state_dict['weights'])
            self.restored = True
            print(f"Model's weights restored")
            if conf['restore_opt']:
                self.optimizer.load_state_dict(state_dict['opt'])
                self.init_epoch = state_dict['epoch'] + 1
                self.init_best_metrics = state_dict['best_metrics']
                self.trial_id = state_dict['trial_id']
                print(f"Training restored at epoch {state_dict['epoch']}")
        elif conf['distil_model']:
            distil_model_name = conf['model'].split('Squared')[1]
            distil_model_cache_path = os.path.join(
                conf['model_cache_path'], conf['dataset'], distil_model_name,
                conf['distil_exp_id'], conf['distil_run'], f"{conf['model_file_id']}.pt"
            )
            distil_state_dict = torch.load(distil_model_cache_path, map_location=self.device)
            weights_state_dict = distil_state_dict['weights']
            if distil_model_name in ['CP', 'RESCAL']:
                ent_embs, rel_embs = weights_state_dict['entity.weight'], weights_state_dict['relation.weight']
                self.model.ent_embeddings.weight.data.copy_(ent_embs)
                self.model.rel_embeddings.weight.data.copy_(rel_embs)
            elif distil_model_name == 'ComplEx':
                ent_embs = weights_state_dict['embeddings.0.weight']
                rel_embs = weights_state_dict['embeddings.1.weight']
                self.model.ent_embeddings.weight.data.copy_(ent_embs)
                self.model.rel_embeddings.weight.data.copy_(rel_embs)
            else:
                assert False, "Something is wrong"
            self.restored = True
            print(f"Restoring weights from {distil_model_name} to {conf['model']}")
            if conf['restore_opt']:
                self.optimizer.load_state_dict(distil_state_dict['opt'])
                print(f"Optimizer state from pre-trained model restored")

        if isinstance(self.model, TypedSquaredKBCModel):
            self.model.set_type_constraint_info(
                self.dataset.pred_to_domains,
                self.dataset.dom_to_types,
                self.dataset.dom_to_preds,
                self.dataset.type_entity_ids,
                device=self.device
            )

        # Set some useful attributes
        self.num_epochs = conf['num_epochs']
        self.num_valid_step = conf['num_valid_step']
        self.patience = conf['patience']
        self.score_rhs = conf['score_rhs']
        self.score_rel = conf['score_rel']
        self.score_lhs = conf['score_lhs']
        self.score_ll = conf['score_ll']
        self.w_rel = conf['w_rel']
        self.w_lhs = conf['w_lhs']
        self.w_rhs = conf['w_rhs']
        self.w_ll = conf['w_ll']
        self.num_workers = conf['num_workers']
        self.persistent_workers = conf['persistent_workers']
        self.show_bar = conf['show_bar']
        self.wandb_enabled = conf['wandb']
        self.wandb_offline = conf['wandb_offline']
        self.wandb_sweep = conf['wandb_sweep']

        # Setup Tensorboard logger
        if conf['tboard_dir']:
            self.writer = SummaryWriter(os.path.join(conf['tboard_dir'], self.sub_run_dir))
        else:
            self.writer = None

        # Setup W&B
        if self.wandb_enabled and not self.wandb_sweep:
            wandb_dir = os.path.join(conf['wandb_dir'], self.sub_run_dir)
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(
                project='tractable-srl',
                name=self.short_alias,
                config=conf,
                dir=wandb_dir,
                mode='offline' if self.wandb_offline else 'online'
            )

    @staticmethod
    def check_config(conf: dict):
        if (conf['restore_model'] or conf['distil_model']) and conf['model_cache_path'] is None:
            raise ValueError("In order to restore a model you have to specify the models cache path")
        if conf['restore_opt'] and not (conf['restore_model'] or conf['distil_model']):
            raise ValueError("In order to restore the optimizer you have to restore the model's weights too")
        if conf['restore_model'] and conf['distil_model']:
            raise ValueError("Only one between --restore_model and --distil_model can be True")
        if conf['distil_model'] and 'Squared' not in conf['model']:
            raise ValueError("--distil_model can only be used with Squared KBC models")
        if conf['distil_model'] and (not conf['distil_exp_id'] or not conf['distil_run']):
            raise ValueError("You have to specify --distil_exp_id and --distil_run when using --distil_model")
        if conf['model'] in TSRL_MODELS_NAMES:
            if not conf['score_rhs'] and not conf['score_rel'] and not conf['score_lhs'] and not conf['score_ll']:
                raise ValueError("At least one between '--score_rhs', '--score_rel', '--score_lhs' and '--score_ll'"
                                 " must be specified for TSRL models")
            if 'NNeg' in conf['model'] and conf['regularizer'] != 'None':
                raise ValueError("No regularizer is supported for TSRL models obtained by monotonic restriction")
        else:
            if not conf['score_rhs'] and not conf['score_rel'] and not conf['score_lhs']:
                raise ValueError("At least one between '--score_rhs', '--score_rel' and '--score_lhs'"
                                 " must be specified for traditional KGE models")
            if conf['score_ll']:
                raise ValueError("The '--score_ll' flag can only be specified for monotonic circuits")

    def log_scalar(self, name: str, value: float, epoch: int):
        if self.writer is not None:
            self.writer.add_scalar(name, value, epoch)
        if self.wandb_enabled:
            wandb.log({name: value}, step=epoch)

    def log_summary(self, summary: dict):
        if self.writer is not None:
            self.writer.add_hparams(
                self.hparams, summary,
                hparam_domain_discrete=self.hparam_domain_discrete,
                run_name=self.trial_id
            )
        if self.wandb_enabled:
            wandb.run.summary.update(summary)

    def log_shutdown(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb_enabled:
            wandb.finish(quiet=True)

    def checkpoint(self, epoch: int, best_metrics: dict, checkpoint_id: Optional[str] = None):
        print(f'Save the model and training statistics at epoch {epoch}')
        state_dict = {
            'weights': self.model.state_dict(),
            'epoch': epoch,
            'best_metrics': best_metrics,
            'trial_id': self.trial_id
        }
        if self.checkpoint_opt:
            state_dict['opt'] = self.optimizer.state_dict()
        if checkpoint_id is None:
            path = self.model_cache_path
        else:
            path = os.path.join(self.model_cache_path, checkpoint_id)
            os.makedirs(path, exist_ok=True)
        torch.save(state_dict, os.path.join(path, f'{self.model_file_id}.pt'))

    def __train_step(self, batch: torch.Tensor) -> Tuple[float, float, float]:
        # Forward the next batch to the model
        if isinstance(self.model, TractableKBCModel):
            predictions = self.model.forward(
                batch, score_rhs=self.score_rhs, score_rel=self.score_rel, score_lhs=self.score_lhs,
                score_ll=self.score_ll
            )
        else:
            predictions, factors = self.model.forward(
                batch, score_rhs=self.score_rhs, score_rel=self.score_rel, score_lhs=self.score_lhs
            )

        # Extract the predictions and initialize the loss
        if isinstance(self.model, TractableKBCModel):
            log_probs, (rhs_scores, rel_scores, lhs_scores) = predictions
            if self.score_ll:
                loss_obj = self.w_ll * self.loss(log_probs, None)
            else:
                loss_obj = 0.0
        else:
            rhs_scores, rel_scores, lhs_scores = predictions
            loss_obj = 0.0

        # Compute the main component of the loss
        if rhs_scores is not None:
            loss_obj = loss_obj + self.w_rhs * self.loss(rhs_scores, batch[:, 2])
        if rel_scores is not None:
            loss_obj = loss_obj + self.w_rel * self.loss(rel_scores, batch[:, 1])
        if lhs_scores is not None:
            loss_obj = loss_obj + self.w_lhs * self.loss(lhs_scores, batch[:, 0])

        # Sum the regularizer's loss, if necessary
        if self.regularizer is not None:
            loss_reg = self.regularizer.penalty(factors)
            loss = loss_obj + loss_reg
        else:
            loss_reg = 0.0
            loss = loss_obj

        # Back-propagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, loss_obj, loss_reg

    def __valid_step(self, epoch: int, best_metrics: dict) -> Tuple[dict, bool, bool]:
        # Compute metrics for all the splits
        res_all, res_all_detailed = [], []
        res = dict()
        diverged = False
        for split in self.dataset.splits:
            # Subsample 1024 triples for computing approximated training MRR
            n_queries = -1 if split != 'train' else 1024
            res_s, div = self.dataset.eval(
                self.model, split, n_queries,
                eval_ll=isinstance(self.model, TractableKBCModel)
            )
            diverged = diverged or div
            res[split] = average_metrics_entity(res_s[0], res_s[1])
            if res_s[2] is not None:
                res[split].update(res_s[2])
            res_all_detailed.append(res_s[3])
        res_detailed = dict(zip(self.dataset.splits, res_all_detailed))

        # Log some metrics for each split
        print(f"Epoch: {epoch}")
        for split in self.dataset.splits:
            format_res_split = res[split].copy()
            format_res_split['mrr'] = round(format_res_split['mrr'], 4)
            format_res_split['hits@[1,3,10]'] = list(map(lambda x: round(x, 4), format_res_split['hits@[1,3,10]']))
            if 'avg_ll' in res[split]:
                format_res_split['avg_ll'] = round(format_res_split['avg_ll'], 4)
                self.log_scalar(f'AvgLL/{split}', res[split]['avg_ll'], epoch)
            self.log_scalar(f'MRR/{split}', res[split]['mrr'], epoch)
            self.log_scalar(f'Hits@1/{split}', res[split]['hits@[1,3,10]'][0], epoch)
            print(f"{split.upper()}: {format_res_split}")

        if 'avg_ll' in res['valid']:
            # Update the best valid AvGLL and save model's checkpoint, if a better model is found
            if res['valid']['avg_ll'] > best_metrics['valid_avg_ll'] + 2e-2:
                # The new validation/test metrics after an improvement in MRR over the validation set
                best_metrics = best_metrics.copy()
                best_metrics['valid_avg_ll'] = res['valid']['avg_ll']
                best_metrics['valid_avg_ll_epoch'] = epoch
                best_metrics['test_avg_ll'] = res['test']['avg_ll']
                # Checkpoint model and predictions
                if self.model_cache_path is not None and self.checkpoint_model:
                    self.checkpoint(epoch, best_metrics, checkpoint_id='gen')
                self.log_scalar(f'BestAvgLL/valid', best_metrics['valid_avg_ll'], epoch)

        # Update the best valid MRR and save model's checkpoint and predictions cache, if a better model is found
        if res['valid']['mrr'] > best_metrics['valid_mrr'] + 2e-4:
            # The new validation/test metrics after an improvement in MRR over the validation set
            best_metrics = best_metrics.copy()
            best_metrics['valid_mrr'] = res['valid']['mrr']
            best_metrics['valid_mrr_epoch'] = epoch
            best_metrics['test_mrr'] = res['test']['mrr']
            best_metrics['test_hits@1'] = res['test']['hits@[1,3,10]'][0]
            best_metrics['test_hits@3'] = res['test']['hits@[1,3,10]'][1]
            best_metrics['test_hits@10'] = res['test']['hits@[1,3,10]'][2]
            # Checkpoint model and predictions
            if self.model_cache_path is not None and self.checkpoint_model:
                checkpoint_id = 'kbc' if 'avg_ll' in res['valid'] else None
                self.checkpoint(epoch, best_metrics, checkpoint_id=checkpoint_id)
            if self.eval_cache_path is not None:
                for s in self.dataset.splits:
                    for m in ['lhs', 'rhs']:
                        torch.save(res_detailed[s][m], os.path.join(self.eval_cache_path, '{}_{}.pt'.format(s, m)))
            self.log_scalar(f'BestMRR/valid', best_metrics['valid_mrr'], epoch)

        # Stop training, if no improvement occurred on the validation metrics after a certain amount of epoch
        should_stop = False
        if self.patience >= 0:
            if (epoch - self.init_epoch + 1 - best_metrics['valid_mrr_epoch']) // self.num_valid_step > self.patience:
                should_stop = True
            if 'avg_ll' in res['valid']:
                avg_ll_should_stop = (epoch - self.init_epoch + 1 - best_metrics['valid_avg_ll_epoch']) \
                                     // self.num_valid_step > self.patience
                should_stop = should_stop and avg_ll_should_stop

        return best_metrics, diverged, should_stop

    def episode(self):
        # Flag indicating if the training diverged
        diverged = False

        # Flag indicating if the training early stopped
        early_stopped = False

        # The best metrics dictionary for validation/test splits
        best_metrics = self.init_best_metrics

        # The training data loader (used to get batches of data)
        train_loader = self.dataset.get_train_loader(self.batch_size, self.num_workers, self.persistent_workers)
        n_batches = len(train_loader)

        # If the model has been restored, run a validation step first
        if self.restored:
            print("Evaluating restored model ...")
            self.model.eval()
            best_metrics, _, _ = self.__valid_step(self.init_epoch - 1, best_metrics)

        # Iterate through all batches inside an epoch
        for epoch in range(self.init_epoch, self.num_epochs + 1):
            # Train
            self.model.train()
            loss = loss_obj = loss_reg = 0.0
            loader = tqdm(train_loader, total=n_batches, leave=False, disable=not self.show_bar)
            start_time = time.perf_counter()
            for (batch,) in loader:
                batch = batch.to(self.device)
                ls, lsobj, lreg = self.__train_step(batch)
                loss += ls
                loss_obj += lsobj
                loss_reg += lreg
            end_time = time.perf_counter()
            self.training_time += (end_time - start_time) / 60.0

            # Log training losses
            self.log_scalar('Loss/train', loss / n_batches, epoch)
            if self.regularizer is not None:
                self.log_scalar('LossObj/train', loss_obj / n_batches, epoch)
                self.log_scalar('LossReg/train', loss_reg / n_batches, epoch)
            print("Average training loss: {:.3f}".format(loss / n_batches))

            # Log parameters' values and their gradients
            if self.log_parameters and self.writer is not None:
                for name, param in self.model.named_parameters():
                    p = param.data.cpu().numpy()
                    p_box = np.quantile(p, q=[0.0, 0.25, 0.5, 0.75, 1.0])
                    self.writer.add_scalars(
                        'Parameters/{}'.format(name),
                        {'min': p_box[0], 'q25': p_box[1], 'q50': p_box[2], 'q75': p_box[3], 'max': p_box[4]},
                        epoch
                    )
                    if param.requires_grad:
                        g = param.grad.cpu().numpy()
                        q_box = np.quantile(np.abs(g), q=[0.0, 0.25, 0.5, 0.75, 1.0])
                        self.writer.add_scalars(
                            'Gradients/{}'.format(name),
                            {'min': q_box[0], 'q25': q_box[1], 'q50': q_box[2], 'q75': q_box[3], 'max': q_box[4]},
                            epoch
                        )

            # Validation phase, if necessary
            if (epoch - self.init_epoch + 1) % self.num_valid_step == 0:
                self.model.eval()
                best_metrics, diverged, should_stop = self.__valid_step(epoch, best_metrics)

                # Check for divergence in evaluating metrics
                if diverged:
                    print("Evaluating, diverged!")
                    break

                # Check for early stopping
                if should_stop:
                    early_stopped = True
                    print("Early stopping!")
                    break

            # Check for training divergence or collapse of embeddings to zero due to regularization
            if best_metrics['valid_mrr'] == 1:
                diverged = True
                print("MRR 1, diverged!")
                break
            if 0.0 < best_metrics['valid_mrr'] < 1e-4:
                if self.regularizer is not None:
                    diverged = True
                    print("0 embedding weight, diverged!")
                    break

        # Add the hyperparameters dictionary and the related metrics
        best_metrics['diverged'] = diverged
        best_metrics['early_stopped'] = early_stopped
        best_metrics['training_time'] = self.training_time
        self.log_summary(best_metrics)

        # Flush and close the stream manually just to be sure
        self.log_shutdown()
