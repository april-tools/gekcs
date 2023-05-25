import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser(
    description="Tractable Statistical Relational Learning (TSRL) Best Models for Link Prediction Script"
)
parser.add_argument(
    'exps_filepath', type=str, help="The experiment results and hyperparameters CSV filepath"
)
parser.add_argument(
    'best_out_filepath', type=str, help="The best experiment results and hyperparameters CSV filepath"
)
parser.add_argument(
    '--best_cmd_filepath', type=str, default="cmd.txt", help="The list of commands of the best experiment runs filepath"
)
parser.add_argument(
    '--best-lower', action='store_true', help="Whether to take as best experiments those with lower validation metrics"
)
parser.add_argument(
    '--exp-id', type=str, default='PLL', choices=['PLL', 'MLE'], help="The experiment ids to take into account"
)
parser.add_argument(
    '--distil-model', action='store_true', help="Whether to consider models with parameters distillation"
)
parser.add_argument(
    '--metric', type=str, default='mrr', help="The validation metric to get the best results"
)
parser.add_argument(
    '--group-by-rank', action='store_true', default=False, help="Whether to group results by rank"
)
parser.add_argument(
    '--digits', type=int, default=3, help="Maximum number of digits after comma"
)
parser.add_argument(
    '--md-decorate', action='store_true', default=False, help="Whether to decorate the commands using Markdown"
)
parser.add_argument(
    '--make-repetitions', action='store_true', default=False, help="Whether to print repetitions for commands"
)
parser.add_argument(
    '--start-repetition', type=int, default=0, help="The starting repetition id",
)
parser.add_argument(
    '--num-repetitions', type=int, default=5, help="The number of repetitions"
)
parser.add_argument(
    '--average-reps', action='store_true', default=False,
    help="Whether to average multiple repetitions and run statistical tests"
)

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the experiments results CSV
    df = pd.read_csv(args.exps_filepath, sep=',')

    # Filter diverged experiments and keep experiments of interested based on exp id
    filter_dict = {'diverged': False, 'distil_model': args.distil_model}
    if not args.average_reps:
        filter_dict['repetition_id'] = 0
        if args.exp_id == 'PLL':
            filter_dict['score_lhs'] = True
            filter_dict['score_rhs'] = True
            filter_dict['score_rel'] = True
            filter_dict['score_ll'] = False
        elif args.exp_id == 'MLE':
            filter_dict['score_lhs'] = False
            filter_dict['score_rhs'] = False
            filter_dict['score_rel'] = False
            filter_dict['score_ll'] = True
        else:
            raise ValueError("Unknown experiment id")
    for k, v in filter_dict.items():
        if k not in df.columns:
            continue
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]

    # Get the best hyperparameters by grouping by dataset, model and learning setting
    if args.average_reps:
        filter_keys = [
            'repetition_id',
            'valid_mrr',
            'test_mrr',
            'test_hits@1',
            'test_hits@3',
            'test_hits@10',
            'valid_avg_ll',
            'test_avg_ll',
            'valid_mrr_epoch',
            'valid_avg_ll_epoch',
            'diverged',
            'training_time']
        groupby_keys = list(filter(lambda c: c not in filter_keys, df.columns))
        group_df = df.groupby(groupby_keys)
        def two_stdev(x):
            return 2.0 * np.std(x)
        agg_df = group_df.agg(['mean', two_stdev]).reset_index()
        group_df = df.groupby(by='dataset')
        num_groups = len(group_df)
        pairwise_pvalues = dict()
        for i, data_df in group_df:
            inner_group_df = data_df.groupby(by=groupby_keys)
            data_df.to_csv('test.csv', index=False)
            num_inner_groups = len(inner_group_df)
            pairwise_vars = list(itertools.product(range(num_inner_groups), range(num_inner_groups)))
            inner_pairwise_pvalues = np.zeros(shape=(num_inner_groups, num_inner_groups), dtype=np.float64)
            ss = [r[1]['test_' + args.metric].tolist() for r in inner_group_df]
            for j, k in pairwise_vars:
                _, p = stats.mannwhitneyu(ss[j], ss[k], alternative='greater')
                inner_pairwise_pvalues[j, k] = p
            pairwise_pvalues[i] = inner_pairwise_pvalues
        nums_inner_groups = set(map(lambda x: x[1].shape[0], pairwise_pvalues.items()))
        assert len(nums_inner_groups) == 1
        num_vars = list(nums_inner_groups)[0]
        df = agg_df
        for j in range(num_vars):
            pvalues = np.concatenate([pairwise_pvalues[i][:, j] for i in pairwise_pvalues.keys()], axis=0)
            df['mannwhitneyu_pvalue_>%d' % j] = pvalues.tolist()
    else:
        groupby_keys = [
            'dataset', 'model',
            'score_lhs', 'score_rhs', 'score_rel', 'score_ll'
        ]
        if args.group_by_rank:
            groupby_keys.append('rank')
        group_df = df.groupby(groupby_keys)['valid_' + args.metric]
        best_idx = group_df.idxmin() if args.best_lower else group_df.idxmax()
        df = df.loc[best_idx]

    # Round float columns
    float_cols = [
        c for (c, t) in df.dtypes.to_dict().items()
        if t in [float, np.dtype('float32'), np.dtype('float64')]
    ]
    df = df.round(dict((k, args.digits) for k in float_cols))

    # Save the best results
    df.to_csv(args.best_out_filepath, sep=',', index=args.average_reps)
    if args.average_reps:
        quit()

    def build_command(r):
        cmd = "python -m kbc.experiment"
        if args.exp_id == 'PLL':
            cmd = f"{cmd} --experiment_id {'distil-' if args.distil_model else ''}PLL"
        elif args.exp_id == 'MLE':
            cmd = f"{cmd} --experiment_id {'distil-' if args.distil_model else ''}MLE"
        else:
            raise ValueError("Unknown experiment id")
        for hp in ['dataset', 'model', 'rank', 'optimizer', 'batch_size', 'learning_rate']:
            cmd = f"{cmd} --{hp} {r[hp]}"
        if args.exp_id == 'PLL':
            cmd = f"{cmd} --score_lhs True --score_rel True --score_rhs True"
        elif args.exp_id == 'MLE':
            cmd = f"{cmd} --score_ll True"
        if args.distil_model:
            cmd = f"{cmd} --distil_model True"
        return cmd

    # Save the list of commands of the best experimental outcomes
    with open(args.best_cmd_filepath, 'w') as f:
        for _, r in df.iterrows():
            cmd = build_command(r)
            if args.md_decorate:
                f.write('**MLE**' if args.exp_id == 'MLE' else '**PLL**')
                f.write('\n')
                f.write('\n')
                f.write("```shell\n")
            if args.make_repetitions:
                for repetition_id in range(args.start_repetition, args.start_repetition + args.num_repetitions):
                    rep_cmd = f"{cmd} --repetition_id {repetition_id}"
                    f.write(f"{rep_cmd}\n")
            else:
                f.write(f"{cmd}\n")
            if args.md_decorate:
                f.write("```\n")
                f.write('\n')
