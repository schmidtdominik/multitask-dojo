import pickle

import numpy as np
from scipy.stats import norm

# with open('dojo/task_baseline_stats.pickle', 'rb') as f:
#     task_baseline_stats = pickle.load(f)


def eval_ge_urp(mean_rets):
    return np.mean([task_baseline_stats[tc]['urp_ret_mean'] <= ret for tc, ret in mean_rets.items()])


def eval_score_diff(mean_rets, reduce=np.mean, clip_eps=2, eps=1):
    diffs = {}
    for tc, ret in mean_rets.items():
        urp_score = task_baseline_stats[tc]['urp_ret_mean']
        diffs[tc] = (ret - urp_score)
        diffs[tc] /= abs(urp_score) if urp_score != 0 else abs(urp_score + eps)
        if clip_eps is not None:
            diffs[tc] = float(np.clip(diffs[tc], -clip_eps, clip_eps))

    return diffs if reduce is None else reduce(list(diffs.values()))


def eval_score_distr(mean_rets, reduce=np.mean, spread_factor=2, eps=1):
    tcs = list(mean_rets.keys())
    params = []

    for tc in tcs:
        urp_mean = task_baseline_stats[tc]['urp_ret_mean']
        urp_std = task_baseline_stats[tc]['urp_ret_std']
        if urp_std == 0: urp_std = eps
        params.append([mean_rets[tc], urp_mean, urp_std])
    params = np.array(params)

    ps = norm.cdf(params[:, 0], params[:, 1], params[:, 2] * spread_factor)
    ps = {tc: p for tc, p in zip(tcs, ps)}

    return ps if reduce is None else reduce(list(ps.values()))


def get_task_percentile_rank(tc: TaskConfig, ret: float):
    urp_mean = task_baseline_stats[tc]['urp_ret_mean']
    urp_std = task_baseline_stats[tc]['urp_ret_std']

    if urp_std == 0: urp_std = 1
    return norm.cdf(ret, urp_mean, urp_std * 2)


def eval_hns(mean_rets, reduce=np.median):
    hns_scores = {}
    for tc, ret in mean_rets.items():
        if tc.game_name in dojo.tasks.hns_ale:
            urp_score = task_baseline_stats[tc]['urp_ret_mean']
            hns_scores[tc] = (ret - urp_score) / (dojo.tasks.hns_ale[tc.game_name] - urp_score)

    if reduce is None:
        return hns_scores
    return reduce(list(hns_scores.values())) if hns_scores else 0
