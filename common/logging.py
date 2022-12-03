import math
import os
import pickle
import time
from collections import defaultdict, namedtuple
from copy import copy
from functools import partial
from pathlib import Path
from typing import List, Dict, NamedTuple

import cv2
import imageio
import numpy as np
import psutil
import wandb
import subprocess

from common.action_wrappers import DOJO_LEGAL_ACTIONS
from common.types import EpisodeRecord


class TiledRecorder:
    def __init__(self, log_dir, record_every):
        self.vid_dir = log_dir / "videos"
        self.vid_dir.mkdir(parents=True, exist_ok=True)
        self.record_every = record_every

        self.writer = None
        self.recorded_frames = 0

    def update(self, obs, n_workers, total_steps, upd_check_func):

        if (
            self.record_every is not None
            and upd_check_func(self.record_every)
            and self.writer is None
        ):
            print("Starting video recording...")
            self.writer = imageio.get_writer(
                self.vid_dir / f"{total_steps}.mp4",
                fps=30,
                macro_block_size=1,
                quality=10,
            )

        if self.writer is not None:
            n, m = 8, n_workers // 8
            vid_upscale = 2
            obs = np.stack(obs)
            obs = (
                obs.reshape(n, m, obs.shape[1], obs.shape[2], -1)
                .transpose(0, 2, 1, 3, 4)
                .reshape(n * obs.shape[1], m * obs.shape[2], -1)
                .squeeze()
            )
            obs = cv2.resize(
                obs,
                dsize=(obs.shape[1] * vid_upscale, obs.shape[0] * vid_upscale),
                interpolation=cv2.INTER_NEAREST,
            )
            self.writer.append_data(obs)
            self.recorded_frames += 1

        if self.recorded_frames >= 60 * 15:  # 60 seconds of gameplay
            print("Finished video recording.")
            self.writer.close()
            self.writer = None
            self.recorded_frames = 0


def ema(ema_x, next_x, alpha=0.9):
    return alpha * ema_x + (1 - alpha) * next_x


class Logger:
    def __init__(
        self,
        log_dir,
        task_ids: List[str],
        metadata: dict,
        n_workers: int,
        record_every: int,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_save: int = 0
        self.recorder = TiledRecorder(self.log_dir, record_every)

        # metadata for this run
        self.n_workers = n_workers
        self.task_ids = task_ids
        self.metadata: dict = metadata

        # current values
        self.global_action_distribution = np.zeros((DOJO_LEGAL_ACTIONS,))
        self.task_frame_count = defaultdict(int)
        self.est_task_returns = dict()
        self.est_task_lengths = {task_id: 900 for task_id in task_ids}

        #self.task_perc_ranks_ema = {tc: 0.5 for tc in task_configs}
        #self.task_perc_ranks = {tc: 0.5 for tc in task_configs}
        #self.task_perc_ranks_slope = {tc: 0.0 for tc in task_configs}

        # actual data
        self.total_steps: int = 0
        self.total_episodes: int = 0
        self.global_action_distributions = []
        self.task_ep_stats: Dict[str, List[EpisodeRecord]] = {
            task_id: [] for task_id in task_ids
        }
        self.task_frame_counts = []
        self.sysinfo = []

    def finalize_task(self, task_id: str, infos: dict):
        self.total_episodes += 1
        metrics = infos["episode_metrics"]

        self.est_task_lengths[task_id] = ema(
            self.est_task_lengths.get(task_id, metrics["length_mdp"]),
            metrics["length_mdp"],
            0.8,
        )
        self.est_task_returns[task_id] = ema(
            self.est_task_returns.get(task_id, metrics["ret"]), metrics["ret"], 0.8
        )

        # percentile_rank = get_task_percentile_rank(task_id, metrics['ret'])
        # self.task_perc_ranks_ema[task_id] = alpha * self.task_perc_ranks_ema[task_id] + (1 - alpha) * percentile_rank
        # slope = (percentile_rank - self.task_perc_ranks[task_id])
        # self.task_perc_ranks[task_id] = percentile_rank
        # self.task_perc_ranks_slope[task_id] = alpha * self.task_perc_ranks_slope[task_id] + (1 - alpha) * slope

        record = EpisodeRecord(
            timestamp=time.time(),
            framestamp=self.total_steps,
            task_id=task_id,
            ret=metrics["ret"],
            ret_clipped=metrics["ret_clipped"],
            length_emu=metrics["length_emu"],
            length_mdp=metrics["length_mdp"],
            action_distribution=metrics["action_distribution"],
            truncated=infos["time_limit_expired"]
            if "time_limit_expired" in infos
            else None,
            aux_rew=metrics.get("aux_rew", 0),
        )
        self.task_ep_stats[task_id].append(record)

    def save(self):
        with open(self.log_dir / "metrics_.pickle", "wb+") as f:
            pickle_data = copy(self.__dict__)
            del pickle_data["recorder"]
            del pickle_data["global_action_distribution"]
            del pickle_data["est_task_returns"]
            pickle.dump(pickle_data, f)
            self.last_save = time.time()
        os.rename(self.log_dir / "metrics_.pickle", self.log_dir / "metrics.pickle")
        print(f"Saved dojo log data to disk! {self.log_dir}")

    def update_step_condition(self, every_n_steps: int) -> bool:
        return math.ceil(self.total_steps / every_n_steps) < (
            (self.total_steps + self.n_workers) / every_n_steps
        )

    def update(self, obs):
        # other conditions below rely on the fact that the step counter moves in steps of n_workers
        assert self.total_steps % self.n_workers == 0

        if self.update_step_condition(50_000) and self.total_steps >= 100_000:
            self.global_action_distributions.append(
                self.global_action_distribution / self.global_action_distribution.sum()
            )
            self.global_action_distribution = np.zeros((DOJO_LEGAL_ACTIONS,))

            wandb.log(
                dict(
                    total_steps=self.total_steps,
                    #eval_ge_urp=eval_utils.eval_ge_urp(self.est_task_returns),
                    #eval_median_hns=eval_utils.eval_hns(self.est_task_returns),
                    #eval_score_diff=eval_utils.eval_score_diff(self.est_task_returns),
                    #eval_score_distr=eval_utils.eval_score_distr(self.est_task_returns),
                    episode_counts=wandb.Histogram(
                        [len(eps) for eps in self.task_ep_stats.values()]
                    ),
                    frame_counts=wandb.Histogram(
                        list(self.task_frame_count.values()), num_bins=256
                    ),
                    action_distribution=wandb.Histogram(
                        np_histogram=(
                            self.global_action_distributions[-1],
                            np.arange(DOJO_LEGAL_ACTIONS + 1),
                        )
                    ),
                )
            )

        if self.update_step_condition(500_000) and self.total_steps > 0:
            self.task_frame_counts.append(copy(self.task_frame_count))

        if time.time() - self.last_save > 60 * 20:
            self.save()

        self.recorder.update(
            obs, self.n_workers, self.total_steps, partial(self.update_step_condition)
        )
