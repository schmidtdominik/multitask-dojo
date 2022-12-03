"""
multi-task dojo is a reinforcement learning environment intended for research in multi-task RL.
"""

import os
import random
from collections import deque, Counter
from copy import copy
from typing import List, Union, Set, Dict

# noinspection PyUnresolvedReferences
import gym, meta_arcade, procgen, procgen
from gym.spaces import Discrete
from scipy.stats import rankdata

import gym_retro_patch
from common.action_wrappers import *
from common.logging import Logger
from common.task_specs import tasks, task_int_ids
from common.types import StepResult
from common.worker import WorkerRef

"""
Note: Newer gym versions introduced changes that break procgen and gym-retro. Neither of which 
is still maintained. The newest version of gym we can safely support is gym[atari]==0.23.1, 
however we then need to awkwardly monkey patch gym.utils.seeding for gym-retro.
"""
gym_retro_patch.apply_patch()


class Dojo:
    def __init__(
        self,
        log_dir=None,
        task_ids=list(tasks.keys()),  # by default use all tasks
        seed=0,
        record_every=10_000_000,
        resolution=(128, 128),
        grayscale=False,
        time_limit=108_000 // 6,
        frame_skip=4,
        clip_rewards=True,
        sticky_prob=0.2,
        n_workers=None,
        min_task_session_len=500,
        scheduler="least_frames",
    ):

        self.n_workers = n_workers if n_workers else len(os.sched_getaffinity(0))
        # due to memory constraints we don't do double buffering,
        # instead we maintain a buffer of `n_repl_workers` additional workers
        # that are used to immediately replace workers that we are waiting for
        self.n_repl_workers = 6
        self.min_task_session_len = min_task_session_len

        self.rng = random.Random(seed)
        self.preproc = dict(
            grayscale=grayscale,
            resolution=resolution,
            time_limit=time_limit,
            frame_skip=frame_skip,
            sticky_prob=sticky_prob,
            clip_rewards=clip_rewards,
        )
        self.action_space = Discrete(DOJO_LEGAL_ACTIONS)
        self.action_space.seed(self.rng.getrandbits(31))

        self.task_ids = task_ids

        if scheduler == "least_frames":
            self.sched_policy = (
                self.scheduler_least_frames_stochastic
                if len(self.task_ids) > 100
                else self.scheduler_least_frames
            )
        elif scheduler == "least_episodes":
            self.sched_policy = self.scheduler_least_episodes

        print(
            f"Initializing Dojo env with {len(self.task_ids)} tasks\n"
            f"Execution settings: n_workers={self.n_workers}, n_repl_workers={self.n_repl_workers}"
        )

        self.logger = Logger(
            log_dir=log_dir,
            task_ids=self.task_ids,
            metadata=dict(seed=seed, preproc=self.preproc),
            n_workers=n_workers,
            record_every=record_every,
        )

        self.active_tasks: Set[str] = set()
        self.worker_refs, self.repl_worker_refs = (
            [],
            [],
        )  # schedule_task refers back to these to find active tasks
        self.worker_refs: List[WorkerRef] = [
            self.schedule_task(WorkerRef()) for _ in range(self.n_workers)
        ]
        self.repl_worker_refs = deque(
            [self.schedule_task(WorkerRef()) for _ in range(self.n_repl_workers)]
        )

    def get_active_task_counts(self) -> Dict[str, int]:
        return Counter(
            [w_ref.task_id for w_ref in self.worker_refs + list(self.repl_worker_refs)]
        )

    def get_running_frame_counts(self) -> Dict[str, int]:
        active_task_counts = self.get_active_task_counts()
        return {
            task_id: self.logger.task_frame_count.get(task_id, 0)
            + max(self.logger.est_task_lengths[task_id], self.min_task_session_len)
            * active_task_counts[task_id]
            for task_id in self.task_ids
        }

    def scheduler_least_frames(self, available_tasks) -> str:
        running_frame_counts = self.get_running_frame_counts()
        task_frames = [
            running_frame_counts.get(task_id, 0) for task_id in available_tasks
        ]
        return available_tasks[np.argmin(task_frames)]

    def scheduler_least_frames_stochastic(self, available_tasks) -> str:
        running_frame_counts = self.get_running_frame_counts()
        norm_task_frames = np.array(
            [running_frame_counts.get(task_id, 0) for task_id in available_tasks]
        )

        task_frame_deficit = np.max(norm_task_frames) - norm_task_frames

        # proportional
        # distr = task_frame_deficit / np.sum(task_frame_deficit)

        # rank-based
        temp = 0.72
        ranks = rankdata(-task_frame_deficit)
        softmax = (1 / ranks) ** (1 / temp)
        distr = softmax / np.sum(softmax)

        i = np.random.choice(np.arange(0, len(available_tasks)), size=1, p=distr)
        return available_tasks[i[0]]

    def scheduler_least_episodes(self, available_tasks) -> str:
        active_tasks_counts = self.get_active_task_counts()
        task_eps = [
            len(self.logger.task_ep_stats[task_id])
            + active_tasks_counts.get(task_id, 0)
            for task_id in available_tasks
        ]
        return available_tasks[np.argmin(task_eps)]

    def schedule_task(self, worker_ref):
        """
        The worker_ref is passed through.
        """
        # shut down loaded task
        self.active_tasks.discard(worker_ref.task_id)
        available_tasks: List[str] = copy(self.task_ids)
        self.rng.shuffle(available_tasks)

        next_task_id: str = self.sched_policy(available_tasks)
        worker_ref.parent_conn.send(
            (
                "load_task",
                next_task_id,
                self.rng.getrandbits(31),
                self.preproc,
                self.logger.log_dir,
            )
        )

        worker_ref.task_id = next_task_id
        # worker_ref.action_mask
        worker_ref.reset_async()
        worker_ref.task_session_length = 0
        worker_ref.task_session_id = random.randint(0, 2**31 - 1)
        self.active_tasks.add(next_task_id)
        return worker_ref

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: Union[np.ndarray, list]):
        for w_ref, a in zip(self.worker_refs, actions):
            self.logger.global_action_distribution[a] += 1
            w_ref.step_async(a)

    def step_wait(self) -> StepResult:
        obs, rewards, dones, infos = zip(*[w_ref.get() for w_ref in self.worker_refs])

        dones = np.stack(
            dones
        )  # do not remove this line, otherwise the iteration below will be incorrect
        obs = list(obs)

        for idx, done in enumerate(dones):
            self.logger.task_frame_count[self.worker_refs[idx].task_id] += 1
            self.worker_refs[idx].task_session_length += 1
            self.logger.total_steps += 1

            if done:
                # log episode stats
                finished_tc = self.worker_refs[idx].task_id
                self.logger.finalize_task(finished_tc, infos[idx])

                if (
                    self.worker_refs[idx].task_session_length
                    > self.min_task_session_len
                ):
                    # schedule new task on worker
                    self.schedule_task(self.worker_refs[idx])
                else:
                    # continue executing the same task
                    self.worker_refs[idx].reset_async()
                    self.worker_refs[idx].task_session_id = random.randint(
                        0, 2**31 - 1
                    )

                # move worker to repl queue
                self.repl_worker_refs.append(self.worker_refs[idx])

                # replace finished worker by next repl worker
                self.worker_refs[idx] = self.repl_worker_refs.popleft()

                # activate new replacement worker, get its first observation, and update action_space size
                obs[idx] = self.worker_refs[idx].get()[0]

        obs = np.stack(obs)
        self.logger.update(obs)

        # these correspond to the (next step) observations in obs
        task_ids = np.array(
            [task_int_ids[w_ref.task_id] for w_ref in self.worker_refs],
            dtype=np.int64,
        )

        return StepResult(
            obs=obs,
            rewards=np.stack(rewards),
            dones=dones,
            infos=infos,
            task_ids=task_ids,
        )

    def close(self):
        print("Shutting Dojo down!")
        self.logger.save()
        for w_ref in self.worker_refs + list(self.repl_worker_refs):
            w_ref.close()
