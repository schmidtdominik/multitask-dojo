"""
Dojo builds on several libraries that provide the underlying environments:
- gym
- gym-retro
- procgen
- meta_arcade

This module translates standardized TaskSpec's into environment specific arguments
and creates and configures an environment instance.
"""

import pickle
from pathlib import Path

import gym, retro, procgen, meta_arcade

from common.action_wrappers import *
from common.env_wrappers import *

from common.types import TaskType

task_type_to_actionmapper = {
    TaskType.ATARI: AtariActionMapper,
    TaskType.RETRO: RetroActionMapper,
    TaskType.PROCGEN: ProcgenActionMapper,
    TaskType.META_ARCADE: MetaArcadeActionMapper,
}


def wrap_generic(
    env,
    tc,
    seed,
    log_dir,
    time_limit,
    frame_skip,
    sticky_prob,
    resolution,
    clip_rewards,
    grayscale,
):
    if tc.type == TaskType.PROCGEN or tc.type == TaskType.META_ARCADE:
        frame_skip = 1

    env = task_type_to_actionmapper[tc.type](env, tc)
    if tc.type == TaskType.ATARI:
        env = NoopResetEnv(env, noop_max=30)
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env, frame_skip=frame_skip)
    if frame_skip > 1:
        env = StickyMaxFrameSkip(
            env,
            seed,
            n_skip=frame_skip,
            sticky_prob=sticky_prob,
            max_obs=(False if tc.type == TaskType.RETRO else True),
        )
    if tc.type == TaskType.RETRO:
        env = AuxiliaryRewardWrapper(env)
    if clip_rewards:
        env = RewardClipWrapper(env, (5 if tc.type == TaskType.PROCGEN else 1))
    env = WarpFrame(env, width=resolution[1], height=resolution[0], grayscale=grayscale)
    return env


def create_env(tc, seed, log_dir, preproc_args):
    if tc.type == TaskType.ATARI:
        env = gym.make(
            tc.name,
            obs_type="rgb",
            frameskip=1,
            repeat_action_probability=0,
            full_action_space=False,
        )
        # TODO: gym now automatically wraps the env in a TimeLimit wrapper!
    elif tc.type == TaskType.RETRO:
        env = retro.make(tc.name, state=tc.args["state"])
    elif tc.type == TaskType.PROCGEN:
        env = gym.make(
            f"procgen-{tc.name.lower()}-v0",
            start_level=seed,
            num_levels=0,
            **tc.args,
        )
    elif tc.type == TaskType.META_ARCADE:
        lib_path = Path(__file__).resolve().parent.parent

        config = (
            str(lib_path / "meta_arcade_level_defs" / tc.name)
            if tc.name.endswith(".json")
            else tc.name
        )
        env = gym.make(
            "MetaArcade-v0", config=config, headless=True, game_ticks_per_step=3
        )
    else:
        raise ValueError(f"Unknown task type {tc.type}")

    return wrap_generic(env, tc, seed, log_dir, **preproc_args)
