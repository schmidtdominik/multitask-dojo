from enum import Enum
from typing import NamedTuple
import numpy as np
from frozendict import frozendict


class StepResult(NamedTuple):
    """ Batched result of a step in the environment. """
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: dict
    task_ids: np.ndarray


class TaskType(Enum):
    ATARI = 'atari'
    RETRO = 'retro'
    PROCGEN = 'procgen'
    META_ARCADE = 'meta_arcade'


class TaskSpec(NamedTuple):
    """ Contains the arguments that are passed to the underlying libraries (gym, gym-retro, ..) to launch an env. """
    name: str
    type: TaskType
    subtype: str
    args: frozendict


class EpisodeRecord(NamedTuple):
    timestamp: float
    framestamp: int
    task_id: str

    ret: float
    ret_clipped: float

    length_emu: int
    length_mdp: int

    action_distribution: np.ndarray
    truncated: bool

    aux_rew: float = 0