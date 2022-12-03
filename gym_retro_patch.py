"""
Note: Newer gym versions introduced changes that break procgen and gym-retro. Neither of which
is still maintained. The newest version of gym we can safely support is gym[atari]==0.23.1,
however we then need to awkwardly monkey patch gym.utils.seeding for gym-retro.

See dojo.py.
"""
import hashlib
import os
import struct
from typing import Optional, List, Union
import gym.error as error

def hash_seed(seed: Optional[int] = None, max_bytes: int = 8) -> int:
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:
    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)
    Args:
        seed: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def create_seed(a: Optional[Union[int, str]] = None, max_bytes: int = 8) -> int:
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.
    Args:
        a: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        bt = a.encode("utf8")
        bt += hashlib.sha512(bt).digest()
        a = _bigint_from_bytes(bt[:max_bytes])
    elif isinstance(a, int):
        a = int(a % 2 ** (8 * max_bytes))
    else:
        raise error.Error(f"Invalid type for seed: {type(a)} ({a})")

    return a


# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bt: bytes) -> int:
    sizeof_int = 4
    padding = sizeof_int - len(bt) % sizeof_int
    bt += b"\0" * padding
    int_count = int(len(bt) / sizeof_int)
    unpacked = struct.unpack(f"{int_count}I", bt)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint: int) -> List[int]:
    # Special case 0
    if bigint < 0:
        raise error.Error(f"Seed must be non-negative, not {bigint}")
    elif bigint == 0:
        return [0]

    ints: List[int] = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints

def apply_patch():
    from gym.utils import seeding
    seeding.hash_seed = hash_seed
    seeding.create_seed = create_seed
    seeding._bigint_from_bytes = _bigint_from_bytes
    seeding._int_list_from_bigint = _int_list_from_bigint