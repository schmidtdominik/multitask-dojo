import signal
from typing import Optional

import numpy as np
import multiprocessing as mp

from common import task_specs
from common.env_setup import create_env
from common.types import TaskType


class WorkerRef:
    """reference to a worker process held by the main process"""

    worker_id = 0

    def __init__(self):
        ctx = mp.get_context("fork")

        self.parent_conn, child_conn = ctx.Pipe()
        self.worker = ctx.Process(target=worker, args=(child_conn, WorkerRef.worker_id))
        WorkerRef.worker_id += 1
        self.worker.start()
        self.task_id: Optional[str] = None
        self.task_session_length: int = 0
        self.task_session_id: int = 0

    def step_async(self, action):
        self.parent_conn.send(("step", action))

    def reset_async(self):
        self.parent_conn.send(("reset",))

    def get(self):
        return self.parent_conn.recv()

    def close(self):
        self.parent_conn.send(None)
        self.worker.join()
        self.worker.close()


def worker(pipe, worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    env = None

    while True:
        try:
            cmd = pipe.recv()
        except OSError as e:
            print(e, f"in worker {worker_id}")
            return
        if cmd is None:
            break
        elif cmd[0] == "load_task":
            if env:
                env.close()
                env = None
            _, task_id, seed, preproc, log_dir = cmd
            task_spec = task_specs.tasks[task_id]

            rng = np.random.default_rng(seed=seed)
            local_seed = int(rng.integers(0, 2**31 - 2))

            env = create_env(task_spec, local_seed, log_dir, preproc)

            env.action_space.seed(local_seed)
            env.observation_space.seed(local_seed)
            if task_spec.type != TaskType.PROCGEN:
                env.seed(local_seed)
        elif cmd[0] == "reset":
            pipe.send((env.reset(), 0.0, False, dict()))
        elif cmd[0] == "step":
            pipe.send(env.step(cmd[1]))
        else:
            raise RuntimeError("Unknown command {}".format(cmd))
