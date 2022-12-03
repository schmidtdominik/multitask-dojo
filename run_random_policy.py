import itertools
import socket
import time

import wandb
from tqdm import tqdm

from dojo import Dojo

if __name__ == '__main__':
    n_workers = 16
    wandb.init(project='dojo_v1', mode='disabled')
    dojo = Dojo(log_dir=f'./data/{int(time.time())}_{socket.gethostname()}_urp/', n_workers=n_workers)

    results = {}

    for step in tqdm(itertools.count(0, 1)):
        obs, rewards, dones, infos, task_ids = dojo.step_wait()
        dojo.step_async([dojo.action_space.sample() for i in range(n_workers)])

    dojo.close()