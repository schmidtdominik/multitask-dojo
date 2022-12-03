"""
Definitions of environment wrappers
"""
import os
import random

import cv2
import gym
import imageio
import numpy as np
from common.action_wrappers import DOJO_LEGAL_ACTIONS

cv2.ocl.setUseOpenCL(False)


class RecordEpisodeStatistics(gym.Wrapper):
    """
    Wrapper that records episode statistics.
    """

    def __init__(self, env, frame_skip):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.reset_stats()
        self.frame_skip = frame_skip

    def reset_stats(self):
        self.episode_return = self.episode_clipped_return = 0.0
        self.episode_length = 0
        self.action_stats = np.zeros((DOJO_LEGAL_ACTIONS,))

    def reset(self, **kwargs):
        observation = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.reset_stats()
        return observation

    def step(self, action: int):
        observation, reward, done, info = super(RecordEpisodeStatistics, self).step(
            action
        )
        self.episode_return += reward
        self.episode_clipped_return += np.clip(reward, -1, 1)
        self.episode_length += 1
        self.action_stats[action] += 1
        if done:
            info["episode_metrics"] = {
                "ret": self.episode_return,
                "ret_clipped": self.episode_clipped_return,
                "length_emu": self.episode_length,
                "length_mdp": self.episode_length / self.frame_skip,
                "action_distribution": self.action_stats / self.action_stats.sum(),
            }
            self.reset_stats()
        return observation, reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["truncated"] = True
        else:
            info["truncated"] = False
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        if env.unwrapped.get_action_meanings()[0] != "NOOP":
            self.override_num_noops = 1

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class StickyMaxFrameSkip(gym.Wrapper):
    def __init__(self, env, seed, n_skip, sticky_prob, max_obs=True):
        super().__init__(env)
        self.n_skip = n_skip
        self.sticky_prob = sticky_prob
        self.current_action = None
        self.rng = np.random.RandomState(seed)
        self.max_obs = max_obs
        if max_obs:
            self.obs_buffer = np.zeros(
                (2,) + env.observation_space.shape, dtype=np.uint8
            )

    def reset(self, **kwargs):
        self.current_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        done = False
        total_reward = 0.0
        for i in range(self.n_skip):
            if self.current_action is None:
                # first step after reset, always use given action
                self.current_action = action
            elif i < self.n_skip - 1:
                # first n-1 substeps, delay with probability=sticky_prob
                if self.rng.rand() > self.sticky_prob:
                    self.current_action = action
            else:
                # last substep, new action definitely kicks in
                self.current_action = action

            obs, reward, done, info = self.env.step(self.current_action)
            total_reward += reward

            if self.max_obs:
                if i == self.n_skip - 2:
                    self.obs_buffer[0] = obs
                if i == self.n_skip - 1:
                    self.obs_buffer[1] = obs
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        if self.max_obs:
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
            obs = self.obs_buffer[0]
        return obs, total_reward, done, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height, grayscale, interp=cv2.INTER_AREA):
        super().__init__(env)
        self.interp = interp
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if obs.shape[0] != self._height or obs.shape[1] != self._width:
            obs = cv2.resize(
                obs, (self._width, self._height), interpolation=self.interp
            )
        if self._grayscale:
            obs = np.expand_dims(obs, -1)
        return obs


class RewardClipWrapper(gym.RewardWrapper):
    def __init__(self, env, clip_v=1):
        super().__init__(env)
        self.clip_v = clip_v
        assert clip_v > 0.1

    def reward(self, reward):
        if reward < 0:
            return np.clip(reward, -self.clip_v, -0.1)
        elif reward > 0:
            return np.clip(reward, 0.1, self.clip_v)
        else:
            return reward


class AuxiliaryRewardWrapper(gym.Wrapper):
    """
    Some gym-retro games give us extra information in the info dict (health was lost, coins/rings
    were collected, etc.) Since many gym-retro environments have very sparse reward functions, we add
    auxiliary rewards based on this information.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reset_vals()

    def reset_vals(self):
        self.episode_aux_rew = 0
        self.stats = {
            v: None
            for v in {"health", "lives", "rings", "coins", "gems", "ammo", "energy"}
        }

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        step_aux_reward = 0

        if done:
            info["episode_metrics"]["aux_rew"] = self.episode_aux_rew
            self.reset_vals()
        else:
            for k, v in info.items():
                if k not in self.stats:
                    continue
                if self.stats[k] is not None:
                    if k in {"health", "lives"} and v < self.stats[k]:
                        # penalty for losing health/lives
                        step_aux_reward -= 0.3
                    elif (
                        k in {"health", "rings", "coins", "gems", "ammo", "energy"}
                        and v > self.stats[k]
                    ):
                        # reward for collecting health/rings/coins/gems/ammo/energy
                        step_aux_reward += 0.3
                self.stats[k] = v
            self.episode_aux_rew += step_aux_reward

        return obs, reward + step_aux_reward, done, info


class RecorderWrapper(gym.Wrapper):
    def __init__(self, env, log_dir, tc):
        super().__init__(env)
        self.writer = None
        self.log_dir = log_dir
        self.gid = (
            tc.game_name.replace("/", "_") + "_" + tc.meta_args.get("state", "") + "_"
        )

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if done and self.writer is not None:
            self.writer.close()
            self.writer = None
        else:
            if self.writer is None:
                self.writer = imageio.get_writer(
                    str(self.log_dir)
                    + f"/videos/{self.gid}_{random.randint(0, 2**31-1)}.mp4",
                    fps=60,
                    macro_block_size=1,
                    quality=10,
                )
            else:
                self.writer.append_data(obs)

        return obs, rew, done, info
