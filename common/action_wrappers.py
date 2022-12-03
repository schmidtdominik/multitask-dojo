"""
Dojo always offers 15 discrete actions, but most games only use a subset of these actions.
This modules defines action wrappers that map actions from dojo's shared action space to each environment's
individual action space. Actions that are invalid for a specific task are mapped to NOOP.
Action mappings roughly preserve action semantics, e.g. dojo action "2" is mapped to "right" where applicable.
"""

import gym
import numpy as np

DOJO_LEGAL_ACTIONS = 15


class RetroActionMapper(gym.ActionWrapper):
    controller_buttons = dict(
        nes=['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'],
        snes=['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R'],
        genesis=['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z'],
        sms=['B', None, None, 'PAUSE', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'],
        gameboy=['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    )

    resolved_actions = dict(
        nes=[[], ['B'], ['UP'], ['RIGHT'], ['LEFT'], ['DOWN'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['A'], ['A', 'B'], ['UP', 'B'], ['DOWN', 'A']],
        snes=[[], ['Y'], ['UP'], ['RIGHT'], ['LEFT'], ['DOWN'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['B'], ['A'], ['X'], ['L'], ['R']],
        genesis=[[], ['B'], ['UP'], ['RIGHT'], ['LEFT'], ['DOWN'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['C'], ['DOWN', 'B'], ['A'], ['Z'], ['X']],
        sms=[[], ['B'], ['UP'], ['RIGHT'], ['LEFT'], ['DOWN'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['A']],
        gameboy=[[], ['B'], ['UP'], ['RIGHT'], ['LEFT'], ['DOWN'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['A'], ['UP', 'DOWN'], ['DOWN', 'A']]
    )

    def __init__(self, env, tc):
        super().__init__(env)

        console = tc.subtype.lower()
        actions = self.resolved_actions[console]
        buttons = self.controller_buttons[console]

        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(np.array(arr, dtype=bool))

        # self.action_space = gym.spaces.Discrete(len(self.resolved_actions[console]))
        self.action_mask = np.array([i < len(self._actions) for i in range(DOJO_LEGAL_ACTIONS)], dtype=bool)

    def action(self, a):
        if a < len(self._actions):
            return self._actions[a].copy()
        else:
            return self._actions[0]


class ProcgenActionMapper(gym.ActionWrapper):

    def __init__(self, env, tc):
        super().__init__(env)

        procgen_actions = [("LEFT", "DOWN"), ("LEFT",), ("LEFT", "UP"), ("DOWN",), (), ("UP",), ("RIGHT", "DOWN"),
                           ("RIGHT",), ("RIGHT", "UP"), ("D",), ("A",), ("W",), ("S",), ("Q",), ("E",)]
        dojo_actions = [(), ("D",), ("UP",), ("RIGHT",), ("LEFT",), ("DOWN",), ("RIGHT", "UP"), ("LEFT", "UP"),
                        ("RIGHT", "DOWN"), ("LEFT", "DOWN"), ("A",), ("W",), ("S",), ("Q",), ("E",)]
        self.action_indices = [procgen_actions.index(a) for a in dojo_actions]

        # self.action_space = gym.spaces.Discrete(len(dojo_actions))
        self.action_mask = np.array([i < len(self.action_indices) for i in range(DOJO_LEGAL_ACTIONS)], dtype=bool)

    def action(self, a):
        return self.action_indices[a]


class AtariActionMapper(gym.ActionWrapper):

    def __init__(self, env, tc):
        super().__init__(env)

        # determine actions that should be made available for this game
        game_actions = env.unwrapped.get_action_meanings()

        dojo_actions = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
                        'UPFIRE', 'DOWNFIRE', 'RIGHTFIRE', 'LEFTFIRE']

        self.action_indices = [game_actions.index(a) if a in game_actions else 0 for a in dojo_actions]
        self.action_mask = np.array(
            [i < len(self.action_indices) and dojo_actions[i] in game_actions for i in range(DOJO_LEGAL_ACTIONS)],
            dtype=bool)

    def action(self, a):
        return self.action_indices[a] if a < len(self.action_indices) else 0


class MetaArcadeActionMapper(gym.ActionWrapper):

    def __init__(self, env, tc):
        super().__init__(env)

        self.mapping = {
            0: 0,
            1: 5,
            2: 1,
            5: 2,
            4: 3,
            3: 4
        }

        # self.action_space = gym.spaces.Discrete(6)
        self.action_mask = np.array([i in self.mapping for i in range(DOJO_LEGAL_ACTIONS)], dtype=bool)

    def action(self, a):
        return self.mapping[int(a)] if int(a) in self.mapping else 0
