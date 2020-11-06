import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from gym_waf.envs.interfaces import LocalInterface, ClassificationFailure
from gym_waf.envs.features import SqlFeatureExtractor

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}


class WafEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, payload, maxturns=20):
        self.orig_payload = payload
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.maxturns = maxturns
        self.feature_extractor = SqlFeatureExtractor()
        self.history = []

        self.reset()

    def step(self, action_index):
        raise NotImplementedError("_step not implemented")

    def _take_action(self, action_index):
        assert action_index < len(ACTION_LOOKUP)
        action = ACTION_LOOKUP[action_index]
        print(action.__name__)
        self.history.append(action)
        self.payload = action(self.payload)

    def reset(self):
        self.turns = 0
        self.payload = self.orig_payload

        print("reset payload: {}".format(self.payload))                

        self.observation_space = self.feature_extractor.extract(self.payload)

        return np.asarray(self.observation_space)

    def render(self, mode='human', close=False):
        pass
