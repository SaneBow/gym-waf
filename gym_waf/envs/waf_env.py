import numpy as np
import gym
import random
from gym_waf.envs.features import SqlFeatureExtractor

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}

SEED = 0


class WafEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, payloads_file, maxturns=20, turn_penalty=0.2):
        """
        Base class for WAF env
        :param payloads: a list of payload strings
        :param maxturns: max mutation before env ends
        """
        self.action_space = gym.spaces.Discrete(len(ACTION_LOOKUP))
        self.maxturns = maxturns
        self.feature_extractor = SqlFeatureExtractor()
        self.history = []
        self.payload_list = None
        self.max_reward = 10.0
        self.min_reward = 0.0
        self.orig_payload = None
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.feature_extractor.shape, dtype=np.float32)
        self.turn_penalty = turn_penalty

        self.payload = None
        self.observation = None
        self.turns = 0

        self._load_payloads(payloads_file)

    def _load_payloads(self, filepath):
        try:
            with open(filepath, 'r') as f:
                self.payload_list = f.read().splitlines()
        except OSError as e:
            print("failed to load dataset from {}".format(filepath))
            raise

    def step(self, action_index):
        raise NotImplementedError("_step not implemented")

    def _check_sqli(self, payload):
        raise NotImplementedError("_check_sqli not implemented")

    def _take_action(self, action_index):
        assert action_index < len(ACTION_LOOKUP)
        action = ACTION_LOOKUP[action_index]
        # print(action.__name__)
        self.history.append(action)
        self.payload = action(self.payload, seed=SEED)

    def _process_reward(self, reward):
        reward = reward - self.turns * self.turn_penalty  # encourage fewer turns
        reward = max(min(reward, self.max_reward), self.min_reward)
        return reward

    def reset(self):
        self.turns = 0

        while True:     # until find one that is SQLi by the interface
            payload = random.choice(self.payload_list)
            if self._check_sqli(payload):
                self.orig_payload = self.payload = payload
                break

        # print("reset payload: {}".format(self.payload))

        self.observation = self.feature_extractor.extract(self.payload)

        return self.observation

    def render(self, mode='human', close=False):
        pass
