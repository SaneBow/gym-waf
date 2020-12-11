from termcolor import colored
import logging

from gym_waf.envs.interfaces import ClassificationFailure
from .waf_env import WafEnv

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}


class WafLabelEnv(WafEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_label(self, payload):
        raise NotImplementedError("_get_label not implemented")

    def _check_sqli(self, payload):
        try:
            label = self._get_label(payload)
        except ClassificationFailure:
            logging.warning("Failed to classify payload: {}".format(colored(repr(self.payload), 'red')))
            label = 0   # assume evasion due to implementation bug in classifier
        self.label = label
        return label

    def step(self, action_index):
        assert self.orig_payload is not None, "please reset() before step()"

        self.turns += 1
        self._take_action(action_index)

        self.observation = self.feature_extractor.extract(self.payload)

        win = False
        # get reward
        if not self._check_sqli(self.payload):
            reward = 10.
            episode_over = True
            win = True
            logging.debug("WIN with payload: {}".format(colored(repr(self.payload), 'green')))
        elif self.turns >= self.maxturns:
            # out of turns :(
            reward = 0.0
            episode_over = True
        else:
            reward = 0.0
            episode_over = False
        reward = self._process_reward(reward)

        if episode_over:
            logging.debug("episode is over: reward = {}!".format(reward))

        return self.observation, reward, episode_over, \
            {"win": win, "original": self.orig_payload, "payload": self.payload, "history": self.history}

