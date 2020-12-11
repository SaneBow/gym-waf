from termcolor import colored
import logging

from gym_waf.envs.interfaces import ClassificationFailure
from .waf_env import WafEnv

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}


class WafScoreEnv(WafEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score = None

    def _get_score(self, payload):
        raise NotImplementedError("_get_score not implemented")

    def _check_sqli(self, payload):
        try:
            score, is_sqli = self._get_score(payload)
        except ClassificationFailure:
            logging.warning("Failed to classify payload: {}".format(colored(repr(self.payload), 'red')))
            score = 0.01
            is_sqli = False
        self.score = score
        return is_sqli

    def step(self, action_index):
        assert self.orig_payload is not None, "please reset() before step()"

        self.turns += 1
        self._take_action(action_index)

        self.observation = self.feature_extractor.extract(self.payload)

        win = False
        # get reward
        if not self._check_sqli(self.payload):
            # we win!
            episode_over = True
            win = True
            logging.debug("WIN with payload: {}".format(colored(repr(self.payload), 'green')))
        elif self.turns >= self.maxturns:
            # out of turns :(
            episode_over = True
        else:
            episode_over = False
        reward = 1. / max(self.score, 0.01)
        reward = self._process_reward(reward)

        if episode_over:
            logging.debug("episode is over: reward = {}!".format(reward))

        return self.observation, reward, episode_over, \
            {"win": win, "original": self.orig_payload, "payload": self.payload, "history": self.history}

