from termcolor import colored
import logging

from gym_waf.envs.interfaces import ClassificationFailure
from .waf_env import WafEnv

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}


class WafScoreEnv(WafEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, payloads_file, maxturns=20, score_threshold=0.1, use_diff_reward=False):
        super().__init__(payloads_file, maxturns=maxturns)
        self.score_threshold = score_threshold
        self.use_diff_reward = use_diff_reward
        self.reward = None
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

        new_reward = self.score_threshold * 10. / max(self.score, self.score_threshold)
        if self.use_diff_reward:
            if self.reward is None:
                self.reward = old_reward = new_reward
            else:
                old_reward = self.reward
                self.reward = new_reward
            step_reward = new_reward - old_reward - self.turn_penalty
        else:
            step_reward = self._process_reward(new_reward)

        if episode_over:
            logging.debug("episode is over: reward = {}!".format(step_reward))

        return self.observation, step_reward, episode_over, \
            {"win": win, "original": self.orig_payload, "payload": self.payload, "history": self.history}

