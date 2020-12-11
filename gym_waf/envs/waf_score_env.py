from termcolor import colored

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
            print("Failed to classify payload: ", colored(repr(self.payload), 'red'))
            score = 0.01
            is_sqli = False
        self.score = score
        return is_sqli

    def step(self, action_index):
        assert self.orig_payload is not None, "please reset() before step()"

        self.turns += 1
        self._take_action(action_index)

        self.observation = self.feature_extractor.extract(self.payload)

        # get reward
        if not self._check_sqli(self.payload):
            # we win!
            episode_over = True
            # print("WIN with payload:", colored(repr(self.payload), 'green'))
        elif self.turns >= self.maxturns:
            # out of turns :(
            episode_over = True
        else:
            episode_over = False
        reward = 1. / self.score
        reward = self._process_reward(reward)

        # if episode_over:
        #     print("episode is over: reward = {}!".format(reward))

        return self.observation, reward, episode_over, {}

