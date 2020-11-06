from termcolor import colored

from gym_waf.envs.interfaces import LocalInterface, ClassificationFailure
from gym_waf.envs.features import SqlFeatureExtractor
from .waf_env import WafEnv

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}

# change this to function to the AV engine to attack
# function should be of the form
# def label_function( bytez ):
#    # returns 0.0 if benign else 1.0 if malware


class WafScoreEnv(WafEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, payload, score_threshold, maxturns=20):
        super(WafScoreEnv, self).__init__(payload, maxturns)        
        interface = LocalInterface()
        self.score_function = interface.get_score   # need to override
        self.score_threshold = score_threshold

    def step(self, action_index):
        self.turns += 1
        self._take_action(action_index)

        # get reward
        try:
            self.score = self.score_function(self.payload)
        except ClassificationFailure:
            print("Failed to classify payload: ", colored(repr(self.payload), 'red'))
            reward = 10.    # assume evasion due to implementation bug in classifier
            episode_over = True
        else:
            self.observation_space = self.feature_extractor.extract(self.payload)
            if self.score < self.score_threshold:
                # we win!
                reward = 1. / self.score
                episode_over = True
                print("WIN with payload:", colored(repr(self.payload), 'green'))
                
            elif self.turns >= self.maxturns:
                # out of turns :(
                reward = 0.0
                episode_over = True
            else:
                reward = 1. / self.score
                episode_over = False

        if episode_over:
            print("episode is over: reward = {}!".format(reward))

        return self.observation_space, reward, episode_over, {}

