import numpy as np
import gym
import gym_waf
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.models import load_model
from stable_baselines import PPO2


def evaluate(policy, env, iterations=100):
    success = []
    failed = []
    for i in range(iterations):
        logging.info("Evaluate: {}/{}".format(i+1, iterations))
        env.seed(np.random.randint(13337))
        obs = env.reset()
        for _ in range(env.maxturns):
            act = policy(obs)
            obs, reward, over, info = env.step(act)
            if over:
                break
        if info["win"]:
            success.append(info["original"])
        else:
            failed.append(info["original"])
    return success, failed  # evasion accuracy is len(success) / len(sha256_holdout)


def gen_dqn_policy(dqn_model):
    def f(obs):
        q_values = dqn_model.predict(obs)[0]
        act = boltzmann_action(q_values) # alternative: best_action
        return act
    return f


# option 1: Boltzmann sampling from Q-function network output
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
boltzmann_action = lambda x: np.argmax(np.random.multinomial(1, softmax(x).flatten()))
# option 2: maximize the Q value, ignoring stochastic action space
best_action = lambda x: np.argmax(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO2 trainer for gym-waf.')
    parser.add_argument('env_id', metavar='ENV', type=str,
                        help='env id (WafBrain-diff-v0, WafLibinjection-v0')
    parser.add_argument('agent', metavar='AGENT', type=str,
                        help='agent type (ppo, dqn, random)')
    parser.add_argument('-m', dest='model_file', metavar='FILE', type=str,
                        help='file path to load model (required except random agent)')
    parser.add_argument('-i', dest='eval_cnt', type=int, default=100,
                        help='number of episodes to evaluate (default 100)')
    args = parser.parse_args()

    env_name = args.env_id
    agent = args.agent
    model_file = args.model_file
    eval_cnt = args.eval_cnt

    if agent != 'random' and model_file is None:
        parser.print_help()

    env = gym.make(env_name)

    if agent == 'random':
        # baseline: choose actions at random
        random_policy = lambda obs: env.action_space.sample()
        random_success, _ = evaluate(random_policy, env)
        print("Success rate (random): {}\n".format(len(random_success) / eval_cnt))

    if agent == 'dqn':
        model = load_model(model_file)
        dqn_success, _ = evaluate(gen_dqn_policy(model), env, iterations=eval_cnt)
        print("Success rate (dqn): {}\n".format(len(dqn_success) / eval_cnt))

    if agent == 'ppo':
        model = PPO2.load(model_file)
        ppo_success, _ = evaluate(lambda obs: model.predict(obs)[0], env, iterations=eval_cnt)
        print("Success rate (ppo): {}\n".format(len(ppo_success) / eval_cnt))

    # let's compare scores
    # with open("log_test_all.txt", 'a') as logfile:
    #     logfile.write("Success rate (random chance): {}\n".format(len(random_success) / eval_iters))
    #     logfile.write("Success rate (dqn): {}\n".format(len(dqn_success) / eval_iters))
    #     logfile.write("Success rate (dqn): {}\n".format(len(dqn_score_success) / eval_iters))
