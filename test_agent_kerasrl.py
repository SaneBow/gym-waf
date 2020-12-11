import numpy as np
import gym
import gym_waf
import os
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.models import load_model


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
    return success, failed # evasion accuracy is len(success) / len(sha256_holdout)


def gen_policy_func(model):
    def f(obs):
        # first, get features from bytez
        q_values = model.predict(obs)[0]
        act = boltzmann_action(q_values) # alternative: best_action
        return act
    return f


# option 1: Boltzmann sampling from Q-function network output
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
boltzmann_action = lambda x: np.argmax(np.random.multinomial(1, softmax(x).flatten()))
# option 2: maximize the Q value, ignoring stochastic action space
best_action = lambda x: np.argmax(x)


if __name__ == '__main__':
    env_name = 'WafBrain-v0'
    env = gym.make(env_name)

    eval_iters = 100

    # baseline: choose actions at random
    random_policy = lambda obs: env.action_space.sample()
    random_success, _ = evaluate(random_policy, env)

    save_dir = 'trained_model'

    # dqn = load_model(os.path.join(save_dir, 'dqn.h5'))
    # dqn_success, _ = evaluate(gen_policy_func(dqn), env)
    #
    # dqn_score = load_model(os.path.join(save_dir, 'dqn_score.h5'))
    # dqn_score_success, _ = evaluate(gen_policy_func(dqn_score), env)

    # let's compare scores
    # with open("log_test_all.txt", 'a') as logfile:
    #     logfile.write("Success rate (random chance): {}\n".format(len(random_success) / eval_iters))
    #     logfile.write("Success rate (dqn): {}\n".format(len(dqn_success) / eval_iters))
    #     logfile.write("Success rate (dqn): {}\n".format(len(dqn_score_success) / eval_iters))

    print("Success rate of random policy: {}\n".format(len(random_success) / eval_iters))
    # print("Success rate (dqn): {}\n".format(len(dqn_success) / eval_iters))
    # print("Success rate (dqn): {}\n".format(len(dqn_score_success) / eval_iters))
