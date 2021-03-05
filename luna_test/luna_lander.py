import gym
# from luna_test.deep_q_test import Agent
from luna_test.deep_q_test_two_networks import Agent
from plot_util import plot_progress
import numpy as np
from logger import Logger

visualize_training = False

env = gym.make('LunarLander-v2')
# env = gym.make('BipedalWalker-v3')
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, target_update=10, n_actions=4, eps_end=0.001, input_dims=[8], learning_rate=0.001, eps_dec=0.004)
# agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, target_update=10, n_actions=4, eps_end=0.01, input_dims=[24], learning_rate=0.001, eps_dec=0.0001)
# agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.02, input_dims=[8], learning_rate=0.001, eps_dec=0.0001)
scores, eps_history = [], []
n_games = 10000

for episode in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    # Logger.debug("Observation:", observation)
    agent.calculate_epsilon(episode)
    while not done:
        if visualize_training:
            env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn(episode)
        observation = observation_

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    # Logger.info('Episode', episode, 'Score %.2f' % score, 'Average score %.2f' % avg_score, 'Epsilon %.2f' % agent.epsilon)

    plot_progress(scores, exploration_rate=agent.epsilon, epsilon=eps_history)


while True:
    score = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_


env.close()

