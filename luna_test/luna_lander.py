import gym
from luna_test.deep_q_test import Agent
from plot_util import plot_progress
import numpy as np

env = gym.make('LunarLander-v2')
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], learning_rate=0.001, eps_dec=0.0001)
# agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.02, input_dims=[8], learning_rate=0.001, eps_dec=0.0001)
scores, eps_history = [], []
n_games = 1500

for i in range(n_games):
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

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    print('Episode', i, 'Score %.2f' % score, 'Average score %.2f' % avg_score, 'Epsilon %.2f' % agent.epsilon)

    plot_progress(scores, agent.epsilon)


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

