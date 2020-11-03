import numpy as np
import random
import time
from plot_util import plot_progress
from logger import Logger
from data_util.experiment_data_classes import Parameters


def train(width: int, length: int, params: Parameters, environment, visualize=False, plot=False, plot_interval=10, plot_moving_avg_period=100):
    q_table = np.zeros((width * length, 4))

    exploration_rate = params.start_exploration_rate
    for episode in range(params.num_episodes):
        if episode % 100 == 0:
            Logger.debug("EPISODE: ", episode)
        state = environment.reset_agent()
        done = False
        rewards_current_episode = 0
        max_reward_current_episode = 0

        for step in range(params.max_steps_per_episode):
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = random.choice(environment.get_agent_possible_actions())
            new_state, reward = environment.agent_perform_action(action)
            q_table[state, action] = q_table[state, action] * (1 - params.learning_rate) + params.learning_rate * (
                        reward + params.discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward
            if max_reward_current_episode < reward:
                max_reward_current_episode = reward

            if visualize:
                # environment.clear()
                # environment.render()
                environment.redraw_agent()
                time.sleep(0.05)

        exploration_rate = params.min_exploration_rate + (params.max_exploration_rate - params.min_exploration_rate) * np.exp(
            -params.exploration_decay_rate * episode)
        # print("Exploration Rate: " + exploration_rate.__str__())
        # print(max_reward_current_episode)
        params.rewards_all_episodes.append(rewards_current_episode)
        params.max_rewards_all_episodes.append(max_reward_current_episode)
        if episode % plot_interval == 0:
            plot_progress(params.rewards_all_episodes, exploration_rate, plot_moving_avg_period)

    return q_table, params
