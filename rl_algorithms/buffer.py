import numpy as np
import random
import time
from plot_util import plot_progress
from logger import Logger
from data_util.experiment_data_classes import Parameters


class Buffer:

    def train(self, width: int, length: int, params: Parameters, environment,visualize=False, plot=False, plot_interval=10, plot_moving_avg_period=100):
        q_table = np.zeros((width * length, 4))

        exploration_rate = params.start_exploration_rate
        eps_history = []
        buffer = []  # list of (state, action, reward, state)-tuples
        for episode in range(params.num_episodes):
            if episode % 1000 == 0:
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
                new_state, reward, _ = environment.agent_perform_action(action)
                sars = (state, action, reward, new_state)

                buffer.append(sars)

                state = new_state
                rewards_current_episode += reward
                if max_reward_current_episode < reward:
                    max_reward_current_episode = reward

                if visualize:
                    # environment.clear()
                    # environment.render()
                    environment.redraw_agent()
                    time.sleep(0.04)

            # for index in range(len(buffer)):
            #     (buffer_state, buffer_action, buffer_reward, buffer_new_state) = buffer.pop(0)
            #     q_table[buffer_state, buffer_action] = q_table[buffer_state, buffer_action] * (1 - params.learning_rate) + params.learning_rate * (
            #                     buffer_reward + params.discount_rate * np.max(q_table[buffer_new_state, :]))

            random.shuffle(buffer)
            while len(buffer) > 0:
                (buffer_state, buffer_action, buffer_reward, buffer_new_state) = buffer.pop(0)
                q_table[buffer_state, buffer_action] = q_table[buffer_state, buffer_action] * (1 - params.learning_rate) + params.learning_rate * (
                                buffer_reward + params.discount_rate * np.max(q_table[buffer_new_state, :]))

            # exploration_rate = params.min_exploration_rate + (params.max_exploration_rate - params.min_exploration_rate) * np.exp(
            #     -params.exploration_decay_rate * episode)

            exploration_rate = params.min_exploration_rate if params.min_exploration_rate >= exploration_rate else params.min_exploration_rate + (
                    params.max_exploration_rate - params.min_exploration_rate) * np.exp(
                -params.exploration_decay_rate * episode)

            eps_history.append(exploration_rate)
            # print("Exploration Rate: " + exploration_rate.__str__())
            # print(max_reward_current_episode)
            params.rewards_all_episodes.append(rewards_current_episode)
            params.max_rewards_all_episodes.append(max_reward_current_episode)
            if episode % plot_interval == 0:
                # plot_progress(params.rewards_all_episodes, exploration_rate, plot_moving_avg_period, epsilon=eps_history)
                # if episode == 6000:
                #     plot_progress(params.rewards_all_episodes, exploration_rate, plot_moving_avg_period,
                #                   epsilon=eps_history, show=True)
                    # plot_progress(params.rewards_all_episodes, average_period=plot_moving_avg_period,
                    #               epsilon=eps_history, width=600, height=300, show=True)
                plot_progress(params.rewards_all_episodes, average_period=plot_moving_avg_period, epsilon=eps_history)
                # plot_progress(params.rewards_all_episodes, average_period=plot_moving_avg_period, epsilon=eps_history, width=600, height=300)

        return q_table, params
