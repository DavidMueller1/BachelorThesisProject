import os
import pickle
import traceback

import dill
import gym
# from luna_test.deep_q_test import Agent
from luna_test.deep_q_test_two_networks import Agent
from plot_util import plot_progress
import numpy as np
from logger import Logger


class ExperimentSaveRepeatLuna:
    def __init__(self, number_of_experiments, dir_name, file_name, params, visualize=False):
        self.number_of_experiments = number_of_experiments
        self.dir_name = dir_name
        self.file_name = file_name
        self.params = params
        self.visualize = visualize
        self.env = gym.make('LunarLander-v2')

        self.result_data = []

    def start_experiments(self):
        try:
            for n in range(0, self.number_of_experiments):
                Logger.status("----------------------------------")
                Logger.status("Initializing experiment %d/%d" % (n + 1, self.number_of_experiments))
                self.result_data.append(self.execute_experiment().rewards_all_episodes)
                Logger.status("Experiment done.\n")
        except Exception as e:
            # e = sys.exc_info()
            Logger.error("An error occured:", e)
            traceback.print_exc()
        finally:
            Logger.info("Saving progress of %d/%d runs..." % (n + 1, self.number_of_experiments))
            self.save_data()

    def execute_experiment(self):
        agent = Agent(gamma=self.params.discount_rate, epsilon=self.params.start_exploration_rate, batch_size=self.params.batch_size,
                      target_update=self.params.target_update, n_actions=4, eps_end=self.params.min_exploration_rate,
                      input_dims=[8], learning_rate=self.params.learning_rate, eps_dec=self.params.exploration_decay_rate)
        # agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, target_update=10, n_actions=4, eps_end=0.01, input_dims=[24], learning_rate=0.001, eps_dec=0.0001)
        # agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.02, input_dims=[8], learning_rate=0.001, eps_dec=0.0001)
        scores, eps_history = [], []
        n_games = self.params.num_episodes

        curr_params = self.params

        for episode in range(n_games):
            score = 0
            done = False
            observation = self.env.reset()
            # Logger.debug("Observation:", observation)
            agent.calculate_epsilon(episode)
            while not done:
                if self.visualize:
                    self.env.render()
                action = agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn(episode)
                observation = observation_

            scores.append(score)
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])

            # Logger.info('Episode', episode, 'Score %.2f' % score, 'Average score %.2f' % avg_score, 'Epsilon %.2f' % agent.epsilon)

            plot_progress(scores, exploration_rate=agent.epsilon, epsilon=eps_history, epsilon_fac=200)

        curr_params.rewards_all_episodes = scores
        # self.params.max_reward_average = max_average

        return curr_params

    def save_data(self):
        result = ResultData(self.params, self.result_data)
        os.chdir(os.path.dirname(__file__))
        with open("../data/learned/" + self.dir_name + "/" + self.file_name, 'wb') as f:
            dill.dump(result, f, pickle.HIGHEST_PROTOCOL)


class ResultData:
    def __init__(self, params, result_data):
        self.params = params
        self.result_data = result_data

