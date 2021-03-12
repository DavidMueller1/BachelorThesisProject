import os
import pickle
import traceback
from pathlib import Path
import dill
import gym
# from luna_test.deep_q_test import Agent
from luna_test.deep_q_test_two_networks import Agent
from plot_util import plot_progress
from plot_util import get_current_average
import numpy as np
from logger import Logger

TARGET_PATH = "../data/learned/new_experiments_path_luna_lander/"


class ExperimentSaveRepeatLuna:
    def __init__(self, number_of_experiments, dir_name, file_name, params, visualize=False, start_number=1):
        self.number_of_experiments = number_of_experiments
        self.dir_name = dir_name
        self.file_name = file_name
        self.params = params
        self.visualize = visualize
        self.start_number = start_number
        self.env = gym.make('LunarLander-v2')

        self.result_data = []

    def start_experiments(self):
        n = self.start_number

        while n <= self.number_of_experiments:
            Logger.status("----------------------------------")
            Logger.status("Initializing experiment %d/%d" % (n, self.number_of_experiments))
            try:
                trained_net, params = self.execute_experiment()
                self.save_single_data(params, params.rewards_all_episodes, trained_net, n)
                n += 1
                Logger.status("Experiment done.\n")
            except Exception as e:
                # e = sys.exc_info()
                Logger.error("An error occured:", e)
                traceback.print_exc()
        Logger.status("\nALL EXPERIMENTS DONE\n\n")

    def execute_experiment(self):
        agent = Agent(gamma=self.params.discount_rate, epsilon=self.params.start_exploration_rate, batch_size=self.params.batch_size,
                      target_update=self.params.target_update, n_actions=4, eps_end=self.params.min_exploration_rate,
                      input_dims=[8], learning_rate=self.params.learning_rate, eps_dec=self.params.exploration_decay_rate)
        # agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, target_update=10, n_actions=4, eps_end=0.01, input_dims=[24], learning_rate=0.001, eps_dec=0.0001)
        # agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.02, input_dims=[8], learning_rate=0.001, eps_dec=0.0001)
        scores, eps_history = [], []
        n_games = self.params.num_episodes

        curr_params = self.params
        max_average = -99999

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

            if episode % 20 == 0:
                plot_progress(scores, exploration_rate=agent.epsilon, epsilon=eps_history, epsilon_fac=200)

            current_average = get_current_average(values=scores)
            if max_average < current_average or episode == 100:
                max_average = current_average
                # Logger.info("New max average:", max_average)
                agent.best_net.load_state_dict(agent.policy_net.state_dict())

        curr_params.rewards_all_episodes = scores
        # self.params.max_reward_average = max_average

        return agent.best_net, curr_params

    def save_single_data(self, params, single_result_data, trained_net, num):
        single_result = SingleResultData(params, single_result_data, trained_net)
        os.chdir(os.path.dirname(__file__))
        Path(TARGET_PATH + self.dir_name).mkdir(parents=True, exist_ok=True)
        file = TARGET_PATH + self.dir_name + "/" + self.file_name + "_exp" + str(num).zfill(2)

        with open(file, 'wb') as f:
            dill.dump(single_result, f, pickle.HIGHEST_PROTOCOL)
        Logger.status("Experiment", num, "saved as \"" + file + "\"")

    # def save_data(self):
    #     result = ResultData(self.params, self.result_data)
    #     os.chdir(os.path.dirname(__file__))
    #     with open("../data/learned/" + self.dir_name + "/" + self.file_name, 'wb') as f:
    #         dill.dump(result, f, pickle.HIGHEST_PROTOCOL)


class ResultData:
    def __init__(self, params, result_data):
        self.params = params
        self.result_data = result_data


class SingleResultData:
    def __init__(self, params, single_result_data, trained_net):
        self.params = params
        self.single_result_data = single_result_data
        self.trained_net = trained_net