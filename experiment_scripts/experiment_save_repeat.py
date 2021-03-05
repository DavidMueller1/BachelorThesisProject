from data_util.experiment_data_classes import DeepQParameters
from visualization_engine_3d.engine import Engine3D
from rl_algorithms import deep_q_2
from rl_algorithms import deep_q_2_modified_reward
from logger import Logger
from pathlib import Path
import sys
import os
import dill
import pickle
import traceback

TARGET_PATH = "../data/learned/new_experiments_path/"

class ExperimentSaveRepeat:
    def __init__(self, number_of_experiments, dir_name, file_name, params, environment):
        self.number_of_experiments = number_of_experiments
        self.dir_name = dir_name
        self.file_name = file_name
        self.params = params
        self.environment = environment

        self.result_data = []

    def start_experiments(self):
        n = 1
        while n <= self.number_of_experiments:
            Logger.status("----------------------------------")
            Logger.status("Beginning experiment %d/%d" % (n, self.number_of_experiments))
            try:
                trained_net, params = self.execute_experiment()
                self.save_single_data(params, params.rewards_all_episodes, trained_net, n)
                n += 1
            except Exception as e:
                Logger.error("An error occurred:", e)
                Logger.info("Trying to repeat experiment...")

        #     self.result_data.append(self.execute_experiment().rewards_all_episodes)
        #     Logger.status("Experiment done.\n")
        # except Exception as e:
        #     # e = sys.exc_info()
        #     Logger.error("An error occured:", e)
        #     traceback.print_exc()
        # finally:
        #     Logger.info("Saving progress of %d/%d runs..." % (n + 1, self.number_of_experiments))
        #     self.save_data()

        # try:
        #     for n in range(0, self.number_of_experiments):
        #         Logger.status("----------------------------------")
        #         Logger.status("Initializing experiment %d/%d" % (n + 1, self.number_of_experiments))
        #         self.result_data.append(self.execute_experiment().rewards_all_episodes)
        #         Logger.status("Experiment done.\n")
        # except Exception as e:
        #     # e = sys.exc_info()
        #     Logger.error("An error occured:", e)
        #     traceback.print_exc()
        # finally:
        #     Logger.info("Saving progress of %d/%d runs..." % (n + 1, self.number_of_experiments))
        #     self.save_data()

    def execute_experiment(self):
        trained_net, params = deep_q_2.train(width=self.environment.grid_width,
                                   length=self.environment.grid_height,
                                   params=self.params,
                                   environment=self.environment,
                                   visualize=False,
                                   show_path_interval=100,
                                   plot=True,
                                   plot_interval=20,
                                   plot_moving_avg_period=100)

        return trained_net, params

    def save_single_data(self, params, single_result_data, trained_net, num):
        single_result = SingleResultData(params, single_result_data, trained_net)
        os.chdir(os.path.dirname(__file__))
        Path(TARGET_PATH + self.dir_name).mkdir(parents=True, exist_ok=True)

        with open(TARGET_PATH + self.dir_name + "/" + self.file_name + "_exp" + str(num).zfill(2), 'wb') as f:
            dill.dump(single_result, f, pickle.HIGHEST_PROTOCOL)

    # def save_data(self):
    #     result = ResultData(self.params, self.result_data)
    #     os.chdir(os.path.dirname(__file__))
    #
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


class AllResultData:
    def __init__(self, params, result_data):
        self.params = params
        self.result_data = result_data