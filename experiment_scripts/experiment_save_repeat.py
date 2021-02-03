from data_util.experiment_data_classes import DeepQParameters
from visualization_engine_3d.engine import Engine3D
from rl_algorithms import deep_q_2
from logger import Logger
import sys
import os
import dill
import pickle
import traceback


class ExperimentSaveRepeat:
    def __init__(self, number_of_experiments, dir_name, file_name, params, environment):
        self.number_of_experiments = number_of_experiments
        self.dir_name = dir_name
        self.file_name = file_name
        self.params = params
        self.environment = environment

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
        _, params = deep_q_2.train(width=self.environment.grid_width,
                                   length=self.environment.grid_height,
                                   params=self.params,
                                   environment=self.environment,
                                   visualize=False,
                                   show_path_interval=100,
                                   plot=True,
                                   plot_interval=20,
                                   plot_moving_avg_period=100)

        return params

    def save_data(self):
        result = ResultData(self.params, self.result_data)
        os.chdir(os.path.dirname(__file__))
        with open("../data/learned/" + self.dir_name + "/" + self.file_name, 'wb') as f:
            dill.dump(result, f, pickle.HIGHEST_PROTOCOL)


class ResultData:
    def __init__(self, params, result_data):
        self.params = params
        self.result_data = result_data