from data_util.experiment_data_classes import DeepQParameters
from init_experiment_assistant import ExperimentAssistant
from experiment_scripts.experiment_save_repeat import ExperimentSaveRepeat
from logger import Logger
import sys
from terrain_generator.terrain_util import get_environment

folder_name = "2021_02_02_massive_different_epsilon"
file_name = "static_epsilon_2"
terrain_file = "test_2"
number_of_experiments = 20

environment = get_environment(terrain_file=terrain_file)

params = DeepQParameters(
            num_episodes=1250,
            max_steps_per_episode=80,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.999,
            target_update=25,

            # start_exploration_rate=1,
            start_exploration_rate=0.2,
            max_exploration_rate=1,
            # min_exploration_rate=0.001,
            min_exploration_rate=0.2,
            # exploration_decay_rate=0.005,
            exploration_decay_rate=0,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params, environment=environment)
experiment_handler.start_experiments()
