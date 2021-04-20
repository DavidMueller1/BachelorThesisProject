from data_util.experiment_data_classes import DeepQParameters
from init_experiment_assistant import ExperimentAssistant
from luna_test.luna_lander_module import ExperimentSaveRepeatLuna
from logger import Logger
import sys
from terrain_generator.terrain_util import get_environment

folder_name = "2021_04_20_decaying_epsilon_start_1"
file_name = "decaying_epsilon_start_1"
number_of_experiments = 10

params = DeepQParameters(
            num_episodes=1800,
            max_steps_per_episode=0,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.99,
            target_update=10,

            start_exploration_rate=1,
            max_exploration_rate=1,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                              file_name=file_name, params=params, start_number=3, epsilon_in_reward=False)
experiment_handler.start_experiments()



folder_name = "2021_04_20_static_epsilon_00"
file_name = "static_epsilon_00"
number_of_experiments = 10

params = DeepQParameters(
            num_episodes=1800,
            max_steps_per_episode=0,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.99,
            target_update=10,

            start_exploration_rate=0,
            max_exploration_rate=0,
            min_exploration_rate=0,
            exploration_decay_rate=0,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                              file_name=file_name, params=params, start_number=1, epsilon_in_reward=False)
experiment_handler.start_experiments()



folder_name = "2021_04_20_epsilon_in_reward_1"
file_name = "epsilon_in_reward_1"
number_of_experiments = 10

params = DeepQParameters(
            num_episodes=1800,
            max_steps_per_episode=0,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.99,
            target_update=10,

            start_exploration_rate=1,
            max_exploration_rate=1,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                              file_name=file_name, params=params, start_number=1, epsilon_in_reward=True)
experiment_handler.start_experiments()



folder_name = "2021_04_20_epsilon_in_reward_05"
file_name = "epsilon_in_reward_05"
number_of_experiments = 10

params = DeepQParameters(
            num_episodes=1800,
            max_steps_per_episode=0,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.99,
            target_update=10,

            start_exploration_rate=0.5,
            max_exploration_rate=0.5,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                              file_name=file_name, params=params, start_number=1, epsilon_in_reward=True)
experiment_handler.start_experiments()