from data_util.experiment_data_classes import DeepQParameters
from init_experiment_assistant import ExperimentAssistant
from luna_test.luna_lander_module import ExperimentSaveRepeatLuna
from logger import Logger
import sys
from terrain_generator.terrain_util import get_environment

folder_name = "2021_04_14_decaying_epsilon_start_1_lr_2"
file_name = "decaying_epsilon_start_1"
number_of_experiments = 10


params = DeepQParameters(
            num_episodes=2500,
            max_steps_per_episode=0,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.002,
            discount_rate=0.99,
            target_update=10,

            # start_exploration_rate=1,
            start_exploration_rate=1,
            max_exploration_rate=1,
            # min_exploration_rate=0.001,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,
            # exploration_decay_rate=0,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                              file_name=file_name, params=params, start_number=1, epsilon_in_reward=False)
experiment_handler.start_experiments()


# folder_name = "2021_03_11_decaying_epsilon_start_05"
# file_name = "decaying_epsilon_start_05"
# number_of_experiments = 20
#
#
# params = DeepQParameters(
#             num_episodes=2200,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.99,
#             target_update=10,
#
#             # start_exploration_rate=1,
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             # min_exploration_rate=0.001,
#             min_exploration_rate=0.001,
#             exploration_decay_rate=0.005,
#             # exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params)
# experiment_handler.start_experiments()


# folder_name = "2021_03_11_static_epsilon_start_1"
# file_name = "decaying_epsilon_start_1"
# number_of_experiments = 20
#
#
# params = DeepQParameters(
#             num_episodes=2200,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.99,
#             target_update=10,
#
#             start_exploration_rate=1,
#             # start_exploration_rate=0,
#             max_exploration_rate=1,
#             min_exploration_rate=0.001,
#             # min_exploration_rate=0,
#             exploration_decay_rate=0.004,
#             # exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params)
# experiment_handler.start_experiments()


# folder_name = "2021_03_26_static_epsilon_00"
# file_name = "static_epsilon_00"
# number_of_experiments = 10
#
#
# params = DeepQParameters(
#             num_episodes=2500,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.99,
#             target_update=10,
#
#             # start_exploration_rate=1,
#             start_exploration_rate=0,
#             # start_exploration_rate=0.2,
#             max_exploration_rate=0,
#             # max_exploration_rate=0.2,
#             min_exploration_rate=0,
#             # min_exploration_rate=0.2,
#             # min_exploration_rate=0,
#             # exploration_decay_rate=0.004,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params, start_number=1)
# experiment_handler.start_experiments()



# folder_name = "2021_03_11_static_epsilon_02"
# file_name = "static_epsilon_02"
# number_of_experiments = 20
#
#
# params = DeepQParameters(
#             num_episodes=2200,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.99,
#             target_update=10,
#
#             # start_exploration_rate=1,
#             # start_exploration_rate=0,
#             start_exploration_rate=0.2,
#             # max_exploration_rate=0,
#             max_exploration_rate=0.2,
#             # min_exploration_rate=0,
#             min_exploration_rate=0.2,
#             # min_exploration_rate=0,
#             # exploration_decay_rate=0.004,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                               file_name=file_name, params=params, start_number=3)
# experiment_handler.start_experiments()

# TODO Check if other experiment is with 0.4 or 0.5

## EPSILON IN REWARD
# folder_name = "2021_03_26_epsilon_in_reward_01"
# file_name = "epsilon_in_reward_01"
# number_of_experiments = 10
#
#
# params = DeepQParameters(
#             num_episodes=2500,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.99,
#             target_update=10,
#
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             min_exploration_rate=0,
#             exploration_decay_rate=0.004,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                               file_name=file_name, params=params, start_number=7, epsilon_in_reward=True)
# experiment_handler.start_experiments()


# folder_name = "2021_03_26_epsilon_in_reward_02"
# file_name = "epsilon_in_reward_02"
# number_of_experiments = 10
#
#
# params = DeepQParameters(
#             num_episodes=2500,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.002,
#             discount_rate=0.99,
#             target_update=10,
#
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             min_exploration_rate=0,
#             exploration_decay_rate=0.004,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                               file_name=file_name, params=params, start_number=6, epsilon_in_reward=True)
# experiment_handler.start_experiments()


# folder_name = "2021_03_26_epsilon_in_reward_relu"
# file_name = "epsilon_in_reward_relu"
# number_of_experiments = 10
#
#
# params = DeepQParameters(
#             num_episodes=2500,
#             max_steps_per_episode=0,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.002,
#             discount_rate=0.99,
#             target_update=100,
#
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             min_exploration_rate=0,
#             exploration_decay_rate=0.004,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeatLuna(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                               file_name=file_name, params=params, start_number=1, epsilon_in_reward=True)
# experiment_handler.start_experiments()