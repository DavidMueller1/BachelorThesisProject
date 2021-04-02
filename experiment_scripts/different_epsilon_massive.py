from data_util.experiment_data_classes import DeepQParameters
from init_experiment_assistant import ExperimentAssistant
from experiment_scripts.experiment_save_repeat import ExperimentSaveRepeat
from logger import Logger
import sys
from terrain_generator.terrain_util import get_environment
from visualization_engine_3d.engine import Rewards, States

# TODO redo
# folder_name = "2021_03_15_epsilon_in_reward_start_1_num_2"
# file_name = "epsilon_in_reward_start_1"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file)
#
# params = DeepQParameters(
#             num_episodes=1250,
#             # num_episodes=50,
#             max_steps_per_episode=80,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=1,
#             # start_exploration_rate=0.0,
#             max_exploration_rate=1,
#             min_exploration_rate=0.001,
#             # min_exploration_rate=0.0,
#             exploration_decay_rate=0.005,
#             # exploration_decay_rate=0.003,
#             # exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=10)
# experiment_handler.start_experiments()



# folder_name = "2021_03_09_epsilon_in_reward_start_05"
# file_name = "epsilon_in_reward_start_05"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file)
#
# params = DeepQParameters(
#             num_episodes=1250,
#             # num_episodes=50,
#             max_steps_per_episode=80,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=0.5,
#             # start_exploration_rate=0.0,
#             max_exploration_rate=0.5,
#             min_exploration_rate=0.001,
#             # min_exploration_rate=0.0,
#             exploration_decay_rate=0.005,
#             # exploration_decay_rate=0.003,
#             # exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1)
# experiment_handler.start_experiments()



# folder_name = "2021_03_09_decaying_epsilon_start_1"
# file_name = "decaying_epsilon_start_1"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file)
#
# params = DeepQParameters(
#             num_episodes=1250,
#             # num_episodes=50,
#             max_steps_per_episode=80,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=1,
#             # start_exploration_rate=0.0,
#             max_exploration_rate=1,
#             # min_exploration_rate=0.001,
#             min_exploration_rate=0.0,
#             exploration_decay_rate=0.005,
#             # exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params, environment=environment, start_number=1)
# experiment_handler.start_experiments()
#
#
#
# folder_name = "2021_03_09_static_epsilon_00"
# file_name = "static_epsilon_00"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file)
#
# params = DeepQParameters(
#             num_episodes=1250,
#             # num_episodes=50,
#             max_steps_per_episode=80,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             # start_exploration_rate=1,
#             start_exploration_rate=0.0,
#             max_exploration_rate=0,
#             # min_exploration_rate=0.001,
#             min_exploration_rate=0.0,
#             # exploration_decay_rate=0.005,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params, environment=environment, start_number=1)
# experiment_handler.start_experiments()
#
#
#
# folder_name = "2021_03_09_static_epsilon_02"
# file_name = "static_epsilon_02"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file)
#
# params = DeepQParameters(
#             num_episodes=1250,
#             # num_episodes=50,
#             max_steps_per_episode=80,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             # start_exploration_rate=1,
#             start_exploration_rate=0.2,
#             max_exploration_rate=0.2,
#             # min_exploration_rate=0.001,
#             min_exploration_rate=0.2,
#             # exploration_decay_rate=0.005,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params, environment=environment, start_number=1)
# experiment_handler.start_experiments()
#
#
#
# folder_name = "2021_03_09_static_epsilon_05"
# file_name = "static_epsilon_05"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file)
#
# params = DeepQParameters(
#             num_episodes=1250,
#             # num_episodes=50,
#             max_steps_per_episode=80,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             # start_exploration_rate=1,
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             # min_exploration_rate=0.001,
#             min_exploration_rate=0.5,
#             # exploration_decay_rate=0.005,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name, file_name=file_name, params=params, environment=environment, start_number=1)
# experiment_handler.start_experiments()


# folder_name = "2021_03_31_simple"
# file_name = "simple"
# terrain_file = "test_2"
# number_of_experiments = 1
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Delta, state_val=States.Minimal, random_spawn=False)
#
# params = DeepQParameters(
#             num_episodes=10000,
#             max_steps_per_episode=100,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=1,
#             max_exploration_rate=1,
#             min_exploration_rate=0.001,
#             exploration_decay_rate=0.001,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
# experiment_handler.start_experiments()


# folder_name = "2021_04_02_heights_in_states"
# file_name = "heights_in_states"
# terrain_file = "test_2"
# number_of_experiments = 1
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Delta, state_val=States.AdjacentHeights, random_spawn=False)
#
# params = DeepQParameters(
#             num_episodes=1500,
#             max_steps_per_episode=100,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=1,
#             max_exploration_rate=1,
#             min_exploration_rate=0.001,
#             exploration_decay_rate=0.005,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
# experiment_handler.start_experiments()


# folder_name = "2021_04_02_random_spawn"
# file_name = "random_spawn"
# terrain_file = "test_2"
# number_of_experiments = 1
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Delta, state_val=States.RelativePosAndHeights, random_spawn=True)
#
# params = DeepQParameters(
#             num_episodes=1500,
#             max_steps_per_episode=100,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=1,
#             max_exploration_rate=1,
#             min_exploration_rate=0.001,
#             exploration_decay_rate=0.005,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
# experiment_handler.start_experiments()


# folder_name = "2021_04_02_random_spawn_exploration"
# file_name = "random_spawn_exploration"
# terrain_file = "test_2"
# number_of_experiments = 1
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Delta, state_val=States.RelativePosAndHeightsAndHighestPointAndSteps, random_spawn=True)
#
# params = DeepQParameters(
#             num_episodes=1500,
#             max_steps_per_episode=100,
#             replay_buffer_size=20000,
#             batch_size=32,
#
#             learning_rate=0.001,
#             discount_rate=0.999,
#             target_update=25,
#
#             start_exploration_rate=1,
#             max_exploration_rate=1,
#             min_exploration_rate=0.001,
#             exploration_decay_rate=0.005,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
# experiment_handler.start_experiments()


folder_name = "2021_04_02_random_spawn_exploration"
file_name = "random_spawn_exploration"
terrain_file = "test_2"
number_of_experiments = 1

environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default, random_spawn=True)

params = DeepQParameters(
            num_episodes=1500,
            max_steps_per_episode=100,
            replay_buffer_size=20000,
            batch_size=32,

            learning_rate=0.001,
            discount_rate=0.999,
            target_update=25,

            start_exploration_rate=1,
            max_exploration_rate=1,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                          file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
experiment_handler.start_experiments()