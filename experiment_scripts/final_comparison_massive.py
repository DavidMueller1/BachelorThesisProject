from data_util.experiment_data_classes import DeepQParameters
from init_experiment_assistant import ExperimentAssistant
from experiment_scripts.experiment_save_repeat import ExperimentSaveRepeat
from logger import Logger
import sys
from terrain_generator.terrain_util import get_environment
from visualization_engine_3d.engine import Rewards, States

SUBFOLDER = "2021_04_03_redo_comparison_terrain_2000_episodes"

EPISODES = 2000
STEPS = 80
REPLAY_BUFFER = 20000
BATCH = 32

LEARNING_RATE = 0.001
DISCOUNT = 0.999
UPDATE_RATE = 25


# folder_name = SUBFOLDER + "/2021_04_02_eps_random_in_reward_1"
# file_name = "redo_spiral_eps_in_reward_1"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
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
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=True)
# experiment_handler.start_experiments()
#
#
# folder_name = SUBFOLDER + "/2021_04_02_eps_random_in_reward_05"
# file_name = "redo_spiral_eps_in_reward_05"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
#
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             min_exploration_rate=0.001,
#             exploration_decay_rate=0.005,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=True)
# experiment_handler.start_experiments()
#
#
# folder_name = SUBFOLDER + "/2021_04_03_decaying_eps_start_1"
# file_name = "decaying_eps_start_1"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
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
#
#
# folder_name = SUBFOLDER + "/2021_04_03_decaying_eps_start_05"
# file_name = "decaying_eps_start_05"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
#
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
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
#
#
# folder_name = SUBFOLDER + "/2021_04_03_static_eps_00"
# file_name = "static_eps_00"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
#
#             start_exploration_rate=0,
#             max_exploration_rate=0,
#             min_exploration_rate=0,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
# experiment_handler.start_experiments()
#
#
# folder_name = SUBFOLDER + "/2021_04_03_static_eps_02"
# file_name = "static_eps_02"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
#
#             start_exploration_rate=0.2,
#             max_exploration_rate=0.2,
#             min_exploration_rate=0.2,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=8, modified_reward=False)
# experiment_handler.start_experiments()
#
#
# folder_name = SUBFOLDER + "/2021_04_03_static_eps_05"
# file_name = "static_eps_05"
# terrain_file = "test_2"
# number_of_experiments = 20
#
# environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)
#
# params = DeepQParameters(
#             num_episodes=EPISODES,
#             max_steps_per_episode=STEPS,
#             replay_buffer_size=REPLAY_BUFFER,
#             batch_size=BATCH,
#
#             learning_rate=LEARNING_RATE,
#             discount_rate=DISCOUNT,
#             target_update=UPDATE_RATE,
#
#             start_exploration_rate=0.5,
#             max_exploration_rate=0.5,
#             min_exploration_rate=0.5,
#             exploration_decay_rate=0,
#
#             rewards_all_episodes=[],
#             max_rewards_all_episodes=[],
#             max_reward_average=0
#         )
#
# experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
#                                           file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=False)
# experiment_handler.start_experiments()


folder_name = SUBFOLDER + "/2021_04_02_eps_in_reward_1"
file_name = "redo_spiral_eps_in_reward_1"
terrain_file = "test_2"
number_of_experiments = 20

environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)

params = DeepQParameters(
            num_episodes=EPISODES,
            max_steps_per_episode=STEPS,
            replay_buffer_size=REPLAY_BUFFER,
            batch_size=BATCH,

            learning_rate=LEARNING_RATE,
            discount_rate=DISCOUNT,
            target_update=UPDATE_RATE,

            start_exploration_rate=1,
            max_exploration_rate=1,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                          file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=True)
experiment_handler.start_experiments()


folder_name = SUBFOLDER + "/2021_04_02_eps_in_reward_05"
file_name = "redo_spiral_eps_in_reward_05"
terrain_file = "test_2"
number_of_experiments = 20

environment = get_environment(terrain_file=terrain_file, reward_val=Rewards.Spiral, state_val=States.Default)

params = DeepQParameters(
            num_episodes=EPISODES,
            max_steps_per_episode=STEPS,
            replay_buffer_size=REPLAY_BUFFER,
            batch_size=BATCH,

            learning_rate=LEARNING_RATE,
            discount_rate=DISCOUNT,
            target_update=UPDATE_RATE,

            start_exploration_rate=0.5,
            max_exploration_rate=0.5,
            min_exploration_rate=0.001,
            exploration_decay_rate=0.005,

            rewards_all_episodes=[],
            max_rewards_all_episodes=[],
            max_reward_average=0
        )

experiment_handler = ExperimentSaveRepeat(number_of_experiments=number_of_experiments, dir_name=folder_name,
                                          file_name=file_name, params=params, environment=environment, start_number=1, modified_reward=True)
experiment_handler.start_experiments()