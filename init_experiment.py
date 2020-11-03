from visualization_engine_3d import engine as render_engine
from init_terrain import generate_random_terrain
from data_util.experiment_data_classes import Terrain
from data_util.experiment_data_classes import Parameters
from data_util.data_loader import load_terrain
from data_util.data_saver import save_terrain
from rl_algorithms.simple import train
from logger import Logger
import random
import numpy as np
import time


plot_training_progress = True  # If True, the training will take longer
plot_interval = 20  # Only refresh plot every n episodes. Will speed up training
plot_moving_average_period = 100  # The period in which the average is cumputed

visualize_training = False  # If True, the training will obviously take much much longer


# RENDERER
renderer = render_engine

# scale = 20
scale = 14
distance = 100


# TERRAIN
terrain_file = "test_2"  # Will generate a new terrain if empty

if terrain_file == "":
    Logger.status("Generating new terrain...")
    terrain: Terrain = generate_random_terrain()

    world = render_engine.Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=1400, height=750)
    world.render()

    Logger.input("Would you like to save the new Terrain? (y/n)")
    q_input = input()
    if q_input == "y":
        Logger.input("Enter a file name: ")
        terrain_file = input()
        save_terrain(terrain_file, terrain)

else:
    Logger.status("Loading terrain from file \"" + terrain_file + "\"...")
    terrain: Terrain = load_terrain(terrain_file)
    world = render_engine.Engine3D(terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=1400, height=750)
    world.render()

Logger.status("Terrain ready. Highest point is", terrain.highest_point)


# TRAINING
Logger.status("Beginning training...")

params = Parameters(
    num_episodes=10000,
    max_steps_per_episode=500,

    # learning_rate=0.1,
    learning_rate=0.5,
    discount_rate=0.99,

    start_exploration_rate=1,
    max_exploration_rate=1,
    min_exploration_rate=0.01,
    exploration_decay_rate=0.0001,

    rewards_all_episodes=[],
    max_rewards_all_episodes=[],
)

q_table, params = train(width=terrain.width, length=terrain.length, params=params, environment=world,
                        visualize=visualize_training, plot=plot_training_progress, plot_interval=plot_interval,
                        plot_moving_avg_period=plot_moving_average_period)
Logger.status("Training done.")


# SHOW RESULT
Logger.info("Zeros in q_table: ", q_table.size - np.count_nonzero(q_table), "/", q_table.size)
Logger.info("Non-Zeros in q_table: ", np.count_nonzero(q_table))
Logger.status("Showing best learned path...")

while True:
    reward_demo = 0
    max_reward_demo = 0
    state = world.reset_agent()
    for step in range(params.max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        action = np.argmax(q_table[state, :])
        new_state, reward = world.agent_perform_action(action)
        state = new_state
        reward_demo += reward
        if max_reward_demo < reward:
            max_reward_demo = reward

        world.redraw_agent()
        time.sleep(0.1)
    time.sleep(5)
