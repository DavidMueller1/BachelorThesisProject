from terrain_generator import random_terrain_generator
from visualization_engine_3d import engine as render_engine
import threading
import numpy as np
import time
import random
import pickle
from data_util import experiment_data_classes as data
import dill

# frequency = 30
# amplitude = 20

n1div = 5
n2div = 4
n3div = 1

n1scale = 20  # landmass height
n2scale = 2  # boulder scale
n3scale = 1  # rock scale

############ Display variables

# scale = 20
scale = 14
distance = 100

############ Land size

width = 50 # map width
length = 50 # map length

############ Noise variables

noise1 = random_terrain_generator.PerlinNoise(width / n1div, length / n1div) # landmass / mountains
noise2 = random_terrain_generator.PerlinNoise(width / n2div, length / n2div) # boulders
noise3 = random_terrain_generator.PerlinNoise(width / n3div, length / n3div) # rocks

############ 3D shapes

points = []
print(points.__class__)
triangles = []


############

for x in range(-int(width / 2), int(width / 2)):
    for y in range(-int(length / 2), int(length / 2)):
        x1 = x + width / 2
        y1 = y + length / 2
        z = noise1.perlin(x1 / n1div, y1 / n1div) * n1scale  # add landmass
        #z += noise2.perlin(x1 / n2div, y1 / n2div) * n2scale  # add boulders
        #z += noise3.perlin(x1 / n3div, y1 / n3div) * n3scale  # add rocks
        points.append([x, y, z])

highest_point = (0, 0, 0)

# for x in range(width):
#     for y in range(length):
#         z = (noise1.perlin(x / n1div, y / n1div) * n1scale) + n1scale / 2  # add landmass
#         if z > highest_point[2]:
#             highest_point = (x, y, z)
#         points.append([x, y, z])


print("HIGHEST POINT:", highest_point)

triangles = random_terrain_generator.RandomTerrain.calculate_triangles(points, width, length)

world = render_engine.Engine3D(points, triangles, grid_width=width, grid_height=length, agent_pos=(0, 0), scale=scale, distance=distance, width=1400, height=750)
world.render()
# world.screen.window.mainloop()

# First Reinforcement Learning Test

q_table = np.zeros((width * length, 4))

render_agent = False

num_episodes = 10000
max_steps_per_episode = 200

learning_rate = 0.1
discount_rate = 0.99

start_exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []
max_rewards_all_episodes = []


def training(event_training_complete):
    exploration_rate = start_exploration_rate
    for episode in range(num_episodes):
        print("EPISODE: ", episode)
        state = world.reset_agent()
        done = False
        rewards_current_episode = 0
        max_reward_current_episode = 0

        for step in range(max_steps_per_episode):
            # print("---")
            # print(q_table)
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = random.choice(world.get_agent_possible_actions())
            new_state, reward = world.agent_perform_action(action)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward
            if max_reward_current_episode < reward:
                max_reward_current_episode = reward

            if render_agent:
                world.clear()
                world.render()
                time.sleep(0.00001)
                input()

        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        # print("Exploration Rate: " + exploration_rate.__str__())
        # print(max_reward_current_episode)
        rewards_all_episodes.append(rewards_current_episode)
        max_rewards_all_episodes.append(max_reward_current_episode)
    event_training_complete.set()


event_training_complete = threading.Event()
training_thread = threading.Thread(target=training, args=(event_training_complete,))
training_thread.start()

# print("********REWARDS********\n")
# print(rewards_all_episodes)
# print("********MAX REWARDS********\n")
# print(max_rewards_all_episodes)

# show learned path
event_training_complete.wait()
print("Zeros in q_table: ", q_table.size - np.count_nonzero(q_table), "/", q_table.size)
print("Non-Zeros in q_table: ", np.count_nonzero(q_table))
print("Showing best learned path...")
reward_demo = 0
max_reward_demo = 0
state = world.reset_agent()
for step in range(max_steps_per_episode):
    exploration_rate_threshold = random.uniform(0, 1)
    action = np.argmax(q_table[state, :])
    new_state, reward = world.agent_perform_action(action)
    state = new_state
    reward_demo += reward
    if max_reward_demo < reward:
        max_reward_demo = reward

    world.clear()
    world.render()
    time.sleep(0.1)

print("********REWARD DEMO********\n")
print(reward_demo)
print("********MAX REWARD DEMO********\n")
print(max_reward_demo)

print("Would you like to save the Q-Table? (y/n)")
q_input = input()
if q_input == "y":
    print("Enter a file name: ")
    file_name = input()
    # np.save("data/" + file_name, q_table)
    data_save = data.DataSave(max_steps_per_episode, q_table, points, width, length)
    with open("data/" + file_name, 'wb') as f:
        dill.dump(data_save, f, pickle.HIGHEST_PROTOCOL)
    # opened_table = np.load(file_name, allow_pickle=True)
