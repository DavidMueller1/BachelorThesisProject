from dataclasses import dataclass
import numpy as np
from rl_algorithms.deep_q import DQN


@dataclass
class Parameters:
    num_episodes: int
    max_steps_per_episode: int

    learning_rate: float
    discount_rate: float

    start_exploration_rate: float
    max_exploration_rate: float
    min_exploration_rate: float
    exploration_decay_rate: float

    rewards_all_episodes: list
    max_rewards_all_episodes: list

@dataclass
class DeepQParameters:
    num_episodes: int
    max_steps_per_episode: int
    replay_buffer_size: int
    batch_size: int

    learning_rate: float
    discount_rate: float
    target_update: int

    start_exploration_rate: float
    max_exploration_rate: float
    min_exploration_rate: float
    exploration_decay_rate: float

    rewards_all_episodes: list
    max_rewards_all_episodes: list


@dataclass
class Terrain:
    width: int
    length: int
    points: list
    highest_point: list
    triangles: list


@dataclass
class Learned:
    q_table: np.ndarray
    parameters: Parameters


@dataclass
class LearnedDeepQ:
    network: DQN
    parameters: Parameters


@dataclass
class LearnedDeepQForSpecificTerrain:
    learned: LearnedDeepQ
    terrainFile: str

@dataclass
class LearnedForSpecificTerrain:
    learned: Learned
    terrainFile: str

    # def __init__(self, max_steps_per_episode, q_table, points, width, length):
    #     self.max_steps_per_episode = max_steps_per_episode
    #     self.q_table = q_table
    #     self.points = points
    #     self.width = width
    #     self.length = length