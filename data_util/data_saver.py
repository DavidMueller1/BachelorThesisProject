from data_util.experiment_data_classes import Terrain
from data_util.experiment_data_classes import LearnedForSpecificTerrain
from data_util.experiment_data_classes import LearnedDeepQForSpecificTerrain
import dill
import pickle
import os
from logger import Logger


def save_terrain(file_name, terrain: Terrain):
    os.chdir(os.path.dirname(__file__))
    with open("../data/terrain/" + file_name, 'wb') as f:
        dill.dump(terrain, f, pickle.HIGHEST_PROTOCOL)


def save_learned_for_terrain(file_name, data: LearnedForSpecificTerrain):
    os.chdir(os.path.dirname(__file__))
    with open("../data/learned/" + file_name, 'wb') as f:
        dill.dump(data, f, pickle.HIGHEST_PROTOCOL)


def save_learned_for_terrain_deep_q(file_name, data: LearnedDeepQForSpecificTerrain):
    os.chdir(os.path.dirname(__file__))

    # with open("data/learned/" + file_name, 'wb') as f:
    with open("../" + file_name, 'wb') as f:
        dill.dump(data, f, pickle.HIGHEST_PROTOCOL)
