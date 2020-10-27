from data_util.experiment_data_classes import Terrain
from data_util.experiment_data_classes import LearnedForSpecificTerrain
import dill
import pickle


def save_terrain(file_name, terrain: Terrain):
    with open("data/terrain/" + file_name, 'wb') as f:
        dill.dump(terrain, f, pickle.HIGHEST_PROTOCOL)


def save_learned_for_terrain(file_name, data: LearnedForSpecificTerrain):
    with open("data/learned/" + file_name, 'wb') as f:
        dill.dump(data, f, pickle.HIGHEST_PROTOCOL)
