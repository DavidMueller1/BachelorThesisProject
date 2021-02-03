from rl_algorithms.deep_q_2 import DeepQNetwork
from visualization_engine_3d.engine import Engine3D
from init_terrain import generate_random_terrain
from data_util.experiment_data_classes import Terrain
from data_util.experiment_data_classes import DeepQParameters
from data_util.experiment_data_classes import LearnedDeepQForSpecificTerrain
from data_util.experiment_data_classes import LearnedDeepQ
from data_util.data_loader import load_terrain
from data_util.data_saver import save_terrain
from data_util.data_saver import save_learned_for_terrain_deep_q
from visualize_util import visualize_best_path_deep_q
from tkinter.filedialog import asksaveasfilename
from rl_algorithms import simple
from rl_algorithms import buffer
# from rl_algorithms import deep_q
from rl_algorithms import deep_q_2
from rl_algorithms import exploration_as_reward
from logger import Logger
import numpy as np
import time
import torch


class ExperimentAssistant:

    def __init__(self, save_folder="temp", save_name="default", terrain_file="",
                 plot_training_progress=False, plot_interval=1, plot_moving_average_period=100, show_path_interval=0,
                 visualize_training=False):

        self.params: DeepQParameters
        self.save_folder = save_folder
        self.save_name = save_name
        self.terrain_file = terrain_file
        self.plot_training_progress = plot_training_progress
        self.plot_interval = plot_interval
        self.plot_moving_average_period = plot_moving_average_period
        self.show_path_interval = show_path_interval
        self.visualize_training = visualize_training

        self.terrain_saved = False
        self.terrain: Terrain
        self.target_net: DeepQNetwork

    def init_experiment(self, params: DeepQParameters, terrain_file=""):
        self.params = params
        if terrain_file != "":
            self.terrain_file = terrain_file

        Logger.status("BEGINNING EXPERIMENT \"" + self.save_folder + ": " + self.save_name + "\"")

        # RENDERER
        # scale = 20
        scale = 14
        distance = 100


        # TERRAIN
        if self.terrain_file == "":
            Logger.status("Generating new terrain...")
            self.terrain: Terrain = generate_random_terrain()

            world = Engine3D(self.terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800)
            world.render()

            Logger.input("Would you like to save the new Terrain? (y/n)")
            if input() == "y":
                Logger.input("Enter a file name: ")
                terrain_file = input()
                save_terrain(terrain_file, self.terrain)
                self.terrain_saved = terrain_file
                Logger.status("Terrain saved as \"" + terrain_file + "\"")

        else:
            self.terrain_saved = self.terrain_file
            Logger.status("Loading terrain from file \"" + self.terrain_file + "\"...")
            self.terrain: Terrain = load_terrain(self.terrain_file)
            world = Engine3D(self.terrain, agent_pos=(0, 0), scale=scale, distance=distance, width=800, height=800, random_spawn=True)
            world.render()

        Logger.status("Terrain ready. Highest point is", self.terrain.highest_point)

        # TRAINING
        Logger.status("Beginning training...")

        self.target_net, self.params = deep_q_2.train(width=self.terrain.width, length=self.terrain.length, params=self.params,
                                            environment=world, visualize=self.visualize_training,
                                            show_path_interval=self.show_path_interval, plot=self.plot_training_progress,
                                            plot_interval=self.plot_interval, plot_moving_avg_period=self.plot_moving_average_period)

        world.screen.window.destroy()
        Logger.status("Training done.")

    def save_experiment_data(self, save_folder="", save_name=""):
        if not self.terrain_saved:
            terrain_file = "terrain_" + self.save_name
            Logger.status("Terrain not saved. Saving as \"%s\"" % terrain_file)
            save_terrain(terrain_file, self.terrain)
            self.terrain_saved = terrain_file

        folder = save_folder if save_folder != "" else self.save_folder
        name = save_name if save_name != "" else self.save_name
        file_name = "data/learned/" + folder + "/" + name
        Logger.status("Saving experiment result under \"%s\"" % file_name)
        save_learned_for_terrain_deep_q(file_name, LearnedDeepQForSpecificTerrain(LearnedDeepQ(self.target_net, self.params), self.terrain_saved))
        Logger.status("Done")
