from terrain_generator.random_terrain_generator import PerlinNoise
from terrain_generator.random_terrain_generator import RandomTerrain
from data_util.experiment_data_classes import Terrain
from logger import Logger

# PARAMS
n1div = 5  # landmass distribution
n1scale = 20  # landmass height

width = 10  # map width
length = 10  # map

# frequency = 30
# amplitude = 20


def generate_random_terrain():
    noise1 = PerlinNoise(width / n1div, length / n1div)

    points = []
    triangles = []

    highest_point = [0, 0, 0]

    for x in range(-int(width / 2), int(width / 2)):
        for y in range(-int(length / 2), int(length / 2)):
            x1 = x + width / 2
            y1 = y + length / 2
            z = noise1.perlin(x1 / n1div, y1 / n1div) * n1scale
            point = [x, y, z]
            points.append(point)
            if z == highest_point[2]:
                Logger.debug(point, " has the same height as ", highest_point)
            if z > highest_point[2]:
                highest_point = point

    Logger.debug("Highest Point: ", highest_point)

    triangles = RandomTerrain.calculate_triangles(points, width, length)

    return Terrain(width, length, points, highest_point, triangles)
