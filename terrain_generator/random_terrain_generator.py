import math
import random
import colorsys


class PerlinNoise:

    def __init__(self, x, y):
        x, y = math.ceil(x) + 1, math.ceil(y) + 1
        self.gradients = []
        for j in range(y):
            self.gradients.append([])
            for i in range(x):
                a = random.uniform(0, 1)
                b = math.sqrt(1 - a ** 2)
                c = [-1, 1][random.randint(0, 1)]
                d = [-1, 1][random.randint(0, 1)]
                self.gradients[j].append([a * c, b * d])

    def dot_grid_gradient(self, ix, iy, x, y):
        dx = x - ix
        dy = y - iy
        return dx * self.gradients[iy][ix][0] + dy * self.gradients[iy][ix][1]

    def interpolate(self, a0, a1, w):
        return a0 + w * (a1 - a0)

    def perlin(self, x, y):
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1

        sx = x - x0
        sy = y - y0

        n0 = self.dot_grid_gradient(x0, y0, x, y)
        n1 = self.dot_grid_gradient(x1, y0, x, y)
        ix0 = self.interpolate(n0, n1, sx)

        n0 = self.dot_grid_gradient(x0, y1, x, y)
        n1 = self.dot_grid_gradient(x1, y1, x, y)
        ix1 = self.interpolate(n0, n1, sx)

        value = self.interpolate(ix0, ix1, sy)
        return value


class RandomTerrain:
    def generate_random_terrain(self, width, length):
        pass

    @staticmethod
    def calculate_triangles(points, width, length):
        triangles = []
        for x in range(width):
            for y in range(length):
                if 0 < x and 0 < y:
                    a, b, c = int(x * length + y), int(x * length + y - 1), int((x - 1) * length + y)  # find 3 points in triangle
                    triangles.append([a, b, c, RandomTerrain.color(points, a, b, c)])

                if x < width - 1 and y < length - 1:
                    a, b, c, = int(x * length + y), int(x * length + y + 1), int((x + 1) * length + y)  # find 3 points in triangle
                    triangles.append([a, b, c, RandomTerrain.color(points, a, b, c)])

        return triangles

    @staticmethod
    def color(points, a, b, c):  # check land type
        z = (points[a][2] + points[b][2] + points[c][2]) / 3  # calculate average height of triangle
        (r, g, b) = colorsys.hsv_to_rgb(((1 - (z / 20)) * 0.8 - 0.45), 1, 1)
        return "#" + '{:02x}'.format(max(int(r * 255), 0)) + '{:02x}'.format(max(int(g * 255), 0)) + '{:02x}'.format(
            max(int(b * 255), 0))
