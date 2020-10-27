import tkinter as tk
import math


class Engine:
    def __init__(self, points, triangles, height=400, width=600, distance=6, scale=100):
        self.window = tk.Tk()
        self.window.title('3D Graphics')
        self.image = tk.Canvas(self.window, width=width, height=height)
        self.image.pack()
        self.height = height
        self.width = width
        self.distance = distance
        self.scale = scale
        self.points = points
        self.triangles = triangles
        self.shapes = []

    def flatten_point(self, point):
        (x, y, z) = (point[0], point[1], point[2])
        projected_x = int(self.height / 2 + ((x * self.distance) / (z + self.distance)) * self.scale)
        projected_y = int(self.height / 2 + ((y * self.distance) / (z + self.distance)) * self.scale)
        return projected_x, projected_y

    def create_triangle(self, points):
        a, b, c = points[0], points[1], points[2]
        coords = [a[0], a[1], b[0], b[1], c[0], c[1]]
        self.shapes.append(self.image.create_polygon(coords, fill="", outline="black"))

    def render(self):
        coords = []

        for point in self.points:
            coords.append(self.flatten_point(point))

        for triangle in self.triangles:
            self.create_triangle((coords[triangle[0]], coords[triangle[1]], coords[triangle[2]]))