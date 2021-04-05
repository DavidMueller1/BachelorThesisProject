import visualization_engine_3d.screen
import visualization_engine_3d.face
import visualization_engine_3d.vertex
from logger import Logger
import random
from data_util.experiment_data_classes import Terrain
import copy
import numpy as np
from enum import Enum


class Rewards(Enum):
    Default = 0
    Spiral = 1
    Delta = 2
    DeltaVisited = 3
    DeltaVisitedDistance = 4
    SpiralEpsilon = 5


class States(Enum):
    Default = 0
    Minimal = 1
    AdjacentHeights = 2
    RelativePosAndHeights = 3
    RelativePosAndHeightsAndHighestPoint = 4
    RelativePosAndHeightsAndHighestPointAndSteps = 5
    RelativePosAndHeightsAndHighestPointAndStepsAndVisited = 6


# For Spiral:
HEIGHT_MULTIPLIKATOR = 40
# VISITED_MULTIPLIKATOR = 1
VISITED_MULTIPLIKATOR = 10
DISTANCE_MULTIPLIKATOR = 1


class Engine3D:

    def __reset_drag(self, event):
        self.__prev = []

    def __drag(self, event):
        if self.__prev:
            self.rotate('y', (event.x - self.__prev[0]) / 20)
            self.rotate('x', (event.y - self.__prev[1]) / 20)
            # self.rotate('z', (event.y - self.__prev[1]) / 20)
            self.clear()
            self.render()
        self.__prev = [event.x, event.y]

    # def __select(self, event):
    #     zeros = self.screen.zeros
    #     event = (event.x - zeros[0], event.y - zeros[1])
    #
    #     possibilities = []
    #     for a in range(-6, 5):
    #         for b in range(-6, 5):
    #             possibilities.append((event[0] + a, event[1] + b))
    #
    #     found = [e for e in possibilities if e in self.flattened]
    #     if found:
    #         self.__moveaxis = None
    #         self.__selected = self.flattened.index(found[0])
    #
    #         i = self.points[self.__selected]
    #         self.__axis = [[copy.deepcopy(i) for a in range(2)] for b in range(3)]
    #
    #         self.__axis[0][0].x -= 40 / self.scale
    #         self.__axis[0][1].x += 40 / self.scale
    #         self.__axis[1][0].y -= 40 / self.scale
    #         self.__axis[1][1].y += 40 / self.scale
    #         self.__axis[2][0].z -= 40 / self.scale
    #         self.__axis[2][1].z += 40 / self.scale
    #
    #         self.__axis = [[point.flatten(self.scale, self.distance) for point in i] for i in self.__axis]
    #         self.__axis = [[[i[0] + zeros[0], i[1] + zeros[1]] for i in j] for j in self.__axis]
    #         self.__axis = [self.screen.create_line(self.__axis[0], 'red'), self.screen.create_line(self.__axis[1], 'green'), self.screen.create_line(self.__axis[2], 'blue')]
    #
    # def __selectx(self, event):
    #     self.__moveaxis = 'x'
    #
    # def __selecty(self, event):
    #     self.__moveaxis = 'y'
    #
    # def __selectz(self, event):
    #     self.__moveaxis = 'z'

    def __agent_move_up(self, event):
        self.agent_last_pos = self.agent_pos
        state, reward, done = self.agent_perform_action(1)
        Logger.debug("REWARD:", reward)
        self.manuel_control_event()
        self.redraw_agent()

    def __agent_move_down(self, event):
        self.agent_last_pos = self.agent_pos
        state, reward, done = self.agent_perform_action(3)
        Logger.debug("REWARD:", reward)
        self.manuel_control_event()
        self.redraw_agent()

    def __agent_move_left(self, event):
        self.agent_last_pos = self.agent_pos
        state, reward, done = self.agent_perform_action(0)
        Logger.debug("REWARD:", reward)
        self.manuel_control_event()
        self.redraw_agent()

    def __agent_move_right(self, event):
        self.agent_last_pos = self.agent_pos
        state, reward, done = self.agent_perform_action(2)
        Logger.debug("REWARD:", reward)
        self.manuel_control_event()
        self.redraw_agent()

    def manuel_control_event(self):
        Logger.debug("STEP")
        Logger.debug("New State:", self.get_state_for_deep_q(step=4, max_steps=10))
        Logger.debug("VISITED:", self.visited_points)
        Logger.debug("DISTANCE:", self.get_agent_distance_from_spawn())
        Logger.debug("------------")

    def __moveup(self, event):
        if self.__selected is not None and self.__moveaxis is not None:
            self.points[self.__selected].move(self.__moveaxis, 0.1)
            self.clear()
            self.render()

    def __movedown(self, event):
        if self.__selected is not None and self.__moveaxis is not None:
            self.points[self.__selected].move(self.__moveaxis, -0.1)
            self.clear()
            self.render()

    def __zoomin(self, event):
        self.scale += 2.5
        self.clear()
        self.render()

    def __zoomout(self, event):
        self.scale -= 2.5
        self.clear()
        self.render()

    def __deselect(self, event):
        if self.__selected is not None:
            self.__selected = None
            self.__axis = [self.screen.delete(line) for line in self.__axis]
            self.__moveaxis = None

    def __cameraleft(self, event):
        self.screen.zeros[0] -= 5
        self.clear()
        self.render()

    def __cameraright(self, event):
        self.screen.zeros[0] += 5
        self.clear()
        self.render()

    def __cameraup(self, event):
        self.screen.zeros[1] -= 5
        self.clear()
        self.render()

    def __cameradown(self, event):
        self.screen.zeros[1] += 5
        self.clear()
        self.render()

    def write_points(self, points):
        self.points = []
        for point in points:
            self.points.append(visualization_engine_3d.vertex.Vertex(point))

    def write_triangles(self, triangles):
        self.triangles = []
        for triangle in triangles:
            if len(triangle) != 4:
                triangle.append('gray')
            self.triangles.append(visualization_engine_3d.face.Face(triangle))

    def __init__(self, terrain: Terrain, agent_pos=(0, 0), width=1000, height=700, distance=6, scale=100, title='3D',
                 background='white', manual_control=False, random_spawn=False, reward_val=Rewards.Default, state_val=States.Default):
        # object parameters
        self.distance = distance
        self.scale = scale
        self.grid_width = terrain.width
        self.grid_height = terrain.length
        self.highest_point = terrain.highest_point
        self.terrain = terrain
        self.random_spawn = random_spawn
        random.seed(0)
        Logger.debug("Random Spawn:", self.random_spawn)

        # initialize display
        self.screen = visualization_engine_3d.screen.Screen(width, height, title, background)
        # self.screen.window.bind('<B1-Motion>', self.__drag)
        # self.__prev = []
        # self.screen.window.bind('<ButtonRelease-1>', self.__reset_drag)

        # self.screen.window.bind('<Up>', self.__zoomin)
        # self.screen.window.bind('<Down>', self.__zoomout)
        self.screen.window.bind('w', self.__cameraup)
        self.screen.window.bind('s', self.__cameradown)
        self.screen.window.bind('a', self.__cameraleft)
        self.screen.window.bind('d', self.__cameraright)

        # this is for editing the model
        # self.__selected = None
        # self.__axis = []
        # self.__moveaxis = None
        # self.screen.window.bind('<ButtonPress-3>', self.__select)
        # self.screen.window.bind('<ButtonRelease-3>', self.__deselect)
        # self.screen.window.bind('x', self.__selectx)
        # self.screen.window.bind('y', self.__selecty)
        # self.screen.window.bind('z', self.__selectz)
        # self.screen.window.bind('<Left>', self.__movedown)
        # self.screen.window.bind('<Right>', self.__moveup)

        # store coordinates
        self.original_points = terrain.points
        self.write_points(terrain.points)
        self.flattened = []

        # store faces
        self.write_triangles(terrain.triangles)

        # self.rotate('y', 24.7)
        # self.rotate('x', 180)
        # self.rotate('z', 1)

        self.agent_pos = agent_pos
        self.agent_start_pos = agent_pos
        self.agent_last_pos = agent_pos

        self.highest_point_vistited_pos = (0, 0)
        self.highest_point_vistited_height = -100

        self.visited_points = [self.original_points[self.get_agent_state()]]

        self.furthes_distance_from_spawn = 0

        self.agent_same_pos_counter = 0
        self.agent_shape = None
        self.path_shapes = []

        self.reward_fun = {
            Rewards.Default: self.get_reward_via_delta,
            Rewards.Spiral: self.get_reward_spiral,
            Rewards.Delta: self.get_reward_via_delta,
            Rewards.DeltaVisitedDistance: self.get_reward_via_delta,
            Rewards.SpiralEpsilon: self.get_reward_for_spiral_with_epsilon
        }
        self.reward_val = reward_val

        self.state_fun = {
            States.Default: self.get_state_default,
            States.Minimal: self.get_state_minimal,
            States.AdjacentHeights: self.get_state_with_heights,
            States.RelativePosAndHeights: self.get_state_relative_pos_and_heights,
            States.RelativePosAndHeightsAndHighestPoint: self.get_state_relative_pos_and_heights_and_highest_point,
            States.RelativePosAndHeightsAndHighestPointAndSteps: self.get_state_relative_pos_and_heights_and_highest_point_and_steps,
            States.RelativePosAndHeightsAndHighestPointAndStepsAndVisited: self.get_state_relative_pos_and_heights_and_highest_point_and_steps_and_visited
        }
        self.state_val = state_val

        if manual_control:
            self.screen.window.bind('<Up>', self.__agent_move_up)
            self.screen.window.bind('<Right>', self.__agent_move_right)
            self.screen.window.bind('<Down>', self.__agent_move_down)
            self.screen.window.bind('<Left>', self.__agent_move_left)
            self.render()
            self.screen.window.mainloop()

    def clear(self):
        # clear display
        self.screen.clear()

    def rotate(self, axis, angle):
        # rotate model around axis
        for point in self.points:
            point.rotate(axis, angle)

    def draw_agent(self):
        (x, y) = self.agent_pos
        agent_point = self.points[self.get_agent_state()].flatten(self.scale, self.distance)

        self.agent_shape = self.screen.create_circle(agent_point, 16, color='purple')
        #self.screen.create_triangle([agent_point, visualizationEngine.graphics.vertex.Vertex((2, 2, 0 + 10)), visualizationEngine.graphics.vertex.Vertex((2, 2, 0 + 10))], 'red')

    def redraw_agent(self):
        (x, y) = self.agent_pos
        agent_point = self.points[self.get_agent_state()].flatten(self.scale, self.distance)

        self.screen.delete(self.agent_shape)
        self.agent_shape = self.screen.create_circle(agent_point, 16, color='purple')
        self.screen.window.update()

    def reset_agent(self):
        if self.random_spawn:
            half_width = int(self.grid_width / 2)
            half_height = int(self.grid_height / 2)
            self.agent_pos = (random.randint(-half_width, half_width - 1), random.randint(-half_height, half_height - 1))
            self.agent_start_pos = self.agent_pos
            self.highest_point_vistited_height = self.get_agent_height()
            self.highest_point_vistited_pos = self.get_agent_relative_pos()
            self.visited_points = [self.original_points[self.get_agent_state()]]
        else:
            self.agent_pos = (0, 0)
        return self.get_agent_state()

    def get_agent_state(self):
        return self.pos_to_state(self.agent_pos)

    def get_agent_relative_pos(self):
        return tuple(np.subtract(self.agent_pos, self.agent_start_pos))

    def get_agent_distance_from_spawn(self):
        (x, y) = self.get_agent_relative_pos()
        return abs(x) + abs(y)

    def get_agent_possible_actions(self):
        (x, y) = self.agent_pos
        actions = []
        if x > -int(self.grid_width / 2):
            actions.append(0)
        if y < int(self.grid_height / 2) - 1:
            actions.append(1)
        if x < int(self.grid_height / 2) - 1:
            actions.append(2)
        if y > -int(self.grid_height / 2):
            actions.append(3)
        return actions

    def agent_perform_action(self, action, is_last_action=False):
        (x, y) = self.agent_pos
        last_state = self.get_agent_state()

        done = is_last_action
        if action == 0:
            self.agent_pos = (max(-self.grid_width / 2, x - 1), y)
        if action == 1:
            self.agent_pos = (x, min(self.grid_height / 2 - 1, y + 1))
        if action == 2:
            self.agent_pos = (min(self.grid_width / 2 - 1, x + 1), y)
        if action == 3:
            self.agent_pos = (x, max(-self.grid_height / 2, y - 1))

        if self.get_agent_height() > self.highest_point_vistited_height:
            self.highest_point_vistited_height = self.get_agent_height()
            self.highest_point_vistited_pos = self.get_agent_relative_pos()

        new_point = False
        if self.original_points[self.get_agent_state()] not in self.visited_points:
            new_point = True
            self.visited_points.append(self.original_points[self.get_agent_state()])

        return self.get_agent_state(), self.reward_fun[self.reward_val](last_state, new_point), done

        # if action == 4:
        #     done = True
        # return self.get_agent_state(), self.get_reward_via_delta(last_state), done
        # return self.get_agent_state(), self.get_reward_via_end_state(is_last_action), done
        # return self.get_agent_state(), self.get_reward_via_visited_points(is_last_action), done
        # return self.get_agent_state(), self.get_reward_via_visited_points(is_last_action) - (self.get_reward_via_distance_from_start() if is_last_action else 0), done
        # return self.get_agent_state(), VISITED_MULTIPLIKATOR if new_point else 0, done
        # return self.get_agent_state(), (VISITED_MULTIPLIKATOR if new_point else 0) - self.get_reward_via_distance_from_start(), done # WORKING 1
        # return self.get_agent_state(), (-VISITED_MULTIPLIKATOR if not new_point else 0) - self.get_reward_via_distance_from_start() + self.get_reward_via_delta(last_state), done # ADDING DELTA HEIGHT
        # return self.get_agent_state(), self.get_reward_via_finish()
        # return self.get_agent_state(), self.points[self.get_agent_state()].z
        # return self.get_agent_state(), self.get_reward_via_delta(last_state) + (VISITED_MULTIPLIKATOR if new_point else 0), done
        # return self.get_agent_state(), self.get_reward_via_delta(last_state) + (VISITED_MULTIPLIKATOR if new_point else 0), done

    def get_reward_spiral(self, last_state, new_point):
        return self.get_reward_via_visited(last_state, new_point) - self.get_reward_via_distance_from_start()

    def get_reward_via_delta(self, last_state, new_point):
        # return self.points[self.get_agent_state()].z * HEIGHT_MULTIPLIKATOR - self.points[last_state].z * HEIGHT_MULTIPLIKATOR
        return self.original_points[self.get_agent_state()][2] * HEIGHT_MULTIPLIKATOR - self.original_points[last_state][2] * HEIGHT_MULTIPLIKATOR

    def get_reward_delta_visited_distance(self, last_state, new_point):
        return self.get_reward_via_delta(last_state, new_point) + self.get_reward_via_visited(last_state, new_point) - self.get_reward_via_distance_from_start()

    def get_reward_via_end_state(self, is_last_state):
        # return self.points[self.get_agent_state()].z if is_last_state else 0
        return (self.points[self.get_agent_state()].z - self.points[self.pos_to_state(self.agent_start_pos)].z) * HEIGHT_MULTIPLIKATOR if is_last_state else 0
        # return (self.highest_point_vistited_height - self.get_agent_height()) * HEIGHT_MULTIPLIKATOR + len(self.visited_points) * VISITED_MULTIPLIKATOR if is_last_state else 0

    def get_reward_via_visited_points(self, is_last_state):
        return len(self.visited_points) * VISITED_MULTIPLIKATOR if is_last_state else 0
        # return len(self.visited_points) * VISITED_MULTIPLIKATOR

    def get_reward_via_visited(self, last_state, new_point):
        return VISITED_MULTIPLIKATOR if new_point else 0

    def get_reward_via_distance_from_start(self):
        # return len(self.visited_points) if is_last_state else 0
        return self.get_agent_distance_from_spawn() * DISTANCE_MULTIPLIKATOR

    def get_reward_via_delta_punish_negative(self, last_state):
        reward = self.original_points[self.get_agent_state()][2] - self.original_points[last_state][2]
        return reward if reward >= 0 else reward * 1.5

    # def get_reward_via_delta(self, last_state):
    #     delta = self.points[self.get_agent_state()].z - self.points[last_state].z
    #     return 1 if delta > 1 else 0

    def get_reward_via_finish(self):
        return 1 if self.agent_pos == tuple(self.highest_point[0:2]) else 0

    def get_reward_for_spiral_with_epsilon(self, last_state, new_point):
        return

    ################################################ STATES FOR DEEP Q ############################################

    def get_state_for_deep_q(self, step=False, max_steps=False,):
        heights = self.get_agent_adjacent_heights()
        relative_pos = self.get_agent_relative_pos()
        adjacent_visited = self.get_agent_adjacent_visited()
        steps_left = (0, )
        if step and max_steps:
            steps_left = (int(max_steps - step), )
        # return np.asarray(self.agent_pos + tuple(heights[0] - x for x in heights), dtype=np.float32)
        # return np.asarray(tuple(heights[0] - x for x in heights), dtype=np.float32)
        # return np.asarray(relative_pos + heights + self.highest_point_vistited_pos, dtype=np.float32)
        # return np.append(np.asarray(relative_pos + heights + self.highest_point_vistited_pos, dtype=np.float32), np.array(self.highest_point_vistited_height * HEIGHT_MULTIPLIKATOR, dtype=np.float32))
        # return np.append(np.asarray(relative_pos + heights + self.highest_point_vistited_pos + adjacent_visited + steps_left + (max_steps if max_steps else 0, ), dtype=np.float32), np.array(self.highest_point_vistited_height * HEIGHT_MULTIPLIKATOR, dtype=np.float32))
        # return np.asarray(relative_pos + heights + self.highest_point_vistited_pos, dtype=np.float32)
        # return np.asarray(relative_pos + heights + self.highest_point_vistited_pos + adjacent_visited + steps_left, dtype=np.float32)
        # return np.asarray(relative_pos + heights + adjacent_visited + steps_left + (max_steps if max_steps else 0, ), dtype=np.float32)
        # return np.asarray(relative_pos + heights + adjacent_visited + steps_left + (max_steps if max_steps else 0, ) + self.highest_point_vistited_pos, dtype=np.float32)

        return self.state_fun[self.state_val](step, max_steps)

    def get_state_default(self, step, max_steps):  # WORKING 1
        relative_pos = self.get_agent_relative_pos()
        adjacent_visited = self.get_agent_adjacent_visited()
        steps_left = (0,)
        if step and max_steps:
            steps_left = (int(max_steps - step),)

        return np.asarray(relative_pos + adjacent_visited + steps_left + (max_steps if max_steps else 0,),
                          dtype=np.float32)

    def get_state_minimal(self, step, max_steps):
        return np.asarray(self.agent_pos, dtype=np.float32)

    def get_state_with_heights(self, step, max_steps):
        heights = self.get_agent_adjacent_heights()
        return np.asarray(self.agent_pos + heights, dtype=np.float32)

    def get_state_relative_pos_and_heights(self, step, max_steps):
        heights = self.get_agent_adjacent_heights()
        relative_pos = self.get_agent_relative_pos()
        return np.asarray(relative_pos + heights, dtype=np.float32)

    def get_state_relative_pos_and_heights_and_highest_point(self, step, max_steps):
        heights = self.get_agent_adjacent_heights()
        relative_pos = self.get_agent_relative_pos()
        return np.append(np.asarray(relative_pos + heights + self.highest_point_vistited_pos, dtype=np.float32), np.array(self.highest_point_vistited_height * HEIGHT_MULTIPLIKATOR, dtype=np.float32))

    def get_state_relative_pos_and_heights_and_highest_point_and_steps(self, step, max_steps):
        heights = self.get_agent_adjacent_heights()
        relative_pos = self.get_agent_relative_pos()
        steps_left = (int(max_steps - step), )
        return np.append(np.asarray(relative_pos + heights + self.highest_point_vistited_pos + steps_left + (max_steps if max_steps else 0, ), dtype=np.float32), np.array(self.highest_point_vistited_height * HEIGHT_MULTIPLIKATOR, dtype=np.float32))

    def get_state_relative_pos_and_heights_and_highest_point_and_steps_and_visited(self, step, max_steps):
        heights = self.get_agent_adjacent_heights()
        relative_pos = self.get_agent_relative_pos()
        adjacent_visited = self.get_agent_adjacent_visited()
        steps_left = (int(max_steps - step), )
        return np.append(np.asarray(relative_pos + heights + self.highest_point_vistited_pos + adjacent_visited + steps_left + (max_steps if max_steps else 0, ), dtype=np.float32), np.array(self.highest_point_vistited_height * HEIGHT_MULTIPLIKATOR, dtype=np.float32))


    def get_agent_height(self):
        return self.original_points[self.get_agent_state()][2]

    def get_agent_adjacent_heights(self):
        (x, y) = self.agent_pos
        half_width = self.terrain.width / 2
        half_height = self.terrain.length / 2
        top = -100 * HEIGHT_MULTIPLIKATOR if y + 1 > half_height - 1 else self.original_points[self.pos_to_state((x, y + 1))][2] * HEIGHT_MULTIPLIKATOR
        bottom = -100 * HEIGHT_MULTIPLIKATOR if y - 1 < -half_height else self.original_points[self.pos_to_state((x, y - 1))][2] * HEIGHT_MULTIPLIKATOR
        right = -100 * HEIGHT_MULTIPLIKATOR if x + 1 > half_width - 1 else self.original_points[self.pos_to_state((x + 1, y))][2] * HEIGHT_MULTIPLIKATOR
        left = -100 * HEIGHT_MULTIPLIKATOR if x - 1 < -half_width else self.original_points[self.pos_to_state((x - 1, y))][2] * HEIGHT_MULTIPLIKATOR

        return (self.get_agent_height() * HEIGHT_MULTIPLIKATOR, top, right, bottom, left)

    def get_agent_adjacent_visited(self):
        mul = VISITED_MULTIPLIKATOR
        (x, y) = self.agent_pos
        half_width = self.terrain.width / 2
        half_height = self.terrain.length / 2
        top = -1 * mul if y + 1 > half_height - 1 else (0 if self.original_points[self.pos_to_state((x, y + 1))] in self.visited_points else 1 * mul)
        bottom = -1 * mul if y - 1 < -half_height else (0 if self.original_points[self.pos_to_state((x, y - 1))] in self.visited_points else 1 * mul)
        right = -1 * mul if x + 1 > half_width - 1 else (0 if self.original_points[self.pos_to_state((x + 1, y))] in self.visited_points else 1 * mul)
        left = -1 * mul if x - 1 < -half_width else (0 if self.original_points[self.pos_to_state((x - 1, y))] in self.visited_points else 1 * mul)

        return (top, right, bottom, left)

    def plot_path(self, path):
        self.clear_path()
        path_points = [self.points[x] for x in path]

        last_point = path_points[0].flatten(self.scale, self.distance)
        for point in path_points[1:]:
            # point.z + 1
            point = point.flatten(self.scale, self.distance)
            self.path_shapes.append(self.screen.create_line([last_point, point], color='white'))
            last_point = point

    def clear_path(self):
        for shape in self.path_shapes:
            self.screen.delete(shape)
        self.path_shapes = []

    def render(self):
        # calculate flattened coordinates (x, y)
        self.flattened = []
        for point in self.points:
            self.flattened.append(point.flatten(self.scale, self.distance))

        # get coordinates to draw triangles
        triangles = []
        for triangle in self.triangles:
            avgZ = -(self.points[triangle.a].z + self.points[triangle.b].z + self.points[triangle.c].z) / 3
            triangles.append((self.flattened[triangle.a], self.flattened[triangle.b], self.flattened[triangle.c], triangle.color, avgZ))

        # sort triangles from furthest back to closest
        triangles = sorted(triangles, key=lambda x: x[4])

        # draw triangles
        for triangle in triangles:
            self.screen.create_triangle(triangle[0:3], triangle[3])

        self.draw_agent()
        # self.screen.image.create_oval(200, 200, 204, 204, fill=
        self.screen.window.update()

    def state_to_pos(self, state):
        x = int(state / self.grid_width) - 25
        y = state % self.grid_width - 25
        return x, y

    def pos_to_state(self, pos):
        (x, y) = pos
        return int((len(self.points) / 2 + (x * self.grid_width + y) + self.grid_height / 2) % len(self.points))

