from datetime import datetime
import time
from logger import Logger


class TimeEstimater:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.start_time = datetime.now()

    def get_time_left(self, current_episode):
        if current_episode == 0:
            return False
        time_passed = datetime.now() - self.start_time
        percentage_done = float(current_episode) / float(self.num_episodes)
        time_full = time_passed / percentage_done
        return time_full - time_passed
