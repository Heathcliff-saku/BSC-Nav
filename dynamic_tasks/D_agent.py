from objnav_benchmark import *


class DynamicNavAgent:
    def __init__(self, nav_memory, benchmark_env):
        super().__init__(nav_memory, benchmark_env)

        self.benchmark_env = benchmark_env

    def move2object(self, obj_name):
        pass
    