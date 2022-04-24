"""
version 1.0
"""

import numpy as np
from tqdm import tqdm

class NaSh():

    _velosity_max = 5
    _slow_probability = 0.5

    def __init__(self, n_cells: int, n_cars: int) -> None:

        self.n_cells = n_cells
        self.n_cars = n_cars
        self.car_position = np.sort(np.random.choice(n_cells-1, size=n_cars, replace=False))
        self.car_velosity = np.zeros(n_cars, dtype=int)
        self.distance = np.zeros(n_cars, dtype=int)
        self.v_max = np.full(n_cars, self._velosity_max, dtype=int)
        self.v_min = np.zeros(n_cars, dtype=int)
        self.stability = False

    def step(self):
        self.distance = np.roll(self.car_position, -1)-self.car_position
        self.distance %= self.n_cells
        self.car_velosity = np.min([self.car_velosity+1, self.v_max, self.distance-1], axis=0)
        slow = np.random.choice([0,1], size=self.n_cars, p=[1-self._slow_probability, self._slow_probability])
        self.car_velosity = np.max([self.car_velosity-slow, self.v_min], axis=0)
        self.car_position += self.car_velosity
        self.car_position %= self.n_cells

    def set_max_velosity(self, v_max:int) -> None:
        self._velosity_max = v_max

    def set_slow_probability(self, p:float) -> None:
        self._slow_probability = p

    def system_stabilization(self, n_step:int):
        for _ in tqdm(range(n_step)):
            self.step()
        self.stability = True

    def system_research(self, n_step:int): 
        self.matrix_position = np.zeros((n_step, self.n_cars), dtype=int)
        self.matrix_velosity = np.zeros((n_step, self.n_cars), dtype=int)
        for step in tqdm(range(n_step)):
            self.matrix_position[step] = self.car_position
            self.step()
            self.matrix_velosity[step] = self.car_velosity

    def avarage_flow(self):
        return self.matrix_velosity.sum()/self.matrix_velosity.shape[0]/self.n_cells

    def digramm_x_t(self):
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velosity[step]
        return road