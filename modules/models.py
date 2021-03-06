"""
version 1.0
"""

import numpy as np


class NaSh():

    _velosity_max = 5
    _slow_probability = 0.5

    def __init__(self, n_cells: int, n_cars: int) -> None:
        """
        n_cells: size of road in cells
        n_cars: number of cars
        """
        self.n_cells = n_cells
        self.n_cars = n_cars
        self.car_position = np.sort(np.random.choice(n_cells, size=n_cars, replace=False))
        self.car_velosity = np.zeros(n_cars, dtype=int)
        self.distance = np.zeros(n_cars, dtype=int)
        self.v_max = np.full(n_cars, self._velosity_max, dtype=int)
        self.v_min = np.zeros(n_cars, dtype=int)
        self.stability = False
        self.matrix_position = None
        self.matrix_velosity = None

    def step(self):
        """Make one time step of model"""
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
        """
        Stabilizes system
        n_step: time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        for _ in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step:int):
        """
        Method save information about cars position and velosity
        n_step: time step for system research
        """ 
        if not self.stability:
            print("Model hasn't stabilizes")
        self.matrix_position = np.zeros((n_step, self.n_cars), dtype=int)
        self.matrix_velosity = np.zeros((n_step, self.n_cars), dtype=int)
        for step in range(n_step):
            self.matrix_position[step] = self.car_position
            self.step()
            self.matrix_velosity[step] = self.car_velosity

    def avarage_flow(self):
        """Calculate avarage flow of model"""
        if self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velosity.sum()/self.matrix_velosity.shape[0]/self.n_cells

    def digramm_x_t(self):
        """
        Method make matrix of road
        size [n_cells, time step research]
        if cell is empty -> None
        if cell with cars -> value of car's velosity
        """
        if self.matrix_position is None or self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velosity[step]
        return road

class NaShAuto():

    _velosity_max = 5
    _slow_probability = 0.5

    def __init__(self, n_cells: int, n_hdr: int, n_adr: int) -> None:
        """
        n_cells: size of road in cells
        n_hdr: number of cars with drivers
        n_adr: number of cars with auto pilot
        """
        self.n_cells = n_cells
        self.n_cars = n_hdr+n_adr
        self.car_position = np.sort(np.random.choice(n_cells, size=self.n_cars, replace=False))
        self.car_velosity = np.zeros(self.n_cars, dtype=int)
        self.distance = np.zeros(self.n_cars, dtype=int)
        self.v_max = np.full(self.n_cars, self._velosity_max, dtype=int)
        self.v_min = np.zeros(self.n_cars, dtype=int)
        self.stability = False
        adr_position = np.random.choice(self.n_cars, size=n_adr, replace=False)
        self.is_driver = np.ones_like(self.car_position)
        self.is_driver[adr_position] = np.zeros(n_adr)

    def step(self):
        """Make one time step of model"""
        self.distance = np.roll(self.car_position, -1)-self.car_position
        self.distance %= self.n_cells
        self.car_velosity = np.min([self.car_velosity+1, self.v_max, self.distance-1], axis=0)
        slow = np.random.choice([0,1], size=self.n_cars, p=[1-self._slow_probability, self._slow_probability])
        self.car_velosity = np.max([self.car_velosity-slow*self.is_driver, self.v_min], axis=0)
        self.car_position += self.car_velosity
        self.car_position %= self.n_cells

    def set_max_velosity(self, v_max:int) -> None:
        self._velosity_max = v_max

    def set_slow_probability(self, p:float) -> None:
        self._slow_probability = p

    def system_stabilization(self, n_step:int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        for _ in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step:int): 
        """
        Method save information about cars position and velosity
        n_step: time step for system research
        """ 
        if not self.stability:
            print("Model hasn't stabilizes")
        self.matrix_position = np.zeros((n_step, self.n_cars), dtype=int)
        self.matrix_velosity = np.zeros((n_step, self.n_cars), dtype=int)
        for step in range(n_step):
            self.matrix_position[step] = self.car_position
            self.step()
            self.matrix_velosity[step] = self.car_velosity

    def avarage_flow(self):
        """Calculate avarage flow of model"""
        if self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velosity.sum()/self.matrix_velosity.shape[0]/self.n_cells

    def digramm_x_t(self):
        """
        Method make matrix of road
        size [n_cells, time step research]
        if cell is empty -> None
        if cell with cars -> value of car's velosity
        """
        if self.matrix_position is None or self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velosity[step]
        return road

class StochasticNFS():

    _velosity_max = 5
    _S = [1, 2]

    def __init__(self, n_cells: int, n_cars: int, prob={"p":0.5, "q":0.5, "r":0.5}, uniform=True) -> None:
        """
        n_cells: size of road in cells
        n_cars: number of cars
        prob: <dict> of model probability
            p - probability for random breaking
            q - probability for slow to start effect
            r - probability for S
        uniform: <bool> uniform distribution of cars' position 
        """
        if n_cars < 2:
            raise ValueError("Minimum 2 cars should be get")
        self.n_cells = n_cells
        self.n_cars = n_cars
        self._p = prob['p']
        self._q = prob['q']
        self._r = prob['r']
        if uniform:
            self.car_position = np.arange(n_cars) * (n_cells//n_cars)
        else:
            self.car_position = np.sort(np.random.choice(n_cells, size=n_cars, replace=False))
        self.car_position_previous = self.car_position.copy()
        self.car_velosity = np.zeros(n_cars, dtype=int)
        self.distance = np.zeros(n_cars, dtype=int)
        self.v_min = np.zeros(n_cars, dtype=int)
        self.stability = False
        self.matrix_position = None
        self.matrix_velosity = None

    def step(self):
        """Make one time step of model"""
        # rule 1
        self.car_velosity = np.min([self.v_max, self.car_velosity+1], axis=0)
        # rule 2
        self.anticipation = np.random.choice(self._S, size=self.n_cars, p=[1-self._r, self._r])
        slow_start = np.random.choice([False, True], size=self.n_cars, p=[1-self._q, self._q])
        i = np.arange(self.n_cars)
        i_next = (i+self.anticipation) % self.n_cars
        distance_anticipation_previous = (self.car_position_previous.take(i_next) - self.car_position_previous + self.n_cells) % self.n_cells - self.anticipation
        if self.n_cars <= max(self._S):
            distance_anticipation_previous[i_next==i] = self.n_cells
        assert distance_anticipation_previous.min() >= 0
        velosity = np.min([self.car_velosity*slow_start, distance_anticipation_previous*slow_start], axis=0)
        self.car_velosity = np.max([self.car_velosity*np.invert(slow_start), velosity], axis=0)
        # rule 3
        distance_anticipation = (self.car_position.take(i_next) - self.car_position + self.n_cells) % self.n_cells - self.anticipation
        if self.n_cars <= max(self._S):
            distance_anticipation[i_next==i] = self.n_cells
        assert distance_anticipation.min() >= 0
        self.car_velosity = np.min([self.car_velosity, distance_anticipation], axis=0)
        # rule 4
        random_break = np.random.choice([0,1], size=self.n_cars, p=[self._p, 1-self._p])
        self.car_velosity = np.max([self.v_min, self.car_velosity-random_break], axis=0)
        # rule 5
        self.distance = np.roll(self.car_position, -1)-self.car_position + self.n_cells
        self.distance %= self.n_cells
        self.car_velosity = np.min([self.car_velosity, np.roll(self.car_velosity, -1)+self.distance-1], axis=0)
        # rule 6
        self.car_position_previous = self.car_position.copy()
        self.car_position += self.car_velosity.astype(int)
        self.car_position %= self.n_cells

    def set_max_velosity(self, v_max:int) -> None:
        self._velosity_max = v_max
        self.v_max = np.full(self.n_cars, self._velosity_max, dtype=int) 

    def system_stabilization(self, n_step:int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        for _ in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step:int):
        """
        Method save information about cars position and velosity
        n_step: time step for system research
        """ 
        if not self.stability:
            print("Model hasn't stabilizes")
        self.matrix_position = np.zeros((n_step, self.n_cars), dtype=int)
        self.matrix_velosity = np.zeros((n_step, self.n_cars), dtype=int)
        for step in range(n_step):
            self.matrix_position[step] = self.car_position
            self.step()
            self.matrix_velosity[step] = self.car_velosity

    def avarage_flow(self):
        """Calculate avarage flow of model"""
        if self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velosity.sum()/self.matrix_velosity.shape[0]/self.n_cells

    def digramm_x_t(self):
        """
        Method make matrix of road
        size [n_cells, time step research]
        if cell is empty -> None
        if cell with cars -> value of car's velosity
        """
        if self.matrix_position is None or self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velosity[step]
        return road

def get_flow(n_cars, n_cells, prob, time_s, time_r):
    """
    Function for calculate flow of model
    n_cars: number of cars
    n_cells: size of road in cells
    prob: probability of slow down effect
    time_s: time step for stabilize model
    time_r: time step for research model
    """
    model = NaSh(n_cells, n_cars)
    model.set_slow_probability(prob)
    model.system_stabilization(time_s)
    model.system_research(time_r)
    return model.avarage_flow()

def get_flow_auto(n_cars, n_adr, n_cells, prob, time_s, time_r):
    """
    Function for calculate flow of model
    n_cars: number of cars
    n_adr: number of cars with auto pilot
    n_cells: size of road in cells
    prob: probability of slow down effect
    time_s: time step for stabilize model
    time_r: time step for research model
    """
    model = NaShAuto(n_cells, n_cars-n_adr, n_adr)
    model.set_slow_probability(prob)
    model.system_stabilization(time_s)
    model.system_research(time_r)
    return model.avarage_flow()

def get_flow_S_NFS(n_cars, n_cells, prob, time_s, time_r, v_max=5, uniform=True):
    """
    Function for calculate flow of model
    n_cars: number of cars
    n_cells: size of road in cells
    prob: probability <dict>
    time_s: time step for stabilize model
    time_r: time step for research model
    """
    if n_cars < 1:
        return 0
    if n_cars == 1:
        return v_max/n_cells
    model = StochasticNFS(n_cells, n_cars, prob, uniform)
    model.set_max_velosity(v_max)
    model.system_stabilization(time_s)
    model.system_research(time_r)
    return model.avarage_flow()