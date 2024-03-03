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
        self.distance = np.roll(self.car_position, -1) - self.car_position
        self.distance %= self.n_cells
        self.car_velosity = np.min([self.car_velosity + 1, self.v_max, self.distance - 1], axis=0)
        slow = np.random.choice([0, 1], size=self.n_cars, p=[1 - self._slow_probability, self._slow_probability])
        self.car_velosity = np.max([self.car_velosity - slow, self.v_min], axis=0)
        self.car_position += self.car_velosity
        self.car_position %= self.n_cells

    def set_max_velosity(self, v_max: int) -> None:
        self._velosity_max = v_max

    def set_slow_probability(self, p: float) -> None:
        self._slow_probability = p

    def system_stabilization(self, n_step: int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        for _ in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step: int):
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
        return self.matrix_velosity.sum() / self.matrix_velosity.shape[0] / self.n_cells

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
        self.n_cars = n_hdr + n_adr
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
        self.distance = np.roll(self.car_position, -1) - self.car_position
        self.distance %= self.n_cells
        self.car_velosity = np.min([self.car_velosity + 1, self.v_max, self.distance - 1], axis=0)
        slow = np.random.choice([0, 1], size=self.n_cars, p=[1 - self._slow_probability, self._slow_probability])
        self.car_velosity = np.max([self.car_velosity - slow * self.is_driver, self.v_min], axis=0)
        self.car_position += self.car_velosity
        self.car_position %= self.n_cells

    def set_max_velosity(self, v_max: int) -> None:
        self._velosity_max = v_max

    def set_slow_probability(self, p: float) -> None:
        self._slow_probability = p

    def system_stabilization(self, n_step: int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        for _ in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step: int):
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
        return self.matrix_velosity.sum() / self.matrix_velosity.shape[0] / self.n_cells

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

    def __init__(self, n_cells: int, n_cars: int, prob={"p": 0.5, "q": 0.5, "r": 0.5}, uniform=True) -> None:
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
            self.car_position = np.arange(n_cars) * (n_cells // n_cars)
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
        self.car_velosity = np.min([self.v_max, self.car_velosity + 1], axis=0)
        # rule 2
        self.anticipation = np.random.choice(self._S, size=self.n_cars, p=[1 - self._r, self._r])
        slow_start = np.random.choice([False, True], size=self.n_cars, p=[1 - self._q, self._q])
        i = np.arange(self.n_cars)
        i_next = (i + self.anticipation) % self.n_cars
        distance_anticipation_previous = (self.car_position_previous.take(
            i_next) - self.car_position_previous + self.n_cells) % self.n_cells - self.anticipation
        if self.n_cars <= max(self._S):
            distance_anticipation_previous[i_next == i] = self.n_cells
        assert distance_anticipation_previous.min() >= 0
        velosity = np.min([self.car_velosity * slow_start, distance_anticipation_previous * slow_start], axis=0)
        self.car_velosity = np.max([self.car_velosity * np.invert(slow_start), velosity], axis=0)
        # rule 3
        distance_anticipation = (self.car_position.take(
            i_next) - self.car_position + self.n_cells) % self.n_cells - self.anticipation
        if self.n_cars <= max(self._S):
            distance_anticipation[i_next == i] = self.n_cells
        assert distance_anticipation.min() >= 0
        self.car_velosity = np.min([self.car_velosity, distance_anticipation], axis=0)
        # rule 4
        random_break = np.random.choice([0, 1], size=self.n_cars, p=[self._p, 1 - self._p])
        self.car_velosity = np.max([self.v_min, self.car_velosity - random_break], axis=0)
        # rule 5
        self.distance = np.roll(self.car_position, -1) - self.car_position + self.n_cells
        self.distance %= self.n_cells
        self.car_velosity = np.min([self.car_velosity, np.roll(self.car_velosity, -1) + self.distance - 1], axis=0)
        # rule 6
        self.car_position_previous = self.car_position.copy()
        self.car_position += self.car_velosity.astype(int)
        self.car_position %= self.n_cells

    def set_max_velosity(self, v_max: int) -> None:
        self._velosity_max = v_max
        self.v_max = np.full(self.n_cars, self._velosity_max, dtype=int)

    def system_stabilization(self, n_step: int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Reccomend choose n_step = 10*n_cells
        """
        for _ in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step: int):
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
        """Calculate average flow of model"""
        if self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velosity.sum() / self.matrix_velosity.shape[0] / self.n_cells

    def digramm_x_t(self):
        """
        Method make matrix of road
        size [n_cells, time step research]
        if cell is empty -> None
        if cell with cars -> value of car's velocity
        """
        if self.matrix_position is None or self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velosity[step]
        return road

    def average_velocity(self):
        if self.matrix_velosity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velosity.sum() / self.matrix_velosity.shape[0] / self.n_cars


class RealisticNFS_old:
    _velocity_max = 5
    #  probability for slow to start effect
    _q = 0.99
    #  probability for S
    _r = 0.99
    _S = [1, 2]
    #  probability for random breaking
    _p = 0.9
    _P1 = 0.999
    _P2 = 0.99
    _P3 = 0.98
    _P4 = 0.01
    # dist
    _D = 5  # 15

    def __init__(self, n_cells: int, n_cars: int, uniform=True) -> None:
        """
        n_cells: size of road in cells
        n_cars: number of cars
        uniform: <bool> uniform distribution of cars' position
        """
        if n_cars < 2:
            raise ValueError("Minimum 2 cars should be get")
        self.n_cells = n_cells
        self.n_cars = n_cars
        if uniform:
            self.car_position = np.arange(n_cars) * (n_cells // n_cars)
        else:
            self.car_position = np.sort(np.random.choice(n_cells, size=n_cars, replace=False))
        self.car_position_previous = self.car_position.copy()
        self.car_velocity = np.zeros(n_cars, dtype=int)
        # self.distance = np.zeros(n_cars, dtype=int)
        self.v_min = np.zeros(n_cars, dtype=int)
        self.stability = False
        self.matrix_position = None
        self.matrix_velocity = None

    def _step_old(self):
        """Make one time step of model"""
        #  step prepare
        self.anticipation = np.random.choice(self._S, size=self.n_cars, p=[1 - self._r, self._r])
        i = np.arange(self.n_cars)
        i_next = (i + self.anticipation) % self.n_cars
        initial_velocity = self.car_velocity.copy()
        #  d(i)
        length = 1
        distance = ((np.roll(self.car_position, -1) - self.car_position + self.n_cells)
                    % self.n_cells - length)

        #  rule 1 acceleration
        #  d >= D and v(i) <= v(i+1)
        distance_filter = distance >= self._D
        velocity_filter = initial_velocity <= np.roll(initial_velocity, -1)
        acceleration = (distance_filter * velocity_filter)  # 0 or 1
        self.car_velocity = np.min([self.v_max, self.car_velocity + acceleration], axis=0)

        #  rule 2 slow-to-start effect
        slow_start = np.random.choice([False, True], size=self.n_cars, p=[1 - self._q, self._q])
        distance_anticipation_previous = (
                (self.car_position_previous.take(i_next) - self.car_position_previous + self.n_cells)
                % self.n_cells - self.anticipation
        )
        if self.n_cars <= max(self._S):
            distance_anticipation_previous[i_next == i] = self.n_cells
        assert distance_anticipation_previous.min() >= 0
        velocity = np.min([self.car_velocity * slow_start, distance_anticipation_previous * slow_start], axis=0)
        self.car_velocity = np.max([self.car_velocity * np.invert(slow_start), velocity], axis=0)

        #  rule 3 perspective effect
        distance_anticipation = (
                (self.car_position.take(i_next) - self.car_position + self.n_cells)
                % self.n_cells - self.anticipation
        )
        if self.n_cars <= max(self._S):
            distance_anticipation[i_next == i] = self.n_cells
        assert distance_anticipation.min() >= 0
        self.car_velocity = np.min([self.car_velocity, distance_anticipation], axis=0)

        #  rule 4 random braking
        min_available_velocity = 1
        velocity_rule_4 = []
        for vel_i, dist_i, vel_next_init, vel_init in zip(
                self.car_velocity, distance,
                np.roll(initial_velocity, -1),
                initial_velocity,
        ):
            if vel_i <= min_available_velocity:
                velocity_rule_4.append(vel_i)
                continue
            if dist_i >= self._D:
                prob = self._P1
            else:
                if vel_init < vel_next_init:
                    prob = self._P2
                elif vel_init == vel_next_init:
                    prob = self._P3
                else:
                    prob = self._P4

            # random_break = np.random.choice([0, 1], size=1, p=[prob, 1 - prob])
            # random_break = np.random.binomial(1, 1-prob)
            if np.random.random() < 1 - prob:
                random_break = 1
            else:
                random_break = 0
            velocity_rule_4.append(vel_i - random_break)
        self.car_velocity = np.array(velocity_rule_4, dtype=object)

        #  rule 5 collision avoidance
        self.dis = distance
        self.car_velocity = np.min([self.car_velocity, np.roll(self.car_velocity, -1) + distance], axis=0)

        #  rule 6 position renewing
        self.car_position_previous = self.car_position.copy()
        self.car_position += self.car_velocity.astype(int)
        self.car_position %= self.n_cells

    def step(self):

        velocity = []
        for i in range(self.n_cars):
            if np.random.random() <= self._r:
                s = 2
            else:
                s = 1

            # if self.stability:
            #     velocity_1 = self._rule_1(i)
            # else:
            #     velocity_1 = self._rule_1_stab(i)
            velocity_1 = self._rule_1_stab(i)

            velocity_2 = self._rule_2(i, velocity_1, s)
            velocity_3 = self._rule_3(i, velocity_2, s)

            # if self.stability:
            #     velocity_4 = self._rule_4(i, velocity_3)
            # else:
            #     velocity_4 = self._rule_4_stab(i, velocity_3)
            velocity_4 = self._rule_4_stab(i, velocity_3)

            velocity.append(velocity_4)

        velocity = self._rule_5(velocity.copy())

        self.car_velocity = np.array(velocity)
        self.car_position_previous = self.car_position.copy()
        self.car_position += self.car_velocity.astype(int)
        self.car_position %= self.n_cells

    def _rule_1(self, i):
        """acceleration"""
        i_next = (i + 1) % self.n_cars
        distance = self._get_distance(i, i_next, 1)
        velocity = self.car_velocity[i]
        velocity_next = self.car_velocity[i_next]
        if (True
            and distance >= self._D
            and velocity <= velocity_next
        ):
            return min(self._velocity_max, velocity + 1)
        return velocity

    def _rule_1_stab(self, i):
        # S-NFS
        velocity = self.car_velocity[i]
        return min(self._velocity_max, velocity + 1)

    def _rule_2(self, i, velocity, s):
        """slow-to-start effect"""
        if np.random.random() > self._q:
            return velocity
        i_next = (i + s) % self.n_cars
        anticipation = self._get_distance_anticipation(i, i_next, s)
        return min(velocity, anticipation)

    def _rule_3(self, i, velocity, s):
        """perspective effect"""
        i_next = (i + s) % self.n_cars
        distance = self._get_distance(i, i_next, s)
        return min(velocity, distance)

    def _rule_4(self, i, velocity):
        """random breaking"""
        i_next = (i + 1) % self.n_cars
        distance = self._get_distance(i, i_next, 1)
        p = self._get_probability(i, i_next, distance)
        if np.random.random() < 1-p and velocity >= 1:
            return max(velocity-1, 1)
        else:
            return velocity

    def _rule_4_stab(self, i, velocity):
        # S-NFS
        if np.random.random() < 1-self._p:
            return max(0, velocity-1)
        else:
            return velocity

    def _rule_5(self, velocity):
        """collision avoidance"""
        velocity_result = []
        for i in range(self.n_cars):
            i_next = (i + 1) % self.n_cars
            distance = self._get_distance(i, i_next, 1)
            velocity_result.append(min(
                velocity[i], distance + velocity[i_next]
            ))
        return velocity_result

    def _get_distance(self, i, i_next, s):
        distance = (self.car_position[i_next] - self.car_position[i] - s)
        return distance + self.n_cells if distance < 0 else distance

    def _get_distance_anticipation(self, i, i_next, s):
        distance = (self.car_position_previous[i_next] - self.car_position_previous[i] - s)
        return distance + self.n_cells if distance < 0 else distance

    def _get_probability(self, i, i_next, distance):
        if distance >= self._D:
            return self._P1
        velocity = int(self.car_velocity[i])
        velocity_next = int(self.car_velocity[i_next])
        if velocity < velocity_next:
            return self._P2
        if velocity == velocity_next:
            return self._P3
        return self._P4

    def set_max_velocity(self, v_max: int) -> None:
        self._velocity_max = v_max
        self.v_max = np.full(self.n_cars, self._velocity_max, dtype=int)

    def system_stabilization(self, n_step: int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Recommend choose n_step = 10*n_cells
        """
        for i in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step: int):
        """
        Method save information about cars position and velocity
        n_step: time step for system research
        """
        if not self.stability:
            print("Model hasn't stabilizes")
        self.matrix_position = np.zeros((n_step, self.n_cars), dtype=int)
        self.matrix_velocity = np.zeros((n_step, self.n_cars), dtype=int)
        for step in range(n_step):
            self.matrix_position[step] = self.car_position
            self.step()
            self.matrix_velocity[step] = self.car_velocity

    def average_flow(self):
        """Calculate average flow of model"""
        if self.matrix_velocity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velocity.sum() / self.matrix_velocity.shape[0] / self.n_cells

    def average_velocity(self):
        if self.matrix_velocity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velocity.sum() / self.matrix_velocity.shape[0] / self.n_cars

    def diagram_x_t(self):
        """
        Method make matrix of road
        size [n_cells, time step research]
        if cell is empty -> None
        if cell with cars -> value of car's velocity
        """
        if self.matrix_position is None or self.matrix_velocity is None:
            raise ValueError("Research system before calculate flow")
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velocity[step]
        return road


class RevisedNFS:
    _velocity_max = 5
    #  probability for slow to start effect
    _q = 0.99
    #  probability for S
    _r = 0.99
    _S = [1, 2]
    #  probability for random breaking
    _p = 0.96
    _P1 = 0.999
    _P2 = 0.99
    _P3 = 0.98
    _P4 = 0.01
    # dist
    _D = 15

    def __init__(self, n_cells: int, n_cars: int, uniform=True) -> None:
        """
        n_cells: size of road in cells
        n_cars: number of cars
        uniform: <bool> uniform distribution of cars' position
        """
        if n_cars < 2:
            raise ValueError("Minimum 2 cars should be get")
        self.n_cells = n_cells
        self.n_cars = n_cars
        if uniform:
            self.car_position = np.arange(n_cars) * (n_cells // n_cars)
        else:
            self.car_position = np.sort(np.random.choice(n_cells, size=n_cars, replace=False))
        self.car_position_previous = self.car_position.copy()
        self.car_velocity = np.zeros(n_cars, dtype=int)
        # self.distance = np.zeros(n_cars, dtype=int)
        self.v_min = np.zeros(n_cars, dtype=int)
        self.stability = False
        self.matrix_position = None
        self.matrix_velocity = None

    def step(self):

        velocity = []
        for i in range(self.n_cars):
            if np.random.random() <= self._r:
                s = 2
            else:
                s = 1

            if self.stability:
                 velocity_1 = self._rule_1(i)
            else:
                 velocity_1 = self._rule_1_stab(i)

            velocity_2 = self._rule_2(i, velocity_1, s)
            velocity_3 = self._rule_3(i, velocity_2, s)

            if self.stability:
                velocity_4 = self._rule_4(i, velocity_3)
            else:
                velocity_4 = self._rule_4_stab(i, velocity_3)

            velocity.append(velocity_4)

        velocity = self._rule_5(velocity.copy())

        self.car_velocity = np.array(velocity)
        self.car_position_previous = self.car_position.copy()
        self.car_position += self.car_velocity.astype(int)
        self.car_position %= self.n_cells

    def _rule_1(self, i):
        """acceleration"""
        i_next = (i + 1) % self.n_cars
        distance = self._get_distance(i, i_next, 1)
        velocity = self.car_velocity[i]
        velocity_next = self.car_velocity[i_next]
        if (distance >= self._D
            or velocity <= velocity_next
        ):
            return min(self._velocity_max, velocity + 1)
        return velocity

    def _rule_1_stab(self, i):
        # S-NFS
        velocity = self.car_velocity[i]
        return min(self._velocity_max, velocity + 1)

    def _rule_2(self, i, velocity, s):
        """slow-to-start effect"""
        if np.random.random() > self._q:
            return velocity
        i_next = (i + s) % self.n_cars
        anticipation = self._get_distance_anticipation(i, i_next, s)
        return min(velocity, anticipation)

    def _rule_3(self, i, velocity, s):
        """perspective effect"""
        i_next = (i + s) % self.n_cars
        distance = self._get_distance(i, i_next, s)
        return min(velocity, distance)

    def _rule_4(self, i, velocity):
        """random breaking"""
        i_next = (i + 1) % self.n_cars
        distance = self._get_distance(i, i_next, 1)
        p = self._get_probability(i, i_next, distance)
        if np.random.random() < 1-p and velocity >= 1:
            return max(velocity-1, 1)
        else:
            return velocity

    def _rule_4_stab(self, i, velocity):
        # S-NFS
        if np.random.random() < 1-self._p:
            return max(0, velocity-1)
        else:
            return velocity

    def _rule_5(self, velocity):
        """collision avoidance"""
        velocity_result = []
        for i in range(self.n_cars):
            i_next = (i + 1) % self.n_cars
            distance = self._get_distance(i, i_next, 1)
            velocity_result.append(min(
                velocity[i], distance + velocity[i_next]
            ))
        return velocity_result

    def _get_distance(self, i, i_next, s):
        distance = (self.car_position[i_next] - self.car_position[i] - s)
        return distance + self.n_cells if distance < 0 else distance

    def _get_distance_anticipation(self, i, i_next, s):
        distance = (self.car_position_previous[i_next] - self.car_position_previous[i] - s)
        return distance + self.n_cells if distance < 0 else distance

    def _get_probability(self, i, i_next, distance):
        if distance >= self._D:
            return self._P1
        velocity = int(self.car_velocity[i])
        velocity_next = int(self.car_velocity[i_next])
        if velocity < velocity_next:
            return self._P2
        if velocity == velocity_next:
            return self._P3
        return self._P4

    def set_max_velocity(self, v_max: int) -> None:
        self._velocity_max = v_max
        self.v_max = np.full(self.n_cars, self._velocity_max, dtype=int)

    def system_stabilization(self, n_step: int):
        """
        Stabilizes system
        n_step: time step for stabilization
        Recommend choose n_step = 10*n_cells
        """
        for i in range(n_step):
            self.step()
        self.stability = True

    def system_research(self, n_step: int):
        """
        Method save information about cars position and velocity
        n_step: time step for system research
        """
        if not self.stability:
            print("Model hasn't stabilizes")
        self.matrix_position = np.zeros((n_step, self.n_cars), dtype=int)
        self.matrix_velocity = np.zeros((n_step, self.n_cars), dtype=int)
        for step in range(n_step):
            self.matrix_position[step] = self.car_position
            self.step()
            self.matrix_velocity[step] = self.car_velocity

    def average_flow(self):
        """Calculate average flow of model"""
        if self.matrix_velocity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velocity.sum() / self.matrix_velocity.shape[0] / self.n_cells

    def average_velocity(self):
        if self.matrix_velocity is None:
            raise ValueError("Research system before calculate flow")
        return self.matrix_velocity.sum() / self.matrix_velocity.shape[0] / self.n_cars

    def diagram_x_t(self):
        """
        Method make matrix of road
        size [n_cells, time step research]
        if cell is empty -> None
        if cell with cars -> value of car's velocity
        """
        if self.matrix_position is None or self.matrix_velocity is None:
            raise ValueError("Research system before calculate flow")
        road = np.full((self.matrix_position.shape[0], self.n_cells), np.nan)
        for step in range(road.shape[0]):
            road[step][self.matrix_position[step]] = self.matrix_velocity[step]
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
    model = NaShAuto(n_cells, n_cars - n_adr, n_adr)
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
        return v_max / n_cells
    model = StochasticNFS(n_cells, n_cars, prob, uniform)
    model.set_max_velosity(v_max)
    model.system_stabilization(time_s)
    model.system_research(time_r)
    return model.avarage_flow()


def get_flow_s_nfs_real(n_cars, n_cells, time_s, time_r, v_max=5, uniform=True):
    """
    Function for calculate flow of model
    n_cars: number of cars
    n_cells: size of road in cells
    time_s: time step for stabilize model
    time_r: time step for research model
    """
    if n_cars < 1:
        return (0, 0)
    if n_cars == 1:
        return (v_max / n_cells, v_max)
    model = RevisedNFS(n_cells, n_cars, uniform)
    model.stability = True
    model.set_max_velocity(v_max)
    model.system_stabilization(time_s)
    model.system_research(time_r)
    return (model.average_flow(), model.average_velocity())
