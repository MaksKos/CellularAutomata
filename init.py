
import numpy as np
from tqdm import tqdm

n_cells = 10000
time0 = 10*n_cells
tau = 100
probability = 0.5
velosity_max = 5
density = 0.05

n_cars = int(n_cells*density)
v_max = np.full(n_cars, velosity_max, dtype=int)
v_min = np.zeros(n_cars, dtype=int)

car_position = np.sort(np.random.choice(n_cells, size=n_cars, replace=False))
car_velosity = np.zeros(n_cars, dtype=int)
distance = np.zeros(n_cars, dtype=int)

for _ in tqdm(range(time0)):
    distance = np.roll(car_position, -1)-car_position
    distance %= n_cells
    car_velosity = np.min([car_velosity+1, v_max, distance-1], axis=0)
    slow = np.random.choice([0,1], size=n_cars, p=[1-probability, probability])
    car_velosity = np.max([car_velosity-slow, v_min], axis=0)
    car_position += car_velosity
    car_position %= n_cells

