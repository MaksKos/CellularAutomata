from tkinter import N
import numpy as np

n_cells = 1000
time0 = 10*n_cells
tau = 100
precision = 0.5

density = 0.05

n_cars = int(n_cells*density)
car_position = np.random.choice(n_cells-1, size=n_cars, replace=False)
car_velosity = np.zeros(n_cars)
road = np.zeros(n_cells)

for index in car_position:
    road[index]=1


for _ in range(tau+time0):
    distance = np.roll(car_position, -1)-car_position
    distance[-1] += n_cells

    

