from tkinter import N
import pandas as pd
import numpy as np
from modules.models import NaShAuto

n_cells = 10000
time_stabil = 10*n_cells
time_research = 100
velosity_max = 5
n_density = 100

probability = 0.5
auto_car = 0.5

density = np.linspace(0, 1, n_density+1)[1:]
flow = np.zeros_like(density)
n_cars = np.int_(n_cells*density)
n_adr = np.int_(n_cars*auto_car)

for i in range(n_density):
    print(f"step: {i}/{n_density}")
    model = NaShAuto(n_cells, n_cars[i]-n_adr[i], n_adr[i])
    model.set_slow_probability(probability)
    model.system_stabilization(time_stabil)
    model.system_research(time_research)
    flow[i] = model.avarage_flow()

result = {
    'density': density,
    "flow": flow,
}

table = pd.DataFrame(result)
table.to_csv(f"data/nash_auto_{auto_car}.csv", index=False)