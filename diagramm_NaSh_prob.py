import pandas as pd
import numpy as np
from modules.models import NaSh, get_flow
from joblib import Parallel, delayed

CORE = 5

n_cells = 10000
time_stabil = 10*n_cells
time_research = 100
probability = 0.5
velosity_max = 5
n_density = 100

density = np.linspace(0, 1, n_density+1)[1:]
cars = np.int_(n_cells*density)
"""
flow = np.zeros_like(density)
for i in range(n_density):
    flow[i] = get_flow(n_cars[i], n_cells, probability,
                        time_stabil, time_research)
"""
flow =  Parallel(n_jobs=CORE)(delayed(get_flow)(car, n_cells, probability, time_stabil, time_research ) for car in cars)

result = {
    'density': density,
    "flow": flow,
}

table = pd.DataFrame(result)
table.to_csv(f"data/nash_probability_{probability}.csv", index=False)
