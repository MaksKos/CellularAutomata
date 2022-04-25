import joblib
import pandas as pd
import numpy as np
from modules.models import NaSh, get_flow
from joblib import Parallel, delayed

CORE = joblib.cpu_count()-2
print("Input slow probability [0...1]:")
probability = float(input())
if probability > 1 or probability < 0:
    raise TypeError('invalid probability [0...1]')

n_cells = 10_000
n_density = 100

time_stabil = 10*n_cells
time_research = 100

density = np.linspace(0, 1, n_density+1)
cars = np.int_(n_cells*density)

flow =  Parallel(n_jobs=CORE, verbose=10)\
                (delayed(get_flow)(car, n_cells, probability, time_stabil, time_research ) for car in cars)

result = {
    'density': density,
    "flow": flow,
}

print('save in file')
table = pd.DataFrame(result)
table.to_csv(f"data/nash_probability_{probability}.csv", index=False)
