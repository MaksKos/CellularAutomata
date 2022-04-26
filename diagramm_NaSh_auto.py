import joblib
import pandas as pd
import numpy as np
from modules.models import NaShAuto, get_flow_auto
from joblib import Parallel, delayed

CORE = joblib.cpu_count()-2
print("Input auto car proportion [0...1]:")
auto_car = float(input())
if auto_car > 1 or auto_car < 0:
    raise TypeError('invalid proportion [0...1]')

n_cells = 10_000
n_density = 100

time_stabil = 10*n_cells
time_research = 100
probability = 0.5

density = np.linspace(0, 1, n_density+1)
cars = np.int_(n_cells*density)
n_adr = np.int_(cars*auto_car)

flow = Parallel(n_jobs=CORE, verbose=10)\
            (delayed(get_flow_auto)(cars[i], n_adr[i], n_cells, 
            probability, time_stabil, time_research ) for i in range(n_density))

result = {
    'density': density,
    "flow": flow,
}

print('save in file')
table = pd.DataFrame(result)
table.to_csv(f"data/nash_auto_{auto_car}.csv", index=False)