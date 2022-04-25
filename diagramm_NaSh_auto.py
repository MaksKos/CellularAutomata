import pandas as pd
import numpy as np
from modules.models import NaShAuto, get_flow

CORE = 5
print("Input auto car proportion [0...1]:")
auto_car = float(input())

n_cells = 10000
time_stabil = 10*n_cells
time_research = 100
velosity_max = 5
n_density = 100

probability = 0.5

density = np.linspace(0, 1, n_density+1)[1:]
cars = np.int_(n_cells*density)
n_adr = np.int_(cars*auto_car)

flow = Parallel(n_jobs=CORE)(delayed(get_flow)(cars[i], n_adr[i], n_cells, probability, time_stabil, time_research ) for i in range(n_density))

result = {
    'density': density,
    "flow": flow,
}

table = pd.DataFrame(result)
table.to_csv(f"data/nash_auto_{auto_car}.csv", index=False)