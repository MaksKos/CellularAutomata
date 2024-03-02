"""This programm for calculate fundamebtal diagramm with different S-NFS model parametrs"""
import joblib
import pandas as pd
import numpy as np
from modules.models import get_flow_S_NFS
from joblib import Parallel, delayed

CORE = joblib.cpu_count()-2

# print("Input r [0...1]:")
# probability_r = float(input())
# if probability_r > 1 or probability_r < 0:
#     raise TypeError('invalid probability r [0...1]')
#
# print("Input q [0...1]:")
# probability_q = float(input())
# if probability_q > 1 or probability_q < 0:
#     raise TypeError('invalid probability q [0...1]')


v_max = 5
p = 0.9
n_cells = 500
n_density = 1000
# prob = {'p':p, "q":probability_q, 'r':probability_r}
prob = {'p':p, "q":0.99, 'r':0.99}

time_stabil = 300
time_research = 50

density = np.random.random(n_density)
cars = np.int_(n_cells*density)
 
flow =  Parallel(n_jobs=CORE, verbose=2)\
                    (delayed(get_flow_S_NFS)(car, n_cells, prob, time_stabil, time_research, v_max, False) for car in cars)

result = {
        'density': density,
        "flow": flow,
}

print('save in file')
table = pd.DataFrame(result)
table.to_csv(f"data/S-NFS_v_{v_max}_p_{p}.csv", index=False)
