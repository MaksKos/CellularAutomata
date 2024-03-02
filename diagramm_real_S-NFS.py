"""This programm for calculate fundamebtal diagramm with different S-NFS model parametrs"""
import joblib
import pandas as pd
import numpy as np
from modules.models import get_flow_s_nfs_real, RealisticNFS
from joblib import Parallel, delayed

CORE = joblib.cpu_count()-2

dist = RealisticNFS._D

new = 'new'
v_max = 5
n_cells = 500
n_density = 10_000

time_stabilisation = 50
time_research = 50

# file_name = f"data/real_s_nfs_v_{v_max}_cells_{n_cells}_dist_{dist}_{new}.csv"
file_name = f"data/real_snfs_test_{n_density}_time_{time_research}.csv"

density = np.random.random(n_density)
density_start = np.linspace(0, 0.08, 100)
density = np.hstack([density_start, density])
cars = np.int_(n_cells*density)
 
result = Parallel(n_jobs=CORE, verbose=2)(delayed(get_flow_s_nfs_real)(car, n_cells, time_stabilisation, time_research, v_max, True) for car in cars)
#result = get_flow_s_nfs_real(300, n_cells, time_stabilisation, time_research, v_max, False)

flow = []
velocity = []
for f, v in result:
    flow.append(f)
    velocity.append(v)

result_dict = {
        'density': density,
        "flow": flow,
        "velocity": velocity,
        "cars": cars,
}

table = pd.DataFrame(result_dict)
table.to_csv(file_name, index=False)
print('Save: ', file_name)