"""Programm for draw x-t diagramm"""
import numpy as np
from modules.models import NaSh, NaShAuto
import matplotlib.pyplot as plt

n_cells = 1000
time0 = 10*n_cells
tau = 1000
p = 0.5

density = 0.4
auto_car = 0.5

n_cars = int(n_cells*density)
n_adr = int(n_cars*auto_car)

model = NaShAuto(n_cells, n_cars-n_adr, n_adr)
model.set_slow_probability(p)
model.system_stabilization(time0)
model.system_research(tau)

road_map = model.digramm_x_t()
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
fs = 16

fig = plt.figure(figsize=(10,6), dpi=300)

time = np.linspace(0, tau-1, tau, dtype=int)
cells = np.linspace(0, n_cells-1, n_cells, dtype=int)
grid = np.meshgrid(cells, time, indexing='xy')

#plt.scatter(grid[0], grid[1], marker='s', c='grey', s=1, alpha=0.2)
#bar_pic=plt.scatter(grid[0], grid[1], marker ='s', c = road_map, cmap = 'RdYlGn', s = 4, alpha = 1, edgecolors='black', linewidths=0.5 )
bar_pic=plt.scatter(grid[0], grid[1], marker ='s', c = road_map, cmap = 'RdYlGn', s = 0.1, alpha = 1 )
plt.xlim(0, n_cells)
plt.ylim(0, tau)
plt.gca().invert_yaxis()
plt.colorbar()
plt.clim(0, 5)
plt.title(f"x-t diagramm with density {density}")
plt.xlabel(r'$\mathrm{cells}$',  fontsize = fs)
plt.ylabel(r'$\mathrm{time}$',  fontsize = fs)
#plt.grid(linewidth=0.5)
plt.show()
fig.savefig(f'data/x_t_{density}.jpg', dpi = 200, pad_inches=0.1, format='jpg', bbox_inches='tight')