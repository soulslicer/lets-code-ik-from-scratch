import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
import copy

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import meshcat_shapes

from ik import *

# Setup robot definition
double_pendulum_robot = {
    "base": {
        "type": "base",
        "dim": [0.025, 0.025, 0.025],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["j0"]
    },
    "j0": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["l0"]
    },
    "l0": {
        "type": "link",
        "dim": [0.025, 0.025, 0.5],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["j1"],
    },
    "j1": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["l1"],
    },
    "l1": {
        "type": "link",
        "dim": [0.025, 0.025, 0.5],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["ef"],
    },
    "ef": {
        "type": "ef",
        "dim": 0.03,
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": [],
    }
}

# Create a new visualizer
vis = meshcat.Visualizer()
vis.open()

# Create robot
robot = Robot(vis, double_pendulum_robot)

# Setup a toy problem
target_input = {"j0": np.pi/2 + 1.0, "j1": np.pi/2 + 0.5}
print(target_input)
robot.fk(target_input, {})
t = robot.frames["ef"]["global_T"]
w = np.zeros((4,4))
w[0:3, 0:3] = 1
w[0:3, 3] = 5
# Visualize it
robot.vis["target"].set_transform(t)
meshcat_shapes.frame(robot.vis["target"])
# Next compute a fk with some offset and the target
inputs = {"j0": 0.3, "j1": 0.3} # 0,0 fails with G
targets = {"ef": {"T": t, "w": w}, "j1": {"range": [-3,3], "w": 5}}
robot.compute(inputs, {}, viz = True)
time.sleep(1)
#sln, loss = robot.solve(inputs, targets, solver = "LM", viz = False)
#assert(loss < 1e-5)

# Generate sample data for the loss landscape
x = np.linspace(-np.pi - 1.0, np.pi + 1.0, 150)
y = np.linspace(-np.pi - 1.0, np.pi + 1.0, 150)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y**2)
for u in range(0, x.shape[0]):
    for v in range(0, y.shape[0]):
        j0 = X[u,v]
        j1 = Y[u,v]
        total_losses, _, _ = robot.compute({"j0": j0, "j1": j1}, targets)
        Z[u,v] = np.sum(total_losses)

# Draw contour plot
fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.title('Loss Contour Plot')
plt.xlabel('j0')
plt.ylabel('j1')

# Visualize the solving process
viz_targets = [[target_input["j0"], target_input["j1"]], [target_input["j0"]-np.pi*2, target_input["j1"]], [target_input["j0"], target_input["j1"]-np.pi*2]]
for vt in viz_targets:
    ax.plot(vt[0], vt[1], marker="o", markersize=3, markeredgecolor="green", markerfacecolor="green")
plt.ion()
def cb(input):
    ax.plot(input["j0"], input["j1"], marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
    plt.pause(0.001)
sln, loss = robot.solve(inputs, targets, solver = "LM", lr = 0.05, lm_damping=0, viz = True, cb=cb, viz_sleep=0)

# Pause
time.sleep(100)
