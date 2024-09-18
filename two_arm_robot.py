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
two_arm_robot = {
    "base": {
        "type": "base",
        "dim": [0.025, 0.025, 0.025],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["j1"]
    },
    "j1": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["l1"]
    },
    "l1": {
        "type": "link",
        "dim": [0.025, 0.025, 0.5],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["j2", "j3"],
    },
    # One arm
    "j2": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["l2"],
    },
    "l2": {
        "type": "link",
        "dim": [0.025, 0.025, 0.5],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["ef2"],
    },
    "ef2": {
        "type": "ef",
        "dim": 0.03,
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": [],
    },
    # Another arm
    "j3": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 1, 0],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["l3"],
    },
    "l3": {
        "type": "link",
        "dim": [0.025, 0.025, 0.5],
        "offset_T": translation_matrix([0, 0, 0.25]),
        "children": ["ef3"],
    },
    "ef3": {
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
robot = Robot(vis, two_arm_robot)

# Create a target input
target_input = robot.get_joint_space_input()
for target_key, target_value in target_input.items():
    target_input[target_key] = 1.0

# Run FK on that input
robot.compute(target_input, {})

# Get some target frames of the EFs
target_output = robot.get_ef_frames_output()

# Attach joint limit
# target_output["j1"] = {"range": [-0.5, 1.5], "w": 5}
# target_output["j2"] = {"range": [-0.5, 1.5], "w": 5}
# target_output["j3"] = {"range": [-0.5, 1.5], "w": 5}

# Run a pass of IK on that 0 with target
init_input = robot.get_joint_space_input()
robot.compute(init_input, target_output, grad = True)

# Solve
robot.vis["ef2_target"].set_transform(target_output["ef2"]["T"])
meshcat_shapes.frame(robot.vis["ef2_target"])
robot.vis["ef3_target"].set_transform(target_output["ef3"]["T"])
meshcat_shapes.frame(robot.vis["ef3_target"])
time.sleep(1)
sln, loss = robot.solve(init_input, target_output, solver = "LM", lr=0.1, viz = True, viz_sleep=0.01)
