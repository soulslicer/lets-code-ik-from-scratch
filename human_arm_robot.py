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

from pynput import keyboard
import sys

from collections import OrderedDict

from ik import *

# Setup robot definition
human_arm_robot = OrderedDict({
    "base": {
        "type": "base",
        "dim": [0.025, 0.025, 0.025],
        "offset_T": translation_matrix([0, 0, 0]),
        "parent": None,
        "children": ["base_yaw"]
    },
    # Able to rotate the base about yaw
    "base_yaw": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 0, 1],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["left_trap", "right_trap"],
    },
    # Left Trap
    "left_trap": {
        "type": "link",
        "dim": [0.025, 0.025, 0.3],
        "offset_T": rotation_matrix(np.pi/2, [1,0,0]) @ translation_matrix([0, 0, 0.15]),
        "children": ["ls_roll"],
    },
    # Left Shoulder Parts
    "ls_roll": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, 0.15]),
        "children": ["ls_pitch"],
    },
    "ls_pitch": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 1, 0],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["ls_yaw"],
    },
    "ls_yaw": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 0, 1],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["left_shoulder"],
    },
    "left_shoulder": {
        "type": "link",
        "dim": [0.025, 0.025, 0.4],
        "offset_T": translation_matrix([0, 0, 0.2]),
        "children": ["la_roll"],
    },
    # Left arm parts
    "la_roll": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, 0.2]),
        "children": ["la_pitch"],
    },
    "la_pitch": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 1, 0],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["la_yaw"],
    },
    "la_yaw": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 0, 1],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["left_arm"],
    },
    "left_arm": {
        "type": "link",
        "dim": [0.025, 0.025, 0.4],
        "offset_T": translation_matrix([0, 0, 0.2]),
        "children": ["left_ef"],
    },
    "left_ef": {
        "type": "ef",
        "dim": [0.025, 0.025, 0.4],
        "offset_T": translation_matrix([0, 0, 0.2]),
        "children": [],
    },

    # Right Trap
    "right_trap": {
        "type": "link",
        "dim": [0.025, 0.025, 0.3],
        "offset_T": rotation_matrix(np.pi/2, [1,0,0]) @ translation_matrix([0, 0, -0.15]),
        "children": ["rs_roll"],
    },
    # Right Shoulder Parts
    "rs_roll": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, -0.15]),
        "children": ["rs_pitch"],
    },
    "rs_pitch": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 1, 0],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["rs_yaw"],
    },
    "rs_yaw": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 0, 1],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["right_shoulder"],
    },
    "right_shoulder": {
        "type": "link",
        "dim": [0.025, 0.025, 0.4],
        "offset_T": translation_matrix([0, 0, -0.2]),
        "children": ["ra_roll"],
    },
    # Right arm parts
    "ra_roll": {
        "type": "joint",
        "dim": 0.03,
        "axis": [1, 0, 0],
        "offset_T": translation_matrix([0, 0, -0.2]),
        "children": ["ra_pitch"],
    },
    "ra_pitch": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 1, 0],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["ra_yaw"],
    },
    "ra_yaw": {
        "type": "joint",
        "dim": 0.03,
        "axis": [0, 0, 1],
        "offset_T": translation_matrix([0, 0, 0]),
        "children": ["right_arm"],
    },
    "right_arm": {
        "type": "link",
        "dim": [0.025, 0.025, 0.4],
        "offset_T": translation_matrix([0, 0, -0.2]),
        "children": ["right_ef"],
    },
    "right_ef": {
        "type": "ef",
        "dim": [0.025, 0.025, 0.4],
        "offset_T": translation_matrix([0, 0, -0.2]),
        "children": [],
    },
})

# Create a new visualizer
vis = meshcat.Visualizer()
vis.open()

# Create robot
robot = Robot(vis, human_arm_robot)

# Create a target input
init_input = robot.get_joint_space_input()
for target_key, target_value in init_input.items():
    init_input[target_key] = 0.5
robot.compute(init_input, {}, False, True)

targets = {
    "left_ef": {"T": translation_matrix([0.5, -0.5, 0.25]), "w": weight_matrix(0, 5)},
    "right_ef": {"T": translation_matrix([0.5, 0.5, -0.25]), "w": weight_matrix(0, 5)},
    # "base_yaw": {"range": [-0.75,0.75], "w": 50},
}
robot.vis["left_ef_target"].set_transform(targets["left_ef"]["T"])
meshcat_shapes.frame(robot.vis["left_ef_target"])
robot.vis["right_ef_target"].set_transform(targets["right_ef"]["T"])
meshcat_shapes.frame(robot.vis["right_ef_target"])

mode = True
def on_press(key):
    global mode
    try:
        if mode is True:
            tf = "left_ef"
        elif mode is False:
            tf = "right_ef"
        if key.char == 'w':
            targets[tf]["T"][0,3] -= 0.01
        if key.char == 's':
            targets[tf]["T"][0,3] += 0.01
        if key.char == 'a':
            targets[tf]["T"][1,3] -= 0.01
        if key.char == 'd':
            targets[tf]["T"][1,3] += 0.01
        if key.char == 'q':
            targets[tf]["T"][2,3] -= 0.01
        if key.char == 'e':
            targets[tf]["T"][2,3] += 0.01
    except AttributeError:
        if 'shift' in str(key):
            mode = not mode
        if "ctrl" in str(key) or "esc" in str(key):
            sys.exit(0)

# Keyboard listener
print("Use WASDQE to move frame in xyz. Press shift to switch to other frame")
listener = keyboard.Listener(
    on_press=on_press,
    on_release=None)
listener.start()

# Constantly be solving
while 1:
    robot.vis["left_ef_target"].set_transform(targets["left_ef"]["T"])
    robot.vis["right_ef_target"].set_transform(targets["right_ef"]["T"])
    init_input, loss = robot.solve(init_input, targets, "PINV", lr=0.01, viz=True, viz_sleep=0.01, max_iter=1)

time.sleep(100)
