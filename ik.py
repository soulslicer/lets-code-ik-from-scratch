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

from collections import OrderedDict
np.set_printoptions(suppress=True)


def weight_matrix(ow : float, tw : float):
    """
    Given an orientation weight and translation weight
    Returns a 4x4 weight matrix to be applied to a 4x4 SO3 matrix
    """
    w = np.zeros((4,4))
    w[0:3, 0:3] = ow
    w[0:3, 3] = tw
    return w


def translation_matrix(direction : np.array):
    """
    Given a 1x3 direction vector
    Returns a 4x4 SO3 matrix with just translation components
    """
    T = np.identity(4)
    T[:3, 3] = direction[:3]
    return T


def skew(v):
    """
    Given a 1x3 direction vector
    Returns a 3x3 skew symmetric matrix 
    """
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def rotation_matrix(theta : float, v : np.array, grad = False):
    """
    Given the axis angle representation of theta and v
    Returns a 4x4 SO3 matrix with just rotation components
    Returns a 4x4 dThetadT matrix as well if requested 
    """
    v = np.array(v).astype(np.float64)
    if np.linalg.norm(v) != 1:
        raise Exception("Unnormalized direction")
    sint = math.sin(theta)
    cost = math.cos(theta)
    R = (1 - cost)* (np.outer(v, v)) + sint*skew(v) + cost*np.eye(3)
    T = np.identity(4)
    T[:3, :3] = R

    if not grad: 
        return T
    else:
        dThetadR = (-sint)*np.eye(3) + sint*np.outer(v, v) + cost*skew(v)
        dThetadT = np.identity(4) * 0
        dThetadT[:3, :3] = dThetadR
        return T, dThetadT

# Verify rotation matrix gradient
h = 1e-5
dThetadT_numerical = (rotation_matrix(0.5 + h, [1, 0, 0]) - rotation_matrix(0.5, [1, 0, 0])) / h
dThetadT_analytical = rotation_matrix(0.5, [1, 0, 0], grad = True)[1]
assert(np.linalg.norm(dThetadT_numerical - dThetadT_analytical) < 1e-5)


# A method that returns 0 within range or the max val otherwise
def joint_limit(x, range, max, skew=10., grad=False):
    """
    A method that returns 0 if x is within range or tends to max otherwise
    """
    # https://www.desmos.com/calculator/aja9qlbmc8
    lr = range[0]
    rr = range[1]
    s = skew
    m = max

    a = np.exp(-s*(x-lr)) # a
    b = np.exp(-s*(-x+rr)) # b
    limit = m * (1 - (1/(1+a)) * (1/(1+b)))
    if not grad:
        return limit
    dLimitdX = m * ((s*b)/((b+1)**2 * (a+1)) - (s*a)/((a+1)**2 * (b+1)))
    return limit, dLimitdX

# Verify joint limit gradient
h = 1e-5
dLimitdX_numerical = (joint_limit(-2 + h, range=[-2,2], max=2.) - joint_limit(-2, range=[-2,2], max=2.)) / h
limit, dLimitdX_analytical = joint_limit(-2 + h, range=[-2,2], max=2., grad=True)
assert(np.linalg.norm(dLimitdX_numerical - dLimitdX_analytical) < 1e-5)


def iterate_frames(frames, starting="base"):
    """
    A helper method to iterate a robot frame tree in a BFS manner so we always visit the parent before child
    """
    stack = [(frames[starting], [starting])]
    while stack:
        curr_node, curr_traj = stack.pop()
        parent_key = None
        if len(curr_traj) >= 2: parent_key = curr_traj[-2]
        parent_node = None
        if parent_key in frames.keys(): parent_node = frames[parent_key]
        yield curr_node, parent_node, curr_traj
        for child in curr_node["children"]:
            if child in frames:
                stack.append((frames[child], curr_traj + [child]))


# Robot class
class Robot:
    def __init__(self, vis, rdf):
        self.vis = vis

        # Setup robot definition
        self.frames = copy.deepcopy(rdf)

        # Populate keys
        for key, frame in self.frames.items():
            frame["key"] = key

        # Did i originate cache
        self.traj_cache = {}
        for curr_node, parent_node, curr_traj in iterate_frames(self.frames):
            curr_key = curr_node["key"]
            self.traj_cache[curr_key] = curr_traj

        # Setup the visualizer
        self.setup_vis()
        
        # Get the robot to visualize the T pose
        inputs = self.get_joint_space_input()
        self.fk(inputs, {}, False, True)

    def get_joint_space_input(self):
        inputs = {}
        for curr_node, parent_node, _ in iterate_frames(self.frames):
            if curr_node["type"] == "joint":
                inputs[curr_node["key"]] = 0
        return inputs
    
    def get_ef_frames_output(self):
        inputs = {}
        for curr_node, parent_node, _ in iterate_frames(self.frames):
            if curr_node["type"] == "ef":
                inputs[curr_node["key"]] = {"T": curr_node["global_T"], "w": weight_matrix(1, 1)}
        return inputs

    def solve(self, inputs, targets, solver = "LM", lr = 1.0, lm_damping = 0.0006, viz = False, cb = None, viz_sleep = 0.1, min_eps = 1e-6, max_iter = 1000):
        # Make a copy of current state vector
        inputs = copy.deepcopy(inputs)
        
        # Solve
        prev_loss = 100
        total_iter = 0
        while 1:
            # Compute IK
            total_losses, total_grads, total_orderings = self.fk(inputs, targets, grad = True, viz = viz)
            # Compute optimal step with gauss newton
            step = None
            if solver == "GN":
                step = (np.linalg.inv(total_grads.T @ total_grads) @ total_grads.T @ total_losses) * lr
            # Compute optimal step with LMA
            elif solver == "LM":
                jtj = total_grads.T @ total_grads
                step = (np.linalg.inv(jtj + np.eye(len(inputs))*jtj*lm_damping)) @ total_grads.T @ total_losses * lr
            # Compute gradient step
            elif solver == "G":
                step = total_grads.T @ total_losses * lr
            # Compute optimal step using pinv
            elif solver == "PINV":
                step = np.linalg.pinv(total_grads) @ total_losses * lr

            # Pass step into state vector
            for idx, key in enumerate(total_orderings):
                inputs[key] = inputs[key] - step[idx]

            # Callback
            if cb:
                cb(inputs)

            # Termination
            total_iter += 1
            loss = np.sum(total_losses)
            eps = np.abs(loss - prev_loss) 
            prev_loss = loss

            # No more updates possible
            if eps < min_eps: break

            # Too many iterations
            if total_iter > max_iter: break
            
            # Sleep if viz on
            if viz: time.sleep(viz_sleep)

        return inputs, prev_loss

    def fk(self, inputs, targets, grad = False, viz = False):

        ##########################################
        # Compute FK
        ##########################################

        # Iterate FK chain
        for curr_node, parent_node, _ in iterate_frames(self.frames):
            curr_key = curr_node["key"]

            # Base frame
            if curr_node["type"] == "base":
                curr_node["global_T"] = curr_node["offset_T"]

            # Joint frame
            elif curr_node["type"] == "joint":
                curr_node["j_T"], curr_node["j_T_grad"] = rotation_matrix(inputs[curr_key], curr_node["axis"], grad=True)
                curr_node["global_T"] = parent_node["global_T"] @ curr_node["offset_T"] @ curr_node["j_T"]

            # Link frame
            elif curr_node["type"] == "link" or curr_node["type"] == "ef":
                curr_node["global_T"] = parent_node["global_T"] @ curr_node["offset_T"]

        ##########################################
        # Visualize FK frames
        ##########################################

        # Draw frames
        if viz:
            for curr_node, parent_node, _ in iterate_frames(self.frames):
                curr_key = curr_node["key"]
                self.vis[curr_key].set_transform(curr_node["global_T"])
                if curr_node["type"] == "ef":
                    meshcat_shapes.frame(self.vis[curr_key])

        ##########################################
        # Compute IK Jacobians
        ##########################################

        # Iterate IK chain
        for j in inputs.keys():
            if not grad: continue

            # Iterate FK chain
            for curr_node, parent_node, _ in iterate_frames(self.frames):
                curr_key = curr_node["key"]

                # Base frame
                if curr_node["type"] == "base":
                    curr_node["global_J_" + j] = curr_node["global_T"]

                # Joint frame
                elif curr_node["type"] == "joint":
                    if curr_key == j:
                        j_grad = curr_node["j_T_grad"]
                    else:
                        j_grad = curr_node["j_T"]
                    curr_node["global_J_" + j] = parent_node["global_J_" + j] @ curr_node["offset_T"] @ j_grad

                # Link frame
                elif curr_node["type"] == "link" or curr_node["type"] == "ef":
                    curr_node["global_J_" + j] = parent_node["global_J_" + j]  @ curr_node["offset_T"]

                # Special case at ef, we zero out the gradient if its not part of the tree
                if curr_node["type"] == "ef":
                    if j not in self.traj_cache[curr_key]:
                        curr_node["global_J_" + j] *= 0

        ##########################################
        # Compute Target Losses and Grads
        ##########################################

        # Generate the ordering that we return losses and gradients with
        joint_orderings = []
        for input_key, _ in inputs.items():
            joint_orderings.append(input_key)

        # Define ef target losses and grads
        ef_losses = []
        ef_grads = []

        # Define joint losses and grads
        joint_losses = np.zeros(len(inputs))
        joint_grads = np.zeros(len(inputs))

        # Iterate over targets
        for target_key, target in targets.items():
            
            # Loss target is EF
            if self.frames[target_key]["type"] == "ef":

                # Compute target loss
                loss = (self.frames[target_key]["global_T"] - target["T"]) * target["w"]
                squared_loss = loss**2
                ef_losses.append(squared_loss.flatten())

                # Iterate each input variable
                if not grad: continue
                grads = []
                for input_key, _ in inputs.items():
                    loss_J = 2 * (loss.flatten()) * self.frames[target_key]["global_J_" + input_key].flatten()
                    grads.append(loss_J)

                grads = np.vstack(grads)
                ef_grads.append(grads.T)

            # Loss target is Joint
            elif self.frames[target_key]["type"] == "joint":

                # Compute joint limit loss
                rloss, rgrad = joint_limit(inputs[target_key], range=target["range"], max=5, skew=16, grad=True)
                joint_losses[joint_orderings.index(target_key)] = rloss * target["w"]
                if not grad: continue
                joint_grads[joint_orderings.index(target_key)] = rgrad * target["w"]

        # Sum and combine the ef losses and joint losses
        ef_losses = np.sum(np.array(ef_losses), axis=0)
        joint_losses = np.array(joint_losses)
        total_losses = np.hstack([ef_losses, joint_losses])
        if not grad: return total_losses, None, joint_orderings

        # Sum and combine the ef grads and joint grads
        ef_grads = np.sum(np.array(ef_grads), axis=0)
        joint_grads = np.array(joint_grads) * np.eye(len(inputs))
        total_grads = np.vstack([ef_grads, joint_grads])

        return total_losses, total_grads, joint_orderings

    def setup_vis(self):
        for name, frame in self.frames.items():
            if frame["type"] == "base":
                self.vis[name].set_object(g.Box(frame["dim"]))
            elif frame["type"] == "link":
                self.vis[name].set_object(g.Box(frame["dim"]))
            elif frame["type"] == "joint":
                self.vis[name].set_object(g.Sphere(frame["dim"]))
            elif frame["type"] == "ef":
                self.vis[name].set_object(g.Sphere(frame["dim"]))

    def fk_m(self, inputs, targets):
        base_transform = self.frames["base"]["offset_T"]
        j1_T, j1_q1_grad = rotation_matrix(inputs["j1"], self.frames["j1"]["axis"], grad=True)
        self.frames["j1"]["global_T"] = base_transform @ self.frames["j1"]["offset_T"] @ j1_T
        self.frames["l1"]["global_T"] = self.frames["j1"]["global_T"]  @ self.frames["l1"]["offset_T"]
        j2_T, j2_q2_grad = rotation_matrix(inputs["j2"], self.frames["j2"]["axis"], grad=True)
        self.frames["j2"]["global_T"] = self.frames["l1"]["global_T"] @ self.frames["j2"]["offset_T"] @ j2_T
        self.frames["l2"]["global_T"] = self.frames["j2"]["global_T"] @ self.frames["l2"]["offset_T"]
        self.frames["ef"]["global_T"] = self.frames["l2"]["global_T"] @ self.frames["ef"]["offset_T"]
        ef_frame = self.frames["ef"]["global_T"]

        # Q1
        j1_q1_grad = base_transform @ self.frames["j1"]["offset_T"] @ j1_q1_grad
        l1_q1_grad = j1_q1_grad  @ self.frames["l1"]["offset_T"]
        j2_q1_grad = l1_q1_grad @ self.frames["j2"]["offset_T"] @ j2_T
        l2_q1_grad = j2_q1_grad @ self.frames["l2"]["offset_T"]
        ef_q1_grad = l2_q1_grad @ self.frames["ef"]["offset_T"] 

        # Q2
        j1_q2_grad = base_transform @ self.frames["j1"]["offset_T"] @ j1_T
        l1_q2_grad = j1_q2_grad  @ self.frames["l1"]["offset_T"]
        j2_q2_grad = l1_q2_grad @ self.frames["j2"]["offset_T"] @ j2_q2_grad
        l2_q2_grad = j2_q2_grad @ self.frames["l2"]["offset_T"]
        ef_q2_grad = l2_q2_grad @ self.frames["ef"]["offset_T"]
