## Lets Code IK (Inverse Kinematics) from Scratch

Code: https://github.com/soulslicer/lets-code-ik-from-scratch

Here is a fun little weekend project. Lets work through a notebook to learn how to do IK from scratch! We will write a barebones IK (Inverse Kinematics) solver for multi-body trees for anyone to understand, while going through the math, code and logic step by step. We will keep it simple. No complex robot representations (URDF, MJCF etc.), no rigid body library to load them and compute derivatives (Mujoco, Pinnochio), no need for lie algebra theory, and no IK libraries or solvers (OSQP). Just barebones python, numpy for the matrix math, and meshcat for the visualization. The goal is to be able to build something like what you see below, and derive all the math from scratch with references! 

Begin here: [Link](https://github.com/soulslicer/lets-code-ik-from-scratch/blob/main/LetsCodeIKFromScratch.ipynb)

![intro](video.gif)

### Requirements

```
pip install meshcat
pip install meshcat-shapes
pip install numpy
pip install opencv-python
pip install matplotlib
pip install jupyter
pip install jupyter-notebook
pip install keyboard
pip install pyinput
```

### Code examples

```
python dp_robot.py
python two_arm_robot.py
python human_arm_robot.py
```

### Notebooks

```
jupyter-notebook $PWD
```