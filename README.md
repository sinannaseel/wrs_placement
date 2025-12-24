# How To do's, cause it can get pretty messed up !!

1. Open the scene in coppeliaSim (Hope you have dq_robotics library installed, that is zmq interface and dqrobotics in python packages)
2. It is assumed that the block shapes are know and the orientation mean the same thing as in real world
3. first phase i have taken all cubes of different color to kinda have less orientation problem
3. Vision model will identify the pose (for now it is just x,y and depth via cv cause beginner lol)
4. set the positions in the coppeliaSim environment
5. the gymnasium interface will read the position of the environment and train to find the sequence which might not collapse everything (shapes for ppo seems irrelevant, minimal, norm of change in position)
6. the train_ppo python file will run and obtain the trained data
7. test_ppo python file will run to identify the sequence
8. robot controller will go towards the object and pick each one up with overhand grasp on the centroid of the blocks

---

## Dependencies

0. whatever dependencies wrs  and realsens3e demands
1. dqrobotics zmq interface
```bash
pip install dqrobotics-interface-coppeliasim-zmq
```

2. gymnasium environment
```bash
pip install gymnasium
```
3. stable_baseline3
```bash
pip install stable-baselines3
```

