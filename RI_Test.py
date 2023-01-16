from stable_baselines3 import PPO
from turtle import delay
import robosuite as suite
import numpy as np
import time
from CustomGym import RoboEnv

model = PPO.load("/home/howl/robosuite/robosuite-1/model.zip", print_system_info=True)

env = RoboEnv(RenderMode = False)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()