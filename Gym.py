#%%
import gym
import numpy as np
import time

# define the time step
dt = 0.025

# create env
env = gym.make('Pendulum-v1', g=2)

# Reset the environment and get the initial state
state = env.reset()

# PID gains
kp = 0.2
ki = 0
kd = 0.1
# PID variables
integral_error = 0
previous_error = 0

# Target altitude
degrees_point = 0

# Run the simulation for 1000 steps
for _ in range(1000):
    # Render the environment
    env.render()

    ## Determine the action using PID control
    # get the current altitude
    x, y = state[[0, 1]]  
    angle = np.arccos(x)
    degrees = np.degrees(angle)
    if x > 0:
        degrees = -degrees
    
    print('angle: ',angle, 'degrees: ',degrees)

    #calculate the error
    error = degrees_point - degrees

    #calculate the integral error by adding the error to the integral (past) error
    integral_error += error

    #calculate the derivative error calculating the error change over time (gradient)
    derivative_error = (error - previous_error)/dt

    #update the previous error
    previous_error = error

    #calculate the action by multiplying the errors by the PID gains and summing them
    action = np.clip(error * kp + integral_error * ki + derivative_error * kd, -5, 5)

    # Execute the action and get the next state, reward, and whether the episode is done
    state, reward, done, _ = env.step([action])

    #render the environment
    env.render()
    # add a delay to slow down the env render
    time.sleep(dt)
    if done:
        state = env.reset()
# %%