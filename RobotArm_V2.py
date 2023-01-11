from turtle import delay
import robosuite as suite
import numpy as np
import time
from robosuite.controllers import load_controller_config

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# define the time step
dt = 0.025

env = suite.make(env_name="Stack",
                 robots="Panda",
                 controller_configs=controller_config,
                 has_renderer=True,
                 has_offscreen_renderer=False,
                 use_camera_obs = False)

# reset the environment
obs = env.reset()
print(obs)

# PID gains X
x_kp = 10
x_ki = 0.01
x_kd = 3

# PID gains Y
y_kp = 10
y_ki = 0.01
y_kd = 3

# PID gains Z
z_kp = 10
z_ki = 0.01
z_kd = 3

# PID variables X
integral_error_x = 0
previous_error_x = 0

# PID variables Y
integral_error_y = 0
previous_error_y = 0

# PID variables Z
integral_error_z = 0
previous_error_z = 0

i = 0
stage = 0
done = False
cubeA_pos = obs['cubeA_pos']
SetPoint = cubeA_pos  

# loop through the simulation
for step in range(100):
  while not done:
      i = i + 1
      print("loop: ", i)
      
      # Retrive all relevant info from simulation
      gripper_pos = obs['robot0_eef_pos']
      distance = SetPoint - gripper_pos

      #calculate the error
      error_x = np.subtract(obs['robot0_eef_pos'][0], SetPoint[0])
      error_y = np.subtract(obs['robot0_eef_pos'][1], SetPoint[1])
      error_z = np.subtract(obs['robot0_eef_pos'][2], SetPoint[2])
      error = [error_x, error_y, error_z]
      mean_error = np.mean(np.abs(error))

      #calculate the integral error by adding the error to the integral (past) error
      integral_error_x += error_x
      integral_error_y += error_y
      integral_error_z += error_z

      #calculate the derivative error calculating the error change over time (gradient)
      derivative_error_x = (error_x - previous_error_x)/dt
      derivative_error_y = (error_y - previous_error_y)/dt
      derivative_error_z = (error_z - previous_error_z)/dt

      previous_error = [previous_error_x, previous_error_y, previous_error_z]
      mean_previous_error = np.mean(np.abs(previous_error))
      print('prev_error_v1: ', previous_error)


      #calculate the action by multiplying the errors by the PID gains and summing them
      action_x = -np.clip(error_x * x_kp + integral_error_x * x_ki + derivative_error_x * x_kd, -5, 5)
      action_y = -np.clip(error_y * y_kp + integral_error_y * y_ki + derivative_error_y * y_kd, -5, 5)
      action_z = -np.clip(error_z * z_kp + integral_error_z * z_ki + derivative_error_z * z_kd, -5, 5)

      #Gripper to cube
      if stage == 0:
        action = [action_x, action_y, action_z, 0, 0, 0, -1]
        
        ##if the mean_error does not decreas significantly, move to next stage
        if np.mean(np.abs(distance))<0.003:
          stage = 1

      #update the previous error
      previous_error_x = error_x
      previous_error_y = error_y
      previous_error_z = error_z
      previous_error = [previous_error_x, previous_error_y, previous_error_z]

      print("distance: ", distance)
      print('MAE: ', mean_error)
      print('PMAE: ', mean_previous_error)
      print('prev_error_v2: ', previous_error)
      print('stage: ', stage)
      print("###########################################################")

      def delay(timesteps,action,env):
        for i in range(timesteps):
          obs, _, _, _ = env.step(action)
          env.render() #if you are using the remote environment you will need to comment this out, or use the remote render code
        return obs

      if stage == 1:
        action = [0,0,0,0,0,0,0.3]
        #Using a helper function to send the gripping action to the environment for 30 time steps
        obs = delay(30,action,env)
        #Advance to the next stage
        stage = 2
    

       # Stage 2: Lift
      if stage == 2:
        z = [gripper_pos[0], gripper_pos[1], 1.1]
        action = [0, 0, 1, 0, 0, 0, 0.1]
        h = gripper_pos - z
        print(h)

        if all(item > -0.005  and item < 0.005  for item in h):
          stage = 3
      
      # Stage 3: Move gripper to cubeB
      if stage == 3:
        #Get cubeB_pos and make it SetPoint. Then edit the z to be a bit above the cube
        cubeB_pos = obs['cubeB_pos']
        #gripper_pos = obs['robot0_eef_pos']
        SetPoint = [cubeB_pos[0], cubeB_pos[1], cubeB_pos[2] + 0.06]
        distance = SetPoint - gripper_pos
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
          
        ##if the mean_error does not decreas significantly, move to next stage
        if np.mean(np.abs(distance))<0.004:
          stage = 4

        #update the previous error
        previous_error_x = error_x
        previous_error_y = error_y
        previous_error_z = error_z
        previous_error = [previous_error_x, previous_error_y, previous_error_z]

        #Stage 4: let CubeA go
        if stage == 4:
          action = [0,0,0,0,0,0, -0.3]
          #Using a helper function to send the gripping action to the environment for 30 time steps
          obs = delay(30,action,env)
          #Advance to the next stage
          stage = 5
        
        #Stage 6: Move arm up and done
        if stage == 5:
          z = [gripper_pos[0], gripper_pos[1], 1.1]
          action = [0, 0, 0.1, 0, 0, 0, 0.1]
          h = gripper_pos - z
          print(h)

          if all(item > -0.005  and item < 0.005  for item in h):
            done = True

      # Execute the action and get the next state, reward, and whether the episode is done
      obs, reward, done, _ = env.step(action)

      #render the environment
      env.render()
      # add a delay to slow down the env render
      time.sleep(dt)
      if done:
         state = env.reset()