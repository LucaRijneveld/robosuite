from turtle import delay
import robosuite as suite
import numpy as np
import time
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation as R

#convert quaternion to roll, pitch, yaw
def quat_to_rpy(q):
    #convert quaternion to roll, pitch, yaw
    rpy = R.from_quat(q).as_euler('xyz', degrees=True)
    #transform yaw to be between -90 and 90
    if rpy[2]>90:
        rpy[2] = rpy[2]-180
    elif rpy[2]<-90:
        rpy[2] = rpy[2]+180
    return rpy

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# define the time step
dt = 0.025

env = suite.make(env_name="PickPlace",
                 robots="Panda",
                 controller_configs=controller_config,
                 has_renderer=True,
                 has_offscreen_renderer=False,
                 use_camera_obs = False)

# reset the environment
obs = env.reset()
print(obs)

#PID CONTROLLER
# PID gains X
x_kp = 12
x_ki = 0.01
x_kd = 3

# PID gains Y
y_kp = 12
y_ki = 0.01
y_kd = 3

# PID gains Z
z_kp = 12
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

#YAW CONTROLLER
#PID gains Yaw
yaw_kp = 0.1
yaw_ki = 0
yaw_kd = 0

# PID variables yaw
integral_error_yaw = 0
previous_error_yaw = 0

i = 0
stage = 0
done = False
Cereal_pos = obs['Cereal_pos']
SetPoint = [Cereal_pos[0], Cereal_pos[1], 1.1]
Cereal_quat = obs['Cereal_quat']
SetQuat = Cereal_quat

# loop through the simulation
for step in range(1000):
    while not done:
      i = i + 1
      print("loop: ", i)
      
      # Retrive all relevant info from simulation
      gripper_pos = obs['robot0_eef_pos']
      gripper_quat = obs['robot0_eef_quat']
      distance = SetPoint - gripper_pos

      Cereal_bin = np.array([0, 0.42, 1])
      Milk_bin = np.array([0.0, 0.18, 1])
      Can_bin = np.array([0.16, 0.18, 1])
        
      robot_yaw = quat_to_rpy(gripper_quat)[2]
      Set_yaw = quat_to_rpy(SetQuat)[2]

      diff = Set_yaw - robot_yaw

      error_yaw = np.subtract(robot_yaw, Set_yaw)
      integral_error_yaw += error_yaw
      derivative_error_yaw = (error_yaw - previous_error_yaw)/dt
      action_yaw = -np.clip(error_yaw * yaw_kp + integral_error_yaw * yaw_ki + derivative_error_yaw * yaw_kd, -5, 5)

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

      #Stage 0: Gripper YAW
      if stage == 0:
        action = [0, 0, 0, 0, 0, action_yaw, 0]

        #if diff does not decreas significantly, move to next stage
        if np.mean(np.abs(diff))<0.003:
          stage = 1
      
      #update the previous error
      previous_error_yaw = error_yaw

      #Stage 1: Gripper to Cereal
      if stage == 1:
        action = [action_x, action_y, action_z, 0, 0, 0, -1]
        
        ##if the mean_error does not decreas significantly, move to next stage
        if np.mean(np.abs(distance))<0.003:
          stage = 2

      if stage == 2:
        SetPoint = [Cereal_pos[0], Cereal_pos[1], Cereal_pos[2]+ 0.05]
        distance = SetPoint - gripper_pos
        action = [action_x, action_y, action_z, 0, 0, 0, 0]

        if np.mean(np.abs(distance))<0.0065:
          stage = 3

      previous_error_x = error_x
      previous_error_y = error_y
      previous_error_z = error_z
      previous_error = [previous_error_x, previous_error_y, previous_error_z]

      print("distance: ", np.mean(np.abs(distance)))
      print('stage: ', stage)
      print("###########################################################")

      def delay(timesteps,action,env):
        for i in range(timesteps):
          obs, _, _, _ = env.step(action)
          env.render() #if you are using the remote environment you will need to comment this out, or use the remote render code
        return obs

      if stage == 3:
        action = [0,0,0,0,0,0,0.3]
        #Using a helper function to send the gripping action to the environment for 30 time steps
        obs = delay(30,action,env)
        #Advance to the next stage
        stage = 4
    
       # Stage 4: Lift
      if stage == 4:
        SetPoint = [gripper_pos[0], gripper_pos[1], 1.5]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.005:
          stage = 5
      
      # Stage 5: Move gripper to Cereal Bin
      if stage == 5:
        SetPoint = [Cereal_bin[0], Cereal_bin[1], 1.1]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.015:
          stage = 6
      
      #Stage 6: Go down to the bin
      if stage == 6:
        SetPoint = [Cereal_bin[0], Cereal_bin[1], Cereal_bin[2]]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.0065:
          stage = 7
      
      #Stage 7: Drop box, Move up again and change YAW to match Milk
      if stage == 7:
        Milk_quat = obs['Milk_quat']
        gripper_quat = obs['robot0_eef_quat']
        SetPoint = [Cereal_bin[0], Cereal_bin[1], 1.1]
        SetQuat = Milk_quat
        action = [action_x, action_y, action_z, 0, 0, action_yaw, -0.5]
        distance = SetPoint - gripper_pos
        diff = Set_yaw - robot_yaw

        if np.mean(np.abs(distance)) and np.mean(np.abs(diff))<0.003:
          stage = 8
      
      #Stage 8: Move to be above the Milk
      if stage == 8:
        Milk_pos = obs['Milk_pos']
        SetPoint = [Milk_pos[0], Milk_pos[1], 1.1]
        action = [action_x, action_y, action_z, 0, 0, 0, 0]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))>0.003:
          stage = 9

      #Stage 9: Move down to Milk
      if stage == 9:
        SetPoint = [Milk_pos[0], Milk_pos[1], Milk_pos[2]+ 0.05]
        distance = SetPoint - gripper_pos
        action = [action_x, action_y, action_z, 0, 0, 0, 0]

        if np.mean(np.abs(distance))<0.006:
          stage = 10
      
      #Stage 10: Grap Milk
      if stage == 10:
        action = [0,0,0,0,0,0,0.3]
        #Using a helper function to send the gripping action to the environment for 30 time steps
        obs = delay(30,action,env)
        #Advance to the next stage
        stage = 11

      #Stage 11: Lift
      if stage == 11:
        SetPoint = [gripper_pos[0], gripper_pos[1], 1.5]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.005:
          stage = 12
      
      # Stage 12: Move gripper to Milk Bin
      if stage == 12:
        SetPoint = [Milk_bin[0], Milk_bin[1], 1.1]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.005:
          stage = 13
      
      #Stage 13: Go down to the bin
      if stage == 13:
        SetPoint = [Milk_bin[0], Milk_bin[1], Milk_bin[2]]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.005:
          stage = 14
      
      #Stage 14: Drop box, Move up again and change YAW to match Milk
      if stage == 14:
        Can_quat = obs['Can_quat']
        gripper_quat = obs['robot0_eef_quat']
        SetPoint = [Milk_bin[0], Milk_bin[1], 1.1]
        SetQuat = Can_quat
        action = [action_x, action_y, action_z, 0, 0, action_yaw, -0.5]
        distance = SetPoint - gripper_pos
        diff = Set_yaw - robot_yaw

        if np.mean(np.abs(distance)) and np.mean(np.abs(diff))<0.003:
          stage = 15    

    # Execute the action and get the next state, reward, and whether the episode is done
      obs, reward, done, _ = env.step(action)

      #render the environment
      env.render()
      # add a delay to slow down the env render
      time.sleep(dt)
      if done:
         state = env.reset()