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

#YAW CONTROLLER
#PID gains Yaw
yaw_kp = 0.1
yaw_ki = 0
yaw_kd = 0

# PID variables yaw
integral_error_yaw = 0
previous_error_yaw = 0

previous_distance = 0

i = 0
stage = 0
done = False
Cereal_pos = obs['Cereal_pos']
SetPoint = [Cereal_pos[0], Cereal_pos[1], 1.1]
Cereal_quat = obs['Cereal_quat']
SetQuat = Cereal_quat
gripper_pos_default = np.array([-0.04378427, -0.08483982,  1.00846165])
gripper_quat_default = np.array([ 0.997486  ,  0.02507683,  0.06619741, -0.00327562])

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

#This Section will be Gripper to Cereal Box to Bin
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

      #Stage 2: Go to Cereal
      if stage == 2:
        SetPoint = [Cereal_pos[0], Cereal_pos[1], Cereal_pos[2]+ 0.04]
        distance = SetPoint - gripper_pos
        action = [action_x, action_y, action_z, 0, 0, 0, 0]

        if np.mean(np.abs(distance))<0.010:
          stage = 3

      previous_distance = distance
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

        #Stage 3: Grip
      if stage == 3:
        action = [0,0,0,0,0,0,0.3]
        #Using a helper function to send the gripping action to the environment for 30 time steps
        obs = delay(30,action,env)
        #Advance to the next stage
        stage = 4
    
      # Stage 4: Lift
      if stage == 4:
        SetPoint = [gripper_pos[0], gripper_pos[1], 1.25]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.008:
          stage = 5
      
      #Stage 5: Reset Arm
      if stage == 5:
        SetPoint = gripper_pos_default
        SetQuat = gripper_quat_default
        action = [action_x, action_y, 0, 0, 0, action_yaw, 0.1]
        distance = SetPoint - gripper_pos
        diff = Set_yaw - robot_yaw

        if np.mean(np.abs(distance))<0.10 and np.mean(np.abs(diff))<0.05:
          stage = 6
      
      
      # Stage 6: Move gripper to Cereal Bin
      if stage == 6:
        SetPoint = [Cereal_bin[0], Cereal_bin[1], 1.1]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.025:
          stage = 7
      
      #Stage 7: Go down to the bin
      if stage == 7:
        SetPoint = [Cereal_bin[0], Cereal_bin[1], Cereal_bin[2]]
        action = [action_x, action_y, action_z, 0, 0, 0, 0]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.012:
          stage = 8

      #Stage 8: Let Go of Box
      if stage == 8:
        action = [0, 0, 0, 0, 0, 0, -1]
        obs = delay(30,action,env)
        stage = 9

      #Stage 9: Reset Arm
      if stage == 9:
        SetPoint = gripper_pos_default
        SetQuat = gripper_quat_default
        action = [action_x, action_y, action_z, 0, 0, action_yaw, 0]
        distance = SetPoint - gripper_pos
        diff = Set_yaw - robot_yaw

        if np.mean(np.abs(distance))<0.10 and np.mean(np.abs(diff))<0.05:
          stage = 10

#From here we are going to try and make the Gripper move from default to Milk to Milk Bin
      
      #Stage 10: Gripper YAW
      if stage == 10:
        Milk_quat = obs['Milk_quat']
        SetQuat = Milk_quat
        action = [0, 0, 0, 0, 0, action_yaw, 0]
        diff = Set_yaw - robot_yaw

        #if diff does not decreas significantly, move to next stage
        if np.mean(np.abs(diff))<0.005:
          stage = 11
      
      #Stage 11: Gripper to Milk
      if stage == 11:
        Milk_pos = obs['Milk_pos']
        SetPoint = [Milk_pos[0], Milk_pos[1], Milk_pos[2], 1.25]
        distance = SetPoint - gripper_pos
        action = [action_x, action_y, action_z, 0, 0, 0, 0]
        
        ##if the mean_error does not decreas significantly, move to next stage
        if np.mean(np.abs(distance))<0.003:
          stage = 12

      #Stage 12: Go to Milk
      if stage == 12:
        SetPoint = [Milk_pos[0], Milk_pos[1], Milk_pos[2]+ 0.04]
        distance = SetPoint - gripper_pos
        action = [action_x, action_y, action_z, 0, 0, 0, 0]

        if np.mean(np.abs(distance))<0.010:
          stage = 13
      
      #Stage 13: Grip
      if stage == 13:
        action = [0,0,0,0,0,0,0.3]
        #Using a helper function to send the gripping action to the environment for 30 time steps
        obs = delay(30,action,env)
        #Advance to the next stage
        stage = 14
    
      # Stage 14: Lift
      if stage == 14:
        SetPoint = [gripper_pos[0], gripper_pos[1], 1.25]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.008:
          stage = 15
      
      #Stage 15: Reset Arm
      if stage == 15:
        SetPoint = gripper_pos_default
        SetQuat = gripper_quat_default
        action = [action_x, action_y, 0, 0, 0, action_yaw, 0.1]
        distance = SetPoint - gripper_pos
        diff = Set_yaw - robot_yaw

        if np.mean(np.abs(distance))<0.10 and np.mean(np.abs(diff))<0.05:
          stage = 16
      
      
      # Stage 16: Move gripper to Milk Bin
      if stage == 16:
        SetPoint = [Milk_bin[0], Milk_bin[1], 1.1]
        action = [action_x, action_y, action_z, 0, 0, 0, 0.1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.025:
          stage = 17
      
      #Stage 17: Go down to the bin
      if stage == 17:
        SetPoint = [Milk_bin[0], Milk_bin[1], Milk_bin[2]]
        action = [action_x, action_y, action_z, 0, 0, 0, 1]
        distance = SetPoint - gripper_pos

        if np.mean(np.abs(distance))<0.012:
          stage = 18

      #Stage 18: Reset Arm
      if stage == 18:
        SetPoint = gripper_pos_default
        SetQuat = gripper_quat_default
        action = [action_x, action_y, action_z, 0, 0, action_yaw, 0.1]
        distance = SetPoint - gripper_pos
        diff = Set_yaw - robot_yaw

        if np.mean(np.abs(distance))<0.10 and np.mean(np.abs(diff))<0.05:
          stage = 19

    # Execute the action and get the next state, reward, and whether the episode is done
      obs, reward, done, _ = env.step(action)

      #render the environment
      env.render()
      # add a delay to slow down the env render
      time.sleep(dt)
      if done:
         state = env.reset()