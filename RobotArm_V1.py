import robosuite as suite
import numpy as np
from robosuite.controllers import load_controller_config

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

env = suite.make(env_name="Lift",
                 robots="Panda",
                 controller_configs=controller_config,
                 has_renderer=True,
                 has_offscreen_renderer=False,
                 use_camera_obs = False)

# reset the environment
obs = env.reset()
print(obs)
# Loop for 100 steps
for i in range(1000):
    diff = np.subtract(obs['robot0_eef_pos'], obs['cube_pos'])
    gripper = obs['robot0_gripper_qpos']
    print(diff)
    #print(diff)
    # create an action that moves the end effector in the x direction by 0.1 
    action = [-diff[0], -diff[1], -diff[2], 0, 0, 0, -1]
    # close gripper

    diff_round = np.round_(diff, decimals = 2)
    if all(item < 0.010  and item > -0.020  for item in diff):
        action = [0, 0, 0, 0, 0, 0, 2]
        print('This is the Robot_Gripper_qpos: ', obs['robot0_gripper_qpos'])
        if all(item < 0.03  and item > -0.03  for item in gripper):
            action = [0, 0, 0.1, 0, 0, 0, 0]
    
    # step the environment
    obs, reward, done, info = env.step(action)

    # render the environment
    env.render()