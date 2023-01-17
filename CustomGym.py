import gym
from gym import spaces
import numpy as np
import robosuite as suite

class RoboEnv(gym.Env):
    def __init__(self, RenderMode = True, Task = 'Lift'): # Add any arguments you need (Environment settings; Render mode  and task are used as examples)
        super(RoboEnv, self).__init__()
        # Initialize environment variables
        self.RenderMode = RenderMode
        self.Task = Task

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low = -1, high = 1, shape=(8,))
        # Example for using image as input:
        self.observation_space = spaces.Box(low = -np.Inf, high = np.Inf,
                                            shape=(35, ), dtype=np.float64)

        # Instantiate the environment
        self.env = suite.make(env_name= self.Task, 
                                robots="Panda",
                                has_renderer=self.RenderMode,
                                has_offscreen_renderer=False,
                                horizon=200,    
                                use_camera_obs=False,)


    def step(self, action):
        # Execute one time step within the environment
        # action = self.env.actionsample()
        # Call the environment step function
        obs, reward, done, _ = self.env.step(action)
        gripper_pos = obs['robot0_eef_pos']
        # You may find it useful to create helper functions for the following
        obs = np.hstack((obs['robot0_proprio-state'],self.targetposition))
        reward = 1 / np.linalg.norm(self.targetposition - gripper_pos)
        #done = self.targetposition
        return obs, reward, done, _

    def reset(self):
        # Reset the state of the environment to an initial state
        # Call the environment reset function
        obs = self.env.reset()
        self.targetposition = np.array([np.random.randint(-5, 5) / 10, np.random.randint(-5, 5) / 10, np.random.randint(8, 13) / 10])
        obs = np.hstack((obs['robot0_proprio-state'], self.targetposition))
        # Process the observation if needed
        # Reset any variables that need to be reset
        return obs

    def render(self, mode='human'):
        # Render the environment to the screen
        # Call the environment render function
        self.env.render()

    def close (self):
        #Close the environment
        # Call the environment close function
        obs = self.env.close()
        return obs