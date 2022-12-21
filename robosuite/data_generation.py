import matplotlib.pyplot as plt
import numpy as np

import robosuite as suite

# creating a robosuite environment instance
env = suite.make(env_name="Stack", # try with other tasks like "Stack","Door", "PickPlace"
                robots="Panda",   # try with other robots like "Sawyer" and "Jaco"
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,                  
                camera_segmentations = 'element' # if you want segmented images as well
                )

obs = env.reset() 
print(obs)

plt.imshow(agentview_images)