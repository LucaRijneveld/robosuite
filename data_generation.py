#%%
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite

import igibson

import os
from collections import OrderedDict

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.viewer import Viewer
from igibson.utils.constants import MAX_CLASS_COUNT
from igibson.utils.mesh_util import ortho, quat2rotmat, xyzw2wxyz

import robosuite.utils.macros as macros
from robosuite.renderers import load_renderer_config
from robosuite.renderers.base import Renderer
from robosuite.renderers.igibson.igibson_utils import TensorObservable, adjust_convention
from robosuite.renderers.igibson.parser import Parser
from robosuite.utils import macros
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.observables import sensor
from robosuite.wrappers import Wrapper



# creating a robosuite environment instance
env = suite.make(env_name="Stack", # try with other tasks like "Stack","Door", "PickPlace"
                robots="Panda",   # try with other robots like "Sawyer" and "Jaco"
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True, 
                camera_names="agentview",                 
                camera_segmentations = 'element', # if you want segmented images as well
                )

                
for i in range(600):
    print(i)
    obs = env.reset()
    img = adjust_convention(obs['agentview_image'], 1) #This flips the image using Igibson which is part of Robosuite I think??? IDK! But you can probably just use Numpy to do this part... Don't spent 4 hours trying to install igibson... it's not worth it... trust me.
    plt.imsave("/home/hive/robosuite/Blocks/{}.jpg".format(i), img)

# %%
