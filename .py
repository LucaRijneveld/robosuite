#%%
import torch

# Load YOLOv5 model for the torch hub, and import your weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/hive/robosuite/best.pt')

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

# Create the environment
env = suite.make(
    "PickPlace",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True, 
    camera_names="agentview",                 
    camera_segmentations = 'element', # if you want segmented images as well,
)

# Reset the environment
obs = env.reset()

# Get the image from the environment
img= adjust_convention(obs['agentview_image'], 1)
image = img[60:170, 0:135]

# Run inference
results = model(image)

# Show the results
results.show()

# Print the results
results.xyxy[0]
# %%
