# Running the simulation
import torch
# Load YOLOv5 model for the torch hub, and import your weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/howl/robosuite/best.pt')
import robosuite as suite
import pandas as pd
from tempfile import TemporaryFile
import numpy as np

# Run once outsite the loop to define some variables
# Create the environment
env = suite.make(
    "PickPlace",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

rows = 0
while rows < 3:
    # Reset the environment
    obs = env.reset()

    # Get the image from the observation
    image = obs['agentview_image']
    image = np.flipud(image)

    # Crop the image to  desired size and position
    from PIL import Image
    # Convert image to array
    image_arr = np.array(image)

    # Crop image
    image_arr = image_arr[75:170, 0:145]

    # Convert array to image
    image = Image.fromarray(image_arr)

    # Run inference
    results = model(image)

    # Put the bboxes in Dataframe
    df = results.pandas().xyxy[0]
    df = df.drop_duplicates(["class"])
    rows = len(df)

# Get quat info from obs
milk_quat = obs['Milk_quat']
can_quat = obs['Can_quat']
cereal_quat = obs['Cereal_quat']

#convert quaternion to roll, pitch, yaw and changing it to optimal number.
from scipy.spatial.transform import Rotation as R

def quat_to_rpy(q):
    #convert quaternion to rotation matrix
    rpy = R.from_quat(q).as_euler('xyz', degrees=True)
    #convert rotation matrix to euler angles
    return rpy

#Get the YAW for Milk, Can and Cereal
milk_yaw = quat_to_rpy(milk_quat)
milk_yaw = milk_yaw[2]
if milk_yaw > 90:
    milk_yaw -= 180
elif milk_yaw < -90:
    milk_yaw += 180

can_yaw = quat_to_rpy(can_quat)
can_yaw = can_yaw[2]
if can_yaw > 90:
    can_yaw -= 180
elif can_yaw < -90:
    can_yaw += 180

cereal_yaw = quat_to_rpy(cereal_quat)
cereal_yaw = cereal_yaw[2]
if cereal_yaw > 90:
    cereal_yaw -= 180
elif cereal_yaw < -90:
    cereal_yaw += 180

# Getting the pos from simulation and combining with yaw for the labels
milk_pos = obs['Milk_pos'].tolist()
milk_labels = np.hstack((milk_pos, milk_yaw))

can_pos = obs['Can_pos'].tolist()
can_labels = np.hstack((can_pos, can_yaw))

cereal_pos = obs['Cereal_pos'].tolist()
cereal_labels = np.hstack((cereal_pos, cereal_yaw))

# Getting the only th rows from results
milk_bboxes = df.loc[df['class'] == 1]
can_bboxes = df.loc[df['class'] == 2]
cereal_bboxes = df.loc[df['class'] == 0]

# Only picking the wanted columns
milk_bboxes = milk_bboxes[["xmin", "ymin", "xmax", "ymax", "class"]]
can_bboxes = can_bboxes[["xmin", "ymin", "xmax", "ymax", "class"]]
cereal_bboxes = cereal_bboxes[["xmin", "ymin", "xmax", "ymax", "class"]]

# Transforming bboxes to desired from
milk_bboxes = milk_bboxes.values
milk_bboxes = milk_bboxes[0]

can_bboxes = can_bboxes.values
can_bboxes = can_bboxes[0]

cereal_bboxes = cereal_bboxes.values
cereal_bboxes = cereal_bboxes[0]

# Encoder for class
if cereal_bboxes[4] == 0:
    ohe_cereal = np.array([1,0,0])
if milk_bboxes[4] == 1:
    ohe_milk = np.array([0,1,0])
if can_bboxes[4] == 2:
    ohe_can = np.array([0,0,1])

# Stacking DataFrame with encoded class
milk_bbox = np.hstack((milk_bboxes[0:4], ohe_milk))
can_bbox = np.hstack((can_bboxes[0:4], ohe_can))
cereal_bbox = np.hstack((cereal_bboxes[0:4], ohe_cereal))

data = np.vstack((milk_bbox,can_bbox,cereal_bbox))
labels = np.vstack((milk_labels,can_labels,cereal_labels))
images = np.vstack(([image_arr],[image_arr],[image_arr]))

for i in range(300):
    rows = 0
    while rows < 3:
        # Reset the environment
        obs = env.reset()

        # Get the image from the observation
        image = obs['agentview_image']
        image = np.flipud(image)

        # Crop the image to  desired size and position
        from PIL import Image
        # Convert image to array
        image_arr = np.array(image)

        # Crop image
        image_arr = image_arr[75:170, 0:145]

        # Convert array to image
        image = Image.fromarray(image_arr)

        # Run inference
        results = model(image)

        df = results.pandas().xyxy[0]
        df = df.drop_duplicates(["class"])
        rows = len(df)

    # Get quat info from obs
    milk_quat = obs['Milk_quat']
    can_quat = obs['Can_quat']
    cereal_quat = obs['Cereal_quat']

    #convert quaternion to roll, pitch, yaw and changing it to optimal number.
    from scipy.spatial.transform import Rotation as R

    def quat_to_rpy(q):
        #convert quaternion to rotation matrix
        rpy = R.from_quat(q).as_euler('xyz', degrees=True)
        #convert rotation matrix to euler angles
        return rpy

    milk_yaw = quat_to_rpy(milk_quat)
    milk_yaw = milk_yaw[2]
    if milk_yaw > 90:
        milk_yaw -= 180
    elif milk_yaw < -90:
        milk_yaw += 180

    can_yaw = quat_to_rpy(can_quat)
    can_yaw = can_yaw[2]
    if can_yaw > 90:
        can_yaw -= 180
    elif can_yaw < -90:
        can_yaw += 180

    cereal_yaw = quat_to_rpy(cereal_quat)
    cereal_yaw = cereal_yaw[2]
    if cereal_yaw > 90:
        cereal_yaw -= 180
    elif cereal_yaw < -90:
        cereal_yaw += 180

    # Getting the pos from simulation and combining with yaw for the labels
    milk_pos = obs['Milk_pos'].tolist()
    milk_labels = np.hstack((milk_pos, milk_yaw))

    can_pos = obs['Can_pos'].tolist()
    can_labels = np.hstack((can_pos, can_yaw))

    cereal_pos = obs['Cereal_pos'].tolist()
    cereal_labels = np.hstack((cereal_pos, cereal_yaw))

    # Getting the only th rows from results
    milk_bboxes = df.loc[df['class'] == 1]
    can_bboxes = df.loc[df['class'] == 2]
    cereal_bboxes = df.loc[df['class'] == 0]

    # Only picking the wanted columns
    milk_bboxes = milk_bboxes[["xmin", "ymin", "xmax", "ymax", "class"]]
    can_bboxes = can_bboxes[["xmin", "ymin", "xmax", "ymax", "class"]]
    cereal_bboxes = cereal_bboxes[["xmin", "ymin", "xmax", "ymax", "class"]]

    # Transforming bboxes to desired from
    milk_bboxes = milk_bboxes.values
    milk_bboxes = milk_bboxes[0]

    can_bboxes = can_bboxes.values
    can_bboxes = can_bboxes[0]

    cereal_bboxes = cereal_bboxes.values
    cereal_bboxes = cereal_bboxes[0]

    # Encoder for class
    if cereal_bboxes[4] == 0:
        ohe_cereal = np.array([1,0,0])
    if milk_bboxes[4] == 1:
        ohe_milk = np.array([0,1,0])
    if can_bboxes[4] == 2:
        ohe_can = np.array([0,0,1])

    # Stacking DataFrame with encoded class
    milk_bbox = np.hstack((milk_bboxes[0:4], ohe_milk))
    can_bbox = np.hstack((can_bboxes[0:4], ohe_can))
    cereal_bbox = np.hstack((cereal_bboxes[0:4], ohe_cereal))

    data = np.vstack((data,milk_bbox,can_bbox,cereal_bbox))
    labels = np.vstack((labels,milk_labels,can_labels,cereal_labels))
    images = np.vstack((images,[image_arr],[image_arr],[image_arr]))

    if i % 10 == 0:
        print(i++10)

np.savez("vision_data_10.npz", data=data, labels=labels, images=images)
data = np.load('vision_data_10.npz')
print(data['data'])