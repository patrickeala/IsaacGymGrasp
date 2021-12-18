import numpy as np
import json
from pathlib import Path

def get_scale(path_to_json):
    with open(path_to_json) as json_file:
        info = json.load(json_file)
        return  info["scale"]

def load_grasp_data(filename):
    data = np.load(filename)
    is_promising = data['is_promising'].squeeze()
    return data['quaternions'], data['translations'], data['obj_pose_relative'], is_promising

def load_as_urdf(obj_name=None, obj_path=None, texture_obj_path=None, mass=3, scale=None):

        if not texture_obj_path:
            texture_obj_path = obj_path

        print(f'Scaling object mesh with {scale}.')

        urdf_txt = f"""<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="{obj_name}">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{texture_obj_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.0000" ixy="0.0" ixz="0.0" iyy="0.000" iyz="0.0" izz="0.0000"/>
    </inertial>
  </link>
</robot>
    """

        save_file = f'{obj_name}.urdf'
        Path('temp_urdf').mkdir(parents=True, exist_ok=True)
        save_path = f'temp_urdf/{save_file}'
        urdf_file = open(save_path, 'w')
        urdf_file.write(urdf_txt)

        return save_file