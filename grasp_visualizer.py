from __future__ import print_function
from pickle import NONE

import mayavi.mlab as mlab
import tf_utils
import numpy as np
import sample
import trimesh
import pickle5 as pickle   
from scipy.spatial.transform import Rotation as R


np.random.seed(10)

def get_color_plasma_org(x):
    import matplotlib.pyplot as plt
    return tuple([x for i, x in enumerate(plt.cm.plasma(x)) if i < 3])

def get_color_plasma(x):
    return tuple([float(1 - x), float(x) , float(0)])

def plot_mesh(mesh):
    assert type(mesh) == trimesh.base.Trimesh
    mlab.triangular_mesh (
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        mesh.faces,
        colormap='Blues'
    )
def draw_scene(
    pc, 
    grasps=[], 
    grasp_scores=None, 
    grasp_color=None, 
    gripper_color=(0,1,0), 
    mesh=None, 
    show_gripper_mesh=False, 
    grasps_selection=None, 
    visualize_diverse_grasps=False, 
    min_seperation_distance=0.03, 
    pc_color=None, 
    plasma_coloring=False):
    """
    Draws the 3D scene for the object and the scene.
    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
        grasp_scores: grasps will be colored based on the scores. If left 
        empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      mesh: If not None, shows the mesh of the object. Type should be trimesh 
         mesh.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp. 
      grasp_selection: if provided, filters the grasps based on the value of 
        each selection. 1 means select ith grasp. 0 means exclude the grasp.
      visualize_diverse_grasps: sorts the grasps based on score. Selects the 
        top score grasp to visualize and then choose grasps that are not within
        min_seperation_distance distance of any of the previously selected
        grasps. Only set it to True to declutter the grasps for better
        visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each 
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizting the 
        pc.
    """
    
    max_grasps = 8000
    grasps = np.array(grasps)

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:

        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0, high=len(grasps), size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]

    
    if mesh is not None:
        if type(mesh) == list:
            for elem in mesh:
                plot_mesh(elem)
        else:
            plot_mesh(mesh)

    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 2], colormap='plasma')
        else:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0.1,0.1,1),scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc_color[:, 0], colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01


    grasp_pc = np.squeeze(tf_utils.get_control_point_tensor(1, False), 0)
    print(grasp_pc.shape)
    grasp_pc[2, 2] = 0.059
    grasp_pc[3, 2] = 0.059

    mid_point = 0.5*(grasp_pc[2, :] + grasp_pc[3, :])

    modified_grasp_pc = []
    modified_grasp_pc.append(np.zeros((3,), np.float32))
    modified_grasp_pc.append(mid_point)
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[4])
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[3])
    modified_grasp_pc.append(grasp_pc[5])

    grasp_pc = np.asarray(modified_grasp_pc)

    def transform_grasp_pc(g):
        output = np.matmul(grasp_pc, g[:3, :3].T)
        output += np.expand_dims(g[:3, 3], 0)

        return output

    
    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        indexes = range(len(grasps))

    print('draw scene ', len(grasps))
    
    selected_grasps_so_far = []
    removed = 0
    

    if grasp_scores is not None:
        min_score = np.min(grasp_scores)        
        max_score = np.max(grasp_scores)
        top5 = np.array(grasp_scores).argsort()[-5:][::-1]

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue
        
        g = grasps[i]
        is_diverse = True
        for prevg in selected_grasps_so_far:
            distance = np.linalg.norm(prevg[:3, 3] - g[:3, 3])
    
            if distance < min_seperation_distance:
                is_diverse = False
                break
        
        if visualize_diverse_grasps:
            if not is_diverse:
                removed += 1
                continue
            else:
                if grasp_scores is not None:
                    print('selected', i, grasp_scores[i], min_score, max_score)
                else:
                    print('selected', i)
                selected_grasps_so_far.append(g)

        if isinstance(gripper_color, list):
            pass
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] - min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None:
                gripper_color = grasp_color[ii]
            else:
                gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                gripper_color = (0.0, 1.0, 0.0)

    
        if show_gripper_mesh:
            gripper_mesh = sample.Object('panda_gripper.obj').mesh
            gripper_mesh.apply_transform(g)
            mlab.triangular_mesh(
                gripper_mesh.vertices[:, 0],
                gripper_mesh.vertices[:, 1],
                gripper_mesh.vertices[:, 2],
                gripper_mesh.faces,
                color=gripper_color,
                opacity=1 if visualize_diverse_grasps else 0.5
            )
        else:
            pts = np.matmul(grasp_pc, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)
            if isinstance(gripper_color, list):
                mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=gripper_color[i], tube_radius=0.003, opacity=1)
            else:
                tube_radius = 0.001                    
                mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=gripper_color, tube_radius=tube_radius, opacity=1)

    print('removed {} similar grasps'.format(removed))  

def get_axis():
    # hacky axis for mayavi
    axis = np.array([[1,0,0], [0,1,0], [0,0,1]])
    axis_x = np.array([np.linspace(0, 0.10, 50), np.zeros(50), np.zeros(50)]).T
    axis_y = np.array([np.zeros(50), np.linspace(0, 0.10, 50), np.zeros(50)]).T
    axis_z = np.array([np.zeros(50), np.zeros(50), np.linspace(0, 0.10, 50)]).T
    axis = np.concatenate([axis_x, axis_y, axis_z], axis=0)
    return axis


category = "mug"
idx = 1
trial = 1


_object_pcl = f"assets/grasp_data/pcls/{category}/{category}{idx:03}.pkl"



with open(_object_pcl, "rb") as f:
    object_pcl = pickle.load(f)
    object_pcl = np.array(object_pcl)
    print(type(object_pcl))
object_pcl = object_pcl[1:5]
all_pcl = object_pcl[0]
for pcl in object_pcl:
    all_pcl = np.concatenate([all_pcl,pcl])
print("pcl shape", all_pcl.shape)

_grasps_data_rest = f"grasp_data_generated/{category}/{category}{idx:03}_isaac/main{trial}_rest.npz"
_grasps_data_main = f"grasp_data_generated/{category}/{category}{idx:03}_isaac/main{trial}.npz"
# _realsense_pcl = "ori_pcl_sugarbox.pkl"
# with open(_realsense_pcl, "rb") as f:
#     realsense_pcl = pickle.load(f,encoding="latin")[:,:3]

# all_pcl = np.concatenate((object_pcl, realsense_pcl), axis=0)

pcl_color = np.array([[0.2,9.3,0.4] for _ in range(len(object_pcl))])

grasps_data_rest = np.load(_grasps_data_rest) 
grasps_data_main = np.load(_grasps_data_main) 
grasps_data = {}
for key in grasps_data_rest:
    grasps_data[key] = np.concatenate([grasps_data_main[key],grasps_data_rest[key]])

pos_indcs = np.argwhere(grasps_data['isaac_labels'] == 1).squeeze()
print(f"num of positives {len(pos_indcs)} out of {len(grasps_data['isaac_labels'])}")
neg_indcs = np.argwhere(grasps_data['isaac_labels'] == 0).squeeze()
pos_indcs = np.random.choice(pos_indcs, size=50)
neg_indcs = np.random.choice(neg_indcs, size=50)

all_indcs = list(pos_indcs) + list(neg_indcs)

# grasps_data = np.load("test_bowl.npz")
grasps = []
qs = grasps_data["quaternions"][all_indcs]
ts = grasps_data["translations"][all_indcs]
labels = grasps_data["isaac_labels"][all_indcs]
# qs = grasps_data["quaternions"]
# ts = grasps_data["translations"]
# labels = grasps_data["is_promising"]
scores = []
for q,t,label in zip(qs,ts,labels):
    scores.append(label)
    grasp = np.eye(4)
    _r = R.from_quat(q)
    _mat = _r.as_matrix()
    grasp[:3,:3] = _mat
    grasp[:3,3] = t
    grasps.append(grasp)


# print("pcl", object_pcl.shape)
# print(object_pcl)
# exit()
mlab.figure(bgcolor=(1,1,1))
draw_scene(pc=all_pcl,grasps=grasps,grasp_scores=scores)
mlab.show()
