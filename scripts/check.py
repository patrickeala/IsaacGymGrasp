# check is_promising
import numpy as np
data_path = 'grasp_data_generated'
cat = 'box'
trial = 1
idx = 0
obj_name = f'{cat}{idx:03}'
fname = f'{data_path}/{cat}/{obj_name}/main{trial}.npz'

# gather the data
quaternions, translations, is_promising = [], [], []
for _ind in indices:
    _quaternions, _translations, obj_pose_relative, _is_promising = utils.load_grasp_data(fname)
    quaternions.append(_quaternions)
    translations.append(_translations)
    is_promising.append(_is_promising)

quaternions = np.concatenate(quaternions)
translations = np.concatenate(translations)
is_promising = np.concatenate(is_promising)

promising_indices = np.argwhere(is_promising==1).squeeze()