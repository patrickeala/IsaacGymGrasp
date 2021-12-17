import pickle5 as pickle 
import os 
import numpy as np

target_dir = "data_labelled"
source_dir = "isaac_test_labels/box"

def combine(source,target):
    with open(target,'rb') as fh:
        target_data = pickle.load(fh)
    with open(source,'rb') as fs:
        source_data = pickle.load(fs)
    print("before: ", len(target_data['translations']))
    print("before: ", len(target_data['isaac_labels']))
    target_data['translations'] = np.append(target_data['translations'],source_data['translations'], 0)
    target_data['quaternions'] = np.append(target_data['quaternions'],source_data['quaternions'], 0)
    target_data['isaac_labels'] = np.append(target_data['isaac_labels'],source_data['isaac_labels'], 0)
    print("after: ", len(target_data['isaac_labels']))
    print("after: ", len(target_data['translations']))
    with open(target, 'wb') as handle:
        pickle.dump(target_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

for source_file_name in os.listdir(f'{source_dir}'):
    obj_name = source_file_name.split('_')[0]
    source = f'{source_dir}/{source_file_name}'
    target = f'{target_dir}/{obj_name}.pkl'
    combine(source,target)


