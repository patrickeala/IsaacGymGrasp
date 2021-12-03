import json
import os


data_dir = "/home/user/isaacgym/python/IsaacGymGrasp/shapenet_training_data"
# os.chdir(data_dir)
# print(os.getcwd())

for object in os.listdir(data_dir):
    print(object)
    for sample in os.listdir(f'{data_dir}/{object}'):
        filename = f'{data_dir}/{object}/{sample}'

        with open(filename) as json_file:
            data = json.load(json_file)

        a = data['path'].split('.', 1)
        new_path = f"{a[0]}_{a[1]}"
        data['path'] = new_path

        with open(filename, 'w') as fp:
            json.dump(data, fp)