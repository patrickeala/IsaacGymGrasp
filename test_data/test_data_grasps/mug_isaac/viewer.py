import numpy as np


data = np.load('main1_rest.npz') 

for key in data:
    print(f"{key}: ",data[key].shape)
    print(f"type {key}", type(data[key]))
    print(f"type {key} element", type(data[key][0][0]))
    # print(data[key])


