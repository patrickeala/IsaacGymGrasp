'''
For each category:
    1. Open each bash file named "experiment_$obj.bash" 
    2. Modify the experiment index "for j in $experiment_index"
    3. Modify the cuda to run if needed to distribute the gpu usage

3. Open the file "experiment_labeller_new.py" and modify the filename of raw grasps file to load and the filename of the labelledgrasps file
    (current setup is "grasps_graspnet_final.npz" and "grasps_graspnet_final_labelled.npz")
4. run "bash experiment_$obj.bash" for each category





'''